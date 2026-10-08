// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHInterpolation.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/SPHInterpolation.hpp"
#include "shammodels/sph/modules/render/exact_integ_helpers.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace {

    using namespace shammodels::sph::modules::details;

    /// index of the per particle data of the interpolation in a sycl::vec<f64, 8>
    enum InterpPartData : int { IpH = 0, IpInvH, IpHHH, IpInvHHH, IpRho, IpInvRho, IpSupport2 };

    /// Bit identical equivalent of `Kernel::W_3d(r, h)` (`norm_3d * f(r / h) / (h * h * h)`)
    /// using the precomputed correctly rounded reciprocals of h and h * h * h
    template<class Kernel>
    inline f64 interp_W_3d(f64 r, const sycl::vec<f64, 8> &pd) {
        f64 q = div_rn(r, pd[IpH], pd[IpInvH]);
        // q >= +0 or NaN, so the M4 kernel function can compare on bit patterns
        return div_rn(
            Kernel::Generator::norm_3d * kernel_f_pos<Kernel>(q), pd[IpHHH], pd[IpInvHHH]);
    }

    /// capacity of the per thread buffer of particles around the interpolation point
    constexpr u32 interp_buf_size = 96;

    /// the warp flushes its buffers once a thread has this many particles in it
    constexpr u32 interp_flush_threshold = 64;

    /// capacity of the per thread queue of leaves found by the traversal
    constexpr u32 interp_leaf_queue_size = 8;

    /// work group size of the column integration kernel
    constexpr u32 interp_group_size = 128;

    /**
     * @brief SPH interpolation at one point per thread, with the threads of a sub-group
     * cooperating to stay converged
     *
     * Each thread computes exactly the same operations, in the same order, as a plain
     * `rtree_for` traversal accumulating every particle whose support contains the point. Only the
     * moment at which each part of the work is done changes:
     *  - (A) the tree is traversed until every thread of the sub-group found a leaf, threads which
     *    already have one keep traversing and queue the next leaves (in traversal order),
     *  - (B) each thread then tests the particles of its oldest leaf (cheap) and buffers the ones
     *    around the point,
     *  - (C) the expensive contributions of the buffered particles are computed by the whole
     *    sub-group at once, in buffer order.
     */
    template<class Tvec, class T, class Kernel, class ParticleLooper>
    inline void interp_warp_cooperative(
        const sycl::nd_item<1> &item,
        u32 npoints,
        const Tvec *__restrict pixel_positions,
        const Tvec *__restrict xyz,
        const shambase::VecComponent<Tvec> *__restrict hpart,
        const T *__restrict partmass_val,
        const sycl::vec<f64, 8> *__restrict part_data,
        const f32 *__restrict hsupport2_up,
        const sycl::vec<f32, 4> *__restrict xyz_rel_f,
        const sycl::vec<f32, 4> *__restrict node_lo_f,
        const sycl::vec<f32, 4> *__restrict node_up_f,
        Tvec center,
        const ParticleLooper &particle_looper,
        const shambase::VecComponent<Tvec> *__restrict hmax,
        T *__restrict render_field,
        u32 *__restrict staging_id,
        u32 *__restrict staging_owner,
        T *__restrict staging_term,
        u32 *__restrict staging_ok,
        Tvec *__restrict points,
        sycl::vec<f32, 4> *__restrict points_f) {

        using Tscal = shambase::VecComponent<Tvec>;

        const auto &traverser = particle_looper.tree_traverser;
        const auto &tree      = traverser.tree_traverser;

        static constexpr u32 tree_depth = std::remove_cvref_t<decltype(traverser)>::tree_depth_max;

        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        auto sg = item.get_sub_group();

        const u32 gid     = item.get_global_linear_id();
        const bool active = gid < npoints;

        T acc = sham::VectorProperties<T>::get_zero();

        Tvec pos_render = pixel_positions[active ? gid : 0];

        // traversal stack (empty for inactive threads)
        std::array<u32, tree_depth> id_stack;
        u32 stack_cursor = tree_depth;
        if (active) {
            stack_cursor           = tree_depth - 1;
            id_stack[stack_cursor] = 0; // On a Karras tree, the root is always 0
        }

        // queue of the leaves found by the traversal, in traversal order
        u32 leaf_queue[interp_leaf_queue_size];
        u32 queue_head = 0;
        u32 queue_cnt  = 0;

        // buffer of the candidate particles (whose support may contain the point), in traversal
        // order
        u32 buf_id[interp_buf_size];
        u32 buf_cnt = 0;

        // Exact test + contribution of the particle id_b at the point p, this is exactly the code
        // of the direct computation. Returns false if the support of the particle does not
        // contain the point
        auto contrib = [&](u32 id_b, const Tvec &p, T &ret) -> bool {
            Tvec dr    = p - xyz[id_b];
            Tscal rab2 = sycl::dot(dr, dr);

            sycl::vec<f64, 8> pd = part_data[id_b];

            // rab2 > h_b * h_b * Rker2 (precomputed), as an integer comparison: rab2 >= +0 and the
            // threshold >= +0, NaN never rejects (as the floating point comparison)
            {
                u64 rb       = f64_bits(rab2);
                u64 hb       = f64_bits(pd[IpSupport2]);
                u64 inf_bits = f64_bits(shambase::get_infty<f64>());
                if (rb > hb && rb <= inf_bits && hb <= inf_bits) {
                    return false;
                }
            }

            Tscal rab = sycl::sqrt(rab2);

            // partmass * val and rho_h(partmass, h_b, hfactd) precomputed per particle
            ret = div_rn_vec(
                partmass_val[id_b] * interp_W_3d<Kernel>(rab, pd), pd[IpRho], pd[IpInvRho]);
            return true;
        };

        // flush computed by this thread only (only used if the buffer is full within a leaf)
        auto flush = [&]() {
            for (u32 i = 0; i < buf_cnt; i++) {
                T term;
                if (contrib(buf_id[i], pos_render, term)) {
                    acc += term;
                }
            }
            buf_cnt = 0;
        };

        const u32 sg_size = sg.get_local_range()[0];
        const u32 lane    = sg.get_local_linear_id();
        const u32 st_base = sg.get_group_linear_id() * sg_size;

        points[st_base + lane] = pos_render;

        // fp32 copy of the point relative to the patch center for the conservative prefilter
        // (kept in local memory, otherwise the compiler rematerializes the conversions in the
        // particle loop)
        {
            Tvec pos_rel = pos_render - center;
            f32 px       = f32(pos_rel.x());
            f32 py       = f32(pos_rel.y());
            f32 pz       = f32(pos_rel.z());

            // w: |f32(p - c)|_1, plus a term covering the fp64 rounding of p in absolute
            // coordinates in the exact node test (only makes the particle prefilter bound larger)
            f32 w = (sycl::fabs(px) + sycl::fabs(py) + sycl::fabs(pz)) * (1.f + 1.f / 1048576.f)
                    + f32(
                        (sycl::fabs(pos_render.x()) + sycl::fabs(pos_render.y())
                         + sycl::fabs(pos_render.z()))
                        * 0x1p-28);

            points_f[st_base + lane] = {px, py, pz, w};
        }

        // Certified fp32 version of the node test
        // `expand_all(hmax * Rkern).contains_asymmetric(p)` (lo <= p && up > p componentwise).
        // Returns 1 if the fp64 test surely returns true, 0 if it surely returns false and 2 if
        // the fp32 evaluation cannot decide (the fp64 test must then be performed).
        // With e = 2^-24, lo_f / up_f the fp32 expanded node box relative to the patch center c,
        // s a bound on |lo_f|_1 + |up_f|_1 (+ the fp64 rounding of the box in absolute
        // coordinates) and p_f, w the same for the point, (p - lo) and (p_f - lo_f) differ by at
        // most e (s + w), so a margin M = 4 e (s + w) (also covering the fp32 rounding of
        // lo_f +- M) decides the comparisons. NaN never decides.
        auto node_test_f32 = [&](u32 node_id) -> u32 {
            sycl::vec<f32, 4> lo = node_lo_f[node_id];
            sycl::vec<f32, 4> up = node_up_f[node_id];
            sycl::vec<f32, 4> pf = points_f[st_base + lane];

            constexpr f32 e = 1.f / 16777216.f; // 2^-24

            f32 m = 4.f * e * (lo.w() + pf.w());

            bool miss = pf.x() < lo.x() - m || pf.y() < lo.y() - m || pf.z() < lo.z() - m
                        || pf.x() > up.x() + m || pf.y() > up.y() + m || pf.z() > up.z() + m;
            if (miss) {
                return 0;
            }

            bool hit = pf.x() > lo.x() + m && pf.y() > lo.y() + m && pf.z() > lo.z() + m
                       && pf.x() < up.x() - m && pf.y() < up.y() - m && pf.z() < up.z() - m;
            if (hit) {
                return 1;
            }

            return 2;
        };

        // flush of the whole sub-group: the buffered candidates of all threads are computed by
        // all the threads (sg_size entries per round, through the staging area), then each
        // thread adds its own results in its buffer order.
        // The entries are staged level by level (entry j of every thread, then entry j + 1, ...)
        // so that in each round every thread only adds a few terms, all the threads together,
        // instead of one thread adding all its consecutive terms alone.
        auto flush_cooperative = [&]() {
            u32 tot    = sycl::reduce_over_group(sg, buf_cnt, sycl::plus<u32>{});
            u32 maxcnt = sycl::reduce_over_group(sg, buf_cnt, sycl::maximum<u32>{});

            // first level with items in the current round, and number of items before it
            u32 lvl_start = 0;
            u32 cnt_start = 0;

            for (u32 r0 = 0; r0 < tot; r0 += sg_size) {
                u32 r1 = r0 + sg_size;

                // walk the levels of this round, calling func(j, w) for my entries in it
                auto for_my_entries = [&](auto &&func) {
                    u32 lvl     = lvl_start;
                    u32 cnt_lvl = cnt_start;
                    while (lvl < maxcnt && cnt_lvl < r1) {
                        u32 part = (buf_cnt > lvl) ? 1 : 0;
                        u32 rank = sycl::exclusive_scan_over_group(sg, part, sycl::plus<u32>{});
                        u32 n    = sycl::reduce_over_group(sg, part, sycl::plus<u32>{});
                        u32 w    = cnt_lvl + rank;
                        if (part && w >= r0 && w < r1) {
                            func(lvl, w - r0);
                        }
                        cnt_lvl += n;
                        lvl++;
                    }
                };

                for_my_entries([&](u32 j, u32 slot) {
                    staging_id[st_base + slot]    = buf_id[j];
                    staging_owner[st_base + slot] = lane;
                });

                sycl::group_barrier(sg);

                if (r0 + lane < tot) {
                    u32 owner = staging_owner[st_base + lane];
                    T term;
                    bool ok = contrib(staging_id[st_base + lane], points[st_base + owner], term);
                    staging_term[st_base + lane] = term;
                    staging_ok[st_base + lane]   = ok;
                }

                sycl::group_barrier(sg);

                for_my_entries([&](u32 j, u32 slot) {
                    if (staging_ok[st_base + slot]) {
                        acc += staging_term[st_base + slot];
                    }
                });

                sycl::group_barrier(sg);

                // advance to the level containing the item r1
                while (lvl_start < maxcnt) {
                    u32 part = (buf_cnt > lvl_start) ? 1 : 0;
                    u32 n    = sycl::reduce_over_group(sg, part, sycl::plus<u32>{});
                    if (cnt_start + n > r1) {
                        break;
                    }
                    cnt_start += n;
                    lvl_start++;
                }
            }

            buf_cnt = 0;
        };

        auto has_nodes = [&]() {
            return stack_cursor < tree_depth;
        };

        while (sycl::any_of_group(sg, has_nodes() || queue_cnt > 0 || buf_cnt > 0)) {

            // (A) traverse until every thread has a leaf or no nodes left
            while (sycl::any_of_group(sg, has_nodes() && queue_cnt == 0)) {
                if (has_nodes() && queue_cnt < interp_leaf_queue_size) {

                    // Pop the top of the stack
                    u32 current_node_id = id_stack[stack_cursor];
                    stack_cursor++;

                    u32 cert = node_test_f32(current_node_id);

                    bool node_hit;
                    if (cert == 2) {
                        // exact test
                        Tscal rint_cell = hmax[current_node_id] * Kernel::Rkern;

                        shammath::AABB<Tvec> node_aabb{
                            traverser.aabb_min[current_node_id],
                            traverser.aabb_max[current_node_id]};

                        node_hit = node_aabb.expand_all(rint_cell).contains_asymmetric(pos_render);
                    } else {
                        node_hit = (cert == 1);
                    }

                    if (node_hit) {
                        if (tree.is_id_leaf(current_node_id)) {
                            leaf_queue[(queue_head + queue_cnt) % interp_leaf_queue_size]
                                = current_node_id;
                            queue_cnt++;
                        } else {
                            u32 lid = tree.get_left_child(current_node_id);
                            u32 rid = tree.get_right_child(current_node_id);

                            id_stack[stack_cursor - 1] = rid;
                            stack_cursor--;

                            id_stack[stack_cursor - 1] = lid;
                            stack_cursor--;
                        }
                    }
                }
            }

            // (B) test the particles of the oldest leaf
            if (queue_cnt > 0) {
                u32 leaf_node = leaf_queue[queue_head];
                queue_head    = (queue_head + 1) % interp_leaf_queue_size;
                queue_cnt--;

                u32 leaf_id = leaf_node - tree.offset_leaf;

                particle_looper.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 id_b) {
                    // Conservative fp32 prefilter: rejects only particles that the exact test
                    // rejects. With e = 2^-24, A = p - x_b, c the patch center:
                    //  - af = f32(p - c) - f32(x_b - c) satisfies |af - A| <= 1.01 e B,
                    //    B = |f32(p - c)|_1 + |f32(x_b - c)|_1 + |af|_1,
                    //  - so ||af|^2 - |A|^2| <= 2.1 e B^2, and the fp32 evaluation r2_32 of |af|^2
                    //    satisfies r2_32 <= |af|^2 (1 + 3.1 e), the fp64 one rab2 >= |A|^2 (1 - 3
                    //    u),
                    // so r2_32 > H_up (1 + 16 e) + 4 e B^2 (+ 1e-30 for fp32 underflows) implies
                    // rab2 > H. NaN / inf (and fp32 overflows) never reject.
                    sycl::vec<f32, 4> xf = xyz_rel_f[id_b];
                    sycl::vec<f32, 4> pf = points_f[st_base + lane];
                    sycl::vec<f32, 3> af{pf.x() - xf.x(), pf.y() - xf.y(), pf.z() - xf.z()};
                    f32 bound = pf.w() + xf.w() + sycl::fabs(af.x()) + sycl::fabs(af.y())
                                + sycl::fabs(af.z());
                    f32 r2f   = sycl::dot(af, af);

                    constexpr f32 e = 1.f / 16777216.f; // 2^-24

                    if (bound < 1e15f
                        && r2f > hsupport2_up[id_b] * (1.f + 16.f * e) + 4.f * e * bound * bound
                                     + 1e-30f) {
                        return;
                    }

                    buf_id[buf_cnt] = id_b;
                    buf_cnt++;

                    if (buf_cnt == interp_buf_size) {
                        flush();
                    }
                });
            }

            // (C) compute the buffered contributions together
            bool lane_done = !has_nodes() && queue_cnt == 0;
            if (sycl::any_of_group(
                    sg, buf_cnt >= interp_flush_threshold || (lane_done && buf_cnt > 0))) {
                flush_cooperative();
            }
        }

        if (active) {
            render_field[gid] += acc;
        }
    }

} // namespace

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHInterpolation<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<Tvec> &interp_points_buf = edges.interp_points.value;
    sham::DeviceBuffer<T> &output_buf                 = edges.interpolated_field.value;

    u32 npoints = interp_points_buf.get_size();
    if (output_buf.get_size() != npoints) {
        output_buf.resize_discard_data(npoints);
    }
    output_buf.fill(sham::VectorProperties<T>::get_zero());

    using u_morton = u32;
    using Tree     = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    Tscal partmass           = edges.gpart_mass.data;
    u32 tree_reduction_level = edges.tree_reduction_level.data;
    sham::DeviceQueue &queue = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched           = shamsys::instance::get_compute_scheduler_ptr();

    part_counts.for_each([&](u64 id, u32 count) {
        if (count == 0) {
            return;
        }

        PatchDataField<Tvec> &pos = edges.positions.get_field(id);
        if (pos.is_empty()) {
            return;
        }

        Tvec bmax = pos.compute_max();
        Tvec bmin = pos.compute_min();

        shammath::AABB<Tvec> aabb(bmin, bmax);

        Tscal infty = std::numeric_limits<Tscal>::infinity();

        aabb.lower[0] = std::nextafter(aabb.lower[0], -infty);
        aabb.lower[1] = std::nextafter(aabb.lower[1], -infty);
        aabb.lower[2] = std::nextafter(aabb.lower[2], -infty);
        aabb.upper[0] = std::nextafter(aabb.upper[0], infty);
        aabb.upper[1] = std::nextafter(aabb.upper[1], infty);
        aabb.upper[2] = std::nextafter(aabb.upper[2], infty);

        u32 obj_cnt = pos.get_obj_cnt();

        Tree tree = Tree::make_empty(dev_sched);
        tree.rebuild_from_positions(pos.get_buf(), obj_cnt, aabb, tree_reduction_level);

        auto &hpart_span = edges.h_part.get_spans().get(id);
        auto &field_span = edges.field_data.get_spans().get(id);
        auto &buf_hpart  = hpart_span.field_ref.get_buf();
        auto &buf_field  = field_span.field_ref.get_buf();

        auto hmax_tree = shamtree::compute_tree_field_max_field<Tscal>(
            tree.structure,
            tree.reduced_morton_set.get_leaf_cell_iterator(),
            shamtree::new_empty_karras_radix_tree_field<Tscal>(),
            buf_hpart);

        auto obj_it = tree.get_object_iterator();

        // per particle quantities independent of the point (same expressions as in the direct
        // computation `partmass * val * W / rho_h(partmass, h_b, hfactd)`, hence same bits)
        sham::DeviceBuffer<T> partmass_val_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<sycl::vec<f64, 8>> part_data_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<f32> hsupport2_up_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<sycl::vec<f32, 4>> xyz_rel_f_buf(obj_cnt, dev_sched);

        Tvec center = (bmin + bmax) / 2;

        sham::kernel_call(
            queue,
            sham::MultiRef{buf_hpart, buf_field, pos.get_buf()},
            sham::MultiRef{partmass_val_buf, part_data_buf, hsupport2_up_buf, xyz_rel_f_buf},
            obj_cnt,
            [partmass, center](
                u32 id_b,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                const Tvec *__restrict xyz,
                T *__restrict partmass_val,
                sycl::vec<f64, 8> *__restrict part_data,
                f32 *__restrict hsupport2_up,
                sycl::vec<f32, 4> *__restrict xyz_rel_f) {
                // position relative to the patch center in fp32 (+ its L1 norm)
                Tvec rel        = xyz[id_b] - center;
                f32 x           = f32(rel.x());
                f32 y           = f32(rel.y());
                f32 z           = f32(rel.z());
                xyz_rel_f[id_b] = {x, y, z, sycl::fabs(x) + sycl::fabs(y) + sycl::fabs(z)};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal h_b = hpart[id_b];

                partmass_val[id_b] = partmass * torender[id_b];

                // same expressions as in the direct computation, and correctly rounded
                // reciprocals of the divisors
                Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);
                Tscal hhh   = h_b * h_b * h_b;
                sycl::vec<f64, 8> pd;
                pd[IpH]         = h_b;
                pd[IpInvH]      = 1 / h_b;
                pd[IpHHH]       = hhh;
                pd[IpInvHHH]    = 1 / hhh;
                pd[IpRho]       = rho_b;
                pd[IpInvRho]    = 1 / rho_b;
                pd[IpSupport2]  = h_b * h_b * Rker2;
                pd[7]           = 0;
                part_data[id_b] = pd;

                // fp32 upper bound of the exact test threshold h_b * h_b * Rker2
                hsupport2_up[id_b] = f32(h_b * h_b * Rker2) * (1.f + 1.f / 1048576.f);
            });

        // fp32 expanded node boxes relative to the patch center for the certified node test
        u32 node_cnt = tree.aabbs.buf_aabb_min.get_size();
        sham::DeviceBuffer<sycl::vec<f32, 4>> node_lo_f_buf(node_cnt, dev_sched);
        sham::DeviceBuffer<sycl::vec<f32, 4>> node_up_f_buf(node_cnt, dev_sched);

        sham::kernel_call(
            queue,
            sham::MultiRef{tree.aabbs.buf_aabb_min, tree.aabbs.buf_aabb_max, hmax_tree.buf_field},
            sham::MultiRef{node_lo_f_buf, node_up_f_buf},
            node_cnt,
            [center](
                u32 id,
                const Tvec *__restrict aabb_min,
                const Tvec *__restrict aabb_max,
                const Tscal *__restrict hmax,
                sycl::vec<f32, 4> *__restrict node_lo_f,
                sycl::vec<f32, 4> *__restrict node_up_f) {
                Tvec lower      = aabb_min[id];
                Tvec upper      = aabb_max[id];
                Tscal rint_cell = hmax[id] * Kernel::Rkern;

                Tvec lo = (lower - center) - rint_cell;
                Tvec up = (upper - center) + rint_cell;

                auto nl1 = [](Tvec v) {
                    return sycl::fabs(v.x()) + sycl::fabs(v.y()) + sycl::fabs(v.z());
                };

                f32 lx = f32(lo.x()), ly = f32(lo.y()), lz = f32(lo.z());
                f32 ux = f32(up.x()), uy = f32(up.y()), uz = f32(up.z());

                f32 s = (sycl::fabs(lx) + sycl::fabs(ly) + sycl::fabs(lz) + sycl::fabs(ux)
                         + sycl::fabs(uy) + sycl::fabs(uz))
                            * (1.f + 1.f / 1048576.f)
                        + f32(
                            (nl1(lower) + nl1(upper) + nl1(center) + 6 * sycl::fabs(rint_cell))
                            * 0x1p-28);

                node_lo_f[id] = {lx, ly, lz, s};
                node_up_f[id] = {ux, uy, uz, 0.f};
            });

        u32 group_cnt     = shambase::group_count(npoints, interp_group_size);
        u32 corrected_len = group_cnt * interp_group_size;

        sham::kernel_call_hndl(
            queue,
            sham::MultiRef{
                interp_points_buf,
                pos.get_buf(),
                buf_hpart,
                partmass_val_buf,
                part_data_buf,
                hsupport2_up_buf,
                xyz_rel_f_buf,
                node_lo_f_buf,
                node_up_f_buf,
                obj_it,
                hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            npoints,
            [=](u32,
                const Tvec *__restrict pixel_positions,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict partmass_val,
                const sycl::vec<f64, 8> *__restrict part_data,
                const f32 *__restrict hsupport2_up,
                const sycl::vec<f32, 4> *__restrict xyz_rel_f,
                const sycl::vec<f32, 4> *__restrict node_lo_f,
                const sycl::vec<f32, 4> *__restrict node_up_f,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                return [=](sycl::handler &cgh) {
                    sycl::local_accessor<u32> staging_id{interp_group_size, cgh};
                    sycl::local_accessor<u32> staging_owner{interp_group_size, cgh};
                    sycl::local_accessor<T> staging_term{interp_group_size, cgh};
                    sycl::local_accessor<u32> staging_ok{interp_group_size, cgh};
                    sycl::local_accessor<Tvec> points{interp_group_size, cgh};
                    sycl::local_accessor<sycl::vec<f32, 4>> points_f{interp_group_size, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<1>{corrected_len, interp_group_size},
                        [=](sycl::nd_item<1> item) {
                            interp_warp_cooperative<Tvec, T, Kernel>(
                                item,
                                npoints,
                                pixel_positions,
                                xyz,
                                hpart,
                                partmass_val,
                                part_data,
                                hsupport2_up,
                                xyz_rel_f,
                                node_lo_f,
                                node_up_f,
                                center,
                                particle_looper,
                                hmax,
                                render_field,
                                &(staging_id[0]),
                                &(staging_owner[0]),
                                &(staging_term[0]),
                                &(staging_ok[0]),
                                &(points[0]),
                                &(points_f[0]));
                        });
                };
            });
    });

    shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
}

template<class Tvec, class T, template<class> class SPHKernel>
std::string shammodels::sph::modules::SPHInterpolation<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHInterpolation<f64_3, f64_3, C6>;
