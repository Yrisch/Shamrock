// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHAzymuthalInteg.cpp
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
#include "shammodels/sph/modules/render/SPHAzymuthalInteg.hpp"
#include "shammodels/sph/modules/render/exact_integ_helpers.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace {

    using namespace shammodels::sph::modules::details;

    /// capacity of the per thread buffer of particles intersecting the ring
    constexpr u32 azymuthal_buf_size = 64;

    /// the warp flushes its buffers once a thread has this many particles in it
    constexpr u32 azymuthal_flush_threshold = 24;

    /// capacity of the per thread queue of leaves found by the traversal
    constexpr u32 azymuthal_leaf_queue_size = 8;

    /// work group size of the column integration kernel
    constexpr u32 azymuthal_group_size = 128;

    /**
     * @brief Azymuthal integration of one ring ray per thread, with the threads of a sub-group
     * cooperating to stay converged
     *
     * Each thread computes exactly the same operations, in the same order, as a plain
     * `rtree_for` traversal accumulating every particle intersecting the ring. Only the moment at
     * which each part of the work is done changes:
     *  - (A) the tree is traversed until every thread of the sub-group found a leaf, threads which
     *    already have one keep traversing and queue the next leaves (in traversal order),
     *  - (B) each thread then tests the particles of its oldest leaf (cheap) and buffers the ones
     *    intersecting the ring,
     *  - (C) the expensive contributions of the buffered particles are computed by the whole
     *    sub-group at once, in buffer order.
     */
    template<class Tvec, class T, class Kernel, class ParticleLooper>
    inline void azymuthal_integ_warp_cooperative(
        const sycl::nd_item<1> &item,
        u32 nring_rays,
        const shammath::RingRay<Tvec> *__restrict ring_rays_ptr,
        const Tvec *__restrict xyz,
        const shambase::VecComponent<Tvec> *__restrict hpart,
        const T *__restrict partmass_val,
        const sycl::vec<f64, 8> *__restrict part_data,
        const f32 *__restrict hsupport_up,
        const sycl::vec<f32, 4> *__restrict xyz_rel_f,
        const sycl::vec<f32, 4> *__restrict node_center_f,
        const sycl::vec<f32, 2> *__restrict node_radius_f,
        Tvec center,
        const ParticleLooper &particle_looper,
        const shambase::VecComponent<Tvec> *__restrict hmax,
        T *__restrict render_field,
        u32 *__restrict staging_id,
        u32 *__restrict staging_owner,
        T *__restrict staging_term,
        u32 *__restrict staging_ok,
        Tvec *__restrict ring_ezs,
        sycl::vec<f32, 4> *__restrict ring_f) {

        using Tscal = shambase::VecComponent<Tvec>;

        const auto &traverser = particle_looper.tree_traverser;
        const auto &tree      = traverser.tree_traverser;

        static constexpr u32 tree_depth = std::remove_cvref_t<decltype(traverser)>::tree_depth_max;

        constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

        auto sg = item.get_sub_group();

        const u32 gid     = item.get_global_linear_id();
        const bool active = gid < nring_rays;

        T acc = sham::VectorProperties<T>::get_zero();

        shammath::RingRay<Tvec> ring_ray = ring_rays_ptr[active ? gid : 0];
        Tvec ez                          = ring_ray.get_ez();

        // traversal stack (empty for inactive threads)
        std::array<u32, tree_depth> id_stack;
        u32 stack_cursor = tree_depth;
        if (active) {
            stack_cursor           = tree_depth - 1;
            id_stack[stack_cursor] = 0; // On a Karras tree, the root is always 0
        }

        // queue of the leaves found by the traversal, in traversal order
        u32 leaf_queue[azymuthal_leaf_queue_size];
        u32 queue_head = 0;
        u32 queue_cnt  = 0;

        // buffer of the candidate particles (that may intersect the ring), in traversal order
        u32 buf_id[azymuthal_buf_size];
        u32 buf_cnt = 0;

        // Exact test + contribution of the particle id_b for the ring (center, e_x, e_y, ez,
        // radius), this is exactly the code of the direct computation. Returns false if the
        // particle does not intersect the ring
        auto contrib = [&](u32 id_b,
                           const Tvec &center,
                           const Tvec &e_x,
                           const Tvec &e_y,
                           const Tvec &e_z,
                           Tscal radius,
                           T &ret) -> bool {
            Tvec r_center = center - xyz[id_b];

            Tscal z_val = sycl::dot(r_center, e_z);
            Tscal x_val = sycl::dot(r_center, e_x);
            Tscal y_val = sycl::dot(r_center, e_y);
            Tscal r_val = sycl::sqrt(x_val * x_val + y_val * y_val);

            Tscal delta_r = r_val - radius;

            Tscal rab2_ring = z_val * z_val + delta_r * delta_r;

            sycl::vec<f64, 8> pd = part_data[id_b];

            // rab2_ring > h_b * h_b * Rker2 (precomputed), as an integer comparison: rab2_ring
            // >= +0 and the threshold >= +0, NaN never rejects (as the floating point comparison)
            {
                u64 rb       = f64_bits(rab2_ring);
                u64 hb       = f64_bits(pd[PdSupport2]);
                u64 inf_bits = f64_bits(shambase::get_infty<f64>());
                if (rb > hb && rb <= inf_bits && hb <= inf_bits) {
                    return false;
                }
            }

            Tscal rab = sycl::sqrt(rab2_ring);

            // partmass * val and rho_h(partmass, h_b, hfactd) precomputed per particle
            // TODO: account for curvature
            ret = div_rn_vec(
                partmass_val[id_b] * IntegZGrid<Kernel, 4>::Y_3d(rab, pd), pd[PdRho], pd[PdInvRho]);
            return true;
        };

        // flush computed by this thread only (only used if the buffer is full within a leaf)
        auto flush = [&]() {
            for (u32 i = 0; i < buf_cnt; i++) {
                T term;
                if (contrib(
                        buf_id[i],
                        ring_ray.center,
                        ring_ray.e_x,
                        ring_ray.e_y,
                        ez,
                        ring_ray.radius,
                        term)) {
                    acc += term;
                }
            }
            buf_cnt = 0;
        };

        const u32 sg_size = sg.get_local_range()[0];
        const u32 lane    = sg.get_local_linear_id();
        const u32 st_base = sg.get_group_linear_id() * sg_size;

        // the ring of another thread of the sub-group is read back from the (read only, cached)
        // ring ray buffer, only the computed e_z is kept in local memory
        ring_ezs[st_base + lane] = ez;
        const u32 sg_gid_base    = gid - lane;

        // fp32 copies of the ring frame for the conservative prefilter, and the Lipschitz
        // constant L = sqrt(|e_x|^2 + |e_y|^2 + |e_z|^2) of the distance to the ring with
        // respect to the position (rounded up). They are kept in local memory (4 entries per
        // thread), otherwise the compiler rematerializes the conversions in the particle loop.
        {
            f32 lip_f    = f32(sycl::sqrt(
                               sycl::dot(ring_ray.e_x, ring_ray.e_x)
                               + sycl::dot(ring_ray.e_y, ring_ray.e_y) + sycl::dot(ez, ez)))
                           * (1.f + 1.f / 1048576.f);
            f32 radius_f = f32(ring_ray.radius);

            // ring center relative to the patch center in fp32 (+ its L1 norm)
            Tvec center_rel = ring_ray.center - center;
            f32 ox          = f32(center_rel.x());
            f32 oy          = f32(center_rel.y());
            f32 oz          = f32(center_rel.z());

            u32 rfi = 4 * (st_base + lane);
            ring_f[rfi + 0]
                = {f32(ring_ray.e_x.x()), f32(ring_ray.e_x.y()), f32(ring_ray.e_x.z()), lip_f};
            ring_f[rfi + 1]
                = {f32(ring_ray.e_y.x()), f32(ring_ray.e_y.y()), f32(ring_ray.e_y.z()), radius_f};
            ring_f[rfi + 2] = {f32(ez.x()), f32(ez.y()), f32(ez.z()), sycl::fabs(radius_f)};
            ring_f[rfi + 3] = {ox, oy, oz, sycl::fabs(ox) + sycl::fabs(oy) + sycl::fabs(oz)};
        }

        // fp32 squared distance to my ring of the point at af (relative to the ring center) and
        // its error bound eg (see the particle prefilter below for the derivation)
        auto ring_dist2_f32 = [&](sycl::vec<f32, 3> af, f32 bound, f32 &eg, f32 &scale) -> f32 {
            u32 rfi               = 4 * (st_base + lane);
            sycl::vec<f32, 4> ex4 = ring_f[rfi + 0];
            sycl::vec<f32, 4> ey4 = ring_f[rfi + 1];
            sycl::vec<f32, 4> ez4 = ring_f[rfi + 2];

            f32 lip_f        = ex4.w();
            f32 radius_f     = ey4.w();
            f32 radius_abs_f = ez4.w();

            f32 an = sycl::fabs(af.x()) + sycl::fabs(af.y()) + sycl::fabs(af.z());

            f32 xf  = af.x() * ex4.x() + af.y() * ex4.y() + af.z() * ex4.z();
            f32 yf  = af.x() * ey4.x() + af.y() * ey4.y() + af.z() * ey4.z();
            f32 zf  = af.x() * ez4.x() + af.y() * ez4.y() + af.z() * ez4.z();
            f32 rf  = sycl::sqrt(xf * xf + yf * yf);
            f32 drf = rf - radius_f;

            constexpr f32 e = 1.f / 16777216.f; // 2^-24

            scale = lip_f * an + radius_abs_f;
            eg    = e * (lip_f * (2.f * (bound + an) + 16.f * an) + 8.f * radius_abs_f);
            return zf * zf + drf * drf;
        };

        // Certified fp32 version of the node test
        // `expand_all(hmax * Rkern).intersect_ring_ray_approx(ring_ray)`, which compares the
        // distance g(P) of the expanded box center P to the ring with the box bounding radius.
        // Returns 1 if the fp64 test surely returns true, 0 if it surely returns false and 2
        // if the fp32 evaluation cannot decide (the fp64 test must then be performed).
        // node_center_f holds f32(P - c) and in w a bound on |f32(P - c)|_1 that also covers the
        // fp64 rounding of P in absolute coordinates, node_radius_f holds fp32 upper / lower
        // bounds of the bounding radius.
        auto node_test_f32 = [&](u32 node_id) -> u32 {
            u32 rfi               = 4 * (st_base + lane);
            sycl::vec<f32, 4> of4 = ring_f[rfi + 3];
            sycl::vec<f32, 4> pc  = node_center_f[node_id];
            sycl::vec<f32, 2> nr  = node_radius_f[node_id];

            sycl::vec<f32, 3> af{of4.x() - pc.x(), of4.y() - pc.y(), of4.z() - pc.z()};
            f32 bound = of4.w() + pc.w();

            f32 eg, scale;
            f32 g2f = ring_dist2_f32(af, bound, eg, scale);

            constexpr f32 e = 1.f / 16777216.f; // 2^-24

            if (!(bound < 1e15f)) {
                return 2;
            }

            f32 slack = 2e-15f * scale * scale + 1e-30f;

            f32 out = nr.x() + eg;
            if (g2f > out * out * (1.f + 16.f * e) + slack) {
                return 0;
            }

            f32 in = nr.y() - eg;
            if (in > 0 && g2f * (1.f + 16.f * e) + slack < in * in) {
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

                    // the owner has candidates, hence is an active thread
                    const shammath::RingRay<Tvec> &owner_ring = ring_rays_ptr[sg_gid_base + owner];

                    T term;
                    bool ok = contrib(
                        staging_id[st_base + lane],
                        owner_ring.center,
                        owner_ring.e_x,
                        owner_ring.e_y,
                        ring_ezs[st_base + owner],
                        owner_ring.radius,
                        term);
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
                if (has_nodes() && queue_cnt < azymuthal_leaf_queue_size) {

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

                        node_hit
                            = node_aabb.expand_all(rint_cell).intersect_ring_ray_approx(ring_ray);
                    } else {
                        node_hit = (cert == 1);
                    }

                    if (node_hit) {
                        if (tree.is_id_leaf(current_node_id)) {
                            leaf_queue[(queue_head + queue_cnt) % azymuthal_leaf_queue_size]
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
                queue_head    = (queue_head + 1) % azymuthal_leaf_queue_size;
                queue_cnt--;

                u32 leaf_id = leaf_node - tree.offset_leaf;

                particle_looper.cell_iterator.for_each_in_leaf_cell(leaf_id, [&](u32 id_b) {
                    // Conservative fp32 prefilter: rejects only particles that the exact test
                    // rejects. With e = 2^-24, A = center - x_b, c the patch center:
                    //  - af = f32(center - c) - f32(x_b - c) satisfies |af - A| <= 1.01 e B,
                    //    B = |f32(center - c)|_1 + |f32(x_b - c)|_1 + |af|_1,
                    //  - g(A) = |(sqrt(x^2 + y^2) - R, z)| the distance to the ring
                    //    ((x, y, z) = (A.e_x, A.e_y, A.e_z)) is L-Lipschitz in A,
                    //  - the fp32 evaluation from af differs from g(af) by at most
                    //    e (16 L |af|_1 + 8 |R|) (rounding of the frame, of the dot products, of
                    //    the sqrt and of r - R),
                    // so with Eg = e (L (2 B + 16 |af|_1) + 8 |R|),
                    // g2_32 > (sqrt(H) + Eg)^2 (1 + 16 e) + 2e-15 (L |af|_1 + |R|)^2 (the later
                    // bounding the fp64 rounding of the exact test) implies that the fp64 test
                    // rejects the particle. NaN / inf never reject.
                    u32 rfi               = 4 * (st_base + lane);
                    sycl::vec<f32, 4> of4 = ring_f[rfi + 3];

                    sycl::vec<f32, 4> xr = xyz_rel_f[id_b];
                    sycl::vec<f32, 3> af{of4.x() - xr.x(), of4.y() - xr.y(), of4.z() - xr.z()};
                    f32 bound = of4.w() + xr.w();

                    f32 eg, scale;
                    f32 g2f = ring_dist2_f32(af, bound, eg, scale);

                    constexpr f32 e = 1.f / 16777216.f; // 2^-24

                    f32 lim = hsupport_up[id_b] + eg;

                    if (bound < 1e15f
                        && g2f > lim * lim * (1.f + 16.f * e) + 2e-15f * scale * scale + 1e-30f) {
                        return;
                    }

                    buf_id[buf_cnt] = id_b;
                    buf_cnt++;

                    if (buf_cnt == azymuthal_buf_size) {
                        flush();
                    }
                });
            }

            // (C) compute the buffered contributions together
            bool lane_done = !has_nodes() && queue_cnt == 0;
            if (sycl::any_of_group(
                    sg, buf_cnt >= azymuthal_flush_threshold || (lane_done && buf_cnt > 0))) {
                flush_cooperative();
            }
        }

        if (active) {
            render_field[gid] += acc;
        }
    }

} // namespace

template<class Tvec, class T, template<class> class SPHKernel>
void shammodels::sph::modules::SPHAzymuthalInteg<Tvec, T, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts = edges.part_counts.indexes;

    edges.positions.check_sizes(part_counts);
    edges.h_part.check_sizes(part_counts);
    edges.field_data.check_sizes(part_counts);

    const sham::DeviceBuffer<shammath::RingRay<Tvec>> &ring_rays_buf = edges.ring_rays.value;
    sham::DeviceBuffer<T> &output_buf = edges.interpolated_field.value;

    u32 nring_rays = ring_rays_buf.get_size();
    if (output_buf.get_size() != nring_rays) {
        output_buf.resize_discard_data(nring_rays);
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

        // per particle quantities independent of the ring ray (same expressions as in the direct
        // computation `partmass * val * Y / rho_h(partmass, h_b, hfactd)`, hence same bits)
        sham::DeviceBuffer<T> partmass_val_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<sycl::vec<f64, 8>> part_data_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<f32> hsupport_up_buf(obj_cnt, dev_sched);
        sham::DeviceBuffer<sycl::vec<f32, 4>> xyz_rel_f_buf(obj_cnt, dev_sched);

        Tvec center = (bmin + bmax) / 2;

        sham::kernel_call(
            queue,
            sham::MultiRef{buf_hpart, buf_field, pos.get_buf()},
            sham::MultiRef{partmass_val_buf, part_data_buf, hsupport_up_buf, xyz_rel_f_buf},
            obj_cnt,
            [partmass, center](
                u32 id_b,
                const Tscal *__restrict hpart,
                const T *__restrict torender,
                const Tvec *__restrict xyz,
                T *__restrict partmass_val,
                sycl::vec<f64, 8> *__restrict part_data,
                f32 *__restrict hsupport_up,
                sycl::vec<f32, 4> *__restrict xyz_rel_f) {
                Tscal h_b = hpart[id_b];

                // position relative to the patch center in fp32 (+ its L1 norm)
                Tvec rel        = xyz[id_b] - center;
                f32 x           = f32(rel.x());
                f32 y           = f32(rel.y());
                f32 z           = f32(rel.z());
                xyz_rel_f[id_b] = {x, y, z, sycl::fabs(x) + sycl::fabs(y) + sycl::fabs(z)};

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                partmass_val[id_b] = partmass * torender[id_b];

                // same expressions as in the direct computation, and correctly rounded
                // reciprocals of the divisors
                Tscal rho_b = shamrock::sph::rho_h(partmass, h_b, Kernel::hfactd);
                Tscal hh    = h_b * h_b;
                sycl::vec<f64, 8> pd;
                pd[PdH]         = h_b;
                pd[PdInvH]      = 1 / h_b;
                pd[PdHH]        = hh;
                pd[PdInvHH]     = 1 / hh;
                pd[PdRho]       = rho_b;
                pd[PdInvRho]    = 1 / rho_b;
                pd[PdSupport2]  = h_b * h_b * Rker2;
                pd[7]           = 0;
                part_data[id_b] = pd;

                // fp32 upper bound of sqrt(h_b * h_b * Rker2) (the exact test threshold)
                hsupport_up[id_b] = f32(h_b * Kernel::Rkern) * (1.f + 1.f / 1048576.f);
            });

        // fp32 centers and bounding radii of the expanded node boxes, relative to the patch
        // center, for the certified node test
        u32 node_cnt = tree.aabbs.buf_aabb_min.get_size();
        sham::DeviceBuffer<sycl::vec<f32, 4>> node_center_f_buf(node_cnt, dev_sched);
        sham::DeviceBuffer<sycl::vec<f32, 2>> node_radius_f_buf(node_cnt, dev_sched);

        sham::kernel_call(
            queue,
            sham::MultiRef{tree.aabbs.buf_aabb_min, tree.aabbs.buf_aabb_max, hmax_tree.buf_field},
            sham::MultiRef{node_center_f_buf, node_radius_f_buf},
            node_cnt,
            [center](
                u32 id,
                const Tvec *__restrict aabb_min,
                const Tvec *__restrict aabb_max,
                const Tscal *__restrict hmax,
                sycl::vec<f32, 4> *__restrict node_center_f,
                sycl::vec<f32, 2> *__restrict node_radius_f) {
                Tscal rint_cell = hmax[id] * Kernel::Rkern;

                shammath::AABB<Tvec> box
                    = shammath::AABB<Tvec>{aabb_min[id], aabb_max[id]}.expand_all(rint_cell);

                Tvec pc    = box.get_center() - center;
                Tscal brad = box.get_radius();

                auto nl1 = [](Tvec v) {
                    return sycl::fabs(v.x()) + sycl::fabs(v.y()) + sycl::fabs(v.z());
                };

                f32 x = f32(pc.x());
                f32 y = f32(pc.y());
                f32 z = f32(pc.z());

                // |f32(P - c)|_1, plus a term covering the fp64 rounding of P in absolute
                // coordinates in the exact test
                f32 w = (sycl::fabs(x) + sycl::fabs(y) + sycl::fabs(z)) * (1.f + 1.f / 1048576.f)
                        + f32((nl1(box.lower) + nl1(box.upper) + nl1(center)) * 0x1p-28);

                node_center_f[id] = {x, y, z, w};
                node_radius_f[id]
                    = {f32(brad) * (1.f + 1.f / 1048576.f), f32(brad) * (1.f - 1.f / 1048576.f)};
            });

        u32 group_cnt     = shambase::group_count(nring_rays, azymuthal_group_size);
        u32 corrected_len = group_cnt * azymuthal_group_size;

        sham::kernel_call_hndl(
            queue,
            sham::MultiRef{
                ring_rays_buf,
                pos.get_buf(),
                buf_hpart,
                partmass_val_buf,
                part_data_buf,
                hsupport_up_buf,
                xyz_rel_f_buf,
                node_center_f_buf,
                node_radius_f_buf,
                obj_it,
                hmax_tree.buf_field},
            sham::MultiRef{output_buf},
            nring_rays,
            [=](u32,
                const shammath::RingRay<Tvec> *__restrict ring_rays_ptr,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const T *__restrict partmass_val,
                const sycl::vec<f64, 8> *__restrict part_data,
                const f32 *__restrict hsupport_up,
                const sycl::vec<f32, 4> *__restrict xyz_rel_f,
                const sycl::vec<f32, 4> *__restrict node_center_f,
                const sycl::vec<f32, 2> *__restrict node_radius_f,
                auto particle_looper,
                const Tscal *__restrict hmax,
                T *__restrict render_field) {
                return [=](sycl::handler &cgh) {
                    sycl::local_accessor<u32> staging_id{azymuthal_group_size, cgh};
                    sycl::local_accessor<u32> staging_owner{azymuthal_group_size, cgh};
                    sycl::local_accessor<T> staging_term{azymuthal_group_size, cgh};
                    sycl::local_accessor<u32> staging_ok{azymuthal_group_size, cgh};
                    sycl::local_accessor<Tvec> ring_ezs{azymuthal_group_size, cgh};
                    sycl::local_accessor<sycl::vec<f32, 4>> ring_f{4 * azymuthal_group_size, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<1>{corrected_len, azymuthal_group_size},
                        [=](sycl::nd_item<1> item) {
                            azymuthal_integ_warp_cooperative<Tvec, T, Kernel>(
                                item,
                                nring_rays,
                                ring_rays_ptr,
                                xyz,
                                hpart,
                                partmass_val,
                                part_data,
                                hsupport_up,
                                xyz_rel_f,
                                node_center_f,
                                node_radius_f,
                                center,
                                particle_looper,
                                hmax,
                                render_field,
                                &(staging_id[0]),
                                &(staging_owner[0]),
                                &(staging_term[0]),
                                &(staging_ok[0]),
                                &(ring_ezs[0]),
                                &(ring_f[0]));
                        });
                };
            });
    });

    shamalgs::collective::reduce_buffer_in_place_sum(output_buf, MPI_COMM_WORLD);
}

template<class Tvec, class T, template<class> class SPHKernel>
std::string shammodels::sph::modules::SPHAzymuthalInteg<Tvec, T, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, M8>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C2>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64, C6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M6>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, M8>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C2>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C4>;
template class shammodels::sph::modules::SPHAzymuthalInteg<f64_3, f64_3, C6>;
