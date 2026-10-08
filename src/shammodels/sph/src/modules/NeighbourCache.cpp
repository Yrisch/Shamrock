// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NeighbourCache.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/make_ndrange.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/NeighbourCache.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/TreeTraversal.hpp"
#include "shamtree/kernels/geometry_utils.hpp"
#include "shamunits/Constants.hpp"

namespace {

    /// capacity of the per thread queue of leaves found by the traversal
    constexpr u32 nc_leaf_queue_size = 32;

    /**
     * @brief Neighbour search traversal of one particle per thread, with the threads of a
     * sub-group cooperating to stay converged
     *
     * Visits exactly the same particles, in the same order, as a plain `rtree_for` traversal
     * with the `sph_radix_cell_crit` node criterion. Only the moment at which each part of the
     * work is done changes:
     *  - (A) the tree is traversed until every thread of the sub-group found a leaf, threads which
     *    already have one keep traversing and queue the next leaves (in traversal order),
     *  - (B) each thread then calls `on_candidate(id_b)` for the particles of its oldest leaf.
     *
     * `node_test(node_id)` must return the `sph_radix_cell_crit` criterion of the node.
     * `stack` is the thread's slice of the shared memory traversal stack (stack_size entries).
     * Inactive threads (beyond the particle count) take part in the votes with an empty stack.
     */
    template<class ParticleLooper, class FuncNodeTest, class FuncCandidate>
    inline void nc_traverse_warp_cooperative(
        const sycl::nd_item<1> &item,
        bool active,
        u32 *__restrict stack,
        u32 stack_size,
        const ParticleLooper &particle_looper,
        FuncNodeTest &&node_test,
        FuncCandidate &&on_candidate) {

        const auto &traverser = particle_looper.tree_traverser;
        const auto &tree      = traverser.tree_traverser;

        auto sg = item.get_sub_group();

        // traversal stack (empty for inactive threads)
        u32 stack_cursor = stack_size;
        if (active) {
            stack_cursor        = stack_size - 1;
            stack[stack_cursor] = 0; // On a Karras tree, the root is always 0
        }

        // queue of the leaves found by the traversal, in traversal order
        u32 leaf_queue[nc_leaf_queue_size];
        u32 queue_head = 0;
        u32 queue_cnt  = 0;

        auto has_nodes = [&]() {
            return stack_cursor < stack_size;
        };

        while (sycl::any_of_group(sg, has_nodes() || queue_cnt > 0)) {

            // (A) traverse until every thread has a leaf or no nodes left
            while (sycl::any_of_group(sg, has_nodes() && queue_cnt == 0)) {
                if (has_nodes() && queue_cnt < nc_leaf_queue_size) {

                    // Pop the top of the stack
                    u32 current_node_id = stack[stack_cursor];
                    stack_cursor++;

                    bool node_hit = node_test(current_node_id);

                    if (node_hit) {
                        if (tree.is_id_leaf(current_node_id)) {
                            leaf_queue[(queue_head + queue_cnt) % nc_leaf_queue_size]
                                = current_node_id;
                            queue_cnt++;
                        } else {
                            u32 lid = tree.get_left_child(current_node_id);
                            u32 rid = tree.get_right_child(current_node_id);

                            stack[stack_cursor - 1] = rid;
                            stack_cursor--;

                            stack[stack_cursor - 1] = lid;
                            stack_cursor--;
                        }
                    }
                }
            }

            // (B) scan the particles of the oldest leaf
            if (queue_cnt > 0) {
                u32 leaf_node = leaf_queue[queue_head];
                queue_head    = (queue_head + 1) % nc_leaf_queue_size;
                queue_cnt--;

                u32 leaf_id = leaf_node - tree.offset_leaf;

                particle_looper.cell_iterator.for_each_in_leaf_cell(leaf_id, on_candidate);
            }
        }
    }

} // namespace

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::start_neighbors_cache() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_npart{obj_cnt};

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);
        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, obj_it},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                auto particle_looper,
                u32 *__restrict neigh_cnt) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = 0;

                particle_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a,
                            inter_box_a_min,
                            inter_box_a_max,
                            node_aabb.lower,
                            node_aabb.upper,
                            int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};
        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, pcache.scanned_cnt, obj_it},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                u32 *__restrict neigh) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                u32 cnt = scanned_neigh_cnt[id_a];

                particle_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        using namespace walker::interaction_crit;

                        return sph_radix_cell_crit(
                            xyz_a,
                            inter_box_a_min,
                            inter_box_a_max,
                            node_aabb.lower,
                            node_aabb.upper,
                            int_r_max_cell);
                    },
                    [&](u32 id_b) {
                        // compute only omega_a
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
            });

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_shared_offload() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        // a depth first traversal holds at most depth + 1 entries in its stack
        u32 tree_depth = tree.get_exact_tree_depth();
        u32 stack_size = tree_depth + 1;

        shamlog_info_ln("Cache", "patch", patch_id, "tree depth =", tree_depth);

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_npart{obj_cnt};

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, obj_it},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [h_tolerance, stack_size](
                u32 n,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                auto particle_looper,
                u32 *__restrict neigh_cnt) {
                return [=](sycl::handler &cgh) {
                    constexpr Tscal Rker2    = Kernel::Rkern * Kernel::Rkern;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal rint_a = hpart[id_a] * h_tolerance;

                        Tvec xyz_a = xyz[id_a];

                        Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                        Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                        u32 cnt = 0;

                        particle_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(
                                    xyz_a,
                                    inter_box_a_min,
                                    inter_box_a_max,
                                    node_aabb.lower,
                                    node_aabb.upper,
                                    int_r_max_cell);
                            },
                            [&](u32 id_b) {
                                // compute only omega_a
                                Tvec dr      = xyz_a - xyz[id_b];
                                Tscal rab2   = sycl::dot(dr, dr);
                                Tscal rint_b = hpart[id_b] * h_tolerance;

                                bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                   && rab2 > rint_b * rint_b * Rker2;

                                cnt += (no_interact) ? 0 : 1;
                            });

                        neigh_cnt[id_a] = cnt;
                    });
                };
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, tree_field_rint, pcache.scanned_cnt, obj_it},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [h_tolerance, stack_size](
                u32 n,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                u32 *__restrict neigh) {
                return [=](sycl::handler &cgh) {
                    constexpr Tscal Rker2    = Kernel::Rkern * Kernel::Rkern;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal rint_a = hpart[id_a] * h_tolerance;

                        Tvec xyz_a = xyz[id_a];

                        Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                        Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                        u32 cnt = scanned_neigh_cnt[id_a];

                        particle_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                                using namespace walker::interaction_crit;

                                return sph_radix_cell_crit(
                                    xyz_a,
                                    inter_box_a_min,
                                    inter_box_a_max,
                                    node_aabb.lower,
                                    node_aabb.upper,
                                    int_r_max_cell);
                            },
                            [&](u32 id_b) {
                                // compute only omega_a
                                Tvec dr      = xyz_a - xyz[id_b];
                                Tscal rab2   = sycl::dot(dr, dr);
                                Tscal rint_b = hpart[id_b] * h_tolerance;

                                bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                   && rab2 > rint_b * rint_b * Rker2;

                                if (!no_interact) {
                                    neigh[cnt] = id_b;
                                }
                                cnt += (no_interact) ? 0 : 1;
                            });
                    });
                };
            });

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_2stages() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree  = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it  = tree.get_object_iterator();
        auto leaf_it = tree.get_traverser();

        u32 leaf_cnt    = tree.get_leaf_cell_count();
        u32 intnode_cnt = tree.get_internal_cell_count();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_nleaf{leaf_cnt};
        sycl::range range_nobj{obj_cnt};
        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        // start by counting number of leaf neighbours

        sham::DeviceBuffer<u32> neigh_count_leaf(
            leaf_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for Nleaf=", leaf_cnt);

        sham::kernel_call(
            q,
            sham::MultiRef{tree_field_rint, leaf_it},
            sham::MultiRef{neigh_count_leaf},
            leaf_cnt,
            [intnode_cnt](
                u32 id_a,
                const Tscal *__restrict rint_tree,
                auto leaf_looper,
                u32 *__restrict neigh_cnt) {
                u32 offset_leaf = intnode_cnt;

                Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                u32 cnt = 0;

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                        Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                        return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                               || BBAA::cella_neigh_b(
                                   leaf_a_bmin_ext,
                                   leaf_a_bmax_ext,
                                   node_aabb.lower,
                                   node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        cnt++;
                    });

                neigh_cnt[id_a] = cnt;
            });

        //{
        //    u32 offset_leaf = intnode_cnt;
        //    sycl::host_accessor neigh_cnt{neigh_count_leaf};
        //    sycl::host_accessor pos_min_cell
        //    {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt)};
        //    sycl::host_accessor pos_max_cell
        //    {shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt)};
        //
        //    for (u32 i = 0; i < 1000; i++) {
        //        if(neigh_cnt[i] > 30){
        //            logger::raw_ln(i, neigh_cnt[i], pos_max_cell[i+offset_leaf] -
        //            pos_min_cell[i+offset_leaf]);
        //        }
        //    }
        //}

        tree::ObjectCache pleaf_cache
            = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);

        // fill ids of leaf neighbours

        NamedStackEntry stack_loc2{"fill cache"};

        sham::kernel_call(
            q,
            sham::MultiRef{tree_field_rint, pleaf_cache.scanned_cnt, leaf_it},
            sham::MultiRef{pleaf_cache.index_neigh_map},
            leaf_cnt,
            [intnode_cnt](
                u32 id_a,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto leaf_looper,
                u32 *__restrict neigh) {
                u32 offset_leaf = intnode_cnt;

                Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                u32 cnt = scanned_neigh_cnt[id_a];

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                        Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                        Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                        return BBAA::cella_neigh_b(leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                               || BBAA::cella_neigh_b(
                                   leaf_a_bmin_ext,
                                   leaf_a_bmax_ext,
                                   node_aabb.lower,
                                   node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        neigh[cnt] = leaf_b;
                        cnt++;
                    });
            });

        // search in which leaf each parts are
        sham::DeviceBuffer<u32> leaf_part_id(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, leaf_it},
            sham::MultiRef{leaf_part_id},
            obj_cnt,
            [intnode_cnt](
                u32 id_a, const Tvec *__restrict xyz, auto leaf_looper, u32 *__restrict found_id) {
                u32 offset_leaf = intnode_cnt;

                Tvec r_a = xyz[id_a];

                u32 found_id_ = i32_max; // to ensure a crash because of out of bound
                                         // access if not found

                leaf_looper.rtree_for(
                    [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                        return BBAA::is_coord_in_range_incl_max(
                            r_a, node_aabb.lower, node_aabb.upper);
                    },
                    [&](u32 leaf_b) {
                        found_id_ = leaf_b - offset_leaf;
                    });

                SHAM_ASSERT(found_id_ < offset_leaf + 1);

                found_id[id_a] = found_id_;
            });

        //{
        //    sycl::host_accessor xyz{buf_xyz};
        //    sycl::host_accessor acc {leaf_part_id};
        //
        //    for(u32 i = 0; i < obj_cnt; i++){
        //        u32 leaf_id = acc[i];
        //        if(leaf_id >= leaf_cnt){
        //            logger::raw_ln("error : i=",i,"r=",xyz[i],"leaf_id=",leaf_id);
        //        }
        //    }
        //}

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, pleaf_cache, obj_it.cell_iterator, leaf_part_id},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh_cnt) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = 0;

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });
                });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc3{"fill cache"};

        sham::kernel_call(
            q,
            sham::MultiRef{
                buf_xyz,
                buf_hpart,
                pleaf_cache,
                pcache.scanned_cnt,
                obj_it.cell_iterator,
                leaf_part_id},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = scanned_neigh_cnt[id_a];

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
                });
            });
        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_2stages_shared_offload() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree  = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it  = tree.get_object_iterator();
        auto leaf_it = tree.get_traverser();

        // a depth first traversal holds at most depth + 1 entries in its stack
        u32 tree_depth = tree.get_exact_tree_depth();
        u32 stack_size = tree_depth + 1;

        shamlog_info_ln("Cache", "patch", patch_id, "tree depth =", tree_depth);

        u32 leaf_cnt    = tree.get_leaf_cell_count();
        u32 intnode_cnt = tree.get_internal_cell_count();

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_nleaf{leaf_cnt};
        sycl::range range_nobj{obj_cnt};
        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        NamedStackEntry stack_loc1{"init cache"};

        // start by counting number of leaf neighbours

        sham::DeviceBuffer<u32> neigh_count_leaf(
            leaf_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for Nleaf=", leaf_cnt);

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{tree_field_rint, leaf_it},
            sham::MultiRef{neigh_count_leaf},
            leaf_cnt,
            [intnode_cnt, stack_size](
                u32 n,
                const Tscal *__restrict rint_tree,
                auto leaf_looper,
                u32 *__restrict neigh_cnt) {
                return [=](sycl::handler &cgh) {
                    u32 offset_leaf          = intnode_cnt;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                        Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                        Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                        Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                        Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                        u32 cnt = 0;

                        leaf_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                                Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                                Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                                return BBAA::cella_neigh_b(
                                           leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                                       || BBAA::cella_neigh_b(
                                           leaf_a_bmin_ext,
                                           leaf_a_bmax_ext,
                                           node_aabb.lower,
                                           node_aabb.upper);
                            },
                            [&](u32 leaf_b) {
                                cnt++;
                            });

                        neigh_cnt[id_a] = cnt;
                    });
                };
            });

        tree::ObjectCache pleaf_cache
            = tree::prepare_object_cache(std::move(neigh_count_leaf), leaf_cnt);

        // fill ids of leaf neighbours

        NamedStackEntry stack_loc2{"fill cache"};

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{tree_field_rint, pleaf_cache.scanned_cnt, leaf_it},
            sham::MultiRef{pleaf_cache.index_neigh_map},
            leaf_cnt,
            [intnode_cnt, stack_size](
                u32 n,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto leaf_looper,
                u32 *__restrict neigh) {
                return [=](sycl::handler &cgh) {
                    u32 offset_leaf          = intnode_cnt;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tscal leaf_a_rint    = rint_tree[offset_leaf + id_a] * Kernel::Rkern;
                        Tvec leaf_a_bmin     = leaf_looper.aabb_min[offset_leaf + id_a];
                        Tvec leaf_a_bmax     = leaf_looper.aabb_max[offset_leaf + id_a];
                        Tvec leaf_a_bmin_ext = leaf_a_bmin - leaf_a_rint;
                        Tvec leaf_a_bmax_ext = leaf_a_bmax + leaf_a_rint;

                        u32 cnt = scanned_neigh_cnt[id_a];

                        leaf_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                                Tvec ext_bmin = node_aabb.lower - int_r_max_cell;
                                Tvec ext_bmax = node_aabb.upper + int_r_max_cell;

                                return BBAA::cella_neigh_b(
                                           leaf_a_bmin, leaf_a_bmax, ext_bmin, ext_bmax)
                                       || BBAA::cella_neigh_b(
                                           leaf_a_bmin_ext,
                                           leaf_a_bmax_ext,
                                           node_aabb.lower,
                                           node_aabb.upper);
                            },
                            [&](u32 leaf_b) {
                                neigh[cnt] = leaf_b;
                                cnt++;
                            });
                    });
                };
            });

        // search in which leaf each parts are
        sham::DeviceBuffer<u32> leaf_part_id(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call_hndl(
            q,
            sham::MultiRef{buf_xyz, leaf_it},
            sham::MultiRef{leaf_part_id},
            obj_cnt,
            [intnode_cnt, stack_size](
                u32 n, const Tvec *__restrict xyz, auto leaf_looper, u32 *__restrict found_id) {
                return [=](sycl::handler &cgh) {
                    u32 offset_leaf          = intnode_cnt;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a = (u32) item.get_global_linear_id();

                        if (id_a >= n) {
                            return;
                        }

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];
                        auto stack_id  = [&stack_ptr](u32 id) -> u32  &{
                            return stack_ptr[id];
                        };

                        Tvec r_a = xyz[id_a];

                        u32 found_id_ = i32_max; // to ensure a crash because of out of
                                                 // bound access if not found

                        leaf_looper.rtree_for(
                            stack_id,
                            stack_size,
                            [&](u32 node_id, shammath::AABB<Tvec> node_aabb) -> bool {
                                return BBAA::is_coord_in_range_incl_max(
                                    r_a, node_aabb.lower, node_aabb.upper);
                            },
                            [&](u32 leaf_b) {
                                found_id_ = leaf_b - offset_leaf;
                            });

                        SHAM_ASSERT(found_id_ < offset_leaf + 1);

                        found_id[id_a] = found_id_;
                    });
                };
            });

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);

        sham::kernel_call(
            q,
            sham::MultiRef{buf_xyz, buf_hpart, pleaf_cache, obj_it.cell_iterator, leaf_part_id},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh_cnt) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = 0;

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        cnt += (no_interact) ? 0 : 1;
                    });
                });

                neigh_cnt[id_a] = cnt;
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc3{"fill cache"};

        sham::kernel_call(
            q,
            sham::MultiRef{
                buf_xyz,
                buf_hpart,
                pleaf_cache,
                pcache.scanned_cnt,
                obj_it.cell_iterator,
                leaf_part_id},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [intnode_cnt, h_tolerance](
                u32 id_a,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                auto acc_neigh_leaf_looper,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                const u32 *__restrict leaf_owner,
                u32 *__restrict neigh) {
                tree::ObjectCacheIterator neigh_leaf_looper(acc_neigh_leaf_looper);

                u32 offset_leaf = intnode_cnt;

                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint_a = hpart[id_a] * h_tolerance;

                Tvec xyz_a = xyz[id_a];

                u32 cnt = scanned_neigh_cnt[id_a];

                u32 leaf_own_a = leaf_owner[id_a];

                neigh_leaf_looper.for_each_object(leaf_own_a, [&](u32 leaf_b) {
                    SHAM_ASSERT(leaf_b >= offset_leaf);

                    particle_looper.for_each_in_leaf_cell(leaf_b - offset_leaf, [&](u32 id_b) {
                        Tvec dr      = xyz_a - xyz[id_b];
                        Tscal rab2   = sycl::dot(dr, dr);
                        Tscal rint_b = hpart[id_b] * h_tolerance;

                        bool no_interact
                            = rab2 > rint_a * rint_a * Rker2 && rab2 > rint_b * rint_b * Rker2;

                        if (!no_interact) {
                            neigh[cnt] = id_b;
                        }
                        cnt += (no_interact) ? 0 : 1;
                    });
                });
            });
        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

using namespace shammath;
template<class Tvec, class Tmorton, template<class> class SPHKernel>
void shammodels::sph::modules::NeighbourCache<Tvec, Tmorton, SPHKernel>::
    start_neighbors_cache_shared_offload_opt() {

    // interface_control
    using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
    using GhostHandleCache = typename GhostHandle::CacheMap;
    using RTree            = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

    shambase::Timer time_neigh;
    time_neigh.start();

    StackEntry stack_loc{};

    // do cache
    auto build_neigh_cache = [&](u64 patch_id) {
        shamlog_debug_ln("BasicSPH", "build particle cache id =", patch_id);

        NamedStackEntry cache_build_stack_loc{"build cache"};

        auto &mfield = storage.merged_xyzh.get().get(patch_id);

        sham::DeviceBuffer<Tvec> &buf_xyz    = mfield.template get_field_buf_ref<Tvec>(0);
        sham::DeviceBuffer<Tscal> &buf_hpart = mfield.template get_field_buf_ref<Tscal>(1);

        sham::DeviceBuffer<Tscal> &tree_field_rint
            = storage.rtree_rint_field.get().get(patch_id).buf_field;

        RTree &tree = storage.merged_pos_trees.get().get(patch_id);
        auto obj_it = tree.get_object_iterator();

        // a depth first traversal holds at most depth + 1 entries in its stack
        u32 tree_depth = tree.get_exact_tree_depth();
        u32 stack_size = tree_depth + 1;

        shamlog_info_ln("Cache", "patch", patch_id, "tree depth =", tree_depth);

        u32 obj_cnt = shambase::get_check_ref(storage.part_counts).indexes.get(patch_id);

        sycl::range range_npart{obj_cnt};

        Tscal h_tolerance = solver_config.htol_up_coarse_cycle;

        // fp32 lower / upper bounds of the interaction threshold rint * rint * Rker2 of every
        // (merged) particle, computed with the same expression as the exact test
        // and the fp32 positions relative to the center of the tree root box (+ their L1 norm)
        u32 merged_cnt = mfield.get_obj_cnt();
        sham::DeviceBuffer<sycl::vec<f32, 2>> thr_f_buf(
            merged_cnt, shamsys::instance::get_compute_scheduler_ptr());
        sham::DeviceBuffer<sycl::vec<f32, 4>> xyz_rel_f_buf(
            merged_cnt, shamsys::instance::get_compute_scheduler_ptr());

        Tvec center = (tree.aabbs.buf_aabb_min.get_val_at_idx(0)
                       + tree.aabbs.buf_aabb_max.get_val_at_idx(0))
                      / 2;

        // fp32 node boxes relative to the root box center for the certified node test:
        // lo (+ scale bound s in w) and up (+ fp32 box_int_sz = rint_tree * Rkern in w)
        u32 node_cnt = tree.aabbs.buf_aabb_min.get_size();
        sham::DeviceBuffer<sycl::vec<f32, 4>> node_lo_f_buf(
            node_cnt, shamsys::instance::get_compute_scheduler_ptr());
        sham::DeviceBuffer<sycl::vec<f32, 4>> node_up_f_buf(
            node_cnt, shamsys::instance::get_compute_scheduler_ptr());

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler().get_queue(),
            sham::MultiRef{tree.aabbs.buf_aabb_min, tree.aabbs.buf_aabb_max, tree_field_rint},
            sham::MultiRef{node_lo_f_buf, node_up_f_buf},
            node_cnt,
            [center](
                u32 id,
                const Tvec *__restrict aabb_min,
                const Tvec *__restrict aabb_max,
                const Tscal *__restrict rint_tree,
                sycl::vec<f32, 4> *__restrict node_lo_f,
                sycl::vec<f32, 4> *__restrict node_up_f) {
                Tvec lower = aabb_min[id];
                Tvec upper = aabb_max[id];
                Tscal r    = rint_tree[id] * Kernel::Rkern;

                Tvec lo = lower - center;
                Tvec up = upper - center;

                auto nl1 = [](Tvec v) {
                    return sycl::fabs(v.x()) + sycl::fabs(v.y()) + sycl::fabs(v.z());
                };

                f32 lx = f32(lo.x()), ly = f32(lo.y()), lz = f32(lo.z());
                f32 ux = f32(up.x()), uy = f32(up.y()), uz = f32(up.z());
                f32 rf = f32(r);

                // bound on the magnitudes of the compared fp32 values, plus a term covering the
                // fp64 rounding of the exact criterion in absolute coordinates
                f32 sc
                    = (sycl::fabs(lx) + sycl::fabs(ly) + sycl::fabs(lz) + sycl::fabs(ux)
                       + sycl::fabs(uy) + sycl::fabs(uz) + 2 * sycl::fabs(rf))
                          * (1.f + 1.f / 1048576.f)
                      + f32((nl1(lower) + nl1(upper) + nl1(center) + 2 * sycl::fabs(r)) * 0x1p-28);

                node_lo_f[id] = {lx, ly, lz, sc};
                node_up_f[id] = {ux, uy, uz, rf};
            });

        sham::kernel_call(
            shamsys::instance::get_compute_scheduler().get_queue(),
            sham::MultiRef{buf_hpart, buf_xyz},
            sham::MultiRef{thr_f_buf, xyz_rel_f_buf},
            merged_cnt,
            [h_tolerance, center](
                u32 id,
                const Tscal *__restrict hpart,
                const Tvec *__restrict xyz,
                sycl::vec<f32, 2> *__restrict thr_f,
                sycl::vec<f32, 4> *__restrict xyz_rel_f) {
                constexpr Tscal Rker2 = Kernel::Rkern * Kernel::Rkern;

                Tscal rint = hpart[id] * h_tolerance;
                f32 thr    = f32(rint * rint * Rker2);

                thr_f[id] = {thr * (1.f - 1.f / 1048576.f), thr * (1.f + 1.f / 1048576.f)};

                Tvec rel      = xyz[id] - center;
                f32 x         = f32(rel.x());
                f32 y         = f32(rel.y());
                f32 z         = f32(rel.z());
                xyz_rel_f[id] = {x, y, z, sycl::fabs(x) + sycl::fabs(y) + sycl::fabs(z)};
            });

        NamedStackEntry stack_loc1{"init cache"};

        using namespace shamrock;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<u32> neigh_count(
            obj_cnt, shamsys::instance::get_compute_scheduler_ptr());

        shamlog_debug_sycl_ln("Cache", "generate cache for N=", obj_cnt);
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{
                buf_xyz,
                buf_hpart,
                thr_f_buf,
                xyz_rel_f_buf,
                node_lo_f_buf,
                node_up_f_buf,
                tree_field_rint,
                obj_it},
            sham::MultiRef{neigh_count},
            obj_cnt,
            [h_tolerance, stack_size, center](
                u32 n,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const sycl::vec<f32, 2> *__restrict thr_f,
                const sycl::vec<f32, 4> *__restrict xyz_rel_f,
                const sycl::vec<f32, 4> *__restrict node_lo_f,
                const sycl::vec<f32, 4> *__restrict node_up_f,
                const Tscal *__restrict rint_tree,
                auto particle_looper,
                u32 *__restrict neigh_cnt) {
                return [=](sycl::handler &cgh) {
                    constexpr Tscal Rker2    = Kernel::Rkern * Kernel::Rkern;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a    = (u32) item.get_global_linear_id();
                        bool active = id_a < n;

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];

                        Tscal rint_a = hpart[active ? id_a : 0] * h_tolerance;

                        sycl::vec<f32, 2> thr_a = thr_f[active ? id_a : 0];
                        sycl::vec<f32, 4> xa_f  = xyz_rel_f[active ? id_a : 0];

                        Tvec xyz_a = xyz[active ? id_a : 0];

                        Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                        Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                        // Certified fp32 version of the node criterion sph_radix_cell_crit
                        // (search box of a overlaps the cell, or a is in the cell expanded by
                        // box_int_sz): every comparison of the fp64 criterion is decided in fp32
                        // with a margin M = 4 e (s_node + s_a) covering the conversions, the fp32
                        // and fp64 roundings (s_* bound the compared magnitudes, absolute
                        // coordinates included); the exact criterion is used when undecided.
                        f32 ra_f = f32(rint_a * Kernel::Rkern);
                        f32 s_a  = (xa_f.w() + ra_f) * (1.f + 1.f / 1048576.f)
                                   + f32(
                                       (sycl::fabs(xyz_a.x()) + sycl::fabs(xyz_a.y())
                                        + sycl::fabs(xyz_a.z()) + sycl::fabs(center.x())
                                        + sycl::fabs(center.y()) + sycl::fabs(center.z())
                                        + rint_a * Kernel::Rkern)
                                       * 0x1p-28);

                        auto node_test = [&](u32 node_id) -> bool {
                            sycl::vec<f32, 4> lo = node_lo_f[node_id];
                            sycl::vec<f32, 4> up = node_up_f[node_id];

                            constexpr f32 e = 1.f / 16777216.f; // 2^-24
                            f32 m           = 4.f * e * (lo.w() + s_a);
                            f32 r           = up.w();

                            // 0: surely false, 1: surely true, 2: undecided
                            auto tri_and = [](u32 a, u32 b) -> u32 {
                                return (a == 0 || b == 0) ? 0 : ((a == 1 && b == 1) ? 1 : 2);
                            };
                            // a <= b decided with the margin
                            auto leq = [m](f32 a, f32 b) -> u32 {
                                return (a < b - m) ? 1 : ((a > b + m) ? 0 : 2);
                            };

                            u32 overlap = 1;
                            u32 inside  = 1;
                            for (int i = 0; i < 3; i++) {
                                f32 x = xa_f[i];
                                // search box of a: [x - ra, x + ra], cell [lo, up]
                                overlap = tri_and(overlap, leq(x - ra_f, up[i]));
                                overlap = tri_and(overlap, leq(lo[i], x + ra_f));
                                // a in the expanded cell [lo - r, up + r]
                                inside = tri_and(inside, leq(lo[i] - r, x));
                                inside = tri_and(inside, leq(x, up[i] + r));
                            }

                            if (overlap == 1 || inside == 1) {
                                return true;
                            }
                            if (overlap == 0 && inside == 0) {
                                return false;
                            }

                            // exact criterion
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            shammath::AABB<Tvec> node_aabb{
                                particle_looper.tree_traverser.aabb_min[node_id],
                                particle_looper.tree_traverser.aabb_max[node_id]};

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(
                                xyz_a,
                                inter_box_a_min,
                                inter_box_a_max,
                                node_aabb.lower,
                                node_aabb.upper,
                                int_r_max_cell);
                        };

                        u32 cnt = 0;

                        nc_traverse_warp_cooperative(
                            item,
                            active,
                            stack_ptr,
                            stack_size,
                            particle_looper,
                            node_test,
                            [&](u32 id_b) {
                                // Certified fp32 version of the interaction test: with
                                // e = 2^-24, c the root box center, xf = f32(x - c) and
                                // af = xa_f - xb_f, |af - dr| <= 1.01 e B with
                                // B = |xa_f|_1 + |xb_f|_1 + |af|_1, so the fp64 rab2 lies within
                                // r2 [1 -+ 16 e] -+ 4 e B^2 of the fp32 r2 = |af|^2, and thr_f
                                // holds bounds of the fp64 thresholds: the result is decided in
                                // fp32 unless r2 is that close to a threshold (or NaN / inf), in
                                // which case the exact test is performed.
                                sycl::vec<f32, 4> xb_f  = xyz_rel_f[id_b];
                                sycl::vec<f32, 2> thr_b = thr_f[id_b];
                                sycl::vec<f32, 3> af{
                                    xa_f.x() - xb_f.x(), xa_f.y() - xb_f.y(), xa_f.z() - xb_f.z()};
                                f32 bound = xa_f.w() + xb_f.w() + sycl::fabs(af.x())
                                            + sycl::fabs(af.y()) + sycl::fabs(af.z());
                                f32 r2f   = sycl::dot(af, af);

                                constexpr f32 e = 1.f / 16777216.f; // 2^-24

                                f32 slack = 4.f * e * bound * bound + 1e-30f;
                                f32 r2_hi = r2f * (1.f + 16.f * e) + slack;
                                f32 r2_lo = r2f * (1.f - 16.f * e) - slack;

                                bool interact;
                                if (bound < 1e15f && (r2_hi < thr_a.x() || r2_hi < thr_b.x())) {
                                    interact = true;
                                } else if (
                                    bound < 1e15f && r2_lo > thr_a.y() && r2_lo > thr_b.y()) {
                                    interact = false;
                                } else {
                                    // exact test
                                    Tvec dr      = xyz_a - xyz[id_b];
                                    Tscal rab2   = sycl::dot(dr, dr);
                                    Tscal rint_b = hpart[id_b] * h_tolerance;

                                    bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                       && rab2 > rint_b * rint_b * Rker2;
                                    interact         = !no_interact;
                                }

                                cnt += (interact) ? 1 : 0;
                            });

                        if (active) {
                            neigh_cnt[id_a] = cnt;
                        }
                    });
                };
            });

        tree::ObjectCache pcache = tree::prepare_object_cache(std::move(neigh_count), obj_cnt);

        NamedStackEntry stack_loc2{"fill cache"};
        sham::kernel_call_hndl(
            q,
            sham::MultiRef{
                buf_xyz,
                buf_hpart,
                thr_f_buf,
                xyz_rel_f_buf,
                node_lo_f_buf,
                node_up_f_buf,
                tree_field_rint,
                pcache.scanned_cnt,
                obj_it},
            sham::MultiRef{pcache.index_neigh_map},
            obj_cnt,
            [h_tolerance, stack_size, center](
                u32 n,
                const Tvec *__restrict xyz,
                const Tscal *__restrict hpart,
                const sycl::vec<f32, 2> *__restrict thr_f,
                const sycl::vec<f32, 4> *__restrict xyz_rel_f,
                const sycl::vec<f32, 4> *__restrict node_lo_f,
                const sycl::vec<f32, 4> *__restrict node_up_f,
                const Tscal *__restrict rint_tree,
                const u32 *__restrict scanned_neigh_cnt,
                auto particle_looper,
                u32 *__restrict neigh) {
                return [=](sycl::handler &cgh) {
                    constexpr Tscal Rker2    = Kernel::Rkern * Kernel::Rkern;
                    constexpr u32 group_size = 256;

                    sycl::local_accessor<u32, 1> stack_local(stack_size * group_size, cgh);

                    cgh.parallel_for(sham::make_ndrange(group_size, n), [=](sycl::nd_item<1> item) {
                        u32 id_a    = (u32) item.get_global_linear_id();
                        bool active = id_a < n;

                        u32 group_id   = (u32) item.get_local_id(0);
                        u32 *stack_ptr = &stack_local[group_id * stack_size];

                        Tscal rint_a = hpart[active ? id_a : 0] * h_tolerance;

                        sycl::vec<f32, 2> thr_a = thr_f[active ? id_a : 0];
                        sycl::vec<f32, 4> xa_f  = xyz_rel_f[active ? id_a : 0];

                        Tvec xyz_a = xyz[active ? id_a : 0];

                        Tvec inter_box_a_min = xyz_a - rint_a * Kernel::Rkern;
                        Tvec inter_box_a_max = xyz_a + rint_a * Kernel::Rkern;

                        // Certified fp32 version of the node criterion sph_radix_cell_crit
                        // (search box of a overlaps the cell, or a is in the cell expanded by
                        // box_int_sz): every comparison of the fp64 criterion is decided in fp32
                        // with a margin M = 4 e (s_node + s_a) covering the conversions, the fp32
                        // and fp64 roundings (s_* bound the compared magnitudes, absolute
                        // coordinates included); the exact criterion is used when undecided.
                        f32 ra_f = f32(rint_a * Kernel::Rkern);
                        f32 s_a  = (xa_f.w() + ra_f) * (1.f + 1.f / 1048576.f)
                                   + f32(
                                       (sycl::fabs(xyz_a.x()) + sycl::fabs(xyz_a.y())
                                        + sycl::fabs(xyz_a.z()) + sycl::fabs(center.x())
                                        + sycl::fabs(center.y()) + sycl::fabs(center.z())
                                        + rint_a * Kernel::Rkern)
                                       * 0x1p-28);

                        auto node_test = [&](u32 node_id) -> bool {
                            sycl::vec<f32, 4> lo = node_lo_f[node_id];
                            sycl::vec<f32, 4> up = node_up_f[node_id];

                            constexpr f32 e = 1.f / 16777216.f; // 2^-24
                            f32 m           = 4.f * e * (lo.w() + s_a);
                            f32 r           = up.w();

                            // 0: surely false, 1: surely true, 2: undecided
                            auto tri_and = [](u32 a, u32 b) -> u32 {
                                return (a == 0 || b == 0) ? 0 : ((a == 1 && b == 1) ? 1 : 2);
                            };
                            // a <= b decided with the margin
                            auto leq = [m](f32 a, f32 b) -> u32 {
                                return (a < b - m) ? 1 : ((a > b + m) ? 0 : 2);
                            };

                            u32 overlap = 1;
                            u32 inside  = 1;
                            for (int i = 0; i < 3; i++) {
                                f32 x = xa_f[i];
                                // search box of a: [x - ra, x + ra], cell [lo, up]
                                overlap = tri_and(overlap, leq(x - ra_f, up[i]));
                                overlap = tri_and(overlap, leq(lo[i], x + ra_f));
                                // a in the expanded cell [lo - r, up + r]
                                inside = tri_and(inside, leq(lo[i] - r, x));
                                inside = tri_and(inside, leq(x, up[i] + r));
                            }

                            if (overlap == 1 || inside == 1) {
                                return true;
                            }
                            if (overlap == 0 && inside == 0) {
                                return false;
                            }

                            // exact criterion
                            Tscal int_r_max_cell = rint_tree[node_id] * Kernel::Rkern;

                            shammath::AABB<Tvec> node_aabb{
                                particle_looper.tree_traverser.aabb_min[node_id],
                                particle_looper.tree_traverser.aabb_max[node_id]};

                            using namespace walker::interaction_crit;

                            return sph_radix_cell_crit(
                                xyz_a,
                                inter_box_a_min,
                                inter_box_a_max,
                                node_aabb.lower,
                                node_aabb.upper,
                                int_r_max_cell);
                        };

                        u32 cnt = active ? scanned_neigh_cnt[id_a] : 0;

                        nc_traverse_warp_cooperative(
                            item,
                            active,
                            stack_ptr,
                            stack_size,
                            particle_looper,
                            node_test,
                            [&](u32 id_b) {
                                // Certified fp32 version of the interaction test: with
                                // e = 2^-24, c the root box center, xf = f32(x - c) and
                                // af = xa_f - xb_f, |af - dr| <= 1.01 e B with
                                // B = |xa_f|_1 + |xb_f|_1 + |af|_1, so the fp64 rab2 lies within
                                // r2 [1 -+ 16 e] -+ 4 e B^2 of the fp32 r2 = |af|^2, and thr_f
                                // holds bounds of the fp64 thresholds: the result is decided in
                                // fp32 unless r2 is that close to a threshold (or NaN / inf), in
                                // which case the exact test is performed.
                                sycl::vec<f32, 4> xb_f  = xyz_rel_f[id_b];
                                sycl::vec<f32, 2> thr_b = thr_f[id_b];
                                sycl::vec<f32, 3> af{
                                    xa_f.x() - xb_f.x(), xa_f.y() - xb_f.y(), xa_f.z() - xb_f.z()};
                                f32 bound = xa_f.w() + xb_f.w() + sycl::fabs(af.x())
                                            + sycl::fabs(af.y()) + sycl::fabs(af.z());
                                f32 r2f   = sycl::dot(af, af);

                                constexpr f32 e = 1.f / 16777216.f; // 2^-24

                                f32 slack = 4.f * e * bound * bound + 1e-30f;
                                f32 r2_hi = r2f * (1.f + 16.f * e) + slack;
                                f32 r2_lo = r2f * (1.f - 16.f * e) - slack;

                                bool interact;
                                if (bound < 1e15f && (r2_hi < thr_a.x() || r2_hi < thr_b.x())) {
                                    interact = true;
                                } else if (
                                    bound < 1e15f && r2_lo > thr_a.y() && r2_lo > thr_b.y()) {
                                    interact = false;
                                } else {
                                    // exact test
                                    Tvec dr      = xyz_a - xyz[id_b];
                                    Tscal rab2   = sycl::dot(dr, dr);
                                    Tscal rint_b = hpart[id_b] * h_tolerance;

                                    bool no_interact = rab2 > rint_a * rint_a * Rker2
                                                       && rab2 > rint_b * rint_b * Rker2;
                                    interact         = !no_interact;
                                }

                                if (interact) {
                                    neigh[cnt] = id_b;
                                }
                                cnt += (interact) ? 1 : 0;
                            });
                    });
                };
            });

        return pcache;
    };

    shambase::get_check_ref(storage.neigh_cache).free_alloc();

    using namespace shamrock::patch;
    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        auto &ncache = shambase::get_check_ref(storage.neigh_cache);
        ncache.neigh_cache.add_obj(cur_p.id_patch, build_neigh_cache(cur_p.id_patch));
    });

    time_neigh.stop();
    storage.timings_details.neighbors += time_neigh.elapsed_sec();
}

template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M4>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M6>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, M8>;

template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C2>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C4>;
template class shammodels::sph::modules::NeighbourCache<f64_3, u32, C6>;
