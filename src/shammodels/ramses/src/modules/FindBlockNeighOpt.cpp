// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindBlockNeighOpt.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Optimised variant of FindBlockNeigh (NeighGraphStrategy::NeighGraphOpt), builds the same
 * block graph
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/make_ndrange.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/FindBlockNeighOpt.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shammodels/ramses/modules/details/neigh_graph_6dir.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <vector>

namespace {

    /// work group size of the fused block graph kernels
    constexpr u32 block_finder_group_size = 64;

    /// number of traversal stack entries (from the bottom) kept in work-group local memory
    constexpr u32 block_finder_shared_stack_depth = 24;

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::AMRBlockFinder {
        public:
        using acc_u32 = sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device>;
        using acc_u8  = sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device>;
        using acc_grid
            = sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device>;

        static constexpr u32 tree_depth = RTree::tree_depth;
        static constexpr u32 _nindex    = 4294967295;

        // radix tree (same data as shamrock::tree::ObjectIterator)
        acc_u32 particle_index_map;
        acc_u32 cell_index_map;
        acc_u32 rchild_id;
        acc_u32 lchild_id;
        acc_u8 rchild_flag;
        acc_u8 lchild_flag;
        acc_grid pos_min_cell;
        acc_grid pos_max_cell;
        u32 leaf_offset;

        acc_grid acc_block_min;
        acc_grid acc_block_max;

        TgridVec dir_offset;

        AMRBlockFinder(
            sycl::handler &cgh,
            const RTree &tree,
            sycl::buffer<TgridVec> &buf_block_min,
            sycl::buffer<TgridVec> &buf_block_max,
            TgridVec dir_offset)
            : particle_index_map{
                  shambase::get_check_ref(tree.tree_morton_codes.buf_particle_index_map),
                  cgh,
                  sycl::read_only},
              cell_index_map{
                  shambase::get_check_ref(tree.tree_reduced_morton_codes.buf_reduc_index_map),
                  cgh,
                  sycl::read_only},
              rchild_id{
                  shambase::get_check_ref(tree.tree_struct.buf_rchild_id), cgh, sycl::read_only},
              lchild_id{
                  shambase::get_check_ref(tree.tree_struct.buf_lchild_id), cgh, sycl::read_only},
              rchild_flag{
                  shambase::get_check_ref(tree.tree_struct.buf_rchild_flag), cgh, sycl::read_only},
              lchild_flag{
                  shambase::get_check_ref(tree.tree_struct.buf_lchild_flag), cgh, sycl::read_only},
              pos_min_cell{
                  shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt),
                  cgh,
                  sycl::read_only},
              pos_max_cell{
                  shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt),
                  cgh,
                  sycl::read_only},
              leaf_offset(tree.tree_struct.internal_cell_count),
              acc_block_min{buf_block_min, cgh, sycl::read_only},
              acc_block_max{buf_block_max, cgh, sycl::read_only},
              dir_offset(std::move(dir_offset)) {}

        template<class IndexFunctor>
        void for_each_other_index(u32 id_a, IndexFunctor &&fct) const {

            // current block AABB
            shammath::AABB<TgridVec> block_aabb{acc_block_min[id_a], acc_block_max[id_a]};

            // The wanted AABB (the block we look for)
            shammath::AABB<TgridVec> check_aabb{
                block_aabb.lower + dir_offset, block_aabb.upper + dir_offset};

            auto node_test = [&](u32 node_id) -> bool {
                return shammath::AABB<TgridVec>{pos_min_cell[node_id], pos_max_cell[node_id]}
                    .get_intersect(check_aabb)
                    .is_volume_not_null();
            };

            auto on_object = [&](u32 id_b) {
                bool interact = shammath::AABB<TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                                    .get_intersect(check_aabb)
                                    .is_volume_not_null()
                                && id_b != id_a;

                if (interact) {
                    fct(id_b);
                }
            };

            // Same depth first traversal as ObjectIterator::rtree_for, but the inner loop stops
            // on the first hit leaf, whose objects are scanned once the warp has reconverged
            u32 stack_cursor = tree_depth - 1;
            std::array<u32, tree_depth> id_stack;
            id_stack[stack_cursor] = 0;

            while (stack_cursor < tree_depth) {

                u32 found_leaf = _nindex;

                while (stack_cursor < tree_depth) {
                    u32 current_node_id = id_stack[stack_cursor];
                    stack_cursor++;

                    if (node_test(current_node_id)) {
                        if (current_node_id >= leaf_offset) {
                            found_leaf = current_node_id;
                            break;
                        }

                        u32 lid = lchild_id[current_node_id]
                                  + leaf_offset * lchild_flag[current_node_id];
                        u32 rid = rchild_id[current_node_id]
                                  + leaf_offset * rchild_flag[current_node_id];

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                }

                if (found_leaf != _nindex) {
                    u32 min_ids = cell_index_map[found_leaf - leaf_offset];
                    u32 max_ids = cell_index_map[found_leaf + 1 - leaf_offset];
                    for (u32 id_s = min_ids; id_s < max_ids; id_s++) {
                        on_object(particle_index_map[id_s]);
                    }
                }
            }
        }
    };

    /**
     * @brief Traversal of the packed i32 records of all the patches, for the 6 directions at once
     *
     * node_rec holds two vec<i32, 4> per tree node: {lower - origin, a} and {upper - origin, b},
     * with (a, b) the (global) left & right child ids of an internal node, or the (global) range
     * of objects of a leaf, a carrying leaf_flag. obj_rec holds two vec<i32, 4> per object, in
     * leaf order: {lower - origin, block id} and {upper - origin, 0}. The coordinates are relative
     * to the first block of the patch of the tree. The intersection tests are translation
     * invariant on integers, so as long as every coordinate fits (checked when building the
     * records) they give exactly the same results as the TgridVec ones of AMRBlockFinder.
     *
     * One traversal serves the 6 directions of OrientedAMRGraph::offset_check: each stack entry
     * carries the mask of the directions for which the node and all its ancestors are hit, so a
     * direction sees exactly the nodes, leaves and objects of its own traversal, in the same
     * order.
     */
    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::AMRBlockFinderI32 {
        public:
        static constexpr u32 tree_depth = RTree::tree_depth;

        /// stack entries are (direction mask << dir_mask_shift) | node id
        static constexpr u32 dir_mask_shift = 26;
        static constexpr u32 node_id_mask   = (u32(1) << dir_mask_shift) - 1;
        static constexpr u32 all_dirs       = 0x3F;

        /// flag of the leaf records, in the first word of their range of objects
        static constexpr u32 leaf_flag = u32(1) << 31;

        /**
         * @brief Call fct(dir_mask, id_b) on every block id_b intersecting the block id_a (box
         * [qlo, qup]) shifted in one of the directions, dir_mask holding these directions (bit d
         * for offset_check[d] = +x, -x, +y, -y, +z, -z). root is the root node of the tree of
         * the patch.
         */
        template<class Func>
        static void for_each_neigh_6dir(
            const sycl::vec<i32, 4> *__restrict node_rec,
            const sycl::vec<i32, 4> *__restrict obj_rec,
            u32 root,
            i32_3 qlo,
            i32_3 qup,
            u32 id_a,
            u32 *sh_stack,
            u32 sh_stride,
            Func &&fct) {

            // directions whose shifted query box intersects the box [lo, up] (non null volume),
            // from the overlaps per axis of the query shifted by -1, 0 or +1
            auto dir_hits = [&](sycl::vec<i32, 4> lo, sycl::vec<i32, 4> up) -> u32 {
                auto ov = [](i32 lo, i32 up, i32 qlo, i32 qup, i32 s) -> bool {
                    return sycl::min(up, qup + s) > sycl::max(lo, qlo + s);
                };

                bool x0 = ov(lo.x(), up.x(), qlo.x(), qup.x(), 0);
                bool y0 = ov(lo.y(), up.y(), qlo.y(), qup.y(), 0);
                bool z0 = ov(lo.z(), up.z(), qlo.z(), qup.z(), 0);

                u32 m = 0;
                if (y0 && z0) {
                    m |= u32(ov(lo.x(), up.x(), qlo.x(), qup.x(), 1)) << 0;
                    m |= u32(ov(lo.x(), up.x(), qlo.x(), qup.x(), -1)) << 1;
                }
                if (x0 && z0) {
                    m |= u32(ov(lo.y(), up.y(), qlo.y(), qup.y(), 1)) << 2;
                    m |= u32(ov(lo.y(), up.y(), qlo.y(), qup.y(), -1)) << 3;
                }
                if (x0 && y0) {
                    m |= u32(ov(lo.z(), up.z(), qlo.z(), qup.z(), 1)) << 4;
                    m |= u32(ov(lo.z(), up.z(), qlo.z(), qup.z(), -1)) << 5;
                }
                return m;
            };

            // Same traversal as AMRBlockFinder::for_each_other_index, the direction mask
            // travelling with the node ids. The stack grows down from tree_depth - 1: its first
            // block_finder_shared_stack_depth entries live in work-group local memory (sh_stack,
            // entry j at sh_stack[j * sh_stride]), deeper ones in id_stack.
            constexpr u32 sh_lo = tree_depth - block_finder_shared_stack_depth;

            std::array<u32, tree_depth> id_stack;

            auto stack_get = [&](u32 idx) -> u32 {
                return (idx >= sh_lo) ? sh_stack[(idx - sh_lo) * sh_stride] : id_stack[idx];
            };
            auto stack_set = [&](u32 idx, u32 val) {
                if (idx >= sh_lo) {
                    sh_stack[(idx - sh_lo) * sh_stride] = val;
                } else {
                    id_stack[idx] = val;
                }
            };

            u32 stack_cursor = tree_depth - 1;
            stack_set(stack_cursor, (all_dirs << dir_mask_shift) | root);

            while (stack_cursor < tree_depth) {

                u32 leaf_dirs = 0;
                u32 obj_begin = 0;
                u32 obj_end   = 0;

                while (stack_cursor < tree_depth) {
                    u32 entry = stack_get(stack_cursor);
                    stack_cursor++;

                    u32 current_node_id = entry & node_id_mask;

                    sycl::vec<i32, 4> lo = node_rec[2 * current_node_id];
                    sycl::vec<i32, 4> up = node_rec[2 * current_node_id + 1];

                    u32 dirs = (entry >> dir_mask_shift) & dir_hits(lo, up);

                    if (dirs != 0) {
                        if ((u32(lo.w()) & leaf_flag) != 0) {
                            leaf_dirs = dirs;
                            obj_begin = u32(lo.w()) & ~leaf_flag;
                            obj_end   = u32(up.w());
                            break;
                        }

                        stack_set(stack_cursor - 1, (dirs << dir_mask_shift) | u32(up.w()));
                        stack_cursor--;

                        stack_set(stack_cursor - 1, (dirs << dir_mask_shift) | u32(lo.w()));
                        stack_cursor--;
                    }
                }

                if (leaf_dirs != 0) {
                    for (u32 id_s = obj_begin; id_s < obj_end; id_s++) {
                        sycl::vec<i32, 4> lo = obj_rec[2 * id_s];
                        sycl::vec<i32, 4> up = obj_rec[2 * id_s + 1];
                        u32 id_b             = u32(lo.w());

                        u32 dirs = (id_b != id_a) ? (leaf_dirs & dir_hits(lo, up)) : 0;

                        if (dirs != 0) {
                            fct(dirs, id_b);
                        }
                    }
                }
            }
        }
    };

    template<class Tvec, class TgridVec, class Tmorton>
    void FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        using Finder = AMRBlockFinderI32;

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = dev_sched->get_queue();

        shambase::DistributedData<OrientedAMRGraph> graph;

        // the fused traversal assumes the unit offsets +x, -x, +y, -y, +z, -z
        bool offsets_ok = true;
        {
            OrientedAMRGraph proto;
            const std::array<TgridVec, 6> unit_offsets{
                TgridVec{1, 0, 0},
                TgridVec{-1, 0, 0},
                TgridVec{0, 1, 0},
                TgridVec{0, -1, 0},
                TgridVec{0, 0, 1},
                TgridVec{0, 0, -1}};
            for (u32 dir = 0; dir < 6; dir++) {
                offsets_ok = offsets_ok && sham::equals(proto.offset_check[dir], unit_offsets[dir]);
            }
        }

        // The patches are batched: the trees and blocks of all the patches are packed in common
        // record buffers, so that one count and one fill kernel build the 6 graphs of every
        // patch (an AMR run has many small patches, whose kernels alone cannot fill the GPU).
        struct PatchInfo {
            u64 id;
            const RTree *tree;
            u32 block_count;
            u32 tot_count;
            u32 internal_count;
            TgridVec origin;
            u32 node_base; // first record of its tree nodes
            u32 obj_base;  // first record of its objects (leaf order) and query boxes (block order)
        };

        std::vector<PatchInfo> patches;
        u64 node_tot = 0;
        u64 obj_tot  = 0;

        edges.trees.trees.for_each([&](u64 id, const RTree &tree) {
            u32 leaf_count     = tree.tree_reduced_morton_codes.tree_leaf_count;
            u32 internal_count = tree.tree_struct.internal_cell_count;
            u32 block_count    = edges.sizes.indexes.get(id);

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(id);

            // records relative to the first block of the patch
            TgridVec origin
                = (block_count > 0) ? block_min.get_buf().get_val_at_idx(0) : TgridVec{};

            patches.push_back(
                {id,
                 &tree,
                 block_count,
                 leaf_count + internal_count,
                 internal_count,
                 origin,
                 u32(node_tot),
                 u32(obj_tot)});

            node_tot += leaf_count + internal_count;
            obj_tot += block_count;
        });

        // global node ids are packed below the direction masks, object ids below the leaf flag
        bool batch_ok
            = offsets_ok && (node_tot <= Finder::node_id_mask) && (obj_tot < Finder::leaf_flag);

        u32 patch_cnt = patches.size();

        sham::DeviceBuffer<sycl::vec<i32, 4>> node_rec(2 * node_tot, dev_sched);
        sham::DeviceBuffer<sycl::vec<i32, 4>> obj_rec(2 * obj_tot, dev_sched);
        sham::DeviceBuffer<sycl::vec<i32, 4>> qry_rec(2 * obj_tot, dev_sched);
        sham::DeviceBuffer<u32> out_of_range(patch_cnt, dev_sched);
        out_of_range.fill(0);

        // bound on the relative coordinates, so that the query boxes (+-1) and every comparison
        // stay far from the i32 limits
        constexpr i64 max_rel = i64(1) << 30;

        auto to_rel = [](TgridVec v, TgridVec origin, bool &bad) -> sycl::vec<i32, 3> {
            TgridVec d = v - origin;
            bad        = bad || (d.x() < -max_rel) || (d.x() > max_rel) || (d.y() < -max_rel)
                         || (d.y() > max_rel) || (d.z() < -max_rel) || (d.z() > max_rel);
            return {i32(d.x()), i32(d.y()), i32(d.z())};
        };

        for (u32 p = 0; p < patch_cnt && batch_ok; p++) {
            const PatchInfo &pi = patches[p];
            if (pi.block_count == 0) {
                continue;
            }

            const RTree &tree = *pi.tree;

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(pi.id);
            PatchDataField<TgridVec> &block_max = edges.spans_block_max.get_refs().get(pi.id);

            sycl::buffer<TgridVec> &tree_bmin
                = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
            sycl::buffer<TgridVec> &tree_bmax
                = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

            {
                sham::EventList deps;
                auto rec  = node_rec.get_write_access(deps);
                auto flag = out_of_range.get_write_access(deps);

                auto e = q.submit(deps, [&, pi, p](sycl::handler &cgh) {
                    sycl::accessor pos_min{tree_bmin, cgh, sycl::read_only};
                    sycl::accessor pos_max{tree_bmax, cgh, sycl::read_only};
                    sycl::accessor lchild_id{
                        shambase::get_check_ref(tree.tree_struct.buf_lchild_id),
                        cgh,
                        sycl::read_only};
                    sycl::accessor rchild_id{
                        shambase::get_check_ref(tree.tree_struct.buf_rchild_id),
                        cgh,
                        sycl::read_only};
                    sycl::accessor lchild_flag{
                        shambase::get_check_ref(tree.tree_struct.buf_lchild_flag),
                        cgh,
                        sycl::read_only};
                    sycl::accessor rchild_flag{
                        shambase::get_check_ref(tree.tree_struct.buf_rchild_flag),
                        cgh,
                        sycl::read_only};
                    sycl::accessor cell_index_map{
                        shambase::get_check_ref(tree.tree_reduced_morton_codes.buf_reduc_index_map),
                        cgh,
                        sycl::read_only};

                    u32 leaf_offset = pi.internal_count;
                    u32 node_base   = pi.node_base;
                    u32 obj_base    = pi.obj_base;
                    TgridVec origin = pi.origin;

                    shambase::parallel_for(
                        cgh, pi.tot_count, "pack i32 node records", [=](u64 gid) {
                            u32 node = (u32) gid;

                            bool bad             = false;
                            sycl::vec<i32, 3> lo = to_rel(pos_min[node], origin, bad);
                            sycl::vec<i32, 3> up = to_rel(pos_max[node], origin, bad);

                            u32 a, b;
                            if (node >= leaf_offset) {
                                a = (obj_base + cell_index_map[node - leaf_offset])
                                    | Finder::leaf_flag;
                                b = obj_base + cell_index_map[node + 1 - leaf_offset];
                            } else {
                                a = node_base + lchild_id[node] + leaf_offset * lchild_flag[node];
                                b = node_base + rchild_id[node] + leaf_offset * rchild_flag[node];
                            }

                            rec[2 * (node_base + node)]     = {lo.x(), lo.y(), lo.z(), i32(a)};
                            rec[2 * (node_base + node) + 1] = {up.x(), up.y(), up.z(), i32(b)};

                            if (bad) {
                                flag[p] = 1;
                            }
                        });
                });

                node_rec.complete_event_state(e);
                out_of_range.complete_event_state(e);
            }

            {
                sham::EventList deps;
                auto orec = obj_rec.get_write_access(deps);
                auto qrec = qry_rec.get_write_access(deps);
                auto flag = out_of_range.get_write_access(deps);
                auto bmin = block_min.get_buf().get_read_access(deps);
                auto bmax = block_max.get_buf().get_read_access(deps);

                auto e = q.submit(deps, [&, pi, p](sycl::handler &cgh) {
                    sycl::accessor particle_index_map{
                        shambase::get_check_ref(tree.tree_morton_codes.buf_particle_index_map),
                        cgh,
                        sycl::read_only};

                    u32 obj_base    = pi.obj_base;
                    TgridVec origin = pi.origin;

                    shambase::parallel_for(
                        cgh, pi.block_count, "pack i32 object records", [=](u64 gid) {
                            u32 id_s = (u32) gid;
                            u32 id_b = particle_index_map[id_s];

                            bool bad             = false;
                            sycl::vec<i32, 3> lo = to_rel(bmin[id_b], origin, bad);
                            sycl::vec<i32, 3> up = to_rel(bmax[id_b], origin, bad);

                            orec[2 * (obj_base + id_s)]     = {lo.x(), lo.y(), lo.z(), i32(id_b)};
                            orec[2 * (obj_base + id_s) + 1] = {up.x(), up.y(), up.z(), 0};

                            // query box of block id_b, in block order
                            qrec[2 * (obj_base + id_b)]     = {lo.x(), lo.y(), lo.z(), 0};
                            qrec[2 * (obj_base + id_b) + 1] = {up.x(), up.y(), up.z(), 0};

                            if (bad) {
                                flag[p] = 1;
                            }
                        });
                });

                obj_rec.complete_event_state(e);
                qry_rec.complete_event_state(e);
                out_of_range.complete_event_state(e);
                block_min.get_buf().complete_event_state(e);
                block_max.get_buf().complete_event_state(e);
            }
        }

        // patches built by the batched kernels, the others (empty, out of range) use the per
        // direction TgridVec path
        std::vector<u32> batched;
        std::vector<u32> flags = out_of_range.copy_to_stdvec();
        for (u32 p = 0; p < patch_cnt && batch_ok; p++) {
            if (patches[p].block_count > 0 && flags[p] == 0) {
                batched.push_back(p);
            }
        }

        u32 S = batched.size();

        if (S > 0) {
            // per batched patch s: first thread (blk_base), root node, first object / query box,
            // first link count slot (cnt_base, 6 * (block_count + 1) slots per patch)
            std::vector<u32> h_blk_base(S + 1), h_root(S), h_qbase(S), h_cnt_base(S + 1);
            h_blk_base[0] = 0;
            h_cnt_base[0] = 0;
            for (u32 s = 0; s < S; s++) {
                const PatchInfo &pi = patches[batched[s]];
                h_blk_base[s + 1]   = h_blk_base[s] + pi.block_count;
                h_cnt_base[s + 1]   = h_cnt_base[s] + 6 * (pi.block_count + 1);
                h_root[s]           = pi.node_base;
                h_qbase[s]          = pi.obj_base;
            }
            u32 B         = h_blk_base[S];
            u32 cnt_total = h_cnt_base[S];

            sham::DeviceBuffer<u32> blk_base(S + 1, dev_sched);
            sham::DeviceBuffer<u32> root(S, dev_sched);
            sham::DeviceBuffer<u32> qbase(S, dev_sched);
            sham::DeviceBuffer<u32> cnt_base(S + 1, dev_sched);
            blk_base.copy_from_stdvec(h_blk_base);
            root.copy_from_stdvec(h_root);
            qbase.copy_from_stdvec(h_qbase);
            cnt_base.copy_from_stdvec(h_cnt_base);

            // patch of thread t: the last s with blk_base[s] <= t
            auto find_patch = [S](const u32 *__restrict blk_base, u32 t) -> u32 {
                u32 lo = 0, hi = S;
                while (hi - lo > 1) {
                    u32 mid = (lo + hi) / 2;
                    if (blk_base[mid] <= t) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return lo;
            };

            // link counts: patch s, direction dir, block i at cnt_base[s] + dir * (n_s + 1) + i,
            // a separator after each (patch, direction) and one final slot
            sham::DeviceBuffer<u32> link_counts(cnt_total + 1, dev_sched);

            {
                sham::EventList deps;
                auto nrec = node_rec.get_read_access(deps);
                auto orec = obj_rec.get_read_access(deps);
                auto qrec = qry_rec.get_read_access(deps);
                auto bb   = blk_base.get_read_access(deps);
                auto rt   = root.get_read_access(deps);
                auto qb   = qbase.get_read_access(deps);
                auto cb   = cnt_base.get_read_access(deps);
                u32 *cnt  = link_counts.get_write_access(deps);

                auto e = q.submit(deps, [&, B, S](sycl::handler &cgh) {
                    sycl::local_accessor<u32, 1> stack_local(
                        block_finder_shared_stack_depth * block_finder_group_size, cgh);

                    cgh.parallel_for(
                        sham::make_ndrange(block_finder_group_size, B), [=](sycl::nd_item<1> item) {
                            u32 t = (u32) item.get_global_linear_id();
                            if (t >= B) {
                                return;
                            }

                            u32 s    = find_patch(bb, t);
                            u32 id_a = t - bb[s];
                            u32 n    = bb[s + 1] - bb[s];

                            sycl::vec<i32, 4> ql = qrec[2 * (qb[s] + id_a)];
                            sycl::vec<i32, 4> qu = qrec[2 * (qb[s] + id_a) + 1];

                            u32 *sh_stack = &stack_local[item.get_local_linear_id()];
                            u32 sh_stride = (u32) item.get_local_range(0);

                            u32 c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;

                            Finder::for_each_neigh_6dir(
                                nrec,
                                orec,
                                rt[s],
                                i32_3{ql.x(), ql.y(), ql.z()},
                                i32_3{qu.x(), qu.y(), qu.z()},
                                id_a,
                                sh_stack,
                                sh_stride,
                                [&](u32 dirs, u32 id_b) {
                                    c0 += (dirs >> 0) & 1;
                                    c1 += (dirs >> 1) & 1;
                                    c2 += (dirs >> 2) & 1;
                                    c3 += (dirs >> 3) & 1;
                                    c4 += (dirs >> 4) & 1;
                                    c5 += (dirs >> 5) & 1;
                                });

                            u32 base                       = cb[s];
                            cnt[base + 0 * (n + 1) + id_a] = c0;
                            cnt[base + 1 * (n + 1) + id_a] = c1;
                            cnt[base + 2 * (n + 1) + id_a] = c2;
                            cnt[base + 3 * (n + 1) + id_a] = c3;
                            cnt[base + 4 * (n + 1) + id_a] = c4;
                            cnt[base + 5 * (n + 1) + id_a] = c5;

                            // separators of the patch, and the final slot
                            if (id_a == n - 1) {
                                for (u32 dir = 0; dir < 6; dir++) {
                                    cnt[base + dir * (n + 1) + n] = 0;
                                }
                            }
                            if (t == B - 1) {
                                cnt[cb[S]] = 0;
                            }
                        });
                });

                node_rec.complete_event_state(e);
                obj_rec.complete_event_state(e);
                qry_rec.complete_event_state(e);
                blk_base.complete_event_state(e);
                root.complete_event_state(e);
                qbase.complete_event_state(e);
                cnt_base.complete_event_state(e);
                link_counts.complete_event_state(e);
            }

            // one scan for all the patches and directions; the offsets of (patch, direction)
            // are the scan minus its value at the start of the (patch, direction)
            sham::DeviceBuffer<u32> scanned
                = shamalgs::numeric::scan_exclusive(dev_sched, link_counts, cnt_total + 1);

            sham::DeviceBuffer<u32> starts(6 * S + 1, dev_sched);
            sham::kernel_call(
                q,
                sham::MultiRef{scanned, blk_base, cnt_base},
                sham::MultiRef{starts},
                6 * S + 1,
                [S](u32 k,
                    const u32 *__restrict sc,
                    const u32 *__restrict bb,
                    const u32 *__restrict cb,
                    u32 *__restrict st) {
                    if (k == 6 * S) {
                        st[k] = sc[cb[S]];
                    } else {
                        u32 s   = k / 6;
                        u32 dir = k % 6;
                        u32 n   = bb[s + 1] - bb[s];
                        st[k]   = sc[cb[s] + dir * (n + 1)];
                    }
                });
            std::vector<u32> h_starts = starts.copy_to_stdvec();

            // graph buffers of every (patch, direction), reached by the fill kernel through
            // tables of pointers
            std::vector<std::unique_ptr<sham::DeviceBuffer<u32>>> offsets(6 * S);
            std::vector<std::unique_ptr<sham::DeviceBuffer<u32>>> links(6 * S);
            for (u32 k = 0; k < 6 * S; k++) {
                u32 n      = patches[batched[k / 6]].block_count;
                offsets[k] = std::make_unique<sham::DeviceBuffer<u32>>(n + 1, dev_sched);
                links[k]   = std::make_unique<sham::DeviceBuffer<u32>>(
                    h_starts[k + 1] - h_starts[k], dev_sched);
            }

            sham::DeviceBuffer<u64> offsets_ptr(6 * S, dev_sched);
            sham::DeviceBuffer<u64> links_ptr(6 * S, dev_sched);

            {
                sham::EventList deps;
                std::vector<u64> h_offsets_ptr(6 * S), h_links_ptr(6 * S);
                for (u32 k = 0; k < 6 * S; k++) {
                    h_offsets_ptr[k] = reinterpret_cast<u64>(offsets[k]->get_write_access(deps));
                    h_links_ptr[k]   = reinterpret_cast<u64>(links[k]->get_write_access(deps));
                }
                offsets_ptr.copy_from_stdvec(h_offsets_ptr);
                links_ptr.copy_from_stdvec(h_links_ptr);

                auto nrec = node_rec.get_read_access(deps);
                auto orec = obj_rec.get_read_access(deps);
                auto qrec = qry_rec.get_read_access(deps);
                auto bb   = blk_base.get_read_access(deps);
                auto rt   = root.get_read_access(deps);
                auto qb   = qbase.get_read_access(deps);
                auto cb   = cnt_base.get_read_access(deps);
                auto sc   = scanned.get_read_access(deps);
                auto st   = starts.get_read_access(deps);
                auto optr = offsets_ptr.get_read_access(deps);
                auto lptr = links_ptr.get_read_access(deps);

                auto e = q.submit(deps, [&, B, S](sycl::handler &cgh) {
                    sycl::local_accessor<u32, 1> stack_local(
                        block_finder_shared_stack_depth * block_finder_group_size, cgh);

                    cgh.parallel_for(
                        sham::make_ndrange(block_finder_group_size, B), [=](sycl::nd_item<1> item) {
                            u32 t = (u32) item.get_global_linear_id();
                            if (t >= B) {
                                return;
                            }

                            u32 s    = find_patch(bb, t);
                            u32 id_a = t - bb[s];
                            u32 n    = bb[s + 1] - bb[s];

                            sycl::vec<i32, 4> ql = qrec[2 * (qb[s] + id_a)];
                            sycl::vec<i32, 4> qu = qrec[2 * (qb[s] + id_a) + 1];

                            u32 *sh_stack = &stack_local[item.get_local_linear_id()];
                            u32 sh_stride = (u32) item.get_local_range(0);

                            u32 base = cb[s];

                            // offsets from the single scan, stored in the graphs here
                            std::array<u32, 6> next_link_idx;
                            std::array<u32 *, 6> ids;
#pragma unroll
                            for (u32 dir = 0; dir < 6; dir++) {
                                u32 k     = 6 * s + dir;
                                u32 *offs = reinterpret_cast<u32 *>(optr[k]);
                                ids[dir]  = reinterpret_cast<u32 *>(lptr[k]);

                                next_link_idx[dir] = sc[base + dir * (n + 1) + id_a] - st[k];
                                offs[id_a]         = next_link_idx[dir];
                                if (id_a == n - 1) {
                                    offs[n] = sc[base + dir * (n + 1) + n] - st[k];
                                }
                            }

                            Finder::for_each_neigh_6dir(
                                nrec,
                                orec,
                                rt[s],
                                i32_3{ql.x(), ql.y(), ql.z()},
                                i32_3{qu.x(), qu.y(), qu.z()},
                                id_a,
                                sh_stack,
                                sh_stride,
                                [&](u32 dirs, u32 id_b) {
#pragma unroll
                                    for (u32 dir = 0; dir < 6; dir++) {
                                        if ((dirs >> dir) & 1) {
                                            ids[dir][next_link_idx[dir]] = id_b;
                                            next_link_idx[dir]++;
                                        }
                                    }
                                });
                        });
                });

                node_rec.complete_event_state(e);
                obj_rec.complete_event_state(e);
                qry_rec.complete_event_state(e);
                blk_base.complete_event_state(e);
                root.complete_event_state(e);
                qbase.complete_event_state(e);
                cnt_base.complete_event_state(e);
                scanned.complete_event_state(e);
                starts.complete_event_state(e);
                offsets_ptr.complete_event_state(e);
                links_ptr.complete_event_state(e);
                for (u32 k = 0; k < 6 * S; k++) {
                    offsets[k]->complete_event_state(e);
                    links[k]->complete_event_state(e);
                }
            }

            for (u32 s = 0; s < S; s++) {
                const PatchInfo &pi = patches[batched[s]];
                OrientedAMRGraph result;
                for (u32 dir = 0; dir < 6; dir++) {
                    u32 k          = 6 * s + dir;
                    u32 link_count = h_starts[k + 1] - h_starts[k];

                    shamlog_debug_ln(
                        "AMR Block Graph",
                        "Patch",
                        pi.id,
                        "direction",
                        dir,
                        "link cnt",
                        link_count);

                    result.graph_links[dir] = std::make_unique<AMRGraph>(AMRGraph{
                        .node_link_offset = std::move(*offsets[k]),
                        .node_links       = std::move(*links[k]),
                        .link_count       = link_count,
                        .obj_cnt          = pi.block_count});
                }
                graph.add_obj(pi.id, std::move(result));
            }
        }

        // the other patches: per direction TgridVec path
        for (u32 p = 0; p < patch_cnt; p++) {
            const PatchInfo &pi = patches[p];
            if (graph.has_key(pi.id)) {
                continue;
            }

            OrientedAMRGraph result;

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(pi.id);
            PatchDataField<TgridVec> &block_max = edges.spans_block_max.get_refs().get(pi.id);

            sycl::buffer<TgridVec> buf_block_min_sycl = block_min.get_buf().copy_to_sycl_buffer();
            sycl::buffer<TgridVec> buf_block_max_sycl = block_max.get_buf().copy_to_sycl_buffer();

            for (u32 dir = 0; dir < 6; dir++) {

                TgridVec dir_offset = result.offset_check[dir];

                AMRGraph rslt = details::compute_neigh_graph_deprecated<AMRBlockFinder>(
                    dev_sched,
                    pi.block_count,
                    *pi.tree,
                    buf_block_min_sycl,
                    buf_block_max_sycl,
                    dir_offset);

                shamlog_debug_ln(
                    "AMR Block Graph",
                    "Patch",
                    pi.id,
                    "direction",
                    dir,
                    "link cnt",
                    rslt.link_count);

                result.graph_links[dir] = std::make_unique<AMRGraph>(std::move(rslt));
            }

            graph.add_obj(pi.id, std::move(result));
        }

        edges.block_neigh_graph.graph = std::move(graph);

        // possible unittest
        /*
        one patch with :
        sz = 1 << 4
        base = 4
        model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

        make a grid of 4^3 blocks, which when merge with interface make 6^3 blocks.
        In each direction one slab will have no links, hence the number of links should always be
        6^3 - 6^2 = 180 which we get here on all directions
        */
    }

    template<class Tvec, class TgridVec, class Tmorton>
    std::string FindBlockNeighOpt<Tvec, TgridVec, Tmorton>::_impl_get_tex() const {

        std::string sizes             = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min         = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max         = get_ro_edge_base(2).get_tex_symbol();
        std::string trees             = get_ro_edge_base(3).get_tex_symbol();
        std::string block_neigh_graph = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Find neighbour blocks

            \begin{align}
            {block_neigh_graph} = \text{FindBlockNeighOpt}({sizes}, {block_min}, {block_max}, {trees})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{trees}", trees);
        shambase::replace_all(tex, "{block_neigh_graph}", block_neigh_graph);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::FindBlockNeighOpt<f64_3, i64_3, u64>;
