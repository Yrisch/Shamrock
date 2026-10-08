// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BlockNeighToCellNeighOpt.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Optimised variant of BlockNeighToCellNeigh (NeighGraphStrategy::NeighGraphOpt), builds the
 * same cell graph
 */

#include "shambase/exception.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/make_ndrange.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/BlockNeighToCellNeighOpt.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shammodels/ramses/modules/details/neigh_graph_6dir.hpp"
#include "shamrock/amr/AMRCell.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"
#include <vector>

namespace {

    /// work group size of the cell graph kernels (a multiple of the block size)
    constexpr u32 cell_graph_group_size = 128;

    /**
     * @brief Call fct on the cells sharing a face with cell id_a in the direction dir_offset
     *
     * Same as AMRLowering::ro_acces::for_each_other_index_safe: candidates are the cells of the
     * block of id_a, then of the blocks linked to it in the block graph of that direction, in
     * link order.
     */
    template<class AMRBlock, class TgridVec, class IndexFunctor>
    inline void for_each_cell_neigh_safe(
        u32 id_a,
        const shammodels::basegodunov::modules::AMRGraph::ro_access &graph_iter,
        const TgridVec *acc_block_min,
        const TgridVec *acc_block_max,
        TgridVec dir_offset,
        IndexFunctor &&fct) {

        const u32 cell_global_id = (u32) id_a;

        const u32 block_id    = cell_global_id / AMRBlock::block_size;
        const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

        // fetch current block info
        const TgridVec cblock_min = acc_block_min[block_id];
        const TgridVec cblock_max = acc_block_max[block_id];
        const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

        std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
        TgridVec lcoord               = {lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]};

        shammath::AABB<TgridVec> current_cell_aabb
            = {cblock_min + lcoord * delta_cell,
               cblock_min + (lcoord + TgridVec{1, 1, 1}) * delta_cell};

        const shammath::AABB<TgridVec> current_cell_aabb_shifted
            = {current_cell_aabb.lower + dir_offset, current_cell_aabb.upper + dir_offset};

        // The cells of a block are a tensor product of Nside slabs per axis, so the volume test
        // of the original (max of the lowers < min of the uppers on every axis) splits into per
        // axis tests of the slabs, done once per block and combined in the original loop order.
        const TgridVec sh_lo = current_cell_aabb_shifted.lower;
        const TgridVec sh_up = current_cell_aabb_shifted.upper;

        auto on_block = [&](u32 block_b) {
            TgridVec block_b_min = acc_block_min[block_b];
            TgridVec block_b_max = acc_block_max[block_b];

            const TgridVec delta_cell_b = (block_b_max - block_b_min) / AMRBlock::Nside;

            std::array<bool, AMRBlock::Nside> ov_x, ov_y, ov_z;
#pragma unroll
            for (u32 l = 0; l < AMRBlock::Nside; l++) {
                TgridVec cell_lo = block_b_min + TgridVec{l, l, l} * delta_cell_b;
                TgridVec cell_up = block_b_min + TgridVec{l + 1, l + 1, l + 1} * delta_cell_b;

                ov_x[l] = sycl::min(cell_up.x(), sh_up.x()) > sycl::max(cell_lo.x(), sh_lo.x());
                ov_y[l] = sycl::min(cell_up.y(), sh_up.y()) > sycl::max(cell_lo.y(), sh_lo.y());
                ov_z[l] = sycl::min(cell_up.z(), sh_up.z()) > sycl::max(cell_lo.z(), sh_lo.z());
            }

#pragma unroll
            for (u32 lx = 0; lx < AMRBlock::Nside; lx++) {
#pragma unroll
                for (u32 ly = 0; ly < AMRBlock::Nside; ly++) {
#pragma unroll
                    for (u32 lz = 0; lz < AMRBlock::Nside; lz++) {

                        u32 idx
                            = block_b * AMRBlock::block_size + AMRBlock::get_index({lx, ly, lz});

                        bool overlap = ov_x[lx] && ov_y[ly] && ov_z[lz] && id_a != idx;

                        if (overlap) {
                            fct(idx);
                        }
                    }
                }
            }
        };

        on_block(block_id);
        graph_iter.for_each_object_link(block_id, [&](u32 block_b) {
            on_block(block_b);
        });
    }

} // namespace

namespace shammodels::basegodunov::modules {

    // here if we want to find the stencil instead of just the common face, we can just the change
    // lowering mask to the cube offseted in the wanted direction instead of just checking if we
    // have a common surface

    // like on the above case with block Nside 2 we get 12^3 - 12^2 = 1584 link count
    // it is a really good possible test

    template<class Tvec, class TgridVec, class Tmorton>
    void BlockNeighToCellNeighOpt<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        using AMRBlock = amr::AMRBlock<Tvec, TgridVec, 1>;

        if (block_nside_pow != 1) {
            shambase::throw_unimplemented("block_nside_pow != 1");
        }

        shambase::DistributedData<OrientedAMRGraph> cell_graph_links;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = dev_sched->get_queue();

        const std::array<TgridVec, 6> off = OrientedAMRGraph{}.offset_check;

        // The links of a direction are ordered by cell and the cells of a block are contiguous,
        // so the scan is done on the link counts of the blocks (8 times smaller): the offset of a
        // cell is the one of its block plus the links of the previous cells of the block, summed
        // in work-group memory (a work group holds whole blocks).
        static_assert(cell_graph_group_size % AMRBlock::block_size == 0);

        // The patches are batched (as in FindBlockNeighOpt): one count kernel, one scan and one
        // fill kernel build the 6 cell graphs of every patch. A thread handles one cell, finds
        // its patch by a binary search on the cell offsets, and reaches the block graphs, block
        // boxes and output buffers of its patch through tables of pointers. Each thread handles
        // the 6 directions of its cell one after the other, each with the same candidates in the
        // same order as AMRLowering's safe mode, so each link list is unchanged.
        struct PatchInfo {
            u64 id;
            u32 block_count;
            const OrientedAMRGraph *block_graph;
        };
        std::vector<PatchInfo> patches;

        edges.block_neigh_graph.graph.for_each([&](u64 id, const OrientedAMRGraph &bg) {
            u32 block_count = edges.sizes.indexes.get(id);
            if (block_count > 0) {
                patches.push_back({id, block_count, &bg});
                return;
            }

            // no cell: empty graphs (as computed by the standard path)
            OrientedAMRGraph result;
            for (u32 dir = 0; dir < 6; dir++) {
                sham::DeviceBuffer<u32> offsets(1, dev_sched);
                offsets.set_val_at_idx(0, 0);
                result.graph_links[dir] = std::make_unique<AMRGraph>(AMRGraph{
                    .node_link_offset = std::move(offsets),
                    .node_links       = sham::DeviceBuffer<u32>(0, dev_sched),
                    .link_count       = 0,
                    .obj_cnt          = 0,
                    .antecedent       = sham::DeviceBuffer<u32>(0, dev_sched)});
            }
            cell_graph_links.add_obj(id, std::move(result));
        });

        u32 S = patches.size();

        if (S > 0) {
            // per patch s: first cell (cell_base) and first block link count slot (cnt_base, a
            // segment of block_count + 1 slots per direction, the last one being a separator)
            std::vector<u32> h_cell_base(S + 1), h_cnt_base(S + 1);
            h_cell_base[0] = 0;
            h_cnt_base[0]  = 0;
            for (u32 s = 0; s < S; s++) {
                h_cell_base[s + 1] = h_cell_base[s] + patches[s].block_count * AMRBlock::block_size;
                h_cnt_base[s + 1]  = h_cnt_base[s] + 6 * (patches[s].block_count + 1);
            }
            u32 C         = h_cell_base[S];
            u32 cnt_total = h_cnt_base[S];

            sham::DeviceBuffer<u32> cell_base(S + 1, dev_sched);
            sham::DeviceBuffer<u32> cnt_base(S + 1, dev_sched);
            cell_base.copy_from_stdvec(h_cell_base);
            cnt_base.copy_from_stdvec(h_cnt_base);

            // patch of thread t: the last s with cell_base[s] <= t
            auto find_patch = [S](const u32 *__restrict cell_base, u32 t) -> u32 {
                u32 lo = 0, hi = S;
                while (hi - lo > 1) {
                    u32 mid = (lo + hi) / 2;
                    if (cell_base[mid] <= t) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return lo;
            };

            auto block_min_buf = [&](u64 id) -> sham::DeviceBuffer<TgridVec> & {
                PatchDataField<TgridVec> &f = edges.spans_block_min.get_refs().get(id);
                return f.get_buf();
            };
            auto block_max_buf = [&](u64 id) -> sham::DeviceBuffer<TgridVec> & {
                PatchDataField<TgridVec> &f = edges.spans_block_max.get_refs().get(id);
                return f.get_buf();
            };

            // read access to the block graphs and block boxes of every patch, returns the tables
            // of pointers (filled the first time)
            sham::DeviceBuffer<u64> bg_offsets_ptr(6 * S, dev_sched);
            sham::DeviceBuffer<u64> bg_links_ptr(6 * S, dev_sched);
            sham::DeviceBuffer<u64> bmin_ptr(S, dev_sched);
            sham::DeviceBuffer<u64> bmax_ptr(S, dev_sched);
            bool tables_filled = false;

            auto block_data_access = [&](sham::EventList &deps) {
                std::vector<u64> h_bgo(6 * S), h_bgl(6 * S), h_bmin(S), h_bmax(S);
                for (u32 s = 0; s < S; s++) {
                    for (u32 dir = 0; dir < 6; dir++) {
                        AMRGraph::ro_access acc
                            = shambase::get_check_ref(patches[s].block_graph->graph_links[dir])
                                  .get_read_access(deps);
                        h_bgo[6 * s + dir] = reinterpret_cast<u64>(acc.node_link_offset);
                        h_bgl[6 * s + dir] = reinterpret_cast<u64>(acc.node_links);
                    }
                    h_bmin[s]
                        = reinterpret_cast<u64>(block_min_buf(patches[s].id).get_read_access(deps));
                    h_bmax[s]
                        = reinterpret_cast<u64>(block_max_buf(patches[s].id).get_read_access(deps));
                }
                if (!tables_filled) {
                    bg_offsets_ptr.copy_from_stdvec(h_bgo);
                    bg_links_ptr.copy_from_stdvec(h_bgl);
                    bmin_ptr.copy_from_stdvec(h_bmin);
                    bmax_ptr.copy_from_stdvec(h_bmax);
                    tables_filled = true;
                }
            };
            auto block_data_complete = [&](sycl::event &e) {
                for (u32 s = 0; s < S; s++) {
                    for (u32 dir = 0; dir < 6; dir++) {
                        shambase::get_check_ref(patches[s].block_graph->graph_links[dir])
                            .complete_event_state(e);
                    }
                    block_min_buf(patches[s].id).complete_event_state(e);
                    block_max_buf(patches[s].id).complete_event_state(e);
                }
            };

            // link counts of the cells (cell_counts[dir * C + t]) and of the blocks (patch s,
            // direction dir, block i at cnt_base[s] + dir * (n_s + 1) + i, plus a final slot)
            sham::DeviceBuffer<u32> cell_counts(6 * C, dev_sched);
            sham::DeviceBuffer<u32> link_counts(cnt_total + 1, dev_sched);

            {
                sham::EventList deps;
                block_data_access(deps);
                auto bgo = bg_offsets_ptr.get_read_access(deps);
                auto bgl = bg_links_ptr.get_read_access(deps);
                auto bmn = bmin_ptr.get_read_access(deps);
                auto bmx = bmax_ptr.get_read_access(deps);
                auto cb  = cell_base.get_read_access(deps);
                auto nb  = cnt_base.get_read_access(deps);
                u32 *cc  = cell_counts.get_write_access(deps);
                u32 *cnt = link_counts.get_write_access(deps);

                auto e = q.submit(deps, [&, off, C, S, cnt_total](sycl::handler &cgh) {
                    constexpr u32 G = cell_graph_group_size;
                    sycl::local_accessor<u32, 1> lc(6 * G, cgh);

                    cgh.parallel_for(sham::make_ndrange(G, C), [=](sycl::nd_item<1> item) {
                        u32 t       = (u32) item.get_global_linear_id();
                        u32 lid     = (u32) item.get_local_linear_id();
                        bool active = t < C;

                        std::array<u32, 6> c{0, 0, 0, 0, 0, 0};

                        u32 s    = (active) ? find_patch(cb, t) : 0;
                        u32 id_a = t - cb[s];

                        if (active) {
                            const TgridVec *bmin = reinterpret_cast<const TgridVec *>(bmn[s]);
                            const TgridVec *bmax = reinterpret_cast<const TgridVec *>(bmx[s]);

#pragma unroll
                            for (u32 dir = 0; dir < 6; dir++) {
                                AMRGraph::ro_access g{
                                    reinterpret_cast<const u32 *>(bgo[6 * s + dir]),
                                    reinterpret_cast<const u32 *>(bgl[6 * s + dir])};
                                for_each_cell_neigh_safe<AMRBlock>(
                                    id_a, g, bmin, bmax, off[dir], [&](u32) {
                                        c[dir]++;
                                    });
                                cc[dir * C + t] = c[dir];
                            }
                        }

#pragma unroll
                        for (u32 dir = 0; dir < 6; dir++) {
                            lc[dir * G + lid] = c[dir];
                        }

                        sycl::group_barrier(item.get_group());

                        if (!active) {
                            return;
                        }

                        u32 n_b   = (cb[s + 1] - cb[s]) / AMRBlock::block_size;
                        u32 block = id_a / AMRBlock::block_size;

                        // first cell of a block : link counts of the block
                        if ((lid % AMRBlock::block_size) == 0) {
                            for (u32 dir = 0; dir < 6; dir++) {
                                u32 sum = 0;
                                for (u32 k = 0; k < AMRBlock::block_size; k++) {
                                    sum += lc[dir * G + lid + k];
                                }
                                cnt[nb[s] + dir * (n_b + 1) + block] = sum;
                                if (block == n_b - 1) {
                                    cnt[nb[s] + dir * (n_b + 1) + n_b] = 0; // separator
                                }
                            }
                        }
                        if (t == C - 1) {
                            cnt[cnt_total] = 0;
                        }
                    });
                });

                block_data_complete(e);
                bg_offsets_ptr.complete_event_state(e);
                bg_links_ptr.complete_event_state(e);
                bmin_ptr.complete_event_state(e);
                bmax_ptr.complete_event_state(e);
                cell_base.complete_event_state(e);
                cnt_base.complete_event_state(e);
                cell_counts.complete_event_state(e);
                link_counts.complete_event_state(e);
            }

            // one scan for all the patches and directions; the offsets of (patch, direction) are
            // the scan minus its value at the start of the (patch, direction)
            sham::DeviceBuffer<u32> scanned
                = shamalgs::numeric::scan_exclusive(dev_sched, link_counts, cnt_total + 1);

            sham::DeviceBuffer<u32> starts(6 * S + 1, dev_sched);
            sham::kernel_call(
                q,
                sham::MultiRef{scanned, cell_base, cnt_base},
                sham::MultiRef{starts},
                6 * S + 1,
                [S, cnt_total](
                    u32 k,
                    const u32 *__restrict sc,
                    const u32 *__restrict cb,
                    const u32 *__restrict nb,
                    u32 *__restrict st) {
                    if (k == 6 * S) {
                        st[k] = sc[cnt_total];
                    } else {
                        u32 s   = k / 6;
                        u32 dir = k % 6;
                        u32 n_b = (cb[s + 1] - cb[s]) / AMRBlock::block_size;
                        st[k]   = sc[nb[s] + dir * (n_b + 1)];
                    }
                });
            std::vector<u32> h_starts = starts.copy_to_stdvec();

            // graph buffers of every (patch, direction): offsets, links and antecedent maps (link
            // -> cell, as AMRGraph::compute_antecedent)
            std::vector<std::unique_ptr<sham::DeviceBuffer<u32>>> offsets(6 * S), links(6 * S),
                ante(6 * S);
            for (u32 k = 0; k < 6 * S; k++) {
                u32 n_cell = patches[k / 6].block_count * AMRBlock::block_size;
                u32 n_link = h_starts[k + 1] - h_starts[k];
                offsets[k] = std::make_unique<sham::DeviceBuffer<u32>>(n_cell + 1, dev_sched);
                links[k]   = std::make_unique<sham::DeviceBuffer<u32>>(n_link, dev_sched);
                ante[k]    = std::make_unique<sham::DeviceBuffer<u32>>(n_link, dev_sched);
            }

            sham::DeviceBuffer<u64> offsets_ptr(6 * S, dev_sched);
            sham::DeviceBuffer<u64> links_ptr(6 * S, dev_sched);
            sham::DeviceBuffer<u64> ante_ptr(6 * S, dev_sched);

            {
                sham::EventList deps;
                std::vector<u64> h_off(6 * S), h_lnk(6 * S), h_ante(6 * S);
                for (u32 k = 0; k < 6 * S; k++) {
                    h_off[k]  = reinterpret_cast<u64>(offsets[k]->get_write_access(deps));
                    h_lnk[k]  = reinterpret_cast<u64>(links[k]->get_write_access(deps));
                    h_ante[k] = reinterpret_cast<u64>(ante[k]->get_write_access(deps));
                }
                offsets_ptr.copy_from_stdvec(h_off);
                links_ptr.copy_from_stdvec(h_lnk);
                ante_ptr.copy_from_stdvec(h_ante);

                block_data_access(deps);
                auto bgo  = bg_offsets_ptr.get_read_access(deps);
                auto bgl  = bg_links_ptr.get_read_access(deps);
                auto bmn  = bmin_ptr.get_read_access(deps);
                auto bmx  = bmax_ptr.get_read_access(deps);
                auto cb   = cell_base.get_read_access(deps);
                auto nb   = cnt_base.get_read_access(deps);
                auto cc   = cell_counts.get_read_access(deps);
                auto sc   = scanned.get_read_access(deps);
                auto st   = starts.get_read_access(deps);
                auto optr = offsets_ptr.get_read_access(deps);
                auto lptr = links_ptr.get_read_access(deps);
                auto aptr = ante_ptr.get_read_access(deps);

                auto e = q.submit(deps, [&, off, C](sycl::handler &cgh) {
                    constexpr u32 G = cell_graph_group_size;
                    sycl::local_accessor<u32, 1> lc(6 * G, cgh);

                    cgh.parallel_for(sham::make_ndrange(G, C), [=](sycl::nd_item<1> item) {
                        u32 t       = (u32) item.get_global_linear_id();
                        u32 lid     = (u32) item.get_local_linear_id();
                        bool active = t < C;

#pragma unroll
                        for (u32 dir = 0; dir < 6; dir++) {
                            lc[dir * G + lid] = (active) ? cc[dir * C + t] : 0;
                        }

                        sycl::group_barrier(item.get_group());

                        if (!active) {
                            return;
                        }

                        u32 s      = find_patch(cb, t);
                        u32 id_a   = t - cb[s];
                        u32 n_cell = cb[s + 1] - cb[s];
                        u32 n_b    = n_cell / AMRBlock::block_size;

                        u32 block      = id_a / AMRBlock::block_size;
                        u32 cell       = id_a % AMRBlock::block_size;
                        u32 block_lid0 = lid - cell;

                        const TgridVec *bmin = reinterpret_cast<const TgridVec *>(bmn[s]);
                        const TgridVec *bmax = reinterpret_cast<const TgridVec *>(bmx[s]);

#pragma unroll
                        for (u32 dir = 0; dir < 6; dir++) {
                            u32 k     = 6 * s + dir;
                            u32 *offs = reinterpret_cast<u32 *>(optr[k]);
                            u32 *ids  = reinterpret_cast<u32 *>(lptr[k]);
                            u32 *ant  = reinterpret_cast<u32 *>(aptr[k]);

                            // offset of the block from the single scan + links of the previous
                            // cells of the block
                            u32 w = sc[nb[s] + dir * (n_b + 1) + block] - st[k];
                            for (u32 kk = 0; kk < cell; kk++) {
                                w += lc[dir * G + block_lid0 + kk];
                            }

                            offs[id_a] = w;
                            if (id_a == n_cell - 1) {
                                offs[n_cell] = sc[nb[s] + dir * (n_b + 1) + n_b] - st[k];
                            }

                            AMRGraph::ro_access g{
                                reinterpret_cast<const u32 *>(bgo[k]),
                                reinterpret_cast<const u32 *>(bgl[k])};
                            for_each_cell_neigh_safe<AMRBlock>(
                                id_a, g, bmin, bmax, off[dir], [&](u32 idx) {
                                    ant[w] = id_a;
                                    ids[w] = idx;
                                    w++;
                                });
                        }
                    });
                });

                block_data_complete(e);
                bg_offsets_ptr.complete_event_state(e);
                bg_links_ptr.complete_event_state(e);
                bmin_ptr.complete_event_state(e);
                bmax_ptr.complete_event_state(e);
                cell_base.complete_event_state(e);
                cnt_base.complete_event_state(e);
                cell_counts.complete_event_state(e);
                scanned.complete_event_state(e);
                starts.complete_event_state(e);
                offsets_ptr.complete_event_state(e);
                links_ptr.complete_event_state(e);
                ante_ptr.complete_event_state(e);
                for (u32 k = 0; k < 6 * S; k++) {
                    offsets[k]->complete_event_state(e);
                    links[k]->complete_event_state(e);
                    ante[k]->complete_event_state(e);
                }
            }

            for (u32 s = 0; s < S; s++) {
                OrientedAMRGraph result;
                u32 n_cell = patches[s].block_count * AMRBlock::block_size;
                for (u32 dir = 0; dir < 6; dir++) {
                    u32 k      = 6 * s + dir;
                    u32 n_link = h_starts[k + 1] - h_starts[k];

                    shamlog_debug_ln(
                        "AMR Cell Graph",
                        "Patch",
                        patches[s].id,
                        "direction",
                        dir,
                        "link cnt",
                        n_link);

                    result.graph_links[dir] = std::make_unique<AMRGraph>(AMRGraph{
                        .node_link_offset = std::move(*offsets[k]),
                        .node_links       = std::move(*links[k]),
                        .link_count       = n_link,
                        .obj_cnt          = n_cell,
                        .antecedent       = std::move(*ante[k])});
                }
                cell_graph_links.add_obj(patches[s].id, std::move(result));
            }
        }

        edges.cell_neigh_graph.graph = std::move(cell_graph_links);
    }

    template<class Tvec, class TgridVec, class Tmorton>
    std::string BlockNeighToCellNeighOpt<Tvec, TgridVec, Tmorton>::_impl_get_tex() const {

        std::string sizes             = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min         = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max         = get_ro_edge_base(2).get_tex_symbol();
        std::string block_neigh_graph = get_ro_edge_base(3).get_tex_symbol();
        std::string cell_neigh_graph  = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Find neighbour blocks

            \begin{align}
            {cell_neigh_graph} &= \text{BlockNeighToCellNeighOpt}({block_neigh_graph}, {sizes}, {block_min}, {block_max})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{block_neigh_graph}", block_neigh_graph);
        shambase::replace_all(tex, "{cell_neigh_graph}", cell_neigh_graph);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::BlockNeighToCellNeighOpt<f64_3, i64_3, u64>;
