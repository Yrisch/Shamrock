// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file neigh_graph_6dir.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Build the offsets of the 6 directional neighbour graphs with a single scan
 * (NeighGraphStrategy::NeighGraphOpt)
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include <array>
#include <memory>
#include <optional>

namespace shammodels::basegodunov::modules::details {

    /**
     * @brief Index of the link count of object i for direction dir in the buffer of the 6
     * directions
     *
     * The counts of a direction are followed by a separator slot, and the buffer ends with one
     * more slot (see neigh_6dir_count_size)
     */
    inline u32 neigh_6dir_count_idx(u32 dir, u32 i, u32 obj_cnt) { return dir * (obj_cnt + 1) + i; }

    /// Size of the buffer of the link counts of the 6 directions
    inline u32 neigh_6dir_count_size(u32 obj_cnt) { return 6 * (obj_cnt + 1) + 1; }

    /**
     * @brief Result of scan_link_counts_6dir
     *
     * node_link_offset is allocated but, except when there is no object, filled by the fill
     * kernel (see neigh_6dir_link_offset).
     */
    struct NeighGraph6DirOffsets {
        std::array<std::unique_ptr<sham::DeviceBuffer<u32>>, 6> node_link_offset;
        std::array<u32, 6> link_count;

        /// exclusive scan of the concatenated counts
        std::unique_ptr<sham::DeviceBuffer<u32>> scanned;
        /// value of the scan at the start of each direction
        std::array<u32, 6> start;
    };

    /**
     * @brief Offset of the first link of object i for direction dir (i == obj_cnt gives the link
     * count of the direction)
     *
     * The fill kernel of the graphs must store it in node_link_offset[dir][i] for every object i,
     * and in node_link_offset[dir][obj_cnt] for the last object. This is exactly the exclusive scan
     * of the counts of the direction alone.
     */
    inline u32 neigh_6dir_link_offset(const u32 *scanned, u32 start, u32 dir, u32 i, u32 obj_cnt) {
        return scanned[neigh_6dir_count_idx(dir, i, obj_cnt)] - start;
    }

    /**
     * @brief Exclusive scan of the link counts of the 6 directions with one scan
     *
     * `counts` (neigh_6dir_count_size(obj_cnt) entries) must hold the link count of object i for
     * direction dir at neigh_6dir_count_idx(dir, i, obj_cnt); the separator slots are set to 0
     * here. The offsets of a direction are the scan of the concatenated counts minus its value at
     * the start of the direction (neigh_6dir_link_offset), written by the fill kernel.
     *
     * The counts may be per group of graph objects (e.g. per block for a graph of cells): the
     * node_link_offset buffers are then sized for graph_obj_cnt objects.
     */
    inline NeighGraph6DirOffsets scan_link_counts_6dir(
        const sham::DeviceScheduler_ptr &dev_sched,
        sham::DeviceBuffer<u32> &counts,
        u32 obj_cnt,
        std::optional<u32> graph_obj_cnt = std::nullopt) {

        auto &q = dev_sched->get_queue();

        u32 stride = obj_cnt + 1;
        u32 len    = neigh_6dir_count_size(obj_cnt);

        // separators : after the counts of each direction, and the last slot
        sham::kernel_call(
            q, sham::MultiRef{}, sham::MultiRef{counts}, 7, [stride](u32 k, u32 *__restrict c) {
                c[k * stride + ((k < 6) ? stride - 1 : 0)] = 0;
            });

        NeighGraph6DirOffsets ret;
        ret.scanned = std::make_unique<sham::DeviceBuffer<u32>>(
            shamalgs::numeric::scan_exclusive(dev_sched, counts, len));

        // start of each direction in the scan (the 7th is the total)
        sham::DeviceBuffer<u32> starts(7, dev_sched);
        sham::kernel_call(
            q,
            sham::MultiRef{*ret.scanned},
            sham::MultiRef{starts},
            7,
            [stride](u32 k, const u32 *__restrict s, u32 *__restrict st) {
                st[k] = s[k * stride];
            });
        std::vector<u32> base = starts.copy_to_stdvec();

        u32 graph_objs = graph_obj_cnt.value_or(obj_cnt);

        for (u32 dir = 0; dir < 6; dir++) {
            ret.node_link_offset[dir]
                = std::make_unique<sham::DeviceBuffer<u32>>(graph_objs + 1, dev_sched);
            ret.link_count[dir] = base[dir + 1] - base[dir];
            ret.start[dir]      = base[dir];

            if (graph_objs == 0) {
                // no fill kernel to write it
                ret.node_link_offset[dir]->set_val_at_idx(0, 0);
            }
        }

        return ret;
    }

} // namespace shammodels::basegodunov::modules::details
