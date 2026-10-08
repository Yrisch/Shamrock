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
 * @file enum_NeighGraphStrategy.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Strategy used to build the block & cell neighbour graphs + json serialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    /**
     * @brief Strategy used to build the block & cell neighbour graphs
     *
     * Both strategies build exactly the same graphs (same links in the same order).
     */
    enum NeighGraphStrategy {
        NeighGraphStandard = 0, ///< FindBlockNeigh + BlockNeighToCellNeigh
        NeighGraphOpt = 1, ///< FindBlockNeighOpt + BlockNeighToCellNeighOpt (optimised variant)
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        NeighGraphStrategy,
        {{NeighGraphStrategy::NeighGraphStandard, "standard"},
         {NeighGraphStrategy::NeighGraphOpt, "opt"}});

} // namespace shammodels::basegodunov
