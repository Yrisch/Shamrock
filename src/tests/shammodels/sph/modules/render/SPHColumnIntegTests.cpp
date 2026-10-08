// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SPHColumnIntegTests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Unit tests for SPHColumnInteg vs host double-loop reference.
 *
 */

#include "shammodels/sph/modules/render/SPHColumnInteg.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/sph/modules/render/SPHRenderTestCommon.hpp"
#include <memory>

namespace {

    std::vector<sph_render_test::Tscal> run_column_integ(
        const sph_render_test::ParticleDataset &global,
        const std::vector<shammath::Ray<sph_render_test::Tvec>> &rays) {
        using namespace sph_render_test;
        using namespace shammodels::sph::modules;

        auto loc = make_round_robin_fields(global);

        auto gpart_mass           = make_gpart_mass();
        auto tree_reduction_level = make_tree_reduction_level();
        auto rays_edge            = make_query_edge<shammath::Ray<Tvec>>("rays", rays);
        auto interpolated_field   = make_output_edge();

        auto node = std::make_shared<SPHColumnInteg<Tvec, Tscal, shammath::M4>>();
        node->set_edges(
            gpart_mass,
            tree_reduction_level,
            loc.part_counts,
            loc.positions,
            loc.h_part,
            loc.field_data,
            rays_edge,
            interpolated_field);
        node->evaluate();

        return interpolated_field->value.copy_to_stdvec();
    }

} // namespace

NEW_TEST(Unittest, "shammodels/sph/modules/render/SPHColumnInteg:vs_direct", -1) {
    using namespace sph_render_test;

    auto global = make_global_dataset();
    auto rays   = make_column_rays();
    auto ref    = reference_column(global, rays);

    auto out = run_column_integ(global, rays);
    REQUIRE_EQUAL(out.size(), ref.size());
    REQUIRE_EQUAL_CUSTOM_COMP(out, ref, [](const auto &a, const auto &b) {
        return almost_equal_vec(a, b);
    });
}

NEW_TEST(Unittest, "shammodels/sph/modules/render/SPHColumnInteg:half_line_vs_direct", -1) {
    using namespace sph_render_test;

    auto global = make_global_dataset();
    auto rays   = make_half_column_rays();
    auto ref    = reference_column(global, rays);

    auto out = run_column_integ(global, rays);
    REQUIRE_EQUAL(out.size(), ref.size());
    REQUIRE_EQUAL_CUSTOM_COMP(out, ref, [](const auto &a, const auto &b) {
        return almost_equal_vec(a, b);
    });

    // the rays start in the middle of the particle box, so the particles with z < 0 must have
    // been dropped compared to the same rays seen as infinite lines
    auto full_lines = make_column_rays();
    auto ref_full   = reference_column(global, full_lines);

    Tscal sum_half = 0;
    Tscal sum_full = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        sum_half += out[i];
        sum_full += ref_full[i];
    }
    REQUIRE(sum_half > 0);
    REQUIRE(sum_half < sum_full);
}
