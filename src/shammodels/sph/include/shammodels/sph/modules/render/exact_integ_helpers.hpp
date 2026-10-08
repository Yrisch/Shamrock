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
 * @file exact_integ_helpers.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Bit identical building blocks of the SPH line / ring integration kernels
 *
 * Helpers used by SPHColumnInteg and SPHAzymuthalInteg to evaluate
 * `partmass * val * Kernel::Y_3d(r, h, 4) / rho_h(partmass, h, hfactd)` faster while producing
 * exactly the same bits: symmetric evaluation of the Y_3d Riemann sum, Markstein divisions
 * using precomputed reciprocals and comparisons of non negative doubles on their bit patterns.
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shammath/sphkernels.hpp"
#include <type_traits>
#include <array>

namespace shammodels::sph::modules::details {

    /// raw bits of a double
    inline u64 f64_bits(f64 v) { return sycl::bit_cast<u64>(v); }

    /// biased exponent field of a double
    inline u32 f64_exp(f64 v) { return u32((f64_bits(v) >> 52) & 0x7ff); }

    /// `a < b` for a >= +0 or NaN and b > 0 finite (integer comparison of the bit patterns,
    /// NaN gives false as the floating point comparison)
    inline bool lt_pos(f64 a, f64 b) { return f64_bits(a) < f64_bits(b); }

    /**
     * @brief Bit identical equivalent of `a / b` given `y = RN(1 / b)`
     *
     * Uses two Markstein refinements of `a * y`: the first one gives a quotient within 1 ulp of
     * a / b, the second one is then correctly rounded (Markstein's theorem, the residuals being
     * exact with fma). This holds without underflow / overflow of the intermediates, so the
     * hardware division is used outside of a safe exponent range (and for zero / inf / NaN /
     * subnormals).
     */
    inline f64 div_rn(f64 a, f64 b, f64 y) {
        i32 ea = i32(f64_exp(a)) - 1023;
        i32 eb = i32(f64_exp(b)) - 1023;
        i32 ed = ea - eb;

        bool safe = ea >= -900 && ea <= 900 && eb >= -450 && eb <= 450 && ed >= -850 && ed <= 850;

        if (safe) [[likely]] {
            f64 q0 = a * y;
            f64 e0 = sycl::fma(-b, q0, a);
            f64 q1 = sycl::fma(e0, y, q0);
            f64 e1 = sycl::fma(-b, q1, a);
            return sycl::fma(e1, y, q1);
        }
        return a / b;
    }

    /// div_rn applied per component (as `T / f64` does)
    template<class T>
    inline T div_rn_vec(T a, f64 b, f64 y) {
        if constexpr (sham::VectorProperties<T>::dimension == 1) {
            return div_rn(a, b, y);
        } else {
            T ret;
#pragma unroll
            for (u32 i = 0; i < sham::VectorProperties<T>::dimension; i++) {
                ret[i] = div_rn(a[i], b, y);
            }
            return ret;
        }
    }

    /**
     * @brief `Kernel::f(q)` for q >= +0 or NaN
     *
     * For M4 this is a copy of `KernelDefM4::f` where the comparisons are done on the bit
     * patterns (exact for q >= +0 or NaN, and avoiding fp64 comparisons), other kernels use
     * `Kernel::f` directly.
     */
    template<class Kernel>
    inline typename Kernel::Tscal kernel_f_pos(typename Kernel::Tscal q) {
        using Tscal = typename Kernel::Tscal;
        if constexpr (
            std::is_same_v<typename Kernel::Generator, shammath::details::KernelDefM4<Tscal>>) {
            Tscal t1 = 2 - q;
            Tscal t2 = 1 - q;

            t1 = t1 * t1 * t1;
            t2 = t2 * t2 * t2;

            constexpr Tscal div1_4 = (1. / 4.);
            t1 *= div1_4;
            t2 *= -1;

            if (lt_pos(q, Tscal(1))) {
                return t1 + t2;
            } else if (lt_pos(q, Tscal(2))) {
                return t1;
            } else
                return 0;
        } else {
            return Kernel::f(q);
        }
    }

    /// index of the per particle data in a sycl::vec<f64, 8>
    enum IntegPartData : int { PdH = 0, PdInvH, PdHH, PdInvHH, PdRho, PdInvRho, PdSupport2 };

    /**
     * @brief Bit identical equivalent of `Kernel::Y_3d(r, h, np)` exploiting the z symmetry
     *
     * `Y_3d` is a Riemann sum over `z = -Rkern, -Rkern + step, ...` of `f(sqrt(x^2 + z^2))`.
     * When this grid is exact and symmetric, the terms at `z` and `-z` are identical, so only
     * the samples with `z <= 0` are evaluated and the sum is then accumulated in the original
     * order, giving the same bits as `Y_3d`.
     */
    template<class Kernel, int np>
    struct IntegZGrid {
        using Tscal = typename Kernel::Tscal;

        static constexpr Tscal start = -Kernel::Rkern;
        static constexpr Tscal end   = Kernel::Rkern;
        static constexpr Tscal step  = Kernel::Rkern / np;

        // same loop as shammath::integ_riemann_sum
        static constexpr int count() {
            int n = 0;
            for (Tscal z = start; z < end; z += step) {
                n++;
            }
            return n;
        }

        static constexpr int n = count();

        static constexpr std::array<Tscal, n> zs() {
            std::array<Tscal, n> ret{};
            int i = 0;
            for (Tscal z = start; z < end; z += step) {
                ret[i++] = z;
            }
            return ret;
        }

        static constexpr std::array<Tscal, n> z = zs();

        // index of the sample with the same z^2 and z <= 0 (-1 if there is none)
        static constexpr std::array<int, n> mirrors() {
            std::array<int, n> ret{};
            for (int i = 0; i < n; i++) {
                ret[i] = -1;
                if (z[i] <= 0) {
                    ret[i] = i;
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    if (z[j] <= 0 && z[j] == -z[i]) {
                        ret[i] = j;
                    }
                }
            }
            return ret;
        }

        static constexpr std::array<int, n> mirror = mirrors();

        static constexpr bool is_symmetric() {
            for (int i = 0; i < n; i++) {
                if (mirror[i] < 0) {
                    return false;
                }
            }
            return true;
        }

        static_assert(is_symmetric(), "the Riemann grid of Y_3d must be exactly symmetric");

        static constexpr Tscal Rkern2 = Kernel::Rkern * Kernel::Rkern;

        // Rkern^2 must be exact for sqrt(Rkern^2) == Rkern (Rkern has a short dyadic expansion)
        static_assert(
            Kernel::Rkern * 8 == Tscal(i64(Kernel::Rkern * 8)) && Kernel::Rkern < 64,
            "Rkern must be a multiple of 1/8 for Rkern^2 to be exact");

        /// pd: per particle data (see IntegPartData)
        static inline Tscal Y_3d(Tscal r, const sycl::vec<f64, 8> &pd) {
            Tscal x = div_rn(r, pd[PdH], pd[PdInvH]);

            // the original loop compiles `x * x + z * z` to fma(z, z, x * x), match it explicitly
            Tscal xx = x * x;

            Tscal fz[n];
#pragma unroll
            for (int i = 0; i < n; i++) {
                if (z[i] <= 0) {
                    if (z[i] == start) {
                        // y >= Rkern^2 always (or NaN), see below
                        fz[i] = 0;
                    } else if (z[i] == 0) {
                        // y = fma(0, 0, xx) = xx = RN(x^2) and in radix 2 with round to nearest
                        // sqrt(RN(x^2)) == |x| as long as x^2 does not underflow (x >= 0 here)
                        Tscal q = x;
                        if (lt_pos(x, Tscal(0x1p-511))
                            || !(lt_pos(x, shambase::get_infty<Tscal>()))) [[unlikely]] {
                            q = sqrt(sycl::fma(z[i], z[i], xx));
                        }
                        fz[i] = lt_pos(xx, Rkern2) ? kernel_f_pos<Kernel>(q) : Tscal{0};
                    } else {
                        Tscal y = sycl::fma(z[i], z[i], xx);
                        // f(q) is exactly 0 for q >= Rkern (and for NaN) and sqrt(y) >= Rkern
                        // whenever y >= Rkern^2, so these samples are exactly 0
                        fz[i] = lt_pos(y, Rkern2) ? kernel_f_pos<Kernel>(sqrt(y)) : Tscal{0};
                    }
                }
            }

            // same accumulation as shammath::integ_riemann_sum
            Tscal acc = {};
#pragma unroll
            for (int i = 0; i < n; i++) {
                acc += fz[mirror[i]] * step;
            }

            return div_rn(Kernel::Generator::norm_3d * acc, pd[PdHH], pd[PdInvHH]);
        }
    };

} // namespace shammodels::sph::modules::details
