/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Config.h"

#if GSPLAT_BUILD_RELOC

#    include <cfloat>
#    include <cmath>

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>

#    include "Common.h"
#    include "Relocation.h"

namespace gsplat
{
// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
template<typename scalar_t>
__global__ void relocation_kernel(
    int N,
    const scalar_t *opacities,
    const scalar_t *scales,
    const int *ratios,
    float min_opacity,
    scalar_t *new_opacities,
    scalar_t *new_scales
)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= N)
    {
        return;
    }

    int n_idx = ratios[idx];

    // Clamp before the scale computation; intentionally deviates from the Eq. 9
    // transparency invariant for near-zero-opacity Gaussians for numerical
    // stability purposes. Preserve the legacy powf path over the old 1..51
    // domain; use log1p/expm1 beyond it to avoid cancellation at large ratios.
    float new_opacity  = n_idx <= 51 ? 1.0f - powf(1.0f - opacities[idx], 1.0f / static_cast<float>(n_idx))
                                     : -expm1f(log1pf(-opacities[idx]) / static_cast<float>(n_idx));
    new_opacity        = fminf(fmaxf(new_opacity, min_opacity), 1.0f - FLT_EPSILON);
    new_opacities[idx] = new_opacity;

    // Equation (9) contains a triangular double sum. Swap the summations and
    // apply the hockey-stick identity:
    //
    //   sum_{i=1}^n sum_{k=0}^{i-1} C(i-1,k) (-1)^k p^(k+1)/sqrt(k+1)
    // = sum_{j=1}^n C(n,j) (-1)^(j-1) p^j/sqrt(j).
    //
    // For well-conditioned cases, generate adjacent terms recursively with
    // Kahan compensation. This is O(n), needs no Pascal table, and avoids the
    // old n<=51 clamp.
    float denom_sum;
    const float np = static_cast<float>(n_idx) * new_opacity;
    if(np <= 8.0f)
    {
        float term       = np; // j = 1
        denom_sum        = term;
        float correction = 0.0f;
        for(int j = 1; j < n_idx; ++j)
        {
            const float jp1  = static_cast<float>(j + 1);
            term            *= -static_cast<float>(n_idx - j) / jp1 * new_opacity * sqrtf(static_cast<float>(j) / jp1);
            const float corrected_term = term - correction;
            const float next_sum       = denom_sum + corrected_term;
            correction                 = (next_sum - denom_sum) - corrected_term;
            denom_sum                  = next_sum;
        }
    }
    else
    {
        // Large n*p makes the alternating series ill-conditioned. Evaluate the
        // equivalent positive integral instead:
        //
        //   2/sqrt(pi) * integral_0^inf [1-(1-p*exp(-x^2))^n] dx.
        //
        // A fixed 96-interval Simpson rule on [0,8] is sufficient here; its
        // tail is negligible in float32. exp(-x^2) is advanced by recurrence,
        // so each node needs only the stable log1p/expm1 pair.
        constexpr int kIntervals      = 96;
        constexpr float kExpRatio0    = 0.9930796124903161f; // exp(-1/144)
        constexpr float kExpRatioStep = 0.9862071167439163f; // exp(-2/144)
        constexpr float kSimpsonScale = 0.031343865752653126f;

        float exp_x2      = 1.0f;
        float exp_ratio   = kExpRatio0;
        float simpson_sum = 0.0f;
        for(int j = 0; j <= kIntervals; ++j)
        {
            const float z       = new_opacity * exp_x2;
            const float value   = -expm1f(static_cast<float>(n_idx) * log1pf(-z));
            const float weight  = (j == 0 || j == kIntervals) ? 1.0f : ((j & 1) ? 4.0f : 2.0f);
            simpson_sum        += weight * value;
            exp_x2             *= exp_ratio;
            exp_ratio          *= kExpRatioStep;
        }
        denom_sum = kSimpsonScale * simpson_sum;
    }
    float coeff = opacities[idx] / denom_sum;
    for(int i = 0; i < 3; ++i)
    {
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
    }
}

void launch_relocation_kernel(
    // inputs
    at::Tensor opacities, // [N]
    at::Tensor scales,    // [N, 3]
    at::Tensor ratios,    // [N]
    at::Tensor binoms,    // legacy compatibility argument; unused
    const int n_max,      // legacy compatibility argument; unused
    float min_opacity,
    // outputs
    at::Tensor new_opacities, // [N]
    at::Tensor new_scales     // [N, 3]
)
{
    static_cast<void>(binoms);
    static_cast<void>(n_max);
    uint32_t N = opacities.size(0);

    int64_t n_elements = N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        opacities.scalar_type(),
        "relocation_kernel",
        [&]()
        {
            relocation_kernel<scalar_t><<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                N,
                opacities.const_data_ptr<scalar_t>(),
                scales.const_data_ptr<scalar_t>(),
                ratios.const_data_ptr<int>(),
                min_opacity,
                new_opacities.data_ptr<scalar_t>(),
                new_scales.data_ptr<scalar_t>()
            );
        }
    );
}
} // namespace gsplat

#endif
