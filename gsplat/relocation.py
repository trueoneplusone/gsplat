# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from torch import Tensor

from .cuda._wrapper import _make_lazy_cuda_func


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
    binoms: Tensor | None = None,
    min_opacity: float = 0.005,
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]
        binoms: Deprecated compatibility argument. The CUDA implementation no
          longer needs a precomputed binomial table.
        min_opacity: Lower clamp applied to the new opacity before computing
          new scales. Defaults to 0.005.

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    N = opacities.shape[0]
    assert scales.shape == (N, 3), scales.shape
    assert ratios.shape == (N,), ratios.shape
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios = ratios.clamp_min(1).int().contiguous()

    # Keep the legacy low-level custom-op signature for callers that invoke it
    # directly, but the new kernel no longer reads the Pascal table.
    if binoms is None:
        binoms = torch.empty(0, device=opacities.device, dtype=opacities.dtype)
        n_max = 0
    else:
        binoms = binoms.contiguous()
        n_max = binoms.shape[0]
    new_opacities, new_scales = _make_lazy_cuda_func("relocation")(
        opacities, scales, ratios, binoms, n_max, min_opacity
    )
    return new_opacities, new_scales
