# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Smoke coverage for the relocation CUDA op (gsplat.relocation.compute_relocation).

The op is reached only through the MCMC densification strategy, which the rest
of the suite does not exercise, so this gives it direct runtime coverage: that
it runs end to end and returns finite, correctly-shaped tensors.

The all-ratios and end-to-end relocation tests also guard against a CUDA
fast-math regression where pow(-1.0f, k) can return NaN for integer k on some
GPU architectures, corrupting relocated scales before the next render step.
"""

import math

import pytest
import torch

from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import relocate

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _binomial_table(n_max: int, device: torch.device) -> torch.Tensor:
    # Lower-triangular Pascal's triangle, matching MCMCStrategy's state init.
    binoms = torch.zeros((n_max, n_max), device=device)
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    return binoms


def _reference_relocation(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    ratios: torch.Tensor,
    binoms: torch.Tensor,
    min_opacity: float,
):
    opacities = opacities.cpu()
    scales = scales.cpu()
    ratios = ratios.cpu().to(torch.int64)
    binoms = binoms.cpu()

    new_opacities = torch.empty_like(opacities)
    new_scales = torch.empty_like(scales)
    eps = torch.finfo(opacities.dtype).eps
    for idx in range(opacities.shape[0]):
        n_idx = int(ratios[idx].item())
        new_opacity = 1.0 - (1.0 - float(opacities[idx])) ** (1.0 / n_idx)
        new_opacity = min(max(new_opacity, min_opacity), 1.0 - eps)
        new_opacities[idx] = new_opacity

        denom_sum = 0.0
        for i in range(1, n_idx + 1):
            for k in range(i):
                sign = 1.0 if k % 2 == 0 else -1.0
                denom_sum += (
                    float(binoms[i - 1, k])
                    * sign
                    * (new_opacity ** (k + 1))
                    / math.sqrt(k + 1)
                )
        new_scales[idx] = scales[idx] * (float(opacities[idx]) / denom_sum)
    return new_opacities.to(opacities.device), new_scales.to(scales.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
@pytest.mark.parametrize("n_max", [5, 51])
def test_compute_relocation_smoke(n_max: int):
    torch.manual_seed(0)
    N = 128
    binoms = _binomial_table(n_max, device)
    opacities = torch.rand(N, device=device) * 0.9 + 0.05
    scales = torch.rand(N, 3, device=device) * 0.9 + 0.1
    # ratios in [1, n_max] mirrors MCMCStrategy's clamped range; the kernel
    # indexes binoms by ratio so the table must be (n_max, n_max) or larger.
    ratios = torch.randint(1, n_max + 1, (N,), device=device).float()

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, binoms, min_opacity=0.0
    )

    assert new_opacities.shape == (N,)
    assert new_scales.shape == (N, 3)
    assert (new_opacities > 0).all() and (new_opacities <= 1).all()
    assert torch.isfinite(new_scales).all()
    assert (new_scales >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_compute_relocation_min_opacity_clamps_before_scale():
    n_max = 8
    binoms = _binomial_table(n_max, device)
    opacities = torch.tensor([1e-6, 2e-3, 0.2], device=device, dtype=torch.float32)
    scales = torch.tensor(
        [[1.0, 0.5, 0.25], [0.25, 0.5, 1.0], [1.5, 0.75, 0.5]],
        device=device,
        dtype=torch.float32,
    )
    ratios = torch.tensor([8.0, 4.0, 2.0], device=device, dtype=torch.float32)
    min_opacity = 0.005

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, binoms, min_opacity=min_opacity
    )
    ref_opacities, ref_scales = _reference_relocation(
        opacities, scales, ratios, binoms, min_opacity
    )

    torch.testing.assert_close(new_opacities.cpu(), ref_opacities, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_scales.cpu(), ref_scales, rtol=1e-5, atol=1e-6)
    assert torch.all(new_opacities[:2] >= min_opacity)
    assert torch.all(new_opacities[:2] <= min_opacity + 1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_compute_relocation_all_ratios_match_reference():
    """Cover every MCMC ratio, including the fast-math sign regression."""
    n_max = 51
    binoms = _binomial_table(n_max, device)
    ratios = torch.arange(1, n_max + 1, device=device, dtype=torch.float32)
    opacities = torch.linspace(0.05, 0.95, n_max, device=device, dtype=torch.float32)
    scales = torch.linspace(
        0.1, 1.6, n_max * 3, device=device, dtype=torch.float32
    ).reshape(n_max, 3)

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, binoms, min_opacity=0.005
    )
    ref_opacities, ref_scales = _reference_relocation(
        opacities, scales, ratios, binoms, 0.005
    )

    assert torch.isfinite(new_opacities).all()
    assert torch.isfinite(new_scales).all()
    torch.testing.assert_close(new_opacities.cpu(), ref_opacities, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_scales.cpu(), ref_scales, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_relocate_many_duplicates_preserves_optimizer_state():
    """Exercise kernel output plus Parameter and optimizer-state writeback."""
    torch.manual_seed(0)
    n_points = 512
    n_alive = 4
    min_opacity = 0.005

    opacity_values = torch.full((n_points,), 1e-3, device=device, dtype=torch.float32)
    opacity_values[-n_alive:] = 0.5

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n_points, 3, device=device)),
            "scales": torch.nn.Parameter(
                torch.log(torch.rand(n_points, 3, device=device) + 0.1)
            ),
            "quats": torch.nn.Parameter(torch.randn(n_points, 4, device=device)),
            "opacities": torch.nn.Parameter(torch.logit(opacity_values)),
        }
    )
    optimizers = {
        name: torch.optim.Adam([parameter], lr=1e-3)
        for name, parameter in params.items()
    }
    # Adam creates its moment tensors lazily. Take a real step so relocation
    # must move and update populated optimizer state, not just empty dicts.
    for name, parameter in params.items():
        parameter.grad = torch.full_like(parameter, 0.1)
        optimizers[name].step()
        optimizers[name].zero_grad(set_to_none=True)
    original_params = dict(params.items())
    dead_mask = opacity_values <= min_opacity

    relocate(
        params=params,
        optimizers=optimizers,
        state={},
        mask=dead_mask,
        binoms=_binomial_table(51, device),
        min_opacity=min_opacity,
    )

    for name, parameter in params.items():
        assert torch.isfinite(parameter).all()
        optimizer = optimizers[name]
        assert optimizer.param_groups[0]["params"][0] is parameter
        assert original_params[name] not in optimizer.state
        assert parameter in optimizer.state
        for key in ("exp_avg", "exp_avg_sq"):
            moment = optimizer.state[parameter][key]
            assert moment.shape == parameter.shape
            assert torch.isfinite(moment).all()
            assert torch.all(moment[:-n_alive] != 0)
            assert torch.any(moment[-n_alive:] == 0)


def _positive_integral_reference_relocation(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    ratios: torch.Tensor,
    min_opacity: float,
):
    """Independent positive-integral reference for the collapsed Equation (9)."""
    opacities = opacities.cpu()
    scales = scales.cpu()
    ratios = ratios.cpu().to(torch.int64)

    new_opacities = torch.empty_like(opacities)
    new_scales = torch.empty_like(scales)
    eps = torch.finfo(opacities.dtype).eps
    intervals = 1024
    h = 8.0 / intervals

    for idx in range(opacities.shape[0]):
        n_idx = int(ratios[idx].item())
        old_opacity = float(opacities[idx])
        if n_idx <= 51:
            new_opacity = 1.0 - (1.0 - old_opacity) ** (1.0 / n_idx)
        else:
            new_opacity = -math.expm1(math.log1p(-old_opacity) / n_idx)
        new_opacity = min(max(new_opacity, min_opacity), 1.0 - eps)
        new_opacities[idx] = new_opacity

        simpson_sum = 0.0
        for sample in range(intervals + 1):
            x = sample * h
            z = new_opacity * math.exp(-(x * x))
            value = -math.expm1(n_idx * math.log1p(-z))
            weight = (
                1.0
                if sample == 0 or sample == intervals
                else (4.0 if sample % 2 else 2.0)
            )
            simpson_sum += weight * value
        denom_sum = (2.0 / math.sqrt(math.pi)) * h * simpson_sum / 3.0
        new_scales[idx] = scales[idx] * (old_opacity / denom_sum)

    return new_opacities, new_scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_compute_relocation_supports_ratios_beyond_legacy_limit():
    ratios = torch.tensor(
        [52, 80, 128, 1024, 2048, 4096, 8192, 65536, 1_000_000],
        device=device,
        dtype=torch.float32,
    )
    opacities = torch.linspace(0.1, 0.8, len(ratios), device=device)
    scales = torch.linspace(
        0.2, 1.4, len(ratios) * 3, device=device, dtype=torch.float32
    ).reshape(-1, 3)

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, min_opacity=0.005
    )
    ref_opacities, ref_scales = _positive_integral_reference_relocation(
        opacities, scales, ratios, min_opacity=0.005
    )

    assert torch.isfinite(new_opacities).all()
    assert torch.isfinite(new_scales).all()
    torch.testing.assert_close(new_opacities.cpu(), ref_opacities, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_scales.cpu(), ref_scales, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_compute_relocation_does_not_clamp_ratio_to_legacy_table_size():
    opacity = torch.tensor([0.9], device=device, dtype=torch.float32)
    scale = torch.ones((1, 3), device=device, dtype=torch.float32)
    ratio_80 = torch.tensor([80.0], device=device)
    ratio_51 = torch.tensor([51.0], device=device)
    legacy_binoms = _binomial_table(51, device)

    opacity_80, _ = compute_relocation(
        opacity, scale, ratio_80, legacy_binoms, min_opacity=0.0
    )
    opacity_51, _ = compute_relocation(
        opacity, scale, ratio_51, legacy_binoms, min_opacity=0.0
    )

    expected_80 = -math.expm1(math.log1p(-float(opacity.item())) / 80.0)
    torch.testing.assert_close(
        opacity_80.cpu(),
        torch.tensor([expected_80], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert not torch.allclose(opacity_80, opacity_51, rtol=1e-5, atol=1e-7)


def _make_cpu_params(opacity_values: torch.Tensor) -> torch.nn.ParameterDict:
    n = len(opacity_values)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
                requires_grad=False,
            ),
            "scales": torch.nn.Parameter(
                torch.zeros((n, 3), dtype=torch.float32), requires_grad=False
            ),
            "quats": torch.nn.Parameter(
                torch.arange(n * 4, dtype=torch.float32).reshape(n, 4),
                requires_grad=False,
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(opacity_values.clone()), requires_grad=False
            ),
        }
    )


def test_relocate_compacts_duplicate_sampled_sources(monkeypatch):
    from gsplat.strategy import ops as gsplat_ops

    params = _make_cpu_params(
        torch.tensor([0.8, 0.6, 1e-3, 1e-3, 1e-3, 1e-3], dtype=torch.float32)
    )
    dead_mask = torch.tensor([False, False, True, True, True, True])
    monkeypatch.setattr(
        gsplat_ops,
        "_multinomial_sample",
        lambda weights, n, replacement=True: torch.tensor([0, 0, 1, 0]),
    )

    calls = {}

    def fake_compute(opacities, scales, ratios, binoms=None, min_opacity=0.005):
        calls["ratios"] = ratios.clone()
        return (
            torch.tensor([0.25, 0.4], dtype=opacities.dtype),
            scales * torch.tensor([[0.5], [0.75]], dtype=scales.dtype),
        )

    monkeypatch.setattr(gsplat_ops, "compute_relocation", fake_compute)
    original_means = params["means"].detach().clone()

    gsplat_ops.relocate(params, {}, {}, dead_mask)

    torch.testing.assert_close(calls["ratios"], torch.tensor([4, 2]))
    updated_opacity = torch.sigmoid(params["opacities"])
    torch.testing.assert_close(updated_opacity[:2], torch.tensor([0.25, 0.4]))
    torch.testing.assert_close(
        updated_opacity[2:], torch.tensor([0.25, 0.25, 0.4, 0.25])
    )
    torch.testing.assert_close(
        params["means"][2:], original_means[torch.tensor([0, 0, 1, 0])]
    )


def test_sample_add_compacts_duplicate_sampled_sources(monkeypatch):
    from gsplat.strategy import ops as gsplat_ops

    params = _make_cpu_params(torch.tensor([0.8, 0.6], dtype=torch.float32))
    monkeypatch.setattr(
        gsplat_ops,
        "_multinomial_sample",
        lambda weights, n, replacement=True: torch.tensor([0, 0, 1, 0]),
    )

    calls = {}

    def fake_compute(opacities, scales, ratios, binoms=None, min_opacity=0.005):
        calls["ratios"] = ratios.clone()
        return (
            torch.tensor([0.25, 0.4], dtype=opacities.dtype),
            scales * torch.tensor([[0.5], [0.75]], dtype=scales.dtype),
        )

    monkeypatch.setattr(gsplat_ops, "compute_relocation", fake_compute)
    original_means = params["means"].detach().clone()

    gsplat_ops.sample_add(params, {}, {}, n=4)

    torch.testing.assert_close(calls["ratios"], torch.tensor([4, 2]))
    assert len(params["means"]) == 6
    torch.testing.assert_close(
        params["means"][2:], original_means[torch.tensor([0, 0, 1, 0])]
    )
    updated_opacity = torch.sigmoid(params["opacities"])
    torch.testing.assert_close(updated_opacity[:2], torch.tensor([0.25, 0.4]))
    torch.testing.assert_close(
        updated_opacity[2:], torch.tensor([0.25, 0.25, 0.4, 0.25])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_compute_relocation_high_opacity_hybrid_matches_positive_reference():
    ratios = torch.tensor([16, 41, 51, 80], device=device, dtype=torch.float32)
    opacities = torch.full((len(ratios),), 0.999999, device=device)
    scales = torch.linspace(
        0.2, 1.1, len(ratios) * 3, device=device, dtype=torch.float32
    ).reshape(-1, 3)

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, min_opacity=0.0
    )
    ref_opacities, ref_scales = _positive_integral_reference_relocation(
        opacities, scales, ratios, min_opacity=0.0
    )

    assert torch.isfinite(new_scales).all()
    torch.testing.assert_close(new_opacities.cpu(), ref_opacities, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(new_scales.cpu(), ref_scales, rtol=2e-5, atol=2e-6)
