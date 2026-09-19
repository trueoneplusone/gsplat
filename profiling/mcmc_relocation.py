# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Profile MCMC relocation math and duplicate-source bookkeeping.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m profiling.mcmc_relocation \
        --source-count 262144 --total 8000000 --alive 100000
"""

import argparse
import math
import statistics
from typing import Callable, Sequence

import torch

from gsplat.relocation import compute_relocation


def _binomial_table(n_max: int, device: torch.device) -> torch.Tensor:
    table = torch.zeros((n_max, n_max), device=device)
    for n in range(n_max):
        for k in range(n + 1):
            table[n, k] = math.comb(n, k)
    return table


def _cuda_median_ms(
    fn: Callable[[], object], warmup: int, repeats: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del result
    return statistics.median(times), min(times), max(times)


def profile_relocation_op(
    source_count: int,
    ratios: Sequence[int],
    opacity: float,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> None:
    opacities = torch.full((source_count,), opacity, dtype=torch.float32, device=device)
    scales = torch.ones((source_count, 3), dtype=torch.float32, device=device)
    legacy_binoms = _binomial_table(51, device)

    print("\nRelocation op")
    print("ratio,median_ms,min_ms,max_ms")
    for ratio in ratios:
        ratio_tensor = torch.full(
            (source_count,), ratio, dtype=torch.int32, device=device
        )

        def run():
            return compute_relocation(
                opacities,
                scales,
                ratio_tensor,
                legacy_binoms,
                min_opacity=0.005,
            )

        median_ms, min_ms, max_ms = _cuda_median_ms(run, warmup, repeats)
        print(f"{ratio},{median_ms:.6f},{min_ms:.6f},{max_ms:.6f}")


def profile_sampling_layout(
    total: int,
    alive: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> None:
    dead = total - alive
    generator = torch.Generator(device=device)
    generator.manual_seed(20260919)

    alive_indices = torch.randperm(total, device=device)[:alive]
    sampled_local = torch.randint(
        alive, (dead,), generator=generator, device=device, dtype=torch.int64
    )
    sampled_global = alive_indices[sampled_local]
    opacities = torch.rand(total, generator=generator, device=device) * 0.8 + 0.1
    scales = torch.rand((total, 3), generator=generator, device=device) + 0.1

    def expanded():
        counts = torch.bincount(sampled_global)
        ratios = counts[sampled_global] + 1
        return (
            counts,
            ratios,
            opacities[sampled_global],
            scales[sampled_global],
        )

    def compact():
        counts = torch.bincount(sampled_local, minlength=alive)
        source_local = counts.nonzero(as_tuple=True)[0]
        source_global = alive_indices[source_local]
        return (
            counts,
            source_local,
            source_global,
            counts[source_local] + 1,
            opacities[source_global],
            scales[source_global],
        )

    def timed_and_peak(fn: Callable[[], object]):
        timing = _cuda_median_ms(fn, warmup, repeats)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = fn()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline
        del result
        return timing, peak_delta

    old_timing, old_peak = timed_and_peak(expanded)
    new_timing, new_peak = timed_and_peak(compact)
    counts = torch.bincount(sampled_local, minlength=alive)
    nonzero = counts[counts > 0]

    print("\nDuplicate-source bookkeeping")
    print(f"total={total} alive={alive} dead={dead}")
    print(
        f"unique_sources={nonzero.numel()} "
        f"mean_copies={nonzero.float().mean().item():.3f} "
        f"max_copies={nonzero.max().item()}"
    )
    print(
        "expanded_ms="
        f"{old_timing[0]:.6f} compact_ms={new_timing[0]:.6f} "
        f"speedup={old_timing[0] / new_timing[0]:.3f}x"
    )
    print(
        f"expanded_peak_mb={old_peak / 2**20:.3f} "
        f"compact_peak_mb={new_peak / 2**20:.3f} "
        f"peak_reduction={old_peak / new_peak:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-count", type=int, default=262_144)
    parser.add_argument(
        "--ratios",
        type=int,
        nargs="+",
        default=[8, 16, 32, 51, 79, 120, 1024, 4096],
    )
    parser.add_argument("--opacity", type=float, default=0.35)
    parser.add_argument("--total", type=int, default=8_000_000)
    parser.add_argument("--alive", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA")
    if not 0 < args.alive <= args.total:
        raise ValueError("--alive must be in (0, --total]")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    profile_relocation_op(
        args.source_count,
        args.ratios,
        args.opacity,
        args.warmup,
        args.repeats,
        device,
    )
    profile_sampling_layout(
        args.total,
        args.alive,
        args.warmup,
        args.repeats,
        device,
    )


if __name__ == "__main__":
    main()
