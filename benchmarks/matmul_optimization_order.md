# Performance-oriented matmul optimization order

This note records the A100 comparison between the Version 21 scalar
shared-memory operand path and the Version 22 `ldmatrix` path. It also proposes
an optimization order that prioritizes end-to-end GEMM throughput instead of
individual instruction counts.

## Measurement context

- Date: 2026-09-21
- GPU: NVIDIA A100-SXM4-40GB (`sm_80`), 108 SMs
- CUDA runtime: 12.8
- PyTorch: 2.11.0+cu128
- CuPy: 14.0.1
- Version 21: commit `27571025a9fc9521b67fa765efa85719ab55566f`
- Version 22: commit `d87828ec288ffaadade9fd426388b991495c58a9`
- Problem: BF16 `4096 x 4096 @ 4096 x 4096`
- Tile: `BM=32`, `BK=16`, `BN=32`
- Timing: three warmups, ten repetitions, CUDA graphs enabled
- Profiling: Nsight Compute with 43 replay passes

The benchmark time below is the primary performance measurement. Nsight
Compute adds replay overhead, so its kernel duration is useful only for a
relative comparison between profiles captured with the same command.

## Results

| Variant | Raw time | Throughput | Registers/thread | Static shared memory | Achieved occupancy |
|---|---:|---:|---:|---:|---:|
| Version 21, scalar `LDS` | 9.8263 ms | 13.99 TFLOP/s | 94 | 8192 B | 26.96% |
| Version 22, padded `ldmatrix` | 15.1414 ms | 9.08 TFLOP/s | 80 | 9728 B | 21.76% |
| Version 22, `ldmatrix` without padding | 15.1311 ms | 9.08 TFLOP/s | 80 | 8192 B | 26.42% |

Relative to Version 21, the padded Version 22 kernel takes about 54% longer
and loses about 35% of its throughput.

The no-padding variant was a temporary controlled experiment. It was not
committed. Its purpose was to restore Version 21's shared-memory footprint
while retaining the grouped `ldmatrix` lowering.

## Nsight Compute comparison

| Metric | Version 21 | Version 22, padded | Version 22, no padding |
|---|---:|---:|---:|
| NCU kernel duration | 11.94 ms | 17.99 ms | 18.02 ms |
| Compute (SM) throughput | 52.67% | 33.19% | 31.53% |
| SM busy | 53.05% | 33.41% | 31.54% |
| Eligible warps per scheduler | 0.97 | 0.49 | 0.42 |
| No eligible warp | 46.78% | 65.05% | 68.46% |
| Long scoreboard stall | 4.03 | 6.58 | 8.80 |
| Instructions executed | 2,973,630,464 | 2,818,736,128 | 2,684,436,480 |
| Excessive shared wavefronts | 238,026,752 | 3,145,728 | 137,363,456 |

Both generated kernels contain 16 static `HMMA.1688.F32.BF16`
instructions. Version 21 loads fragments through scalar shared-memory
instructions. The grouped Version 22 kernel replaces those operand loads with
two `LDSM.16.M88.4` and two `LDSM.16.MT88.4` instructions.

## What improved

Version 22 successfully implemented the intended hardware fragment-loading
mechanism:

- A and B fragments are loaded cooperatively by a warp.
- Four reusable fragment groups require only four static `LDSM` instructions.
- Transposed B fragments are produced directly by `ldmatrix.trans`.
- Padded layouts reduce excessive shared wavefronts by about 98.7% relative
  to Version 21.
- The kernel executes about 5% fewer instructions and uses fewer registers.
- Sampled numerical results remain exact relative to the Version 21 result.

These are useful compiler and code-generation improvements, but they did not
improve end-to-end throughput.

## Why the kernel became slower

The current kernel assigns one warp to an entire CUDA block and performs
synchronous global-to-shared staging. That warp eventually reaches a short
dependency chain:

```text
shared-memory barrier -> LDSM -> HMMA
```

There are not enough other ready warps or independent instructions to hide
the fragment-load latency. The scheduler evidence is direct:

- eligible warps per scheduler fell from 0.97 to 0.49;
- cycles with no eligible warp increased from 46.78% to 65.05%;
- long-scoreboard stalls increased from 4.03 to 6.58;
- SM busy fell from 53.05% to 33.41%.

The scalar Version 21 path executes more instructions and suffers severe bank
conflicts, but its independent scalar loads, packing operations, and address
work expose more instruction-level parallelism. For this unusually small
one-warp block, that extra independent work hides latency better than the much
shorter `LDSM -> HMMA` chain.

Padding is not the root cause. Removing it restores the theoretical and
achieved occupancy, but performance remains at 9.08 TFLOP/s. Without padding,
bank conflicts and long-scoreboard stalls increase enough to cancel the
occupancy gain. Padding therefore exchanges shared-memory capacity for lower
load latency; neither side of that trade-off fixes the lack of runnable work.

`ldmatrix` is not intrinsically slower than scalar fragment loading. It is
being used by a kernel topology that cannot exploit it yet.

## Recommended optimization order

The pedagogical order used to introduce individual compiler mechanisms is
valid, but a performance-oriented implementation should follow the order
below.

### 0. Preserve a trustworthy baseline

Before changing the kernel, retain Version 21 as a selectable reference path.
For every optimization, record:

- correctness and maximum error;
- raw and CUDA-graph time;
- generated CUDA, PTX, and SASS;
- registers, shared memory, and occupancy;
- eligible-warps and stall counters.

An optimization should not be called successful solely because it removes
instructions or bank conflicts.

### 1. Introduce a multi-warp CTA and a larger block tile

This should have preceded the `ldmatrix` performance work. Several warps
should cooperate on one larger C tile, for example by assigning each warp a
`32 x 32` output subtile inside a larger block tile. The staged A and B data
can then be reused by multiple warps.

The important outcomes are:

- more ready warps while another warp waits on a dependency;
- more reuse per global-memory load;
- a useful block-level `BM x BN` independent of the warp MMA tile;
- an explicit `warp_id -> output subtile` mapping.

Exit criterion: the kernel must have materially more eligible warps per
scheduler without losing correctness or exceeding resource limits.

### 2. Vectorize and coalesce global-to-shared staging

The current cooperative loader moves scalar elements synchronously. Before
introducing asynchronous copies, establish a good access pattern with aligned
vector loads and stores, such as 128-bit transactions where alignment and
boundary masks permit them.

Exit criterion: inspect SASS and memory counters to verify fewer load/store
instructions, coalesced global traffic, and no correctness regressions on edge
tiles.

### 3. Integrate `ldmatrix` into the multi-warp kernel

Once the CTA has enough independent warps, use Version 22's aligned fragment
addressing and grouped `ldmatrix.x1`, `x2`, and `x4` lowering. Keep the scalar
fragment loader available as an A/B benchmark path until `ldmatrix` wins in
end-to-end timing.

Do not spend more time reducing four static `LDSM` instructions in the current
one-warp kernel. The profile shows that their count is not the limiting metric.

Exit criterion: `ldmatrix` must outperform the scalar fragment path under the
same tile shape, staging strategy, and launch configuration.

### 4. Tune the shared-memory layout under the real CTA geometry

Choose padding or swizzling only after the number of warps, block tile, and
stage count are known. A layout that removes every bank conflict can still be
slower if its larger allocation reduces the number of resident CTAs.

Measure at least:

- excessive shared wavefronts;
- long-scoreboard stalls;
- resident blocks and achieved occupancy;
- end-to-end time.

Exit criterion: the chosen layout must beat both the unpadded layout and the
scalar Version 21 baseline. Zero bank conflicts are not themselves an exit
criterion.

### 5. Build a real software pipeline

The existing ping-pong buffers provide storage for multiple stages, but the
kernel still needs a schedule that overlaps work:

```text
load K tile i + 1 while computing K tile i
```

First validate this overlap with ordinary vectorized loads and explicit stage
rotation. Arrange barriers so that a stage is neither consumed before it is
ready nor overwritten while a warp still uses it.

Exit criterion: the profile must show lower long-scoreboard pressure and more
continuous issue of MMA instructions.

### 6. Replace the pipeline copies with `cp.async`

`cp.async` should optimize an already-correct multi-stage schedule, not define
the schedule from scratch. Add commit/wait group management, zero filling for
masked edges, and architecture guards for `sm_80+`.

Exit criterion: global-to-shared latency overlaps tensor-core computation and
the asynchronous path beats the synchronous pipelined path.

### 7. Remove or reduce the shared-memory epilogue

The current tensor-core result is redistributed through shared memory before
the row-major store. Improve fragment-to-output ownership, use vectorized
stores when possible, and avoid a full spill/reload cycle for layouts that can
be stored directly.

Exit criterion: fewer epilogue shared-memory instructions and barriers with
unchanged output semantics.

### 8. Add autotuning

Only after the main execution pipeline is sound should `mytriton.autotune`
search configurations such as:

- block `BM`, `BN`, and `BK`;
- warp tile shape;
- number of warps;
- number of pipeline stages;
- shared-memory padding or swizzle;
- vector width.

Reject configurations that exceed register, shared-memory, thread, or target
constraints before benchmarking them. Cache the winner by problem shape,
dtype, and CUDA target.

### 9. Add advanced specializations last

After the basic GEMM is competitive, consider split-K, persistent CTAs,
architecture-specific warp-group instructions, and shape-specific epilogues.
These features should not compensate for a poorly scheduled base kernel.

## Immediate conclusion for Version 22

Version 22 is a useful infrastructure milestone: it provides correct hardware
fragment addressing and grouped `ldmatrix` lowering. It is not a performance
win in the current one-warp kernel and should be documented as a temporary
regression.

The next performance lesson should not further micro-optimize `LDSM` count.
It should introduce a larger multi-warp CTA tile and use the existing scalar
and `ldmatrix` paths as competing baselines. Vectorized global-to-shared loads
should follow immediately, before software pipelining and `cp.async`.
