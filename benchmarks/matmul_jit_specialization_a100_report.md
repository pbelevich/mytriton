# Version 29: JIT Runtime Specialization on A100

Date: 2026-10-26

Branch: `version-29-jit-specialization`

Base: Version 28 commit `5b11db4`

## Result

Version 29 adds bounded runtime facts to the JIT compilation key and uses them
to generate a cheaper aligned variant of the SM80 static GEMM pipeline. The
specialized variant removes repeated pointer and leading-dimension alignment
tests from the hot `cp.async` predicate. It deliberately retains per-CTA bounds
checks and the masked scalar fallback, because divisibility alone does not prove
that a caller did not overlaunch the grid.

The change is correct on full and irregular tiles, but the A100 measurements do
not establish a standalone throughput improvement. It is compiler groundwork
for a later launch-contract specialization that can safely remove the remaining
bounds branch and fallback code from an aligned binary.

## Environment

- NVIDIA A100-SXM4-40GB, compute capability 8.0, 108 SMs
- PyTorch 2.11.0+cu130
- CUDA 13.0
- CuPy 14.0.1
- BF16 inputs and FP32 accumulation
- `BM=128`, `BK=32`, `BN=128`, four warps
- raw already-compiled kernel timing with CUDA events

## Same-session comparison

The first 30-sample run used the exact Version 28 base commit and Version 29 in
the same Colab session:

| Shape | Version 28 | Version 29 | Raw change | torch in V28 run | torch in V29 run |
|---:|---:|---:|---:|---:|---:|
| `4096^3` | 159.78 TFLOP/s | 176.60 TFLOP/s | +10.5% | 196.80 TFLOP/s | 217.89 TFLOP/s |
| `8192^3` | 207.65 TFLOP/s | 208.43 TFLOP/s | +0.4% | 262.18 TFLOP/s | 244.98 TFLOP/s |

The apparent `4096^3` improvement is not attributable to Version 29: torch
accelerated by essentially the same amount between those two measurements.
At `8192^3`, where each sample is longer, the two mytriton kernels are nearly
identical despite a material torch clock-state shift.

A second interleaved 50-sample sequence produced the following results:

| Order | Variant | Raw time | Raw throughput | torch throughput |
|---:|:---|---:|---:|---:|
| 1 | Version 29 | 0.7583 ms | 181.25 TFLOP/s | 198.84 TFLOP/s |
| 2 | Version 28 | 0.7025 ms | 195.65 TFLOP/s | 244.92 TFLOP/s |
| 3 | Version 29 | 0.8079 ms | 170.11 TFLOP/s | 199.14 TFLOP/s |
| 4 | Version 28 | 0.7680 ms | 178.96 TFLOP/s | 199.14 TFLOP/s |

The direction changes with the A100 clock state and benchmark order. Version
29 leaves the reported resource footprint unchanged at 213 registers per
thread, 49,152 bytes of shared memory, and 12.5% estimated occupancy. The
responsible conclusion is therefore **performance-neutral within the variance
of this Colab session**, rather than claiming either the best speedup or the
worst regression from these runs.

## Generated-code difference

Version 28 puts bounds, stride alignment, and pointer alignment into every
static-pipeline fast-path predicate. Version 29 records facts for runtime
arguments and emits an aligned cache variant whose predicate contains only the
four tile-bound checks. Calls whose pointers or dimensions do not satisfy the
required facts select a separate generic cache entry with the original guards.

The implementation caps pointer alignment at 16 bytes and integer divisibility
at 128. This avoids caching complete addresses or scalar values while retaining
the facts currently useful to 128-bit copies and `128 x 32 x 128` tiles.

## Correctness

The complete A100 suite excluding unavailable MLIR bindings reached 614 passed
tests before one old cache-cardinality assertion failed. The assertion expected
all tested dimensions to share one artifact; Version 29 intentionally creates
four bounded divisibility classes. After updating that contract, the affected
SM80 execution test passed for `K=32/64/96/128` and the irregular
`117 x 29 @ 29 x 109` case. The local full quality gate reports 622 passed and
4 architecture-dependent skips.

## Next step

Alignment specialization removes work from the predicate but leaves the
uniform CTA bounds branch and fallback body in the binary. A truly branch-free
aligned kernel needs an explicit launch relation in the cache key: the compiler
must prove that the grid contains exactly the valid CTA rectangle for the
runtime M and N. That proof should be added before deleting edge handling; doing
so from divisibility alone would make overlaunched kernels access memory out of
bounds.
