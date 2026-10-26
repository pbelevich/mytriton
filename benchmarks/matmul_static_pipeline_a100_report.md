# Version 27: Static Register Pipeline on A100

Date: 2026-10-25

Branch: `version-27-register-pipeline`

Baseline: `a6c650c` (`Version 26: Add pipelined Tensor Core GEMM`)

## Summary

Version 27 specializes the SM80 `128x32x128` Tensor Core tile with four
warps, warp-raked operand copies, precomputed shared-memory addresses, a
static three-phase stage schedule, and two live `K=16` register-fragment
sets. The final 30-sample BF16 `4096^3` run reaches **187.19 TFLOP/s**, or
86.8% of the simultaneously measured `torch.mm`. A same-day run reached
202.14 TFLOP/s. FP16 reaches **203.67 TFLOP/s**.

Using the conservative final BF16 result, this is 1.28x Version 26's 146.21
TFLOP/s result. More importantly, the generated hot loop no longer performs
dynamic shared-stage selection or reconstructs every `ldmatrix` address for
every K tile.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| Compute capability | 8.0 (`sm_80`) |
| SMs | 108 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| CuPy | 14.0.1 |
| Input | row-major BF16 or FP16 |
| Accumulator/output | FP32 |
| CTA tile | `128x32x128` |
| Warp layout | `2x2`, four `64x64` warp tiles |
| Timing | CUDA events, precompiled raw kernel |

The final tables use five warmups and 30 measured launches. CUDA Graph replay
is reported separately. Nsight Compute profiles the exact NVRTC cubin loaded
by CuPy.

## What changed

### Static three-phase shared schedule

Version 26 selected shared stages with `tile % 3`. Version 27 emits three
source-level phases whose stage numbers are constants:

```text
phase 0: write stage 2 | read stage 0
phase 1: write stage 0 | read stage 1
phase 2: write stage 1 | read stage 2
```

The generated loop advances by three K tiles. This removes stage modulo from
the hot address expressions and gives NVCC three explicit dependency chains.

### Warp-raked copies

Four warps cooperate on the A and B producers. Each warp owns 32 A rows and
eight B rows; every lane issues four aligned 16-byte copies for A and four for
B. This keeps global accesses coalesced while writing the existing XOR-swizzled
shared layout.

### Precomputed fragment addresses

Before the loop, the kernel computes:

- one shared store address for A and one for B per lane;
- four A `ldmatrix.x4` addresses per warp;
- four B `ldmatrix.x4.trans` addresses per warp;
- the address deltas for the second `K=16` fragment.

The loop then adds constant stage offsets rather than rebuilding swizzle,
matrix-selection, and shared-address-conversion expressions.

### Register fragments

For `BK=32`, both `K=16` operand sets are loaded before their corresponding
MMA groups. A controlled sequential-fragment variant reached 195.65 TFLOP/s
while the buffered variant reached 202.14 TFLOP/s, but the simultaneous
`torch.mm` rates differed enough that both variants had essentially the same
relative performance. The report therefore does not attribute the overall
gain to source-order fragment buffering alone. Static scheduling, address
hoisting, warp-raked copies, and lower cross-warp duplication are the combined
change that matters.

### Safe tails remain

The warp-raked `cp.async` path requires a full aligned tile. Partial tiles,
odd leading dimensions, or misaligned addresses use the Version 26 masked
cooperative producer and zero-fill invalid elements. The same generated
kernel passes both `128x128x32` and `117x109x29` execution cases.

## BF16 results

### Final size sweep

| Shape | torch raw | mytriton raw | mytriton graph | raw TFLOP/s | Relative to torch |
|---|---:|---:|---:|---:|---:|
| `1024^3` | 0.0379 ms | 0.0563 ms | 0.0420 ms | 38.13 | 67.3% |
| `2048^3` | 0.1485 ms | 0.1823 ms | 0.1679 ms | 94.25 | 81.5% |
| `4096^3` | 0.6369 ms | 0.7342 ms | 0.7178 ms | **187.19** | **86.8%** |
| `8192^3` | 4.1124 ms | 5.3048 ms | 5.2859 ms | **207.27** | 77.5% |

The public wrapper is intentionally excluded from throughput conclusions. At
`4096^3` it takes 1.9108 ms because Python-side launch overhead is still much
larger than a sub-millisecond kernel.

### Same-day variability

The shared Colab A100 changed clock state between runs:

| Run | torch TFLOP/s | mytriton TFLOP/s | Same-run ratio |
|---|---:|---:|---:|
| buffered experiment | 251.82 | **202.14** | 80.3% |
| final 30-sample sweep | 215.78 | **187.19** | 86.8% |
| profiling setup | 198.11 | 159.78 | 80.7% |

For this reason, the final 30-sample value is the headline and same-run ratios
are reported alongside absolute throughput.

## FP16 result

| Implementation | Median time | Throughput | Relative to torch |
|---|---:|---:|---:|
| `torch.mm` | 0.5448 ms | 252.29 TFLOP/s | 100% |
| mytriton raw | 0.6748 ms | **203.67 TFLOP/s** | **80.7%** |
| mytriton graph | 0.6861 ms | 200.32 TFLOP/s | 79.4% |

## Nsight Compute

NCU measured the BF16 kernel at 840.58 us. The table compares the exact V26
and V27 NVRTC cubins:

| Metric | Version 26 | Version 27 |
|---|---:|---:|
| Registers/thread | 126 | 213 |
| Static shared memory | 49.15 KiB | 49.15 KiB |
| Theoretical occupancy | 25.00% | 12.50% |
| Achieved occupancy | 23.95% | 12.44% |
| Compute throughput | 55.36% | **67.61%** |
| Memory throughput | 49.69% | 47.64% |
| Issue slots busy | 42.49% | 28.56% |
| Active warps/scheduler | 3.83 | 1.99 |
| Eligible warps/scheduler | 0.75 | 0.37 |
| No eligible warp | 55.47% | 69.95% |
| Warp cycles/issued instruction | 8.60 | **6.61** |
| Long-scoreboard stall | 0.20 | 0.37 inst/warp |

Version 27 deliberately trades occupancy for more work and reuse per warp.
Despite half as many resident warps, SM compute utilization rises and each
issued instruction requires fewer scheduler cycles. The remaining 69.95%
no-eligible-warp rate shows why this is not the end state: the next backend
work should tune instruction ordering and the epilogue, then select schedules
per shape instead of treating one tile as universal.

## Validation

- Local `make`: 604 passed, 4 skipped; Ruff and mypy pass.
- A100 CUDA execution suite: 32 passed. Three MLIR execution variants are
  unavailable in the Colab image and were excluded from the CUDA result.
- A100 full and irregular production-tile execution cases pass.
- BF16 and FP16 benchmark samples match the PyTorch FP32-accumulation
  reference with zero sampled maximum error for the measured square shapes.
- Other schedules retain the Version 26 dynamic pipeline.
- SM75 retains the synchronous staging fallback.

## Reproduction

Upload the current checkout and run the default BF16 benchmark:

```bash
./tools/sync_colab.sh
./benchmarks/run_matmul_colab.sh
```

Run the complete A100/Nsight Compute workflow:

```bash
./benchmarks/profile_matmul_colab.sh benchmarks/artifacts/version27-profile
```

The benchmark defaults are `128x32x128`, four warps, and BF16 inputs.

## Conclusion

Version 27 demonstrates that a fast GEMM pipeline is more than adding
`cp.async`. The compiler must expose a predictable instruction schedule,
choose a cooperative copy mapping that matches the shared layout, and stop
recomputing fragment addresses in the hot loop. Those changes lift mytriton
from Version 26's 146.21 TFLOP/s final result to 187.19 TFLOP/s, with more
than 200 TFLOP/s observed for both BF16 and FP16.

The next milestone should optimize grid/epilogue behavior and make schedule
selection shape-aware before turning the measured choices into
`mytriton.autotune`.
