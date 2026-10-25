# Version 26: Pipelined Tensor Core GEMM on A100

Date: 2026-10-25

Branch: `version-26-pipelined-tensor-core-gemm`

Baseline: `25f6e63` (`Version 25: Add SM80 production tensor tile`)

## Summary

Version 26 replaces synchronous operand staging with a three-stage SM80
`cp.async` pipeline. The selected `128x32x128`, eight-warp BF16 kernel reaches
146.21 TFLOP/s in the final 30-sample run and reached 162.49 TFLOP/s in the
warp-count sweep. The conservative final result is a 2.31x improvement over
Version 25's 63.19 TFLOP/s default and 73.6% of the simultaneously measured
198.55 TFLOP/s `torch.mm` result.

The improvement comes from overlapping the global-to-shared transfer of a
future K tile with `ldmatrix` and Tensor Core work on the current tile. The
larger CTA tile now has enough independent work to benefit from its operand
reuse rather than being dominated by synchronous staging latency.

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
| Problem | `4096 x 4096 @ 4096 x 4096` |
| Timing | CUDA events, precompiled raw kernel |

The final BF16 run used 10 warmups and 30 measured launches. CUDA Graph timing
was recorded separately. Nsight Compute used the exact NVRTC cubin loaded by
CuPy.

## What changed

The generated kernel now has:

- three shared-memory stages for A and B;
- two-tile asynchronous prologue;
- 16-byte `cp.async.cg.shared.global.L2::128B` copies on aligned full tiles;
- scalar zero-filling fallback for partial or misaligned vectors;
- explicit `cp.async.commit_group` and `cp.async.wait_group 1` operations;
- overlap between the next global copy and current `ldmatrix`/MMA work;
- K-sliced operand-fragment consumption;
- an eight-warp `128x32x128` benchmark default.

The scalar fallback is necessary for shapes such as `117x109x29`: row strides
of 29 or 109 elements do not preserve 16-byte alignment. It keeps the fast
path free of per-vector bounds checks while retaining correct masked tails.

## BF16 results

### Final measurement

| Implementation | Median time | Throughput | Relative to torch |
|---|---:|---:|---:|
| `torch.mm` | 0.6922 ms | 198.55 TFLOP/s | 100% |
| mytriton raw | 0.9400 ms | **146.21 TFLOP/s** | **73.6%** |
| mytriton CUDA Graph | 0.9226 ms | 148.97 TFLOP/s | 75.0% |
| mytriton wrapper | 1.8995 ms | 72.35 TFLOP/s | 36.4% |

The public Python wrapper still includes launch-path overhead. Compiler kernel
optimization is therefore evaluated with the raw precompiled launch.

### Warp-count sweep

| CTA tile | Warps | Registers/thread | Static shared | Occupancy estimate | Raw TFLOP/s |
|---|---:|---:|---:|---:|---:|
| `128x32x128` | 4 | 221 | 48 KiB | 12.5% | 118.83 |
| `128x32x128` | 8 | 126 | 48 KiB | 25.0% | **162.49** |
| `128x32x128` | 16 | 95 | 48 KiB | 25.0% | 107.33 |

Eight warps are the best compromise. Four warps assign a `64x64` accumulator
tile to each warp and create excessive register pressure. Sixteen warps reduce
the warp tile, but do not increase occupancy beyond 25% and perform more
fragment and scheduling work.

Timing varies with the shared Colab A100 clock state: repeated eight-warp runs
ranged from 130.56 to 162.49 TFLOP/s. The report uses the 30-sample 146.21
TFLOP/s result as the headline rather than the maximum observation.

## FP16 result

The same pipeline is type-independent:

| Implementation | Median time | Throughput |
|---|---:|---:|
| `torch.mm` FP16 | 0.5417 ms | 253.72 TFLOP/s |
| mytriton FP16 raw | 0.8274 ms | **166.11 TFLOP/s** |

## Nsight Compute

NCU measured the eight-warp BF16 kernel at 1.03 ms:

| Metric | Version 25, 4 warps | Version 26, 8 warps |
|---|---:|---:|
| Registers/thread | 202 | 126 |
| Static shared memory | 32 KiB | 48 KiB |
| Theoretical occupancy | 12.50% | 25.00% |
| Achieved occupancy | 12.31% | 23.95% |
| Compute throughput | 15.20% | 55.36% |
| Memory throughput | 17.67% | 49.69% |
| Issue slots busy | 13.04% | 42.49% |
| Active warps/scheduler | 1.97 | 3.83 |
| Eligible warps/scheduler | 0.16 | 0.75 |
| No eligible warp | 86.45% | 55.47% |
| Warp cycles/issued instruction | 14.53 | 8.60 |
| Long scoreboard stall | 9.6 | 0.20 inst/warp |

The counters confirm the intended causal change: the kernel spends much less
time waiting for synchronous operand transport, issues instructions more
often, and keeps substantially more Tensor Core work in flight.

## Correctness and validation

- Full `128x128x32` and irregular `117x109x29` execution cases pass on A100.
- BF16 and FP16 `4096^3` benchmark outputs have zero maximum sampled error
  against the PyTorch reference under the benchmark's comparison method.
- SM75 retains synchronous staging because `cp.async` is target-gated to
  `sm_80+`.
- Partial and misaligned vectors use scalar masked zero-fill staging.

## Reproduction

Upload the current checkout and run the default benchmark:

```bash
./tools/sync_colab.sh
./benchmarks/run_matmul_colab.sh
```

Run the complete profiling workflow on the remote A100:

```bash
colabrun --timeout 1800 \
  --cwd /content/mytriton \
  --notebook tests/mytriton_colab_tests.ipynb \
  bash benchmarks/profile_matmul_a100.sh benchmarks/artifacts/version26-profile
```

## Remaining gap

Version 26 is not yet a cuBLAS replacement. More than half of scheduler cycles
still have no eligible warp, and the compiler emits a generic dynamic-stage
pipeline rather than a fully unrolled architecture-specific schedule. The
next optimization should interleave `ldmatrix` for the next register fragment
with MMA on the current fragment, reduce address-generation work, and tune the
epilogue and stage schedule per shape.
