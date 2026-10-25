# Version 23 Vectorized Staging Performance on NVIDIA A100

## Summary

Version 23 replaces scalar FP16/BF16 global-to-shared operand copies with
aligned 128-bit copies while retaining a masked scalar fallback for partial
or misaligned vectors. On an NVIDIA A100-SXM4-40GB, the best `64 x 16 x 64`
four-warp matmul improves from approximately **22.9-24.2 TFLOP/s** in Version
22 to **38.5-40.8 TFLOP/s**.

Compared with Version 22, vectorized staging provides:

- **1.68x speedup** for both 4096 and 8192 square matrix multiplication;
- approximately **40.5% lower raw kernel latency**;
- identical measured FP16 and BF16 kernel performance;
- zero sampled error relative to the PyTorch FP32-accumulating reference;
- 64 registers per thread instead of 95 for the best Version 22 tile;
- an estimated occupancy increase from 31.2% to 37.5%.

The optimization closes a meaningful part of the gap to cuBLAS, but the
kernel remains about 5-7x slower than `torch.mm`. Nsight Compute shows that
the next bottleneck is no longer the global-to-shared copy width: MMA operand
fragments are still assembled through many scalar shared-memory loads.

## Benchmark metadata

- **Date:** October 24, 2026
- **Branch:** `version-23-vectorized-staging`
- **Base commit:** `1ea29866f8008d561c0e63b24a924c25c2700875`
- **Version 22 implementation:** `fbb9411d997923b25309076eae090944314f0db1`
- **GPU:** NVIDIA A100-SXM4-40GB
- **Compute capability:** 8.0 (`sm_80`)
- **Streaming multiprocessors:** 108
- **PyTorch:** 2.11.0+cu130
- **PyTorch CUDA build:** 13.0
- **CuPy:** 14.2.0
- **Tile:** `BM=64`, `BK=16`, `BN=64`
- **Warps per CTA:** 4
- **Warmup iterations:** 5 for BF16, 10 for the confirming FP16 run
- **Measured iterations:** 20 for BF16, 50 for the confirming FP16 run
- **Timing method:** CUDA events around the already compiled raw kernel

The Version 22 reference values come from
[`matmul_multi_warp_a100_report.md`](matmul_multi_warp_a100_report.md), which
used the same GPU, tile, launch configuration, and raw-kernel timing method.
PyTorch timings vary with GPU clocks and process state, so they are shown only
as context and are not used to calculate the Version 22-to-Version 23 speedup.

## Validation

The complete local suite passed on an RTX 2080:

```text
580 passed, 3 skipped
```

The CUDA 13 Colab A100 suite passed with CUDA execution required and optional
MLIR tests deselected because the runtime does not provide MLIR bindings:

```text
572 passed, 11 deselected
```

This includes native SM80 BF16 execution, FP16 execution, masked edge tiles,
CUDA-core low-precision dot, Tensor Core dot, and multi-warp CTA matmul.

## Performance

### BF16

| Matrix size | Version 22 raw | Version 23 raw | V22 TFLOP/s | V23 TFLOP/s | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 x 4096 x 4096 | 5.6735 ms | 3.3659 ms | 24.22 | **40.83** | **1.686x** | 40.7% |
| 8192 x 8192 x 8192 | 47.9841 ms | 28.5911 ms | 22.91 | **38.46** | **1.678x** | 40.4% |

Both runs reported a maximum sampled absolute error of `0.000e+00`.

### FP16

| Matrix size | Version 22 raw | Version 23 raw | V22 TFLOP/s | V23 TFLOP/s | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 x 4096 x 4096 | 5.6771 ms | 3.3679 ms | 24.21 | **40.81** | **1.686x** | 40.7% |
| 8192 x 8192 x 8192 | 47.9534 ms | 28.5957 ms | 22.93 | **38.45** | **1.677x** | 40.4% |

The nearly identical FP16 and BF16 results are expected: both formats use
two-byte elements, eight-element vector copies, the same CTA/warp layout, and
the corresponding `mma.sync.m16n8k8` instruction.

## Tile and warp sweep

The established Version 22 tiles were measured again after vectorization.

| Tile | Warps | 4096 TFLOP/s | 8192 TFLOP/s | Registers | Shared memory | Occupancy estimate |
|---|---:|---:|---:|---:|---:|---:|
| `32x16x32` | 4 | 20.49 | 20.07 | 40 | 8 KiB | 75.0% |
| `32x32x32` | 4 | 19.97 | 20.20 | 40 | 12 KiB | 75.0% |
| `64x16x64` | 4 | **40.83** | **38.46** | 64 | 24 KiB | 37.5% |
| `64x32x64` | 4 | 33.27 | 33.54 | 64 | 32 KiB | 25.0% |

Larger rectangular tiles did not improve the result:

| Tile | 4096 TFLOP/s | Registers | Shared memory | Occupancy estimate |
|---|---:|---:|---:|---:|
| `128x16x64` | 38.20 | 96 | 44 KiB | 18.8% |
| `64x16x128` | 32.64 | 96 | 44 KiB | 18.8% |

For `64x16x64`, two warps reached 32.59 TFLOP/s and eight warps reached 34.22
TFLOP/s. Four warps therefore remain the best schedule for this implementation.

## Generated code and SASS

Each thread in the best CTA stages eight adjacent low-precision values, or 16
bytes, from each operand. The generated fast path uses `uint4`:

```cuda
*reinterpret_cast<uint4 *>(&shared[index]) =
    *reinterpret_cast<const uint4 *>(&global[source_index]);
```

The fast path is guarded by full-vector bounds and 16-byte source alignment.
Partial edge vectors and misaligned source addresses execute an unrolled scalar
fallback, preserving the previous masking and zero-fill semantics.

The exact NVRTC cubin used by Nsight Compute contains one wide load/store pair
for each operand staging site:

```text
LDG.E.128
STS.128
LDG.E.128
STS.128
```

The scalar `LDG.E.U16` and `STS.U16` instructions remain in the binary because
the edge fallback is compiled into the same kernel. Interior aligned tiles take
the 128-bit path.

## Nsight Compute findings

The best BF16 kernel was profiled with Nsight Compute 2025.3.1. Profiling uses
multiple replay passes, so its 4.22 ms duration is diagnostic and must not be
compared directly with the normal benchmark median.

Key metrics:

| Metric | Value |
|---|---:|
| Registers per thread | 64 |
| Static shared memory | 24,576 bytes |
| Achieved occupancy | 36.0% |
| Theoretical occupancy | 37.5% |
| Memory throughput utilization | 93.64% |
| DRAM throughput utilization | 12.21% |
| Compute throughput utilization | 35.94% |
| Warp cycles per issued instruction | 15.96 |
| Long-scoreboard stalls | 5.93 cycles/instruction |
| MIO-throttle stalls | 2.79 cycles/instruction |
| Short-scoreboard stalls | 2.23 cycles/instruction |

High aggregate memory-pipeline utilization together with low DRAM utilization
means the kernel is not simply exhausting HBM bandwidth. The SASS still
contains many scalar `LDS.U16` instructions that assemble MMA fragments from
shared memory, and long-scoreboard stalls dominate. This identifies
warp-cooperative `ldmatrix` fragment loading as the next optimization target.

## Reproduction

Upload the current working tree to the Colab runtime selected in VS Code:

```bash
./tools/sync_colab.sh
```

Run the default best-tile benchmark:

```bash
./benchmarks/run_matmul_colab.sh
```

Capture generated CUDA, PTX, the exact NVRTC cubin, SASS, resource usage, and
an Nsight Compute report:

```bash
./benchmarks/profile_matmul_colab.sh
```

The sync script infers the `cuda12` or `cuda13` project extra from the remote
PyTorch CUDA build unless `MYTRITON_COLAB_EXTRAS` is explicitly provided.

## Conclusion

Version 23 validates the intended optimization order. Multi-warp CTA tiling in
Version 22 created enough reuse to make wider cooperative copies valuable;
Version 23 then removes most scalar global-to-shared staging overhead without
changing MMA semantics or thread ownership. The measured result is a stable
approximately 1.68x kernel speedup.

The next step should preserve this `64x16x64`, four-warp baseline and replace
scalar shared-to-register fragment assembly with `ldmatrix`, rather than
revisiting global-memory staging or increasing CTA size.
