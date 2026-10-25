# Version 22 Multi-Warp Matmul Performance on NVIDIA A100

## Summary

Version 22 improves the `mytriton` Tensor Core matrix multiplication from
approximately **12.6–14.1 TFLOP/s** to **22.9–24.2 TFLOP/s** on an NVIDIA
A100-SXM4-40GB. The best configuration is a `64 x 16 x 64` CTA tile executed
by four warps.

Compared with Version 21, the best Version 22 kernel provides:

- **1.72x speedup** for `4096 x 4096` matrix multiplication;
- **1.82x speedup** for `8192 x 8192` matrix multiplication;
- **42–45% lower kernel latency**;
- identical measured performance for FP16 and BF16 inputs;
- zero sampled error relative to the PyTorch FP32-accumulating reference.

The optimization is meaningful, but the kernel remains approximately
**9–12x slower than `torch.mm`**. Multi-warp CTA tiling fixes an important
structural limitation; it does not yet provide the efficient fragment loads
and asynchronous memory pipeline of a production GEMM.

## Benchmark metadata

- **Date:** October 24, 2026
- **Version 21 commit:** `27571025a9fc9521b67fa765efa85719ab55566f`
- **Version 22 commit:** `fbb9411d997923b25309076eae090944314f0db1`
- **Pull request:** [#23](https://github.com/pbelevich/mytriton/pull/23)
- **GPU:** NVIDIA A100-SXM4-40GB
- **Compute capability:** 8.0 (`sm_80`)
- **Streaming multiprocessors:** 108
- **PyTorch:** 2.11.0+cu130
- **PyTorch CUDA build:** 13.0
- **CuPy CUDA runtime:** 13.2
- **CuPy:** 14.2.0
- **Warmup iterations:** 5
- **Measured iterations:** 20
- **Timing method:** CUDA events
- **Correctness reference:** `torch.mm` with FP32 output

The repository was checked out independently at the Version 21 tag and the
Version 22 pull-request commit. This avoids comparing the new kernel against
stale benchmark results or an earlier experimental branch.

## Validation

The Version 22 pull-request checkout was tested on the A100 with CUDA execution
required:

```text
556 passed, 11 deselected in 3.29s
```

The deselected tests contain `mlir` in their node IDs. The Colab runtime did
not provide the optional MLIR Python bindings. All available CUDA unit,
code-generation, FP16 execution, and native BF16 execution tests passed,
including the new multi-warp CTA tests.

Every reported benchmark configuration produced:

```text
max sampled absolute error = 0.000e+00
```

The benchmark checks 8,192 output elements against PyTorch for each matrix and
tile configuration.

## Methodology

Three timings are available in `benchmark_matmul.py`:

- **wrapper:** normal `mytriton` launch path;
- **raw:** direct launch of the already compiled CuPy kernel;
- **graph:** replay of the raw launch through a CUDA Graph.

This report uses the median **raw kernel time** for Version 21 versus Version
22. Raw timing removes Python and compilation-cache lookup overhead while
retaining the generated CUDA kernel exactly as executed by the public wrapper.

The Version 21 reference configuration is its best established one-warp tile:

```text
BM=32, BK=16, BN=32
```

Version 22 was measured with four warps and these CTA tiles:

```text
32 x 16 x 32
32 x 32 x 32
64 x 16 x 64
64 x 32 x 64
```

A control run of the Version 22 compiler with `num_warps=1` reproduced Version
21 performance almost exactly. This confirms that the measured difference
comes from the multi-warp schedule rather than unrelated compiler changes.

## BF16 results

### 4096 x 4096 x 4096

PyTorch performance varied between benchmark processes, so it is shown as
context rather than used to compute the Version 21-to-Version 22 speedup.

| Version / tile | Warps | Raw time | TFLOP/s | Registers | Shared memory | Estimated occupancy | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Version 21, `32x16x32` | 1 | 9.7756 ms | 14.06 | 94 | 8,192 B | 28.1% | 0.000e+00 |
| Version 22, `32x16x32` | 4 | 12.3290 ms | 11.15 | 56 | 8,192 B | 56.2% | 0.000e+00 |
| Version 22, `32x32x32` | 4 | 9.5176 ms | 14.44 | 56 | 12,288 B | 56.2% | 0.000e+00 |
| Version 22, `64x16x64` | 4 | **5.6735 ms** | **24.22** | 95 | 24,576 B | 31.2% | 0.000e+00 |
| Version 22, `64x32x64` | 4 | 6.8024 ms | 20.20 | 96 | 32,768 B | 25.0% | 0.000e+00 |

For the best Version 22 tile:

```text
speedup          = 9.7756 / 5.6735 = 1.723x
throughput gain  = 72.3%
latency reduction = 42.0%
```

The corresponding Version 22 run measured `torch.mm` at 221.12 TFLOP/s. The
best `mytriton` kernel was therefore approximately 9.1x slower when comparing
the raw kernel with the eager PyTorch timing from the same process.

### 8192 x 8192 x 8192

| Version / tile | Warps | Raw time | TFLOP/s | Registers | Shared memory | Estimated occupancy | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Version 21, `32x16x32` | 1 | 87.4588 ms | 12.57 | 94 | 8,192 B | 28.1% | 0.000e+00 |
| Version 22, `32x16x32` | 4 | 106.2815 ms | 10.35 | 56 | 8,192 B | 56.2% | 0.000e+00 |
| Version 22, `32x32x32` | 4 | 76.9352 ms | 14.29 | 56 | 12,288 B | 56.2% | 0.000e+00 |
| Version 22, `64x16x64` | 4 | **47.9841 ms** | **22.91** | 95 | 24,576 B | 31.2% | 0.000e+00 |
| Version 22, `64x32x64` | 4 | 58.6481 ms | 18.75 | 96 | 32,768 B | 25.0% | 0.000e+00 |

For the best Version 22 tile:

```text
speedup          = 87.4588 / 47.9841 = 1.823x
throughput gain  = 82.3%
latency reduction = 45.1%
```

The corresponding Version 22 run measured `torch.mm` at 273.32 TFLOP/s. The
best `mytriton` kernel was approximately 11.9x slower.

## FP16 results

FP16 and BF16 produced effectively identical `mytriton` kernel performance.
Both types lower to the same `mma.sync.m16n8k8` tile structure and use the same
shared-memory and accumulator layouts.

### 4096 x 4096 x 4096

| Version / tile | Warps | Raw time | TFLOP/s | Max error |
|---|---:|---:|---:|---:|
| Version 21, `32x16x32` | 1 | 9.7766 ms | 14.06 | 0.000e+00 |
| Version 22, `32x16x32` | 4 | 12.3382 ms | 11.14 | 0.000e+00 |
| Version 22, `32x32x32` | 4 | 9.5160 ms | 14.44 | 0.000e+00 |
| Version 22, `64x16x64` | 4 | **5.6771 ms** | **24.21** | 0.000e+00 |
| Version 22, `64x32x64` | 4 | 6.8173 ms | 20.16 | 0.000e+00 |

The best configuration is **1.722x faster**, increases throughput by **72.2%**,
and reduces latency by **41.9%**.

### 8192 x 8192 x 8192

| Version / tile | Warps | Raw time | TFLOP/s | Max error |
|---|---:|---:|---:|---:|
| Version 21, `32x16x32` | 1 | 87.4332 ms | 12.58 | 0.000e+00 |
| Version 22, `32x16x32` | 4 | 106.2651 ms | 10.35 | 0.000e+00 |
| Version 22, `32x32x32` | 4 | 76.9367 ms | 14.29 | 0.000e+00 |
| Version 22, `64x16x64` | 4 | **47.9534 ms** | **22.93** | 0.000e+00 |
| Version 22, `64x32x64` | 4 | 58.6378 ms | 18.75 | 0.000e+00 |

The best configuration is **1.823x faster**, increases throughput by **82.3%**,
and reduces latency by **45.2%**.

## Control result: Version 22 with one warp

The Version 22 compiler was also launched with the old tile and
`num_warps=1`:

| Size | Version 21 | Version 22 control | Difference |
|---:|---:|---:|---:|
| 4096 | 14.06 TFLOP/s | 14.05 TFLOP/s | -0.1% |
| 8192 | 12.57 TFLOP/s | 12.58 TFLOP/s | +0.1% |

The one-warp path is therefore performance-neutral. Version 22 does not obtain
its speedup from an unrelated code-generation change.

## Why the larger CTA is faster

Multi-warp execution alone is not sufficient. Dividing the old `32 x 32`
output tile among four warps makes the kernel slower even though the estimated
occupancy increases from 28.1% to 56.2%.

The useful change is the combination of multiple warps with a larger
`64 x 64` CTA output tile:

1. **Four times more output per CTA.** A `64 x 64` tile computes 4,096 output
   elements instead of 1,024.
2. **Four times fewer CTAs.** For a square matrix, changing the output tile
   from `32 x 32` to `64 x 64` reduces the two-dimensional grid by four.
3. **More operand reuse.** The four warps share one staged `64 x BK` A tile and
   one `BK x 64` B tile. Separate one-warp CTAs would redundantly load the same
   rows of A and columns of B for neighboring output tiles.
4. **Warp-local register pressure remains manageable.** Each warp owns a
   `32 x 32` portion of the CTA result rather than the complete `64 x 64`
   result.
5. **Launch and synchronization work is amortized.** Grid traversal, shared
   staging, and block-level coordination cover more arithmetic per CTA.

The `BK=16` configuration is faster than `BK=32`. Increasing BK reduces the
number of outer K iterations, but raises shared-memory consumption from 24 KiB
to 32 KiB and reduces estimated occupancy from 31.2% to 25.0%. In the current
synchronous pipeline, the larger K tile does not compensate for that loss.

## Why the same-size four-warp tile is slower

For `BM=32`, `BN=32`, and four warps, each warp owns only a `16 x 16` output
tile. The CTA computes no more output than the Version 21 one-warp block, but it
introduces additional active warps, warp-coordinate arithmetic, and CTA-level
coordination. The smaller per-warp tile also provides less instruction-level
work with which to hide shared-memory fragment-load latency.

The result is a regression:

- BF16 4096: `14.06 -> 11.15 TFLOP/s`;
- BF16 8192: `12.57 -> 10.35 TFLOP/s`.

This is why `num_warps=4` should not be interpreted as a universal speedup.
The block tile and warp decomposition must be selected together.

## Remaining gap to a production GEMM

The best Version 22 kernel reaches only about 8–11% of the throughput observed
from PyTorch/cuBLAS in these runs. The generated kernel still has several
known limitations:

- scalar shared-memory fragment loads and explicit 16-bit packing;
- no `ldmatrix` operand loading;
- synchronous global-to-shared copies;
- no `cp.async` pipeline;
- no overlap between loading the next K tile and computing the current tile;
- no shared-memory swizzle designed for Tensor Core fragment access;
- fixed scheduling instead of autotuning;
- shared-memory accumulator redistribution before the final store.

The measurements suggest the following optimization order:

1. retain the `64 x 16 x 64`, four-warp CTA as the new reference topology;
2. introduce `ldmatrix` for warp-fragment loads from shared memory;
3. vectorize global-to-shared copies;
4. add `cp.async` and a multi-stage K pipeline;
5. add layout variants and autotuning for BM, BK, BN, and `num_warps`;
6. profile register dependencies, scheduler stalls, and shared-memory behavior
   with Nsight Compute after each structural change.

## Reproduction

Version 21 baseline:

```bash
python benchmarks/benchmark_matmul.py \
    --sizes 4096 8192 \
    --tiles 32x16x32 \
    --dtype bf16 \
    --warmup 5 \
    --repeats 20
```

Version 22 tile sweep:

```bash
python benchmarks/benchmark_matmul.py \
    --sizes 4096 8192 \
    --tiles 32x16x32 32x32x32 64x16x64 64x32x64 \
    --dtype bf16 \
    --num-warps 4 \
    --warmup 5 \
    --repeats 20
```

Replace `--dtype bf16` with `--dtype fp16` to reproduce the FP16 tables.

## Conclusion

Version 22 demonstrates that CTA topology matters more than adding an isolated
instruction-level optimization. Four warps assigned to the old output tile are
slower, while the same four warps applied to a larger `64 x 64` CTA tile nearly
double throughput by improving operand reuse and reducing the number of CTAs.

The implementation advances `mytriton` from approximately 14 TFLOP/s to
approximately 24 TFLOP/s without changing numerical results. It establishes a
better foundation for the next performance stages: `ldmatrix`, vectorized
copies, asynchronous staging, and autotuning.
