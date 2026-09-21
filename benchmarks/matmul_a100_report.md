# A100 Matrix Multiplication Benchmark Report

## Snapshot

- **Benchmark date:** September 20, 2026
- **Artifact capture:** September 21, 2026 at 01:24 UTC
- **mytriton version:** Version 21 — composable warp MMA tiles
- **Commit:** [`5ef72da062e100829816a0d3216c0bbc2f654377`](https://github.com/pbelevich/mytriton/commit/5ef72da062e100829816a0d3216c0bbc2f654377)
- **Branch at the time of measurement:** `version-21-composable-warp-mma`
- **GPU:** NVIDIA A100-SXM4-40GB
- **Compute capability:** 8.0 (`sm_80`)
- **Streaming multiprocessors:** 108
- **PyTorch:** 2.11.0+cu128
- **CUDA reported by PyTorch:** 12.8
- **CuPy:** 14.0.1

The compiler and runtime sources under test correspond to the commit above.
The independent benchmark driver is available as
[benchmark_matmul.py](benchmark_matmul.py); it does not modify the compiler or
the generated kernel.

## Objective

The goal was to measure how far the Version 21 tensor-core matrix
multiplication kernel is from `torch.mm` on an A100, and then separate launch
overhead from problems inside the generated CUDA kernel.

The benchmark therefore measures three mytriton paths:

1. **Wrapper:** the normal public `mytriton` launch path.
2. **Raw:** direct launch of the already compiled CuPy `RawKernel`.
3. **Graph:** replay of the raw launch from a CUDA Graph.

If wrapper and raw differ, the overhead is outside the CUDA kernel. If raw and
graph differ, CUDA launch latency matters. The remaining difference between
graph/raw and `torch.mm` is caused primarily by the generated kernel.

## Methodology

- Input matrices use BF16 unless stated otherwise.
- Accumulation and output use FP32.
- `torch.mm(a, b, out=out, out_dtype=torch.float32)` is used for an
  equivalent mixed-precision reference.
- Every reported timing is the median of 10 measurements after 3 warm-up
  launches.
- Timings use CUDA events and exclude first-time mytriton and NVRTC
  compilation.
- Throughput is calculated as `2 * M * N * K / time`.
- CUDA Graph capture is performed only after compilation and warm-up.
- Correctness is checked against `torch.mm` using 8,192 evenly distributed
  output samples per configuration.
- The tile sweep covers all combinations of `BM in {16, 32}`,
  `BK in {16, 32}`, and `BN in {16, 32}`.

The main run can be reproduced with:

```bash
python benchmarks/benchmark_matmul.py \
    --sizes 4096 8192 \
    --dtype bf16 \
    --warmup 3 \
    --repeats 10 \
    --output matmul-a100-bf16.json
```

## BF16 results

### 4096 x 4096 x 4096

`torch.mm` took **0.6308 ms**, or **217.89 TFLOP/s**. Its CUDA Graph replay
took 0.6134 ms.

| BM | BK | BN | Wrapper, ms | Raw, ms | Graph, ms | Raw TFLOP/s | Slowdown vs. PyTorch | Registers/thread | Static shared memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 16 | 31.0825 | 30.6975 | 30.6755 | 4.48 | 48.7x | 48 | 3,072 B |
| 16 | 16 | 32 | 16.3492 | 16.0113 | 15.9488 | 8.58 | 25.4x | 72 | 5,120 B |
| 16 | 32 | 16 | 22.0703 | 21.6463 | 21.7006 | 6.35 | 34.3x | 56 | 5,120 B |
| 16 | 32 | 32 | 24.9231 | 24.5120 | 24.4951 | 5.61 | 38.9x | 64 | 8,192 B |
| 32 | 16 | 16 | 12.6797 | 12.3034 | 12.2875 | 11.17 | 19.5x | 64 | 5,120 B |
| **32** | **16** | **32** | **10.1432** | **9.7818** | **9.7526** | **14.05** | **15.5x** | **94** | **8,192 B** |
| 32 | 32 | 16 | 16.2596 | 15.8868 | 15.8531 | 8.65 | 25.2x | 72 | 8,192 B |
| 32 | 32 | 32 | 24.4480 | 24.0604 | 24.0456 | 5.71 | 38.1x | 80 | 12,288 B |

### 8192 x 8192 x 8192

`torch.mm` took **3.9593 ms**, or **277.70 TFLOP/s**. Its measured CUDA Graph
replay took 4.1795 ms.

| BM | BK | BN | Wrapper, ms | Raw, ms | Graph, ms | Raw TFLOP/s | Slowdown vs. PyTorch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 16 | 245.2874 | 244.8753 | 244.7068 | 4.49 | 61.8x |
| 16 | 16 | 32 | 125.4349 | 124.9551 | 124.9654 | 8.80 | 31.6x |
| 16 | 32 | 16 | 188.6925 | 188.0878 | 188.1303 | 5.85 | 47.5x |
| 16 | 32 | 32 | 184.2621 | 183.8751 | 183.9442 | 5.98 | 46.4x |
| 32 | 16 | 16 | 114.8462 | 114.3762 | 114.3588 | 9.61 | 28.9x |
| **32** | **16** | **32** | **87.8751** | **87.3651** | **87.3426** | **12.59** | **22.1x** |
| 32 | 32 | 16 | 150.3616 | 149.8629 | 149.8788 | 7.34 | 37.9x |
| 32 | 32 | 32 | 181.6637 | 181.1881 | 181.0872 | 6.07 | 45.8x |

All sampled correctness checks passed. The sampled maximum absolute error was
zero for every BF16 configuration in this run.

## FP16 cross-check

The best tile was also tested with FP16 inputs at 4096 x 4096 x 4096:

| Implementation | Time | Throughput |
|---|---:|---:|
| `torch.mm` | 0.6963 ms | 197.38 TFLOP/s |
| mytriton raw, 32 x 16 x 32 | 9.8038 ms | 14.02 TFLOP/s |

The raw FP16 and BF16 mytriton results are effectively identical. The main
bottleneck is therefore not the BF16 instruction variant or conversion path.

## Nsight Compute investigation

NVIDIA Nsight Compute 2025.1.1 (`ncu`) was run against the best BF16 tile,
`BM=32`, `BK=16`, `BN=32`, at 4096 x 4096 x 4096. The profiled launch was the
direct raw-kernel launch rather than the Python wrapper.

The primary profiling command was:

```bash
ncu \
    --target-processes all \
    --kernel-name 'regex:.*matmul_kernel.*' \
    --launch-skip 2 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --section WarpStateStats \
    --section SchedulerStats \
    python benchmarks/benchmark_matmul.py \
        --sizes 4096 \
        --tiles 32x16x32 \
        --warmup 0 \
        --repeats 1 \
        --no-graphs
```

Additional passes collected `ComputeWorkloadAnalysis`, `InstructionStats`,
`MemoryWorkloadAnalysis_Tables`, and `SourceCounters`.

Profiling requires many kernel replays and substantially distorts wall-clock
timings. The performance tables above therefore come from normal execution;
`ncu` was used only to diagnose the kernel.

The complete artifact capture can be reproduced with:

```bash
./benchmarks/profile_matmul_a100.sh
```

Each invocation creates a timestamped directory under `benchmarks/artifacts`
containing the benchmark JSON, generated CUDA, source metadata, CuPy's exact
NVRTC cubin, a readable PTX build of the same source, `cuobjdump` SASS and
resource reports, `nvdisasm` output, the raw `.ncu-rep`, and a text export of
all collected profiler sections. These generated captures are intentionally
git-ignored.

### Lowered CUDA, PTX, and SASS

The generated CUDA source for the best tile has SHA-256:

```text
3c086da1e14cff5914a00ebec6eac216f0d1bf1408df50405184484c374deb2f
```

The source declares exactly three shared arrays:

```cuda
__shared__ __nv_bfloat16 dot_lhs[1024];
__shared__ __nv_bfloat16 dot_rhs[1024];
__shared__ float mma_result[1024];
```

The two 1,024-element BF16 arrays contain two 512-element ping-pong stages.
Together they occupy 4,096 bytes. The FP32 result redistribution tile occupies
another 4,096 bytes.

The runtime K-loop still has the synchronous structure:

```text
select stage
scalar global-to-shared A loop
scalar global-to-shared B loop
barrier
ordinary shared loads and manual packing
16 x mma.sync
```

After the loop, the epilogue stores 32 accumulator values per lane to the FP32
shared tile, executes a second barrier, reloads those values in the ordinary
output layout, and performs masked global stores.

The PTX generated independently from the saved CUDA source contains:

| PTX operation | Static count |
|---|---:|
| `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32` | 16 |
| `ld.shared` | 64 |
| `st.shared` | 46 |
| `ld.global` | 14 |
| `st.global` | 32 |
| `bar.sync` | 2 |
| `ldmatrix` | 0 |
| `cp.async` | 0 |

This PTX is useful as a readable representation but is not assumed to be
bit-identical to CuPy's NVRTC pipeline. For the final machine-code analysis,
the capture script extracts the exact cubin loaded by CuPy's `RawKernel` cache
and passes it to `cuobjdump` and `nvdisasm`. Its SHA-256 is:

```text
0284d1bd5278243007b5a0d75100c2323c3ae9a5a75f7345ecf356bf36dadaf3
```

The resulting SASS confirms that the intended tensor-core instructions survive
compilation:

```text
HMMA.1688.F32.BF16 ...
```

There are exactly 16 static `HMMA.1688.F32.BF16` instructions. The relevant
static instruction mix is:

| SASS family | Static count | Role |
|---|---:|---|
| `HMMA.1688.F32.BF16` | 16 | Tensor-core computation |
| `LDS`/`LDS.U16` | 56 | Operand and epilogue shared loads |
| `LDG.E.U16` | 14 | Scalar global operand loads |
| `STS.U16`/`STS.64` | 30 | Operand staging and accumulator spill |
| `STG.E` | 32 | Output stores |
| `BAR.SYNC.DEFER_BLOCKING` | 2 | Staging and epilogue barriers |
| `IMAD` family | 177 | Address and index arithmetic |
| `ISETP` family | 78 | Bounds and control predicates |
| `LDGSTS` | 0 | No asynchronous global-to-shared copy |

The 56 shared loads split naturally into two groups. Before the MMA sequence,
the machine code uses 16 scalar `LDS.U16` instructions and eight 32-bit `LDS`
instructions to assemble the unique A and B fragments. After the second
barrier, 32 additional `LDS` instructions implement the shared-memory result
redistribution. No local-memory loads or stores appear, matching the reported
zero local bytes and confirming that the 94-register kernel does not spill.

The compiler does recover some reuse that was not explicit in the CUDA source:
A fragments are reused across N atoms and B fragments across M atoms. However,
the operands are still fed by ordinary shared loads rather than `LDMATRIX`, and
the large integer/predicate instruction population remains.

Nsight Compute reported 2,973,630,464 dynamically executed instructions. The
4096 problem executes approximately 67.1 million warp-level MMA instructions:

```text
16 MMA per K tile
* 256 K tiles
* 16,384 CUDA blocks
= 67,108,864 MMA instructions
```

The measured dynamic ratio is therefore roughly 44 executed instructions per
MMA. This includes staging, address generation, loop control, synchronization,
and the epilogue, and makes the feeding overhead visible even before comparing
against a tuned library kernel.

### Key profiler metrics

| Metric | Observed value |
|---|---:|
| Kernel grid | 128 x 128 blocks |
| Threads per block | 32 |
| Registers per thread | 94 |
| Static shared memory per block | 8.19 KB |
| Driver shared memory per block | 1.02 KB |
| Theoretical occupancy | 28.12% |
| Achieved occupancy | 26.94% |
| Achieved active warps per SM | 17.24 |
| Compute throughput | 52.73% |
| DRAM throughput | 4.21% |
| L2 hit rate | 90.06% |
| Scheduler cycles with no eligible warp | 46.66% |
| Average active warps per scheduler | 4.34 |
| Average eligible warps per scheduler | 0.97 |

The warp-state analysis attributed approximately 4.0 of the 8.1 cycles
between issued instructions to scoreboard dependencies on L1TEX operations.
This is about half of the issue interval.

### Shared-memory conflicts

The strongest profiler finding was the shared-memory access pattern:

- 101,187,584 shared-load requests were observed.
- Loads suffered an average **3.3-way bank conflict**.
- 236,581,993 bank conflicts were reported.
- Conflicts represented approximately **70%** of shared-load wavefronts.
- Source counters reported 238,026,752 excessive shared wavefronts, or about
  **50%** of all shared wavefronts.

This agrees with the current lowering. Low-precision MMA operands receive no
shared-memory padding because row padding is currently enabled only for F32 in
[`cuda_codegen.py`](../src/mytriton/cuda_codegen.py#L1464-L1471). Operand
fragments are then assembled using ordinary scalar shared-memory expressions
and manual packing in
[`emit_mma_warp_tile_operand_registers`](../src/mytriton/cuda_codegen.py#L460-L614),
rather than `ldmatrix`.

## Bottleneck analysis

### 1. The output tile is too small

The best Version 21 kernel computes only a 32 x 32 output tile per block, and
each block contains exactly one warp. The generated-code contract explicitly
requires 32 threads in
[`emit_mma_warp_tile_operand_registers`](../src/mytriton/cuda_codegen.py#L473-L477).

The tile's idealized global-memory arithmetic intensity, ignoring the output,
is only:

```text
BM * BN / (BM + BN) = 32 * 32 / (32 + 32) = 16 FLOP/byte
```

Larger multi-warp CTA tiles would reuse each A and B tile across substantially
more output values.

### 2. Shared-memory loads have severe bank conflicts

This is directly confirmed by `ncu`. Padding or swizzling the BF16/FP16
shared-memory layout should be addressed before drawing conclusions about the
maximum performance of the current MMA organization.

### 3. There is no `ldmatrix` lowering

MMA input fragments are loaded as individual C++ shared-memory elements and
packed into 32-bit registers. Besides causing bank conflicts, this emits much
more address and integer work than the hardware fragment-load instruction.
Nsight Compute reported the ALU pipeline as the most heavily utilized
pipeline, which is consistent with excessive address and packing work.

### 4. Double buffering does not overlap copying and compute

The current implementation selects one of two shared-memory stages using the
K-loop parity in
[`emit_for_range`](../src/mytriton/cuda_codegen.py#L1993-L2035). However, each
iteration still performs scalar cooperative loads followed by
`__syncthreads()` and then MMA. There is no `cp.async`, asynchronous commit or
wait group, or software-pipeline prologue/steady-state/epilogue.

The two buffers prevent immediate stage aliasing, but they do not yet hide
global-memory latency behind tensor-core work.

### 5. Global-to-shared copies are scalar

Cooperative staging uses a per-thread scalar loop with division, modulo,
bounds checks, and one element assignment in
[`emit_cooperative_load`](../src/mytriton/cuda_codegen.py#L1414-L1441). Vectorized
loads and `cp.async` would reduce instruction overhead and enable overlap.

### 6. The accumulator is spilled through shared memory

Before the ordinary output store, the MMA accumulator is converted into a
row-major shared buffer and synchronized in
[`emit_mma_accumulator_spill`](../src/mytriton/cuda_codegen.py#L866-L936). For a
32 x 32 FP32 result this consumes another 4,096 bytes of shared memory per
block. Together with double-buffered operands and CUDA's per-block reserved
shared memory, this limits the profiled kernel to 18 resident one-warp blocks
per SM.

### 7. Public-launch overhead matters for small matrices

At 1024 x 1024 x 1024 with the best tile:

- normal mytriton wrapper: 1.9517 ms;
- direct raw launch: 0.2171 ms;
- CUDA Graph raw replay: 0.2028 ms.

The public launcher performs signature binding, runtime argument analysis,
DLPack wrapping, cache lookup, and returns deep copies of the expression and
SSA graphs on every invocation. The final copies are visible in
[`compiler.py`](../src/mytriton/compiler.py#L223-L227). A runtime-only fast path
would materially improve small-kernel latency.

For the large matrices in the main benchmark, wrapper overhead is only about
0.4–0.5 ms and does not explain the 15–22x gap. Raw and CUDA Graph timings are
nearly identical, so normal CUDA launch overhead is also not the large-matrix
bottleneck.

## Recommended optimization order

1. Add a conflict-free padded or swizzled FP16/BF16 shared-memory layout.
2. Lower shared-memory MMA fragment loads to `ldmatrix`.
3. Introduce multi-warp CTA tiles and larger block-level M/N tiles.
4. Implement a real `cp.async` multistage pipeline with overlapping copy and
   MMA execution.
5. Avoid preloading every MMA atom's operands at once; load and consume
   fragments in a register-conscious order.
6. Store accumulator fragments without a full shared-memory round trip where
   possible.
7. Add a runtime-only launch path that does not reconstruct or copy IR.
8. Autotune tile sizes, warp count, stage count, and layout after these
   primitives exist.

## Conclusion

Version 21 successfully composes multiple `mma.sync.m16n8k8` instructions and
produces correct BF16 and FP16 matrix multiplication results. The A100 run also
shows that issuing tensor-core instructions is only the starting point for a
competitive GEMM.

The best current kernel reaches approximately 12.6–14.1 TFLOP/s, while
`torch.mm` reaches approximately 218–278 TFLOP/s in the same mixed-precision
comparison. The profiler points to a concrete path forward: fix shared-memory
bank conflicts, adopt `ldmatrix`, increase block-level reuse with multiple
warps, and turn the existing two-stage storage into a true asynchronous
pipeline.
