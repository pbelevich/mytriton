# A100 Multi-Warp CTA Matrix Multiplication Report

## Snapshot

- **Benchmark date:** September 21, 2026
- **Development branch:** `version-22-multi-warp-cta`
- **Base commit:** [`27571025a9fc9521b67fa765efa85719ab55566f`](https://github.com/pbelevich/mytriton/commit/27571025a9fc9521b67fa765efa85719ab55566f)
- **Source state:** uncommitted Version 22 working tree uploaded directly to Colab
- **Uploaded archive SHA-256:** `0543fb2bcfc8e893e6531983946710164b64c44c453c37644894b7dea02b146f`
- **GPU:** NVIDIA A100-SXM4-40GB
- **Compute capability:** 8.0 (`sm_80`)
- **PyTorch:** 2.11.0+cu128
- **CUDA reported by PyTorch:** 12.8
- **CuPy:** 14.0.1
- **Nsight Compute:** 2025.1.1

This report evaluates the Version 22 multi-warp CTA implementation against the
single-warp Version 21 baseline in
[matmul_a100_report.md](matmul_a100_report.md). The current working tree was
uploaded to `/content/mytriton` and installed in editable mode, so the A100 ran
the local compiler changes rather than an older package from GitHub or PyPI.

## Objective

Version 21 assigned one warp to one `32 x 32` output tile. Version 22 groups
four warps into one CTA and lets them compute a shared `64 x 64` output tile:

```text
warp 0 -> rows  0..31, columns  0..31
warp 1 -> rows  0..31, columns 32..63
warp 2 -> rows 32..63, columns  0..31
warp 3 -> rows 32..63, columns 32..63
```

The larger CTA tile reuses each staged A tile across two warp columns and each
B tile across two warp rows. The benchmark measures whether that additional
reuse helps before adding `ldmatrix`, shared-memory swizzling, or `cp.async`.

## Methodology

- Inputs use BF16; accumulation and output use FP32.
- `torch.mm(a, b, out=out, out_dtype=torch.float32)` is the reference.
- Times are medians of 10 measurements after 3 warm-up launches.
- CUDA events exclude first-time mytriton and NVRTC compilation.
- `Raw` launches the already compiled CuPy kernel directly.
- `Graph` replays the raw launch through a CUDA Graph.
- Correctness is checked against PyTorch on 8,192 output samples.
- The sweep compares CTA tiles `32x32`, `32x64`, `64x32`, and `64x64`, all
  with `BK=16`.

The run can be reproduced after uploading the working tree with:

```bash
./tools/sync_colab.sh

./benchmarks/run_matmul_colab.sh \
    --sizes 4096 8192 \
    --tiles 32x16x32 32x16x64 64x16x32 64x16x64 \
    --dtype bf16 \
    --warmup 3 \
    --repeats 10 \
    --artifacts-dir benchmarks/artifacts/colab-multi-warp \
    --output benchmarks/artifacts/colab-multi-warp.json
```

## Benchmark results

### 4096 x 4096 x 4096

`torch.mm` took **0.6917 ms**, or **198.69 TFLOP/s**. Its CUDA Graph replay
took 0.6738 ms.

| BM | BK | BN | Warps | Wrapper, ms | Raw, ms | Graph, ms | Raw TFLOP/s | Slowdown vs. PyTorch | Registers/thread | Static shared memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 32 | 1 | 12.4687 | 11.2486 | 9.8053 | 12.22 | 16.3x | 94 | 8,192 B |
| 32 | 16 | 64 | 2 | 8.1034 | 7.7117 | 7.6974 | 17.82 | 11.1x | 96 | 14,336 B |
| 64 | 16 | 32 | 2 | 8.0620 | 7.6698 | 7.6477 | 17.92 | 11.1x | 96 | 14,336 B |
| **64** | **16** | **64** | **4** | **6.0810** | **5.6975** | **5.6745** | **24.12** | **8.2x** | **95** | **24,576 B** |

### 8192 x 8192 x 8192

`torch.mm` took **3.8835 ms**, or **283.12 TFLOP/s**. Its measured CUDA Graph
replay took 4.0617 ms.

| BM | BK | BN | Warps | Raw, ms | Raw TFLOP/s | Slowdown vs. PyTorch |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 32 | 1 | 87.6349 | 12.55 | 22.6x |
| 32 | 16 | 64 | 2 | 62.1512 | 17.69 | 16.0x |
| 64 | 16 | 32 | 2 | 71.0277 | 15.48 | 18.3x |
| **64** | **16** | **64** | **4** | **48.0558** | **22.88** | **12.4x** |

All sampled correctness checks passed with a maximum absolute error of zero.

Raw time and throughput are the primary comparison because they isolate the
generated CUDA kernel from Python wrapper overhead.

## Version 21 comparison

The closest like-for-like Version 21 result used a one-warp `32x16x32` tile:

| Problem | Version 21 raw | Version 22 `64x16x64` raw | Speedup |
|---|---:|---:|---:|
| 4096 cubed | 14.05 TFLOP/s | 24.12 TFLOP/s | 1.72x |
| 8192 cubed | 12.59 TFLOP/s | 22.88 TFLOP/s | 1.82x |

The same-run `32x16x32` reference gives an even larger 1.97x improvement at
4096 cubed. The difference between historical and same-run baselines is normal
run-to-run/environment variation; the same-run sweep is the cleaner tile-size
comparison.

The multi-warp CTA is therefore a real improvement, but it does not close the
gap to a tuned GEMM. The best kernel remains 8.2x slower than PyTorch at 4096
and 12.4x slower at 8192.

## Nsight Compute profile

The best `64x16x64` BF16 configuration was profiled at 4096 cubed with:

```bash
MYTRITON_PROFILE_TILE=64x16x64 \
    bash benchmarks/profile_matmul_a100.sh \
    /content/mytriton/benchmarks/artifacts/profile-multi-warp
```

The normal, uninstrumented run embedded in the capture measured **5.6929 ms**
and **24.14 TFLOP/s**. `ncu` replayed the kernel 43 times to collect all
sections, so the instrumented wall-clock time is intentionally ignored.

The generated source and exact NVRTC cubin have these hashes:

| Artifact | SHA-256 |
|---|---|
| Generated CUDA | `471eccd8eab2f9fa95643020a54520f8fee2cf480137a1c83578b38751ecdc88` |
| NVRTC cubin | `a2198d3889e0916d45f4288b5905cbd1f3491de488ed3830c948acbb5e0e6beb` |

### Key metrics

| Metric | Observed value |
|---|---:|
| Grid | 64 x 64 blocks |
| Threads per block | 128 |
| Registers per thread | 95 |
| Static shared memory per block | 24.58 KB |
| Total shared memory per block, including driver | 25.60 KB |
| Theoretical occupancy | 31.25% |
| Achieved occupancy | 29.94% |
| Achieved active warps per SM | 19.16 |
| Compute throughput | 53.10% |
| Memory throughput | 62.20% |
| DRAM throughput | 6.95% |
| Tensor utilization (`Tensor (All)`) | 8.15% |
| Global-memory throughput | 108.10 GB/s |
| Active warps per scheduler | 4.80 |
| Eligible warps per scheduler | 1.04 |
| Issued warps per scheduler | 0.54 |
| Cycles per issued instruction | 8.98 |
| Long-scoreboard stall cycles per issued instruction | 3.39 |

Low DRAM utilization rules out global bandwidth as the primary bottleneck.
The scheduler has several active warps but usually only about one eligible
warp, while long-scoreboard dependencies consume roughly 38% of the interval
between issued instructions. The kernel is waiting for shared-memory operand
loads much more than it is waiting for HBM.

### Shared-memory conflicts

Nsight Compute identified the same fundamental feeding problem seen in
Version 21:

- 101,187,584 shared-load requests;
- an average **3.3-way bank conflict**;
- 236,573,177 reported bank conflicts;
- conflicts in about **70.04%** of shared-load wavefronts;
- 238,026,752 excessive shared-load wavefronts;
- an estimated 43.78% speedup opportunity from removing the reported load
  conflicts.

The hot operand instructions are scalar `LDS.U16` loads. Each generates the
equivalent of four shared-memory wavefronts for one warp request. The 32-bit
`LDS` accesses exhibit a smaller two-way conflict pattern. These accesses are
implemented in
[`emit_mma_warp_tile_operand_registers`](../src/mytriton/cuda_codegen.py#L486),
where individual 16-bit shared elements are loaded and packed manually.

The total number of active warps doing the GEMM is essentially unchanged:
Version 21 launched many one-warp blocks, while Version 22 launches one quarter
as many four-warp blocks. Consequently, enlarging the CTA improves cross-warp
A/B reuse but does not fix the per-warp shared-load pattern. This explains why
the speedup is substantial without approaching tensor-core peak throughput.

### Lowered instruction mix

The independently generated PTX contains:

| PTX operation | Static count |
|---|---:|
| `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32` | 16 |
| `ldmatrix` | 0 |
| `cp.async` | 0 |

The exact CuPy/NVRTC SASS contains:

| SASS family | Static count |
|---|---:|
| `HMMA.1688.F32.BF16` | 16 |
| `LDS.U16` | 16 |
| Other `LDS` | 40 |
| `LDG.E.U16` | 14 |
| `STS.U16` | 14 |
| `STS.64` | 16 |
| `STG.E` | 32 |
| `BAR.SYNC` | 2 |

The tensor-core operations are present and survive NVRTC compilation, but
their fragments are still supplied through scalar shared loads. There is no
`LDMATRIX`, `LDGSTS`, or asynchronous copy pipeline.

## Conclusions

1. **Multi-warp CTA tiling works.** A four-warp `64x64` CTA nearly doubles the
   throughput of the same-run one-warp `32x32` tile.
2. **The kernel is not DRAM-bound.** DRAM throughput is only 6.95%, while
   shared/L1TEX activity and scoreboard stalls are high.
3. **Shared operand loading is now the clearest bottleneck.** Scalar 16-bit
   loads produce severe bank conflicts and keep tensor utilization low.
4. **Occupancy is useful but limited.** The kernel reaches about 30% occupancy;
   95 registers per thread and 25.6 KB of shared memory per block prevent more
   resident warps.
5. **The next optimization should target fragment loading and layout.** A
   compatible conflict-free/swizzled shared layout and `ldmatrix` lowering
   should be developed together and validated with `ncu` before introducing a
   `cp.async` multistage pipeline.

The optimization order justified by these measurements is therefore:

```text
multi-warp CTA reuse (done)
    -> shared layout + ldmatrix
    -> vectorized global-to-shared copies
    -> cp.async and true stage overlap
    -> accumulator epilogue improvements
    -> autotuning
```
