# Version 25 SM80 Production Tensor Tile on NVIDIA A100

## Summary

Version 25 introduces the compiler structures needed by the eventual SM80
matmul fast path:

- target-aware `mma.sync.aligned.m16n8k16` for FP16 and BF16;
- a `128 x 128 x 32` CTA split into four `64 x 64` warp tiles;
- grouped `ldmatrix.x4` and `ldmatrix.x4.trans` operand loads;
- a 16-segment shared-memory XOR mapping;
- a direct masked register-to-global FP32 epilogue.

The implementation is correct on A100 for complete and irregular tiles, but
the performance experiment falsifies the original expectation that a larger
tile alone would reach 130--190 TFLOP/s. The four-warp production tile reaches
**36.08 TFLOP/s** for BF16 `4096^3`, below the Version 24 result. The best
current configuration remains `64 x 16 x 64` with four warps, now at
**63.19 TFLOP/s**, up from 59.95 TFLOP/s in Version 24.

Nsight Compute identifies latency rather than bandwidth or spills as the main
limit: 86.45% of scheduler cycles have no eligible warp, and each warp spends
about 9.6 cycles per issued instruction waiting on an L1TEX scoreboard
dependency. The large tile should remain an experimental fast-path foundation
until `cp.async`, multiple shared stages, and register-fragment double
buffering overlap operand transport with MMA.

## Benchmark metadata

- **Date:** October 25, 2026 EDT
- **Branch:** `version-25-production-tensor-tile`
- **Base commit:** `5464472` (Version 24 report)
- **Implementation commit:** the commit containing this report
- **GPU:** NVIDIA A100-SXM4-40GB
- **Compute capability:** 8.0 (`sm_80`)
- **Streaming multiprocessors:** 108
- **PyTorch:** 2.11.0+cu130
- **PyTorch CUDA build:** 13.0
- **CuPy:** 14.0.1
- **Primary shape:** BF16 `4096 x 4096 x 4096`, FP32 output
- **Timing:** CUDA events around the compiled raw kernel
- **Warmup / samples:** 3 / 10

Absolute clocks varied across profiling runs, so configuration comparisons
use raw-kernel medians and resource data from the same runtime session whenever
possible.

## Correctness and validation

The complete local suite passed on an RTX 2080:

```text
604 passed, 4 skipped
```

The SM80-only production execution test passed on A100 for both:

```text
128 x 128 x 32   complete CTA
117 x 109 x 29   M, N, and K tails with an odd output stride
```

All available Tensor Core execution tests passed on A100:

```text
8 passed, 117 deselected
```

This covers FP16 and BF16, K=8 fallback, K=16 native MMA, single-warp tiles,
multi-warp CTA tiles, masked edges, and the direct epilogue. The benchmark
reported zero sampled error for every configuration in the tile sweep.

## What changed

### Native SM80 K=16 MMA

SM80 prefers:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

One lane supplies four packed A registers and two packed B registers. The
instruction still returns four FP32 accumulator values for a `16 x 8` result
fragment. K=8 tiles and SM75 retain the previous `m16n8k8` lowering.

For a `64 x 64 x 32` warp tile, the static instruction grid is:

```text
M atoms: 64 / 16 = 4
N atoms: 64 / 8  = 8
K atoms: 32 / 16 = 2
MMA instructions per warp: 4 x 8 x 2 = 64
```

### Direct epilogue

The previous generic path spilled the complete CTA result into shared memory
so that the ordinary output register layout could reload it. A `128 x 128`
FP32 result needs 64 KiB. Combined with 32 KiB of staged operands, the kernel
requested 96 KiB of static shared memory and failed the compiler's 48 KiB
conservative limit.

Version 25 recognizes the canonical row-major output pointer and mask in SSA,
then maps the MMA fragment directly to global coordinates. Each lane stores
two adjacent result pairs. Aligned even-stride rows use `float2`; odd strides
and boundary columns use scalar stores. Static shared memory therefore falls
from 96 KiB to 32 KiB, and the extra block barrier disappears.

## Tile sweep

All entries below are BF16 `4096^3` with FP32 output on the same A100 runtime.

| CTA tile | Warps | Raw time | Raw TFLOP/s | Registers/thread | Shared memory | Occupancy estimate |
|---|---:|---:|---:|---:|---:|---:|
| `64 x 16 x 64` | 4 | 2.1750 ms | **63.19** | 64 | 8 KiB | 50.0% |
| `128 x 16 x 64` | 4 | 2.3532 ms | 58.41 | 96 | 12 KiB | 31.2% |
| `64 x 16 x 128` | 4 | 2.4044 ms | 57.16 | 96 | 12 KiB | 31.2% |
| `128 x 16 x 128` | 4 | 3.0208 ms | 45.50 | 168 | 16 KiB | 18.8% |
| `64 x 32 x 64` | 4 | 2.3916 ms | 57.47 | 71 | 16 KiB | 43.8% |
| `128 x 32 x 64` | 4 | 2.8201 ms | 48.74 | 122 | 24 KiB | 25.0% |
| `64 x 32 x 128` | 4 | 2.8472 ms | 48.27 | 125 | 24 KiB | 25.0% |
| `128 x 32 x 128` | 4 | 3.8093 ms | 36.08 | 202 | 32 KiB | 12.5% |

The established `64 x 16 x 64` default improves from Version 24's 59.95 to
63.19 TFLOP/s, a **1.05x** speedup. The larger production geometry is a
**0.57x** regression relative to that current default.

## Warp-count ablation for the 128 x 128 CTA

Increasing the warp count reduces each warp's accumulator tile and its
register pressure:

| CTA tile | Warps | Warp tile | Raw TFLOP/s | Registers/thread | Occupancy estimate |
|---|---:|---|---:|---:|---:|
| `128 x 32 x 128` | 4 | `64 x 64` | 36.08 | 202 | 12.5% |
| `128 x 32 x 128` | 8 | `64 x 32` | 59.61 | 121 | 25.0% |
| `128 x 32 x 128` | 16 | `32 x 32` | 54.37 | 71 | 25.0% |
| `128 x 16 x 128` | 16 | `32 x 32` | **66.18** | 64 | 50.0% |

Sixteen warps recover modest throughput, but this is not the intended final
schedule. The independently researched pipelined kernel reaches over 230
TFLOP/s with four `64 x 64` warp tiles. Replacing the planned schedule merely
to hide today's synchronous latency would optimize the intermediate compiler
rather than build toward the proven design.

## Nsight Compute diagnosis

The profiled kernel used `128 x 32 x 128`, four warps, and BF16 `4096^3`.

| Metric | Value |
|---|---:|
| Duration | 3.77 ms |
| Compute throughput | 15.20% |
| Memory throughput | 17.67% |
| DRAM throughput | 4.49% |
| Issue slots busy | 13.04% |
| Registers/thread | 202 |
| Static shared memory | 32.77 KiB |
| Theoretical occupancy | 12.50% |
| Achieved occupancy | 12.31% |
| Active warps/scheduler | 1.97 |
| Eligible warps/scheduler | 0.16 |
| Scheduler cycles with no eligible warp | 86.45% |
| Warp cycles/issued instruction | 14.53 |
| Long-scoreboard stall | 9.6 cycles/instruction |
| Local-memory spill requests | 0 |

Neither DRAM bandwidth nor local-memory spills explain the runtime. The
dominant long-scoreboard stall comes from serialized operand movement and
consumption: cooperative global loads, a barrier, `ldmatrix`, and dependent
MMA execute without an asynchronous or register-level pipeline. With only two
resident four-warp CTAs per SM, the schedulers rarely have another ready warp.

## Corrected optimization order

The measurement changes the interpretation of Version 25, not the proven
destination:

1. **Version 25:** land `m16n8k16`, the production CTA/warp model, direct
   epilogue, and correctness coverage. Keep `64 x 16 x 64` as the default.
2. **Next:** add `cp.async` and a three-stage compact shared-memory ring.
3. **Then:** double-buffer A/B register fragments so `ldmatrix` for the next
   K atom overlaps current MMA work.
4. **Then:** tune source/SASS scheduling and expose measured alternatives
   through `mytriton.autotune`.

The key lesson is that tile reuse is potential performance, not realized
performance. A larger tile reduces global-memory traffic per output, but it
also lengthens dependency chains and consumes registers. It wins only when
the compiler overlaps those chains.

## Reproduction

Upload the current working tree and run the production correctness gate:

```bash
./tools/sync_colab.sh

colabrun \
  --timeout 600 \
  --notebook tests/mytriton_colab_tests.ipynb \
  --cwd /content/mytriton \
  python -m pytest \
  tests/test_tensor_cores.py::test_sm80_production_cta_executes_with_direct_epilogue \
  -v -s
```

Run the primary benchmark:

```bash
./benchmarks/run_matmul_colab.sh \
  --sizes 4096 \
  --tiles 128x32x128 \
  --dtype bf16 \
  --num-warps 4 \
  --warmup 3 \
  --repeats 10
```

Run the tile sweep:

```bash
./benchmarks/run_matmul_colab.sh \
  --sizes 4096 \
  --tiles \
    64x16x64 128x16x64 64x16x128 128x16x128 \
    64x32x64 128x32x64 64x32x128 128x32x128 \
  --dtype bf16 \
  --num-warps 4 \
  --warmup 3 \
  --repeats 10 \
  --no-graphs
```
