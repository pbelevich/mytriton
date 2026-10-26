# Changelog

Each release tag marks one lesson-sized compiler milestone. The project does
not use semantic versioning yet; `verN` records the order in which features
were built.

## Unreleased

### Version 29: JIT runtime specialization

- Added bounded pointer-alignment and integer-divisibility facts inferred from
  NumPy, CuPy, and PyTorch runtime arguments.
- Included specialization facts in the compilation cache key so one kernel can
  safely keep distinct aligned and generic CUDA variants.
- Removed redundant runtime pointer and stride alignment guards from the
  specialized SM80 static pipeline while preserving per-CTA boundary checks
  and the scalar zero-fill fallback for edge tiles.
- Added cache-behavior, generated-CUDA, and irregular-shape regression tests.
- Recorded same-session A100 results; the guard-only change is performance
  neutral within observed Colab clock-state variance.

### Version 28: CUDA event autotuning

- Added public `Config` and `autotune` APIs with Triton-style decorator order,
  compile-time meta-parameter injection, `num_warps`, and runtime tuning keys.
- Added accurate CUDA-event timing on the active CuPy or PyTorch stream after
  compilation, so Python tracing and artifact-copy overhead do not influence
  schedule selection.
- Added a winner cache separated by backend, CUDA architecture, device, runtime
  types, and user-selected key values, plus `best_config`, timing records, and
  cache reset support.
- Added a five-configuration A100 GEMM sweep and reproducible Colab runner.
  BF16 selects `64x32x64` for `1024^3`/`2048^3` and `128x32x128` for
  `4096^3`/`8192^3`; the latter reaches 210.21 TFLOP/s at `8192^3`, 86.7% of
  torch/cuBLAS in the same run.

### Version 27: Static register pipeline

- Added a specialized SM80 `128x32x128`, four-warp lowering with a static
  three-phase shared-stage schedule and no stage modulo in the hot loop.
- Added warp-raked global-to-shared copies and precomputed shared addresses for
  both cooperative stores and `ldmatrix` fragment loads.
- Preloaded both `BK=32` K fragments before their MMA groups while retaining
  the target-gated dynamic pipeline for other schedules and architectures.
- Preserved irregular-shape correctness through the existing masked scalar
  zero-fill producer.
- Improved final A100 BF16 `4096^3` raw throughput from 146.21 TFLOP/s in
  Version 26 to 187.19 TFLOP/s, with 202.14 TFLOP/s observed in a same-day
  run. FP16 reaches 203.67 TFLOP/s.

### Version 26: Pipelined Tensor Core GEMM

- Added target-aware SM80 `cp.async` lowering with three shared-memory stages,
  explicit commit/wait groups, and overlap between the next global-memory tile
  and current `ldmatrix`/MMA work.
- Added a scalar zero-filling fallback for partial or misaligned vectors, so
  irregular matrix tails retain the same correctness as synchronous staging.
- Consumed operand fragments one K slice at a time and selected eight warps for
  the `128x32x128` benchmark default, reducing the production configuration to
  126 registers per thread at 25% measured/theoretical occupancy.
- Improved A100 BF16 `4096^3` raw throughput from Version 25's 63.19 TFLOP/s to
  146.21 TFLOP/s in the final 30-sample run, with 162.49 TFLOP/s observed in
  the warp-count sweep. FP16 reaches 166.11 TFLOP/s.
- Added reproducible benchmark/NCU defaults and full/tail A100 execution
  coverage for the pipelined kernel.

### Version 25: SM80 production Tensor Core tile

- Added target-aware `mma.sync.m16n8k16` lowering for FP16 and BF16 on
  `sm_80+`, while preserving the `m16n8k8` path for `sm_75` and K=8 tiles.
- Extended warp and CTA layouts, grouped `ldmatrix.x4` operand loading, and
  shared-memory swizzling to the `128x128x32` CTA and `64x64` warp tiles.
- Replaced the 64 KiB shared-memory accumulator redistribution tile with a
  direct masked register-to-global FP32 epilogue using aligned `float2` stores
  and a scalar fallback for odd strides and boundary columns.
- Added full and irregular-shape SM80 execution coverage for the production
  tile; all FP16/BF16 Tensor Core execution tests pass on A100.
- Improved the established `64x16x64` BF16 kernel from 59.95 to 63.19
  TFLOP/s. The experimental four-warp `128x32x128` tile reaches 36.08 TFLOP/s;
  profiling shows that it needs asynchronous and register-level pipelining
  before it should replace the current default.

### Version 24: Warp-cooperative `ldmatrix` fragments

- Replaced scalar FP16/BF16 shared-memory reads and manual 16-bit packing with
  target-aware `ldmatrix.sync.aligned.m8n8` operand loads.
- Added non-transposed A loads, transposed B loads, shared-address conversion,
  and `.x1`/`.x2`/`.x4` matrix groups.
- Reused each loaded A or B register fragment across all matching physical
  `mma.sync` instructions instead of reloading it for every M/N combination.
- Added generated-PTX validation, architecture checks, and FP16 execution
  coverage for single-warp, composable, and multi-warp Tensor Core tiles.
- Added a shared 16-byte-segment XOR swizzle used consistently by vectorized
  staging and `ldmatrix`, without increasing shared-memory allocation.
- Reduced A100 shared-load bank conflicts by 99.3% and improved the primary
  `4096^3` BF16 kernel from 40.83 to 59.95 TFLOP/s (1.47x); FP16 reaches
  59.88 TFLOP/s.

### Version 23: Vectorized global-to-shared staging

- Added 16-byte-aligned low-precision shared buffers and cooperative 128-bit
  FP16/BF16 operand copies with masked scalar fallback for boundary and
  misaligned vectors.
- Added generated-CUDA, F16/BF16 execution, alignment, and fallback coverage
  for vectorized CUDA-core and Tensor Core dot staging.
- Improved the best `64x16x64`, four-warp A100 matmul from approximately
  22.9-24.2 TFLOP/s to 38.5-40.8 TFLOP/s, with matching FP16/BF16 results.
- Added reproducible Colab upload, benchmark, and Nsight Compute wrappers;
  Colab CUDA extras are inferred from the remote PyTorch CUDA version.

### Version 22: Multi-warp Tensor Core CTAs

- Added Triton-style `num_warps` launch configuration for CUDA tensor-core
  matrix multiplication.
- Added inferred block-to-warp decomposition, multi-warp CTA layouts,
  cooperative 128-thread operand staging, warp-local MMA accumulation, and
  shared-memory result redistribution.
- Added compilation-cache separation by launch configuration and CUDA
  execution coverage for masked FP16/BF16 multi-warp matrix multiplication.

## [ver21](https://github.com/pbelevich/mytriton/tree/ver21) — 2026-09-20

- Added composable one-warp Tensor Core tiles that lower larger FP16/BF16
  `tl.dot` operations into grids of `mma.sync.m16n8k8` instructions, carry
  multiple accumulator fragments across runtime K-loops, and redistribute the
  complete logical result through shared memory.

## [ver20](https://github.com/pbelevich/mytriton/tree/ver20) — 2026-09-19

- Lowered canonical FP16/BF16 `tl.dot` to target-aware Tensor Core
  `mma.sync` with FP32 loop-carried accumulation.

## [ver19](https://github.com/pbelevich/mytriton/tree/ver19) — 2026-09-07

- Added FP16/BF16 types, numeric promotion, explicit casts, typed pointers,
  CUDA conversions, and mixed-precision dot semantics.

## [ver18](https://github.com/pbelevich/mytriton/tree/ver18) — 2026-09-07

- Added shared-memory padding, unrolled dot loops, and two-stage ping-pong
  buffers for canonical runtime K-loops.

## [ver17](https://github.com/pbelevich/mytriton/tree/ver17) — 2026-09-01

- Added zero-copy PyTorch tensor interoperability through DLPack and execution
  on the current PyTorch CUDA stream.

## [ver16](https://github.com/pbelevich/mytriton/tree/ver16) — 2026-08-31

- Added per-thread register tiles, register-valued loop carries, and masked
  stores for several output elements per CUDA thread.

## [ver15](https://github.com/pbelevich/mytriton/tree/ver15) — 2026-08-22

- Implemented working CUDA-core `tl.dot` with shared operands, FP32 FMA
  accumulation, K-loop traversal, and masked edge tiles.

## [ver14](https://github.com/pbelevich/mytriton/tree/ver14) — 2026-08-22

- Added canonical matrix-load matching, cooperative shared-memory staging,
  zero filling, and block synchronization.

## [ver13](https://github.com/pbelevich/mytriton/tree/ver13) — 2026-08-22

- Added public `tl.dot`, rank-two type inference, typed SSA, verifier rules,
  and optimizer support.

## [ver12](https://github.com/pbelevich/mytriton/tree/ver12) — 2026-08-22

- Separated logical tiles from CUDA thread layouts and introduced projected,
  cooperative, and reduction-aware mappings.

## [ver11](https://github.com/pbelevich/mytriton/tree/ver11) — 2026-07-18

- Added `tl.empty`, `tl.full`, `tl.zeros`, and public compiler dtype objects.

## [ver10](https://github.com/pbelevich/mytriton/tree/ver10) — 2026-07-18

- Added structured runtime `range` loops with captures, loop-carried values,
  `iter_args`, nested regions, and `yield`.

## [ver9](https://github.com/pbelevich/mytriton/tree/ver9) — 2026-07-18

- Replaced direct symbolic execution with a Python AST frontend and
  compile-time loop unrolling.

## [ver8](https://github.com/pbelevich/mytriton/tree/ver8) — 2026-07-10

- Added rank-two blocks, dimension expansion, broadcasting, masks, and tiled
  rank-two CUDA kernels.

## [ver7](https://github.com/pbelevich/mytriton/tree/ver7) — 2026-06-30

- Added an experimental MLIR GPU/NVVM backend for rank-one elementwise
  kernels and cubin execution.

## [ver6](https://github.com/pbelevich/mytriton/tree/ver6) — 2026-06-25

- Added reductions, stable softmax, static ranges, long-row reduction, and a
  first naive matrix multiplication.

## [ver5](https://github.com/pbelevich/mytriton/tree/ver5) — 2026-06-25

- Added SSA verification, constant folding, common subexpression elimination,
  and dead-code elimination.

## [ver4](https://github.com/pbelevich/mytriton/tree/ver4) — 2026-06-25

- Added arithmetic, extrema, selection, exponentiation, and activation
  kernels including ReLU and sigmoid.

## [ver3](https://github.com/pbelevich/mytriton/tree/ver3) — 2026-06-24

- Added CUDA C++ generation, CuPy compilation, kernel launch, and optional GPU
  execution.

## [ver2](https://github.com/pbelevich/mytriton/tree/ver2) — 2026-06-24

- Added type inference and lowering from the expression tree to typed SSA.

## [ver1](https://github.com/pbelevich/mytriton/tree/ver1) — 2026-06-23

- Added symbolic tracing, Triton-like launch syntax, initial tests, and CI.
