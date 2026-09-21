# Changelog

Each release tag marks one lesson-sized compiler milestone. The project does
not use semantic versioning yet; `verN` records the order in which features
were built.

## [ver22](https://github.com/pbelevich/mytriton/tree/ver22) — 2026-09-21

- Added multi-warp Tensor Core CTA tiles with a two-dimensional warp grid,
  shared A/B tile reuse, target-aware block sizing, loop-carried per-warp MMA
  accumulators, and shared-memory result redistribution.
- Replaced scalar Tensor Core operand packing with aligned, padded
  shared-memory tiles and grouped `ldmatrix.x1`, `x2`, and `x4` loads,
  including transposed B-fragment loading and target validation.
- Added FP16 and BF16 execution coverage for partial `64 x 64` output tiles,
  Colab upload and benchmark helpers, and an A100 benchmark/`ncu` profile of
  the four-warp kernel.

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
