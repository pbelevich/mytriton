# Documentation

The root [README](../README.md) is the short project overview. The detailed
compiler notes live in the [implementation guide](implementation.md).

## Compiler frontend and IR

- [AST frontend](implementation.md#ast-frontend)
- [Runtime `for` loops](implementation.md#runtime-for-loops)
- [Block factory functions](implementation.md#block-factory-functions)
- [`tl.dot` semantics](implementation.md#tldot-semantics)
- [SSA verification and optimization](implementation.md#ssa-verification-and-optimization)

## CUDA lowering

- [CUDA tile layouts](implementation.md#cuda-tile-layouts)
- [Shared-memory CUDA-core dot](implementation.md#shared-memory-cuda-core-dot)
- [Shared-memory layout optimization](implementation.md#shared-memory-layout-optimization)
- [Low-precision dot](implementation.md#low-precision-types-and-mixed-precision-dot)
- [Tensor-core `mma.sync`](implementation.md#tensor-core-dot)
- [Composable warp MMA tiles](implementation.md#composable-warp-mma-tiles)
- [Multi-warp CTA tiles](implementation.md#multi-warp-cta-tiles)
- [Vectorized global-to-shared staging](implementation.md#vectorized-global-to-shared-staging)
- [Warp-cooperative `ldmatrix` fragments](implementation.md#warp-cooperative-ldmatrix-fragments)
- [SM80 production tensor tile](implementation.md#sm80-production-tensor-tile)
- [Pipelined Tensor Core GEMM](implementation.md#pipelined-tensor-core-gemm)
- [Static register pipeline](implementation.md#static-register-pipeline)
- [A100 Version 23 performance report](../benchmarks/matmul_vectorized_staging_a100_report.md)
- [A100 Version 24 `ldmatrix` report](../benchmarks/matmul_ldmatrix_a100_report.md)
- [A100 Version 25 production-tile report](../benchmarks/matmul_production_tile_a100_report.md)
- [Version 25 visual report — English](../benchmarks/version25_production_tensor_tile_a100_en.html)
- [Version 25 visual report — Russian](../benchmarks/version25_production_tensor_tile_a100_ru.html)
- [A100 Version 26 pipeline report](../benchmarks/matmul_pipeline_a100_report.md)
- [Version 26 visual report — English](../benchmarks/version26_pipeline_a100_en.html)
- [Version 26 visual report — Russian](../benchmarks/version26_pipeline_a100_ru.html)
- [A100 Version 27 static-pipeline report](../benchmarks/matmul_static_pipeline_a100_report.md)
- [Version 27 visual report — English](../benchmarks/version27_static_pipeline_a100_en.html)
- [Version 27 visual report — Russian](../benchmarks/version27_static_pipeline_a100_ru.html)

## Runtime and development

- [PyTorch tensor interoperability](implementation.md#pytorch-tensor-interoperability)
- [End-to-end compiler example](implementation.md#example)
- [Current limitations](implementation.md#current-limitations)
- [Development workflow](implementation.md#development)
- [Version changelog](../CHANGELOG.md)
- [Colab test notebook](../tests/mytriton_colab_tests.ipynb)
