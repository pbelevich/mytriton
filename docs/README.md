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

## Runtime and development

- [PyTorch tensor interoperability](implementation.md#pytorch-tensor-interoperability)
- [End-to-end compiler example](implementation.md#example)
- [Current limitations](implementation.md#current-limitations)
- [Development workflow](implementation.md#development)
- [Version changelog](../CHANGELOG.md)
- [Colab test notebook](../tests/mytriton_colab_tests.ipynb)
