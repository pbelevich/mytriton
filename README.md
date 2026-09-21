# mytriton

[![CI](https://github.com/pbelevich/mytriton/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pbelevich/mytriton/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B%20%2B%20PTX-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/140oC_PrW1BOgaw7mV9Or3Xa3lCrwcrz3)

`mytriton` is a small, executable compiler built to explore how a
Triton-style Python kernel becomes GPU code. It parses kernel source, builds a
symbolic IR, lowers it to verified typed SSA, and emits CUDA C++ with inline
PTX. The current development version includes shared-memory tiled matrix
multiplication, FP16/BF16 inputs, FP32 accumulation, and composable one-warp
Tensor Core tiles built from NVIDIA `mma.sync` instructions.

```text
Python kernel
    -> AST frontend
    -> expression IR
    -> typed SSA
    -> verification + optimization
    -> CUDA C++ / PTX
```

This is a learning compiler, not a replacement for production Triton. The
implementation deliberately keeps every compiler layer small enough to read.

## Releases

Each tag captures one self-contained step in the compiler's evolution. See the
full [changelog](CHANGELOG.md) and the accompanying
[implementation guide](docs/implementation.md).

| Version | Milestone |
|:--|:--|
| [`ver1`](https://github.com/pbelevich/mytriton/tree/ver1) | [Symbolic tracing and JIT](https://pbelevich.github.io/2026/06/22/My_Triton_From_Scratch_Part_1_Symbolic_Tracing.html) |
| [`ver2`](https://github.com/pbelevich/mytriton/tree/ver2) | [Typed SSA and inference](https://pbelevich.github.io/2026/06/23/My_Triton_From_Scratch_Part_2_Typed_SSA.html) |
| [`ver3`](https://github.com/pbelevich/mytriton/tree/ver3) | [CUDA generation and execution](https://pbelevich.github.io/2026/06/24/My_Triton_From_Scratch_Part_3_CUDA_Lowering.html) |
| [`ver4`](https://github.com/pbelevich/mytriton/tree/ver4) | [Math operations and activations](https://pbelevich.github.io/2026/06/25/My_Triton_From_Scratch_Part_4_Elementwise_Ops.html) |
| [`ver5`](https://github.com/pbelevich/mytriton/tree/ver5) | [Verification and SSA optimization](https://pbelevich.github.io/2026/06/26/My_Triton_From_Scratch_Part_5_Verification.html) |
| [`ver6`](https://github.com/pbelevich/mytriton/tree/ver6) | [Reductions, softmax, naive matmul](https://pbelevich.github.io/2026/06/27/My_Triton_From_Scratch_Part_6_Reductions.html) |
| [`ver7`](https://github.com/pbelevich/mytriton/tree/ver7) | [Experimental MLIR CUDA backend](https://pbelevich.github.io/2026/06/28/My_Triton_From_Scratch_Part_7_Minimal_MLIR.html) |
| [`ver8`](https://github.com/pbelevich/mytriton/tree/ver8) | [Rank-two tiles and broadcasting](https://pbelevich.github.io/2026/06/29/My_Triton_From_Scratch_Part_8_Rank_2_Tiles.html) |
| [`ver9`](https://github.com/pbelevich/mytriton/tree/ver9) | [AST-based Python frontend](https://pbelevich.github.io/2026/08/01/My_Triton_From_Scratch_Part_9_AST_Frontend.html) |
| [`ver10`](https://github.com/pbelevich/mytriton/tree/ver10) | [Structured runtime for loops](https://pbelevich.github.io/2026/08/08/My_Triton_From_Scratch_Part_10_Runtime_For_Loops.html) |
| [`ver11`](https://github.com/pbelevich/mytriton/tree/ver11) | [Typed block factory functions](https://pbelevich.github.io/2026/08/15/My_Triton_From_Scratch_Part_11_Block_Factory_Functions.html) |
| [`ver12`](https://github.com/pbelevich/mytriton/tree/ver12) | [Explicit CUDA tile layouts](https://pbelevich.github.io/2026/08/22/My_Triton_From_Scratch_Part_12_CUDA_Tile_Layouts.html) |
| [`ver13`](https://github.com/pbelevich/mytriton/tree/ver13) | [`tl.dot` semantics and SSA](https://pbelevich.github.io/2026/08/23/My_Triton_From_Scratch_Part_13_Dot_Semantics.html) |
| [`ver14`](https://github.com/pbelevich/mytriton/tree/ver14) | [Shared-memory operand staging](https://pbelevich.github.io/2026/08/29/My_Triton_From_Scratch_Part_14_Shared_Memory_Tiles.html) |
| [`ver15`](https://github.com/pbelevich/mytriton/tree/ver15) | [CUDA-core tiled matrix multiplication](https://pbelevich.github.io/2026/08/30/My_Triton_From_Scratch_Part_15_CUDA_Core_Dot.html) |
| [`ver16`](https://github.com/pbelevich/mytriton/tree/ver16) | [Per-thread register tiles](https://pbelevich.github.io/2026/09/05/My_Triton_From_Scratch_Part_16_Register_Tiles.html) |
| [`ver17`](https://github.com/pbelevich/mytriton/tree/ver17) | [PyTorch tensor interoperability](https://pbelevich.github.io/2026/09/06/My_Triton_From_Scratch_Part_17_PyTorch_Tensor_Interoperability.html) |
| [`ver18`](https://github.com/pbelevich/mytriton/tree/ver18) | [Shared-memory layout optimization](https://pbelevich.github.io/2026/09/07/My_Triton_From_Scratch_Part_18_Shared_Memory_Optimization.html) |
| [`ver19`](https://github.com/pbelevich/mytriton/tree/ver19) | [Mixed-precision types and casts](https://pbelevich.github.io/2026/09/12/My_Triton_From_Scratch_Part_19_Mixed_Precision_Types.html) |
| [`ver20`](https://github.com/pbelevich/mytriton/tree/ver20) | [Tensor-core `mma.sync` lowering](https://pbelevich.github.io/2026/09/13/My_Triton_From_Scratch_Part_20_Tensor_Cores.html) |
| [`ver21`](https://github.com/pbelevich/mytriton/tree/ver21) | [Composable warp MMA tiles](docs/implementation.md#composable-warp-mma-tiles) |

## Tensor-core matmul

This is a complete `mytriton` kernel. FP16 A and B tiles are staged through
shared memory, multiplied by Tensor Cores, and accumulated in FP32 across the
runtime K-loop.

```python
import torch

import mytriton as triton
import mytriton.language as tl


@triton.jit
def matmul_kernel(
    a,
    b,
    out,
    M,
    N,
    K,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    offsets_n = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
    offsets_k = tl.arange(0, BK)

    accumulator = tl.zeros((BM, BN), tl.float32)

    for k_base in range(0, K, BK):
        a_rows = offsets_m
        a_columns = k_base + offsets_k[None, :]
        a_values = tl.load(
            a + a_rows * K + a_columns,
            mask=(a_rows < M) & (a_columns < K),
            other=0.0,
        )

        b_rows = k_base + offsets_k[:, None]
        b_columns = offsets_n
        b_values = tl.load(
            b + b_rows * N + b_columns,
            mask=(b_rows < K) & (b_columns < N),
            other=0.0,
        )

        accumulator = accumulator + tl.dot(a_values, b_values)

    output_pointers = out + offsets_m * N + offsets_n
    output_mask = (offsets_m < M) & (offsets_n < N)
    tl.store(output_pointers, accumulator, mask=output_mask)


M, N, K = 128, 128, 128
BM, BK, BN = 32, 16, 16

torch.manual_seed(0)
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
out = torch.empty((M, N), device="cuda", dtype=torch.float32)

grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
_, _, cuda_source = matmul_kernel[grid](
    a,
    b,
    out,
    M,
    N,
    K,
    BM=BM,
    BK=BK,
    BN=BN,
)

expected = a.float() @ b.float()
torch.testing.assert_close(out, expected, rtol=3e-3, atol=3e-3)
```

A logical `32 x 16 x 16` dot is composed from eight physical `m16n8k8`
instructions. Each instruction has the canonical form:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5}, "
    "{%6}, "
    "{%0, %1, %2, %3};"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(b0)
);
```

FP16 MMA requires `sm_75+`; native BF16 MMA requires `sm_80+`. Unsupported
types, shapes, and targets retain the CUDA-core fallback. See the
[implementation guide](docs/implementation.md#tensor-core-dot) for fragment
layouts, composition, accumulation, and current performance limitations.

## Documentation

- [Documentation index](docs/README.md)
- [Compiler and backend implementation](docs/implementation.md)
- [Version changelog](CHANGELOG.md)
- [Colab test notebook](tests/mytriton_colab_tests.ipynb)
- [My Triton From Scratch blog](https://pbelevich.github.io/)

The implementation guide contains the detailed material previously kept in
this README: the AST frontend, runtime loops, block constructors, CUDA tile
layouts, `tl.dot`, shared-memory staging and optimization, PyTorch
interoperability, mixed precision, Tensor Core fragments, backend limitations,
and development notes.

## Getting started

Requirements:

- Python 3.10 or newer;
- NumPy 2 or newer;
- CuPy matching the CUDA toolkit for GPU execution;
- optionally, PyTorch for zero-copy tensor launches.

Clone the project and install the compiler:

```bash
git clone https://github.com/pbelevich/mytriton.git
cd mytriton
python -m pip install -e .
```

Install the matching CUDA extra:

```bash
# CUDA 12
python -m pip install -e ".[cuda12]"

# CUDA 13
python -m pip install -e ".[cuda13]"
```

NumPy and CPU Torch arrays compile kernels and return generated source. CuPy
arrays and CUDA Torch tensors additionally compile and execute the generated
kernel. PyTorch is optional and should be installed separately with the build
appropriate for the local CUDA environment.

For a ready-to-run A100 workflow, open the
[demo notebook](https://colab.research.google.com/drive/140oC_PrW1BOgaw7mV9Or3Xa3lCrwcrz3).

Install development dependencies and run the complete local check suite:

```bash
python -m pip install -e ".[dev]"
make check
```

GPU execution tests can be required explicitly:

```bash
MYTRITON_REQUIRE_CUDA=1 python -m pytest -v -s
```

GitHub Actions runs Ruff, formatting checks, mypy, unit/codegen tests, wheel
building, and typed-package validation on Python 3.10 and 3.13.
