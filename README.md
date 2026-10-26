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
multiplication, FP16/BF16 inputs, FP32 accumulation, and multi-warp Tensor Core
CTA tiles built from NVIDIA `mma.sync` instructions.
Low-precision operands are cooperatively staged with aligned 128-bit
`cp.async` copies, then loaded into warp fragments with `ldmatrix`.
Triton-style `Config` and `autotune` APIs select a CUDA schedule per problem
shape and cache the measured winner.

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
| [`ver22`](https://github.com/pbelevich/mytriton/tree/ver22) | [Multi-warp Tensor Core CTAs](docs/implementation.md#multi-warp-cta-tiles) |
| [`ver23`](https://github.com/pbelevich/mytriton/tree/ver23) | [Vectorized operand staging](docs/implementation.md#vectorized-global-to-shared-staging) |
| [`ver24`](https://github.com/pbelevich/mytriton/tree/ver24) | [`ldmatrix` operand fragments](docs/implementation.md#warp-cooperative-ldmatrix-fragments) |
| [`ver25`](https://github.com/pbelevich/mytriton/tree/ver25) | [SM80 production tensor tile](docs/implementation.md#sm80-production-tensor-tile) |
| [`ver26`](https://github.com/pbelevich/mytriton/tree/ver26) | [Three-stage `cp.async` pipeline](docs/implementation.md#pipelined-tensor-core-gemm) |
| [`ver27`](https://github.com/pbelevich/mytriton/tree/ver27) | [Static register pipeline](docs/implementation.md#static-register-pipeline) |
| [`ver28`](https://github.com/pbelevich/mytriton/tree/ver28) | [CUDA event autotuning](docs/implementation.md#cuda-event-autotuning) |
| [`ver29`](https://github.com/pbelevich/mytriton/tree/ver29) | [JIT runtime specialization](docs/implementation.md#jit-runtime-specialization) |

## Tensor-core matmul

This is a complete `mytriton` kernel. FP16 A and B tiles are staged through
shared memory, multiplied by Tensor Cores, and accumulated in FP32 across the
runtime K-loop.

```python
import torch

import mytriton as triton
import mytriton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BK": 32, "BN": 64}, num_warps=4),
        triton.Config({"BM": 128, "BK": 32, "BN": 128}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
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


M, N, K = 4096, 4096, 4096

torch.manual_seed(0)
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
out = torch.empty((M, N), device="cuda", dtype=torch.float32)

grid = lambda meta: (
    triton.cdiv(M, meta["BM"]),
    triton.cdiv(N, meta["BN"]),
)
_, _, cuda_source = matmul_kernel[grid](
    a,
    b,
    out,
    M,
    N,
    K,
)

expected = a.float() @ b.float()
torch.testing.assert_close(out, expected, rtol=3e-3, atol=3e-3)
```

The `128 x 32 x 128` CTA tile is divided into a `2 x 2` warp grid. Each warp
owns a `64 x 64` output tile. SM80 executes physical `m16n8k16`
instructions while FP16 SM75 tiles retain `m16n8k8`:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%0, %1, %2, %3};"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
);
```

Before MMA, each thread asynchronously moves eight adjacent FP16/BF16 values
into aligned shared memory:

```cuda
asm volatile(
    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
    : : "r"(shared_address), "l"(global_pointer) : "memory"
);
```

Each warp then loads the shared A and B tiles directly into the packed
register layout required by `mma.sync`:

```cuda
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
    : "r"(shared_address)
    : "memory"
);
```

On A100 this raises the best measured kernel throughput from about 24 to 41
TFLOP/s. See the
[Version 23 performance report](benchmarks/matmul_vectorized_staging_a100_report.md)
for the complete methodology, tile sweep, SASS, and Nsight Compute analysis.
Version 24 combines `ldmatrix` with a 16-byte XOR-swizzled shared layout. The
same tile reaches 59.95 TFLOP/s for BF16 and 59.88 TFLOP/s for FP16 at
`4096^3`, while shared-load bank conflicts fall by 99.3%. See the
[Version 24 performance report](benchmarks/matmul_ldmatrix_a100_report.md).
Version 25 adds native SM80 `m16n8k16`, a direct register-to-global epilogue,
and a correct `128 x 128 x 32` experimental CTA tile. The established
`64 x 16 x 64` tile remains the fastest default at 63.19 TFLOP/s until an
asynchronous pipeline can overlap operand transport with MMA. See the
[Version 25 performance report](benchmarks/matmul_production_tile_a100_report.md).
Version 26 adds that three-stage pipeline, aligned `cp.async` loads with a
correct scalar tail fallback, and K-sliced fragment consumption. With eight
warps the `128 x 32 x 128` BF16 kernel reaches 146.21 TFLOP/s in the final
30-sample run and 162.49 TFLOP/s in the configuration sweep on A100. See the
[Version 26 performance report](benchmarks/matmul_pipeline_a100_report.md).
Version 27 specializes that production tile with a static three-phase stage
schedule, warp-raked copies, precomputed shared addresses, and two live K
fragment sets. The final BF16 `4096^3` run reaches 187.19 TFLOP/s, with
202.14 TFLOP/s observed in a same-day run; FP16 reaches 203.67 TFLOP/s. See the
[Version 27 performance report](benchmarks/matmul_static_pipeline_a100_report.md).
Version 28 adds CUDA-event autotuning and a per-device, per-dtype, per-shape
winner cache. It selects `64 x 32 x 64` for smaller square matrices and the
static `128 x 32 x 128` pipeline for large matrices. On A100, BF16 `8192^3`
reaches 210.21 TFLOP/s, 86.7% of torch/cuBLAS in the same run. See the
[Version 28 performance report](benchmarks/matmul_autotune_a100_report.md).
Version 29 specializes JIT cache entries by safe runtime pointer-alignment and
integer-divisibility facts. Aligned GEMM variants omit redundant alignment
guards while retaining boundary checks and the scalar fallback for edge CTAs.
The same-session A100 measurements are performance-neutral after accounting
for clock-state variation; see the
[Version 29 performance report](benchmarks/matmul_jit_specialization_a100_report.md).

FP16 MMA requires `sm_75+`; native BF16 MMA requires `sm_80+`. Unsupported
types, shapes, and targets retain the CUDA-core fallback. See the
[implementation guide](docs/implementation.md#cuda-event-autotuning)
for fragment layouts, shared-memory mapping, JIT specialization, composition,
and accumulation.

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
