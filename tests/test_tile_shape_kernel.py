import inspect
from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSALowering, SSAPrinter
from mytriton.ssa_verification import SSAVerifier
from mytriton.trace import I32, PTR_F32, Param, trace


def tile_shape_kernel(out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    offs_m = tl.arange(0, BM)[:, None]
    offs_n = tl.arange(0, BN)[None, :]

    offsets = offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    tl.store(out + offsets, offsets, mask=mask)


def test_tile_shape_kernel_lowers_rank2_offsets_without_cuda_execution():
    signature = inspect.signature(tile_shape_kernel)
    bound = signature.bind(object(), 7, 11, BM=16, BN=32)
    params = [
        Param("out", PTR_F32),
        Param("M", I32),
        Param("N", I32),
    ]

    ops, _ = trace(
        tile_shape_kernel,
        signature,
        bound.arguments,
        runtime_params=params,
    )
    ssa_ops = SSALowering().lower(ops)

    SSAVerifier(block_size=16 * 32).verify(ssa_ops)

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = arange {start=0, end=16} : vector<16 x i32>
        %1 = expand_dims %0 {axis=1} : block<16x1 x i32>
        %2 = mul %1, N : block<16x1 x i32>
        %3 = arange {start=0, end=32} : vector<32 x i32>
        %4 = expand_dims %3 {axis=0} : block<1x32 x i32>
        %5 = add %2, %4 : block<16x32 x i32>
        %6 = addptr out, %5 : block<16x32 x ptr<f32>>
        %7 = cmp_lt %1, M : block<16x1 x bool>
        %8 = cmp_lt %4, N : block<1x32 x bool>
        %9 = and %7, %8 : block<16x32 x bool>
        store %6, %5, %9
        """
    ).rstrip("\n")


@triton.jit
def matrix_add_2d_kernel(x, y, out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]

    offsets = offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    lhs = tl.load(x + offsets, mask=mask, other=0.0)
    rhs = tl.load(y + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, lhs + rhs, mask=mask)


@triton.jit
def invalid_unparenthesized_mask_kernel(
    x,
    out,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    offs_m = tl.arange(0, BM)[:, None]
    offs_n = tl.arange(0, BN)[None, :]

    mask = offs_m < M & offs_n < N
    offsets = offs_m * N + offs_n

    tl.store(out + offsets, tl.load(x + offsets, mask=mask, other=0.0), mask=mask)


@pytest.mark.codegen
def test_matrix_add_2d_generates_rank2_cuda_source_without_execution(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N = 19, 37
    x = np.zeros((M * N,), dtype=np.float32)
    y = np.zeros((M * N,), dtype=np.float32)
    out = np.zeros((M * N,), dtype=np.float32)

    _, ssa_ops, cuda_src = matrix_add_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](x, y, out, M, N, BM=16, BN=32)

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 16 : i32
        %2 = arange {start=0, end=16} : vector<16 x i32>
        %3 = expand_dims %2 {axis=1} : block<16x1 x i32>
        %4 = add %1, %3 : block<16x1 x i32>
        %5 = mul %4, N : block<16x1 x i32>
        %6 = program_id {axis=1} : i32
        %7 = mul %6, 32 : i32
        %8 = arange {start=0, end=32} : vector<32 x i32>
        %9 = expand_dims %8 {axis=0} : block<1x32 x i32>
        %10 = add %7, %9 : block<1x32 x i32>
        %11 = add %5, %10 : block<16x32 x i32>
        %12 = addptr x, %11 : block<16x32 x ptr<f32>>
        %13 = cmp_lt %4, M : block<16x1 x bool>
        %14 = cmp_lt %10, N : block<1x32 x bool>
        %15 = and %13, %14 : block<16x32 x bool>
        %16 = load %12, %15, 0.0 : block<16x32 x f32>
        %17 = addptr y, %11 : block<16x32 x ptr<f32>>
        %18 = load %17, %15, 0.0 : block<16x32 x f32>
        %19 = add %16, %18 : block<16x32 x f32>
        %20 = addptr out, %11 : block<16x32 x ptr<f32>>
        store %20, %19, %15
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void matrix_add_2d_kernel(float* x, float* y, float* out, int M, int N) {
            int tile_i = threadIdx.x / 32;
            int tile_j = threadIdx.x % 32;
            int v0 = blockIdx.x;
            int v1 = (v0 * 16);
            int v3 = tile_i;
            int v4 = (v1 + v3);
            int v5 = (v4 * N);
            int v6 = blockIdx.y;
            int v7 = (v6 * 32);
            int v9 = tile_j;
            int v10 = (v7 + v9);
            int v11 = (v5 + v10);
            bool v13 = (v4 < M);
            bool v14 = (v10 < N);
            bool v15 = (v13 && v14);
            float v16 = (v15 ? x[v11] : 0.0f);
            float v18 = (v15 ? y[v11] : 0.0f);
            float v19 = (v16 + v18);
            if (v15) {
                out[v11] = v19;
            }
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.codegen
@pytest.mark.parametrize(("BM", "BN"), [(1, 8), (4, 1)])
def test_matrix_add_2d_generates_cuda_for_degenerate_rank2_tiles(
    monkeypatch,
    BM,
    BN,
):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N = 3, 7
    x = np.zeros((M * N,), dtype=np.float32)
    y = np.zeros((M * N,), dtype=np.float32)
    out = np.zeros((M * N,), dtype=np.float32)

    matrix_add_2d_kernel.clear_cache()
    _, _, cuda_src = matrix_add_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](x, y, out, M, N, BM=BM, BN=BN)

    assert f"int tile_i = threadIdx.x / {BN};" in cuda_src
    assert f"int tile_j = threadIdx.x % {BN};" in cuda_src
    assert "float v16 = (v15 ? x[v11] : 0.0f);" in cuda_src


def test_unparenthesized_mask_error_mentions_parentheses():
    M, N = 3, 7
    x = np.zeros((M * N,), dtype=np.float32)
    out = np.zeros((M * N,), dtype=np.float32)

    invalid_unparenthesized_mask_kernel.clear_cache()
    with pytest.raises(TypeError, match="wrap each comparison in parentheses"):
        invalid_unparenthesized_mask_kernel[(1, 1)](
            x,
            out,
            M,
            N,
            BM=4,
            BN=8,
        )


def test_matrix_add_2d_executes_with_cupy_when_cuda_is_available():
    cp = pytest.importorskip("cupy")

    if not triton.cuda_available():
        pytest.skip("CUDA GPU is not available")

    M, N = 19, 37
    size = M * N

    x = cp.arange(size, dtype=cp.float32)
    y = cp.arange(size, dtype=cp.float32) * cp.float32(2.0)
    out = cp.zeros(size, dtype=cp.float32)

    matrix_add_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](x, y, out, M, N, BM=16, BN=32)

    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_allclose(out, x + y)
