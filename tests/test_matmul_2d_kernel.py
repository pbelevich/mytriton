from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def matmul_2d_kernel(
    a,
    b,
    c,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]

    c_offsets = offs_m * N + offs_n
    c_mask = (offs_m < M) & (offs_n < N)

    acc = tl.zeros((BM, BN), tl.float32)

    for k in tl.static_range(0, K):
        a_offsets = offs_m * K + k
        b_offsets = k * N + offs_n

        a_mask = offs_m < M
        b_mask = offs_n < N

        a_values = tl.load(a + a_offsets, mask=a_mask, other=0.0)
        b_values = tl.load(b + b_offsets, mask=b_mask, other=0.0)

        acc = acc + a_values * b_values

    tl.store(c + c_offsets, acc, mask=c_mask)


@pytest.mark.codegen
def test_matmul_2d_generates_rank2_cuda_source_without_execution(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 5, 7, 3
    BM, BN = 4, 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    matmul_2d_kernel.clear_cache()
    _, ssa_ops, cuda_src = matmul_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K=K, BM=BM, BN=BN)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(4, 8), dtype=f32} : block<4x8 x f32>
        %1 = program_id {axis=0} : i32
        %2 = mul %1, 4 : i32
        %3 = arange {start=0, end=4} : vector<4 x i32>
        %4 = expand_dims %3 {axis=1} : block<4x1 x i32>
        %5 = add %2, %4 : block<4x1 x i32>
        %6 = mul %5, 3 : block<4x1 x i32>
        %8 = addptr a, %6 : block<4x1 x ptr<f32>>
        %9 = cmp_lt %5, M : block<4x1 x bool>
        %10 = load %8, %9, 0.0 : block<4x1 x f32>
        %11 = mul 0, N : i32
        %12 = program_id {axis=1} : i32
        %13 = mul %12, 8 : i32
        %14 = arange {start=0, end=8} : vector<8 x i32>
        %15 = expand_dims %14 {axis=0} : block<1x8 x i32>
        %16 = add %13, %15 : block<1x8 x i32>
        %17 = add %11, %16 : block<1x8 x i32>
        %18 = addptr b, %17 : block<1x8 x ptr<f32>>
        %19 = cmp_lt %16, N : block<1x8 x bool>
        %20 = load %18, %19, 0.0 : block<1x8 x f32>
        %21 = mul %10, %20 : block<4x8 x f32>
        %22 = add %0, %21 : block<4x8 x f32>
        %24 = add %6, 1 : block<4x1 x i32>
        %25 = addptr a, %24 : block<4x1 x ptr<f32>>
        %27 = load %25, %9, 0.0 : block<4x1 x f32>
        %28 = mul 1, N : i32
        %29 = add %28, %16 : block<1x8 x i32>
        %30 = addptr b, %29 : block<1x8 x ptr<f32>>
        %32 = load %30, %19, 0.0 : block<1x8 x f32>
        %33 = mul %27, %32 : block<4x8 x f32>
        %34 = add %22, %33 : block<4x8 x f32>
        %36 = add %6, 2 : block<4x1 x i32>
        %37 = addptr a, %36 : block<4x1 x ptr<f32>>
        %39 = load %37, %9, 0.0 : block<4x1 x f32>
        %40 = mul 2, N : i32
        %41 = add %40, %16 : block<1x8 x i32>
        %42 = addptr b, %41 : block<1x8 x ptr<f32>>
        %44 = load %42, %19, 0.0 : block<1x8 x f32>
        %45 = mul %39, %44 : block<4x8 x f32>
        %46 = add %34, %45 : block<4x8 x f32>
        %47 = mul %5, N : block<4x1 x i32>
        %48 = add %47, %16 : block<4x8 x i32>
        %49 = addptr c, %48 : block<4x8 x ptr<f32>>
        %52 = and %9, %19 : block<4x8 x bool>
        store %49, %46, %52
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void matmul_2d_kernel(float* a, float* b, float* c, int M, int N) {
            int tile_i = threadIdx.x / 8;
            int tile_j = threadIdx.x % 8;
            float v0 = 0.0f;
            int v1 = blockIdx.x;
            int v2 = (v1 * 4);
            int v4 = tile_i;
            int v5 = (v2 + v4);
            int v6 = (v5 * 3);
            bool v9 = (v5 < M);
            float v10 = (v9 ? a[v6] : 0.0f);
            int v11 = (0 * N);
            int v12 = blockIdx.y;
            int v13 = (v12 * 8);
            int v15 = tile_j;
            int v16 = (v13 + v15);
            int v17 = (v11 + v16);
            bool v19 = (v16 < N);
            float v20 = (v19 ? b[v17] : 0.0f);
            float v21 = (v10 * v20);
            float v22 = (v0 + v21);
            int v24 = (v6 + 1);
            float v27 = (v9 ? a[v24] : 0.0f);
            int v28 = (1 * N);
            int v29 = (v28 + v16);
            float v32 = (v19 ? b[v29] : 0.0f);
            float v33 = (v27 * v32);
            float v34 = (v22 + v33);
            int v36 = (v6 + 2);
            float v39 = (v9 ? a[v36] : 0.0f);
            int v40 = (2 * N);
            int v41 = (v40 + v16);
            float v44 = (v19 ? b[v41] : 0.0f);
            float v45 = (v39 * v44);
            float v46 = (v34 + v45);
            int v47 = (v5 * N);
            int v48 = (v47 + v16);
            bool v52 = (v9 && v19);
            if (v52) {
                c[v48] = v46;
            }
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.execution
def test_matmul_2d_execution(cp):
    M, N, K = 19, 23, 5
    BM, BN = 8, 16

    a = cp.arange(M * K, dtype=cp.float32).reshape(M, K)
    b = cp.arange(K * N, dtype=cp.float32).reshape(K, N)
    c = cp.zeros((M, N), dtype=cp.float32)

    matmul_2d_kernel.clear_cache()
    matmul_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K=K, BM=BM, BN=BN)

    cp.cuda.Stream.null.synchronize()

    expected = a @ b
    cp.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)
