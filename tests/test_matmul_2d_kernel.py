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

    # symbolic zero tile: block<BM x BN x f32>
    acc = c_offsets * 0.0

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
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 4 : i32
        %2 = arange {start=0, end=4} : vector<4 x i32>
        %3 = expand_dims %2 {axis=1} : block<4x1 x i32>
        %4 = add %1, %3 : block<4x1 x i32>
        %5 = mul %4, N : block<4x1 x i32>
        %6 = program_id {axis=1} : i32
        %7 = mul %6, 8 : i32
        %8 = arange {start=0, end=8} : vector<8 x i32>
        %9 = expand_dims %8 {axis=0} : block<1x8 x i32>
        %10 = add %7, %9 : block<1x8 x i32>
        %11 = add %5, %10 : block<4x8 x i32>
        %12 = mul %11, 0.0 : block<4x8 x f32>
        %13 = mul %4, 3 : block<4x1 x i32>
        %15 = addptr a, %13 : block<4x1 x ptr<f32>>
        %16 = cmp_lt %4, M : block<4x1 x bool>
        %17 = load %15, %16, 0.0 : block<4x1 x f32>
        %18 = mul 0, N : i32
        %19 = add %18, %10 : block<1x8 x i32>
        %20 = addptr b, %19 : block<1x8 x ptr<f32>>
        %21 = cmp_lt %10, N : block<1x8 x bool>
        %22 = load %20, %21, 0.0 : block<1x8 x f32>
        %23 = mul %17, %22 : block<4x8 x f32>
        %24 = add %12, %23 : block<4x8 x f32>
        %26 = add %13, 1 : block<4x1 x i32>
        %27 = addptr a, %26 : block<4x1 x ptr<f32>>
        %29 = load %27, %16, 0.0 : block<4x1 x f32>
        %30 = mul 1, N : i32
        %31 = add %30, %10 : block<1x8 x i32>
        %32 = addptr b, %31 : block<1x8 x ptr<f32>>
        %34 = load %32, %21, 0.0 : block<1x8 x f32>
        %35 = mul %29, %34 : block<4x8 x f32>
        %36 = add %24, %35 : block<4x8 x f32>
        %38 = add %13, 2 : block<4x1 x i32>
        %39 = addptr a, %38 : block<4x1 x ptr<f32>>
        %41 = load %39, %16, 0.0 : block<4x1 x f32>
        %42 = mul 2, N : i32
        %43 = add %42, %10 : block<1x8 x i32>
        %44 = addptr b, %43 : block<1x8 x ptr<f32>>
        %46 = load %44, %21, 0.0 : block<1x8 x f32>
        %47 = mul %41, %46 : block<4x8 x f32>
        %48 = add %36, %47 : block<4x8 x f32>
        %49 = addptr c, %11 : block<4x8 x ptr<f32>>
        %52 = and %16, %21 : block<4x8 x bool>
        store %49, %48, %52
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void matmul_2d_kernel(float* a, float* b, float* c, int M, int N) {
            int tile_i = threadIdx.x / 8;
            int tile_j = threadIdx.x % 8;
            int v0 = blockIdx.x;
            int v1 = (v0 * 4);
            int v3 = tile_i;
            int v4 = (v1 + v3);
            int v5 = (v4 * N);
            int v6 = blockIdx.y;
            int v7 = (v6 * 8);
            int v9 = tile_j;
            int v10 = (v7 + v9);
            int v11 = (v5 + v10);
            float v12 = (v11 * 0.0f);
            int v13 = (v4 * 3);
            bool v16 = (v4 < M);
            float v17 = (v16 ? a[v13] : 0.0f);
            int v18 = (0 * N);
            int v19 = (v18 + v10);
            bool v21 = (v10 < N);
            float v22 = (v21 ? b[v19] : 0.0f);
            float v23 = (v17 * v22);
            float v24 = (v12 + v23);
            int v26 = (v13 + 1);
            float v29 = (v16 ? a[v26] : 0.0f);
            int v30 = (1 * N);
            int v31 = (v30 + v10);
            float v34 = (v21 ? b[v31] : 0.0f);
            float v35 = (v29 * v34);
            float v36 = (v24 + v35);
            int v38 = (v13 + 2);
            float v41 = (v16 ? a[v38] : 0.0f);
            int v42 = (2 * N);
            int v43 = (v42 + v10);
            float v46 = (v21 ? b[v43] : 0.0f);
            float v47 = (v41 * v46);
            float v48 = (v36 + v47);
            bool v52 = (v16 && v21);
            if (v52) {
                c[v11] = v48;
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
