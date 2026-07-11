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
        %5 = mul %4, 3 : block<4x1 x i32>
        %7 = addptr a, %5 : block<4x1 x ptr<f32>>
        %8 = cmp_lt %4, M : block<4x1 x bool>
        %9 = load %7, %8, 0.0 : block<4x1 x f32>
        %10 = mul 0, N : i32
        %11 = program_id {axis=1} : i32
        %12 = mul %11, 8 : i32
        %13 = arange {start=0, end=8} : vector<8 x i32>
        %14 = expand_dims %13 {axis=0} : block<1x8 x i32>
        %15 = add %12, %14 : block<1x8 x i32>
        %16 = add %10, %15 : block<1x8 x i32>
        %17 = addptr b, %16 : block<1x8 x ptr<f32>>
        %18 = cmp_lt %15, N : block<1x8 x bool>
        %19 = load %17, %18, 0.0 : block<1x8 x f32>
        %21 = add %5, 1 : block<4x1 x i32>
        %22 = addptr a, %21 : block<4x1 x ptr<f32>>
        %24 = load %22, %8, 0.0 : block<4x1 x f32>
        %25 = mul 1, N : i32
        %26 = add %25, %15 : block<1x8 x i32>
        %27 = addptr b, %26 : block<1x8 x ptr<f32>>
        %29 = load %27, %18, 0.0 : block<1x8 x f32>
        %31 = add %5, 2 : block<4x1 x i32>
        %32 = addptr a, %31 : block<4x1 x ptr<f32>>
        %34 = load %32, %8, 0.0 : block<4x1 x f32>
        %35 = mul 2, N : i32
        %36 = add %35, %15 : block<1x8 x i32>
        %37 = addptr b, %36 : block<1x8 x ptr<f32>>
        %39 = load %37, %18, 0.0 : block<1x8 x f32>
        %40 = mul %4, N : block<4x1 x i32>
        %41 = add %40, %15 : block<4x8 x i32>
        %42 = mul %41, 0.0 : block<4x8 x f32>
        %43 = mul %9, %19 : block<4x8 x f32>
        %44 = add %42, %43 : block<4x8 x f32>
        %45 = mul %24, %29 : block<4x8 x f32>
        %46 = add %44, %45 : block<4x8 x f32>
        %47 = mul %34, %39 : block<4x8 x f32>
        %48 = add %46, %47 : block<4x8 x f32>
        %49 = addptr c, %41 : block<4x8 x ptr<f32>>
        %52 = and %8, %18 : block<4x8 x bool>
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
            int v5 = (v4 * 3);
            bool v8 = (v4 < M);
            float v9 = (v8 ? a[v5] : 0.0f);
            int v10 = (0 * N);
            int v11 = blockIdx.y;
            int v12 = (v11 * 8);
            int v14 = tile_j;
            int v15 = (v12 + v14);
            int v16 = (v10 + v15);
            bool v18 = (v15 < N);
            float v19 = (v18 ? b[v16] : 0.0f);
            int v21 = (v5 + 1);
            float v24 = (v8 ? a[v21] : 0.0f);
            int v25 = (1 * N);
            int v26 = (v25 + v15);
            float v29 = (v18 ? b[v26] : 0.0f);
            int v31 = (v5 + 2);
            float v34 = (v8 ? a[v31] : 0.0f);
            int v35 = (2 * N);
            int v36 = (v35 + v15);
            float v39 = (v18 ? b[v36] : 0.0f);
            int v40 = (v4 * N);
            int v41 = (v40 + v15);
            float v42 = (v41 * 0.0f);
            float v43 = (v9 * v19);
            float v44 = (v42 + v43);
            float v45 = (v24 * v29);
            float v46 = (v44 + v45);
            float v47 = (v34 * v39);
            float v48 = (v46 + v47);
            bool v52 = (v8 && v18);
            if (v52) {
                c[v41] = v48;
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
