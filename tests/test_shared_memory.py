from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def shared_roundtrip_2d_kernel(x, out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    local_m = tl.arange(0, BM)[:, None]
    local_n = tl.arange(0, BN)[None, :]

    offs_m = pid_m * BM + local_m
    offs_n = pid_n * BN + local_n

    global_offsets = offs_m * N + offs_n
    local_offsets = local_m * BN + local_n

    mask = (offs_m < M) & (offs_n < N)

    tile = tl._shared_array((BM, BN), dtype=tl.float32)

    values = tl.load(x + global_offsets, mask=mask, other=0.0)
    tl.store(tile + local_offsets, values)

    tl._barrier()

    out_values = tl.load(tile + local_offsets)
    tl.store(out + global_offsets, out_values, mask=mask)


@pytest.mark.codegen
def test_shared_roundtrip_2d_generates_cuda_source_without_execution(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N = 19, 37
    BM, BN = 8, 16

    x = np.zeros((M, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    shared_roundtrip_2d_kernel.clear_cache()
    _, ssa_ops, cuda_src = shared_roundtrip_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](x, out, M, N, BM=BM, BN=BN)

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 8 : i32
        %2 = arange {start=0, end=8} : vector<8 x i32>
        %3 = expand_dims %2 {axis=1} : block<8x1 x i32>
        %4 = add %1, %3 : block<8x1 x i32>
        %5 = mul %4, N : block<8x1 x i32>
        %6 = program_id {axis=1} : i32
        %7 = mul %6, 16 : i32
        %8 = arange {start=0, end=16} : vector<16 x i32>
        %9 = expand_dims %8 {axis=0} : block<1x16 x i32>
        %10 = add %7, %9 : block<1x16 x i32>
        %11 = add %5, %10 : block<8x16 x i32>
        %12 = addptr x, %11 : block<8x16 x ptr<f32>>
        %13 = cmp_lt %4, M : block<8x1 x bool>
        %14 = cmp_lt %10, N : block<1x16 x bool>
        %15 = and %13, %14 : block<8x16 x bool>
        %16 = load %12, %15, 0.0 : block<8x16 x f32>
        %17 = shared_alloc {shape=(8, 16), dtype=f32} : ptr<shared, f32>
        %18 = mul %3, 16 : block<8x1 x i32>
        %19 = add %18, %9 : block<8x16 x i32>
        %20 = addptr %17, %19 : block<8x16 x ptr<shared, f32>>
        store %20, %16, none
        barrier
        %22 = load %20, none, none : block<8x16 x f32>
        %23 = addptr out, %11 : block<8x16 x ptr<f32>>
        store %23, %22, %15
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void shared_roundtrip_2d_kernel(float* x, float* out, int M, int N) {
            __shared__ float smem17[128];

            int tile_i = threadIdx.x / 16;
            int tile_j = threadIdx.x % 16;
            int v0 = blockIdx.x;
            int v1 = (v0 * 8);
            int v3 = tile_i;
            int v4 = (v1 + v3);
            int v5 = (v4 * N);
            int v6 = blockIdx.y;
            int v7 = (v6 * 16);
            int v9 = tile_j;
            int v10 = (v7 + v9);
            int v11 = (v5 + v10);
            bool v13 = (v4 < M);
            bool v14 = (v10 < N);
            bool v15 = (v13 && v14);
            float v16 = (v15 ? x[v11] : 0.0f);
            int v18 = (v3 * 16);
            int v19 = (v18 + v9);
            smem17[v19] = v16;
            __syncthreads();
            float v22 = smem17[v19];
            if (v15) {
                out[v11] = v22;
            }
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.execution
def test_shared_roundtrip_2d_cuda_execution(cp, monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    M, N = 19, 37
    BM, BN = 8, 16

    x = cp.arange(M * N, dtype=cp.float32).reshape(M, N)
    out = cp.zeros((M, N), dtype=cp.float32)

    shared_roundtrip_2d_kernel.clear_cache()
    shared_roundtrip_2d_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](x, out, M, N, BM=BM, BN=BN)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(out, x)
