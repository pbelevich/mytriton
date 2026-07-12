import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAValue


@triton.jit
def matmul_2d_dot_kernel(
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

    acc = c_offsets * 0.0

    for k in tl.static_range(0, K):
        a_offsets = offs_m * K + k
        b_offsets = k * N + offs_n

        a_values = tl.load(a + a_offsets, mask=offs_m < M, other=0.0)
        b_values = tl.load(b + b_offsets, mask=offs_n < N, other=0.0)

        acc = acc + tl.dot(a_values, b_values)

    tl.store(c + c_offsets, acc, mask=c_mask)


@pytest.mark.codegen
def test_matmul_2d_dot_generates_dot_ssa_and_cuda_source(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 5, 7, 3
    BM, BN = 4, 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    matmul_2d_dot_kernel.clear_cache()
    _, ssa_ops, cuda_src = matmul_2d_dot_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K=K, BM=BM, BN=BN)

    dot_ops = [op for op in ssa_ops if op.opcode == "dot"]
    assert len(dot_ops) == K

    for op in dot_ops:
        assert op.result is not None
        assert str(op.result.ty) == "block<4x8 x f32>"

    assert "int tile_i = threadIdx.x / 8;" in cuda_src
    assert "int tile_j = threadIdx.x % 8;" in cuda_src

    # Dot MVP lowers to scalar multiply for the current output element.
    assert " * " in cuda_src


def test_matmul_2d_dot_executes_with_cupy_when_cuda_is_available():
    cp = pytest.importorskip("cupy")

    if not triton.cuda_available():
        pytest.skip("CUDA GPU is not available")

    M, N, K = 19, 23, 5
    BM, BN = 8, 16

    a = cp.arange(M * K, dtype=cp.float32).reshape(M, K)
    b = cp.arange(K * N, dtype=cp.float32).reshape(K, N)
    c = cp.zeros((M, N), dtype=cp.float32)

    matmul_2d_dot_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K=K, BM=BM, BN=BN)

    cp.cuda.Stream.null.synchronize()

    expected = a @ b
    cp.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)


@triton.jit
def dot_full_shape_kernel(
    a, b, c, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    k = tl.arange(0, BK)

    lhs = tl.load(a + m * BK + k[None, :])
    rhs = tl.load(b + k[:, None] * BN + n)

    out = tl.dot(lhs, rhs)
    tl.store(c + m * BN + n, out)


@pytest.mark.codegen
def test_dot_full_k_generates_single_dot_op(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    BM, BN, BK = 4, 8, 16

    a = np.zeros((BM, BK), dtype=np.float32)
    b = np.zeros((BK, BN), dtype=np.float32)
    c = np.zeros((BM, BN), dtype=np.float32)

    dot_full_shape_kernel.clear_cache()
    _, ssa_ops, cuda_src = dot_full_shape_kernel[lambda meta: (1, 1)](
        a, b, c, BM=BM, BN=BN, BK=BK
    )

    dot_ops = [op for op in ssa_ops if op.opcode == "dot"]
    assert len(dot_ops) == 1

    dot = dot_ops[0]
    assert dot.result is not None
    assert isinstance(dot.operands[0], SSAValue)
    assert isinstance(dot.operands[1], SSAValue)
    assert str(dot.operands[0].ty) == "block<4x16 x f32>"
    assert str(dot.operands[1].ty) == "block<16x8 x f32>"
    assert str(dot.result.ty) == "block<4x8 x f32>"

    assert "int tile_i = threadIdx.x / 8;" in cuda_src
    assert "int tile_j = threadIdx.x % 8;" in cuda_src
    assert cuda_src.count(" += ") >= BK
