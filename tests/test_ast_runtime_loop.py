from textwrap import dedent
from typing import Any, cast

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ast_frontend import ASTFrontendError
from mytriton.ssa import SSAForRange, SSALowering, SSAPrinter
from mytriton.trace import (
    PTR_F32,
    Const,
    ForRange,
    LoopIndex,
    Param,
    Store,
    TopLevelOp,
)


@triton.jit
def matmul_runtime_k_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]

    c_offsets = offs_m * N + offs_n
    c_mask = (offs_m < M) & (offs_n < N)

    acc = tl.zeros((BM, BN), tl.float32)  # acc = c_offsets * 0.0

    for k in range(K):
        a_offsets = offs_m * K + k
        b_offsets = k * N + offs_n

        a_values = tl.load(a + a_offsets, mask=offs_m < M, other=0.0)
        b_values = tl.load(b + b_offsets, mask=offs_n < N, other=0.0)

        acc = acc + a_values * b_values

    tl.store(c + c_offsets, acc, mask=c_mask)


@triton.jit
def two_carried_values_kernel(out, n):
    first = 0
    second = 1

    for k in range(n):
        first = first + k
        second = second + first

    tl.store(out, first + second)


@triton.jit
def annotated_carried_value_kernel(out, n):
    acc: Any = 0

    for k in range(n):
        acc: Any = acc + k  # type: ignore[no-redef]

    tl.store(out, acc)


@triton.jit
def no_carried_values_kernel(out, n):
    for k in range(n):
        tl.store(out + k, k)


@triton.jit
def sequential_runtime_loops_kernel(out, n):
    acc = 0

    for i in range(n):
        acc = acc + i

    for j in range(n):
        acc = acc + j

    tl.store(out, acc)


@triton.jit
def nested_runtime_loops_kernel(out, n):
    offset = tl.program_id(0)
    acc: Any = 0

    for i in range(n):
        for j in range(n):
            acc = acc + offset + i + j

    tl.store(out + offset, acc)


@triton.jit
def invalid_runtime_step_kernel(out, n, STEP: tl.constexpr):
    acc = 0

    for k in range(0, n, STEP):
        acc = acc + k

    tl.store(out, acc)


@triton.jit
def assigned_induction_variable_kernel(out, n):
    for k in range(n):
        k += 1

    tl.store(out, 0.0)


def compile_scalar_kernel(monkeypatch, kernel, *args, backend="cuda", **kwargs):
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", backend)
    kernel.clear_cache()
    return kernel[(1,)](*args, **kwargs)


@pytest.mark.codegen
def test_runtime_for_preserves_source_order_for_multiple_carried_values(monkeypatch):
    out = np.zeros(1, dtype=np.float32)

    _, ssa_ops, _ = compile_scalar_kernel(
        monkeypatch,
        two_carried_values_kernel,
        out,
        3,
    )

    loops = [op for op in ssa_ops if isinstance(op, SSAForRange)]
    assert len(loops) == 1
    assert loops[0].carried_inputs == (Const(0), Const(1))


@pytest.mark.execution
def test_runtime_for_executes_multiple_carried_values(cp, monkeypatch):
    n = 4
    out = cp.zeros(1, dtype=cp.float32)

    compile_scalar_kernel(monkeypatch, two_carried_values_kernel, out, n)
    cp.cuda.Stream.null.synchronize()

    first = 0
    second = 1
    for k in range(n):
        first += k
        second += first

    cp.testing.assert_allclose(out, cp.asarray([first + second], dtype=cp.float32))


@pytest.mark.codegen
def test_runtime_for_supports_annotated_carried_assignment(monkeypatch):
    out = np.zeros(1, dtype=np.float32)

    _, ssa_ops, cuda_src = compile_scalar_kernel(
        monkeypatch,
        annotated_carried_value_kernel,
        out,
        3,
    )

    assert any(isinstance(op, SSAForRange) for op in ssa_ops)
    assert "for (int" in cuda_src


@pytest.mark.codegen
def test_runtime_for_without_carried_values(monkeypatch):
    out = np.zeros(3, dtype=np.float32)

    _, ssa_ops, cuda_src = compile_scalar_kernel(
        monkeypatch,
        no_carried_values_kernel,
        out,
        3,
    )

    loop = next(op for op in ssa_ops if isinstance(op, SSAForRange))
    assert loop.carried_inputs == ()
    assert loop.results == ()
    assert "for (int" in cuda_src


@pytest.mark.codegen
def test_sequential_runtime_for_loops(monkeypatch):
    out = np.zeros(1, dtype=np.float32)

    _, _, cuda_src = compile_scalar_kernel(
        monkeypatch,
        sequential_runtime_loops_kernel,
        out,
        3,
    )

    assert cuda_src.count("for (int") == 2


@pytest.mark.codegen
def test_nested_runtime_for_loops(monkeypatch):
    out = np.zeros(1, dtype=np.float32)

    _, ssa_ops, cuda_src = compile_scalar_kernel(
        monkeypatch,
        nested_runtime_loops_kernel,
        out,
        3,
    )

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %9 = for %1 in range(0, n, 1) iter_args(%2 = 0) : i32 {
          %8 = for %3 in range(0, n, 1) iter_args(%4 = %2) : i32 {
            %5 = add %4, %0 : i32
            %6 = add %5, %1 : i32
            %7 = add %6, %3 : i32
            yield %7
          }
          yield %8
        }
        %10 = addptr out, %0 : ptr<f32>
        store %10, %9, none
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void nested_runtime_loops_kernel(float* out, int n) {
            int v0 = blockIdx.x;
            int v9 = 0;
            for (int v1 = 0; v1 < n; v1 += 1) {
                int v8 = v9;
                for (int v3 = 0; v3 < n; v3 += 1) {
                    int v5 = (v8 + v0);
                    int v6 = (v5 + v1);
                    int v7 = (v6 + v3);
                    v8 = v7;
                }
                v9 = v8;
            }
            out[v0] = v9;
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.execution
def test_nested_runtime_for_loops_execute(cp, monkeypatch):
    n = 4
    out = cp.zeros(1, dtype=cp.float32)

    compile_scalar_kernel(
        monkeypatch,
        nested_runtime_loops_kernel,
        out,
        n,
    )
    cp.cuda.Stream.null.synchronize()

    expected = n * n * (n - 1)
    cp.testing.assert_allclose(out, cp.asarray([expected], dtype=cp.float32))


def test_nested_runtime_for_cache_returns_independent_copies(monkeypatch):
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    out = np.zeros(1, dtype=np.float32)

    nested_runtime_loops_kernel.clear_cache()
    _, first_ssa, first_src = nested_runtime_loops_kernel[(1,)](out, 3)
    outer_loop = next(op for op in first_ssa if isinstance(op, SSAForRange))
    assert isinstance(outer_loop, SSAForRange)
    outer_loop.body.clear()

    _, cached_ssa, cached_src = nested_runtime_loops_kernel[(1,)](out, 3)
    cached_outer_loop = next(op for op in cached_ssa if isinstance(op, SSAForRange))
    assert isinstance(cached_outer_loop, SSAForRange)

    assert cached_outer_loop.body
    assert cached_src == first_src


def test_for_range_lowering_restores_state_after_error():
    invalid_body_op = cast(TopLevelOp, Const(1))
    loop = ForRange(
        index=LoopIndex("k"),
        start=Const(0),
        stop=Const(1),
        step=Const(1),
        captures=(),
        body=[invalid_body_op],
        carried_inputs=(),
        carried_args=(),
        carried_outputs=(),
    )
    lowering = SSALowering()

    with pytest.raises(TypeError, match="Cannot lower operation"):
        lowering.lower([loop])

    assert lowering.ops == []
    assert lowering.memo == {}
    assert lowering.next_id == 0
    assert lowering.type_inference.types == {}

    assert lowering.lower([Store(Param("out", PTR_F32), Const(1.0), None)])


@pytest.mark.parametrize("step", [0, -1])
def test_runtime_for_rejects_non_positive_step(monkeypatch, step):
    out = np.zeros(1, dtype=np.float32)

    with pytest.raises(ASTFrontendError, match="only positive step"):
        compile_scalar_kernel(
            monkeypatch,
            invalid_runtime_step_kernel,
            out,
            3,
            STEP=step,
        )


def test_runtime_for_rejects_assignment_to_induction_variable(monkeypatch):
    out = np.zeros(1, dtype=np.float32)

    with pytest.raises(
        ASTFrontendError,
        match="assignment to runtime loop induction variable is not supported",
    ):
        compile_scalar_kernel(
            monkeypatch,
            assigned_induction_variable_kernel,
            out,
            3,
        )


def test_mlir_backend_rejects_runtime_for(monkeypatch):
    out = np.zeros(1, dtype=np.float32)

    with pytest.raises(TypeError, match="MLIR MVP does not support runtime for loops"):
        compile_scalar_kernel(
            monkeypatch,
            two_carried_values_kernel,
            out,
            3,
            backend="mlir",
        )


@pytest.mark.codegen
def test_ast_frontend_lowers_runtime_range_to_cuda_for_loop(monkeypatch):
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 5, 7, 3
    BM, BN = 4, 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    matmul_runtime_k_kernel.clear_cache()
    _, ssa_ops, cuda_src = matmul_runtime_k_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K, BM=BM, BN=BN)

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 4 : i32
        %2 = arange {start=0, end=4} : vector<4 x i32>
        %3 = expand_dims %2 {axis=1} : block<4x1 x i32>
        %4 = add %1, %3 : block<4x1 x i32>
        %5 = program_id {axis=1} : i32
        %6 = mul %5, 8 : i32
        %7 = arange {start=0, end=8} : vector<8 x i32>
        %8 = expand_dims %7 {axis=0} : block<1x8 x i32>
        %9 = add %6, %8 : block<1x8 x i32>
        %10 = zeros {shape=(4, 8), dtype=f32} : block<4x8 x f32>
        %25 = for %11 in range(0, K, 1) iter_args(%12 = %10) : block<4x8 x f32> {
          %13 = mul %4, K : block<4x1 x i32>
          %14 = add %13, %11 : block<4x1 x i32>
          %15 = addptr a, %14 : block<4x1 x ptr<f32>>
          %16 = cmp_lt %4, M : block<4x1 x bool>
          %17 = load %15, %16, 0.0 : block<4x1 x f32>
          %18 = mul %11, N : i32
          %19 = add %18, %9 : block<1x8 x i32>
          %20 = addptr b, %19 : block<1x8 x ptr<f32>>
          %21 = cmp_lt %9, N : block<1x8 x bool>
          %22 = load %20, %21, 0.0 : block<1x8 x f32>
          %23 = mul %17, %22 : block<4x8 x f32>
          %24 = add %12, %23 : block<4x8 x f32>
          yield %24
        }
        %26 = mul %4, N : block<4x1 x i32>
        %27 = add %26, %9 : block<4x8 x i32>
        %28 = addptr c, %27 : block<4x8 x ptr<f32>>
        %29 = cmp_lt %4, M : block<4x1 x bool>
        %30 = cmp_lt %9, N : block<1x8 x bool>
        %31 = and %29, %30 : block<4x8 x bool>
        store %28, %25, %31
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void matmul_runtime_k_kernel(float* a, float* b, float* c, int M, int N, int K) {
            int tile_i = threadIdx.x / 8;
            int tile_j = threadIdx.x % 8;
            int v0 = blockIdx.x;
            int v1 = (v0 * 4);
            int v3 = tile_i;
            int v4 = (v1 + v3);
            int v5 = blockIdx.y;
            int v6 = (v5 * 8);
            int v8 = tile_j;
            int v9 = (v6 + v8);
            float v10 = 0.0f;
            float v25 = v10;
            for (int v11 = 0; v11 < K; v11 += 1) {
                int v13 = (v4 * K);
                int v14 = (v13 + v11);
                bool v16 = (v4 < M);
                float v17 = (v16 ? a[v14] : 0.0f);
                int v18 = (v11 * N);
                int v19 = (v18 + v9);
                bool v21 = (v9 < N);
                float v22 = (v21 ? b[v19] : 0.0f);
                float v23 = (v17 * v22);
                float v24 = (v25 + v23);
                v25 = v24;
            }
            int v26 = (v4 * N);
            int v27 = (v26 + v9);
            bool v29 = (v4 < M);
            bool v30 = (v9 < N);
            bool v31 = (v29 && v30);
            if (v31) {
                c[v27] = v25;
            }
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.execution
def test_ast_frontend_runtime_range_matmul_executes_with_cupy_when_cuda_is_available(
    monkeypatch,
):
    cp = pytest.importorskip("cupy")

    if not triton.cuda_available():
        pytest.skip("CUDA GPU is not available")

    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")

    M, N, K = 19, 23, 5
    BM, BN = 8, 16

    a = cp.arange(M * K, dtype=cp.float32).reshape(M, K)
    b = cp.arange(K * N, dtype=cp.float32).reshape(K, N)
    c = cp.zeros((M, N), dtype=cp.float32)

    matmul_runtime_k_kernel.clear_cache()
    matmul_runtime_k_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K, BM=BM, BN=BN)

    cp.cuda.Stream.null.synchronize()

    expected = a @ b
    cp.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)
