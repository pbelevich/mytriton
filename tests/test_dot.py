from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ast_frontend import trace
from mytriton.block_shapes import cuda_threads_per_block
from mytriton.optim import CSEPass, DCEPass
from mytriton.ssa import SSAItem, SSALowering, SSAOp, SSAPrinter, SSAValue
from mytriton.ssa_verification import CompileError, SSAVerifier
from mytriton.trace import F32, I32, BlockType, Dot, Zeros
from mytriton.type_inference import TypeInference


@triton.jit
def dot_semantics_kernel(
    out,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    lhs = tl.zeros((BM, BK), tl.float32)
    rhs = tl.zeros((BK, BN), tl.float32)
    result = tl.dot(lhs, rhs)

    offsets_m = tl.arange(0, BM)[:, None]
    offsets_n = tl.arange(0, BN)[None, :]
    offsets = offsets_m * BN + offsets_n

    tl.store(out + offsets, result)


def infer_type(value) -> BlockType:
    ty = TypeInference().infer(value.expr)
    assert isinstance(ty, BlockType)
    return ty


def test_dot_builds_expression_tree_node() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16, 8), tl.float32)

    result = tl.dot(lhs, rhs)

    assert isinstance(result.expr, Dot)
    assert result.expr.lhs is lhs.expr
    assert result.expr.rhs is rhs.expr
    assert isinstance(result.expr.lhs, Zeros)
    assert isinstance(result.expr.rhs, Zeros)


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "expected_shape"),
    [
        ((4, 16), (16, 8), (4, 8)),
        ((1, 16), (16, 8), (1, 8)),
        ((4, 16), (16, 1), (4, 1)),
        ((1, 1), (1, 1), (1, 1)),
    ],
)
def test_dot_type_inference(
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int],
    expected_shape: tuple[int, int],
) -> None:
    lhs = tl.zeros(lhs_shape, tl.float32)
    rhs = tl.zeros(rhs_shape, tl.float32)

    result = tl.dot(lhs, rhs)

    assert infer_type(result) == BlockType(expected_shape, F32)


def test_dot_rejects_non_rank2_lhs() -> None:
    lhs = tl.zeros((16,), tl.float32)
    rhs = tl.zeros((16, 8), tl.float32)

    with pytest.raises(
        TypeError,
        match="dot lhs must be a rank-2 block",
    ):
        TypeInference().infer(tl.dot(lhs, rhs).expr)


def test_dot_rejects_non_rank2_rhs() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16,), tl.float32)

    with pytest.raises(
        TypeError,
        match="dot rhs must be a rank-2 block",
    ):
        TypeInference().infer(tl.dot(lhs, rhs).expr)


def test_dot_rejects_mismatched_inner_dimensions() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((8, 4), tl.float32)

    with pytest.raises(
        TypeError,
        match="dot inner dimensions must match",
    ):
        TypeInference().infer(tl.dot(lhs, rhs).expr)


def test_dot_rejects_non_f32_lhs() -> None:
    lhs = tl.zeros((4, 16), tl.int32)
    rhs = tl.zeros((16, 8), tl.float32)

    with pytest.raises(
        TypeError,
        match="dot lhs must have f32 elements",
    ):
        TypeInference().infer(tl.dot(lhs, rhs).expr)


def test_dot_rejects_non_f32_rhs() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16, 8), tl.int32)

    with pytest.raises(
        TypeError,
        match="dot rhs must have f32 elements",
    ):
        TypeInference().infer(tl.dot(lhs, rhs).expr)


def test_dot_lowers_to_ssa() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16, 8), tl.float32)
    result = tl.dot(lhs, rhs)

    lowering = SSALowering()
    ssa_result = lowering.lower_expr(result.expr)

    assert str(ssa_result) == "%2"
    assert ssa_result.ty == BlockType((4, 8), F32)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(4, 16), dtype=f32} : block<4x16 x f32>
        %1 = zeros {shape=(16, 8), dtype=f32} : block<16x8 x f32>
        %2 = dot %0, %1 : block<4x8 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(lowering.ops) == expected_ssa


def test_dot_ssa_reuses_shared_operand() -> None:
    lhs = tl.zeros((4, 4), tl.float32)
    result = tl.dot(lhs, lhs)

    lowering = SSALowering()
    lowering.lower_expr(result.expr)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(4, 4), dtype=f32} : block<4x4 x f32>
        %1 = dot %0, %0 : block<4x4 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(lowering.ops) == expected_ssa


def test_dot_ssa_passes_verification() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16, 8), tl.float32)
    result = tl.dot(lhs, rhs)

    lowering = SSALowering()
    lowering.lower_expr(result.expr)

    verified = SSAVerifier(block_size=32).verify(lowering.ops)

    assert verified == lowering.ops


def make_dot_ssa(
    lhs_ty: BlockType,
    rhs_ty: BlockType,
    result_ty: BlockType,
) -> list[SSAItem]:
    lhs = SSAValue(id=0, ty=lhs_ty)
    rhs = SSAValue(id=1, ty=rhs_ty)
    result = SSAValue(id=2, ty=result_ty)

    return [
        SSAOp(
            opcode="zeros",
            result=lhs,
            attrs={
                "shape": lhs_ty.shape,
                "dtype": lhs_ty.element,
            },
        ),
        SSAOp(
            opcode="zeros",
            result=rhs,
            attrs={
                "shape": rhs_ty.shape,
                "dtype": rhs_ty.element,
            },
        ),
        SSAOp(
            opcode="dot",
            operands=(lhs, rhs),
            result=result,
        ),
    ]


def test_dot_verifier_rejects_wrong_result_shape() -> None:
    ops = make_dot_ssa(
        lhs_ty=BlockType((4, 16), F32),
        rhs_ty=BlockType((16, 8), F32),
        result_ty=BlockType((4, 7), F32),
    )

    with pytest.raises(
        CompileError,
        match=r"expected block<4x8 x f32>, got block<4x7 x f32>",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_dot_verifier_rejects_mismatched_inner_dimensions() -> None:
    ops = make_dot_ssa(
        lhs_ty=BlockType((4, 16), F32),
        rhs_ty=BlockType((8, 4), F32),
        result_ty=BlockType((4, 4), F32),
    )

    with pytest.raises(
        CompileError,
        match="dot inner dimensions must match",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_dot_verifier_rejects_non_rank2_lhs() -> None:
    ops = make_dot_ssa(
        lhs_ty=BlockType((16,), F32),
        rhs_ty=BlockType((16, 8), F32),
        result_ty=BlockType((4, 8), F32),
    )

    with pytest.raises(
        CompileError,
        match="dot lhs must be a rank-2 block",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_dot_verifier_rejects_non_rank2_rhs() -> None:
    ops = make_dot_ssa(
        lhs_ty=BlockType((4, 16), F32),
        rhs_ty=BlockType((16,), F32),
        result_ty=BlockType((4, 8), F32),
    )

    with pytest.raises(
        CompileError,
        match="dot rhs must be a rank-2 block",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_dot_verifier_rejects_non_f32_lhs() -> None:
    ops = make_dot_ssa(
        lhs_ty=BlockType((4, 16), I32),
        rhs_ty=BlockType((16, 8), F32),
        result_ty=BlockType((4, 8), F32),
    )

    with pytest.raises(
        CompileError,
        match="dot lhs must have f32 elements",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_dot_verifier_rejects_non_f32_rhs() -> None:
    ops = make_dot_ssa(
        lhs_ty=BlockType((4, 16), F32),
        rhs_ty=BlockType((16, 8), I32),
        result_ty=BlockType((4, 8), F32),
    )

    with pytest.raises(
        CompileError,
        match="dot rhs must have f32 elements",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_ast_frontend_lowers_dot_to_verified_ssa() -> None:
    out = np.empty(4 * 8, dtype=np.float32)

    bound = dot_semantics_kernel.signature.bind(
        out,
        BM=4,
        BK=16,
        BN=8,
    )

    ops, _ = trace(
        dot_semantics_kernel.fn,
        dot_semantics_kernel.signature,
        bound.arguments,
    )

    ssa_ops = SSALowering().lower(ops)
    threads_per_block = cuda_threads_per_block(ssa_ops)

    assert threads_per_block == 32
    SSAVerifier(threads_per_block).verify(ssa_ops)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(4, 16), dtype=f32} : block<4x16 x f32>
        %1 = zeros {shape=(16, 8), dtype=f32} : block<16x8 x f32>
        %2 = dot %0, %1 : block<4x8 x f32>
        %3 = arange {start=0, end=4} : vector<4 x i32>
        %4 = expand_dims %3 {axis=1} : block<4x1 x i32>
        %5 = mul %4, 8 : block<4x1 x i32>
        %6 = arange {start=0, end=8} : vector<8 x i32>
        %7 = expand_dims %6 {axis=0} : block<1x8 x i32>
        %8 = add %5, %7 : block<4x8 x i32>
        %9 = addptr out, %8 : block<4x8 x ptr<f32>>
        store %9, %2, none
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa


def test_cuda_backend_rejects_dot_before_lowering(monkeypatch) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    out = np.empty(4 * 8, dtype=np.float32)

    dot_semantics_kernel.clear_cache()

    with pytest.raises(
        TypeError,
        match=r"CUDA lowering for tl\.dot is not implemented",
    ):
        dot_semantics_kernel[(1,)](
            out,
            BM=4,
            BK=16,
            BN=8,
        )


def test_cse_reuses_duplicate_dot() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16, 8), tl.float32)

    first = tl.dot(lhs, rhs)
    second = tl.dot(lhs, rhs)
    result = first + second

    lowering = SSALowering()
    lowering.lower_expr(result.expr)

    optimized = CSEPass().run(lowering.ops)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(4, 16), dtype=f32} : block<4x16 x f32>
        %1 = zeros {shape=(16, 8), dtype=f32} : block<16x8 x f32>
        %2 = dot %0, %1 : block<4x8 x f32>
        %4 = add %2, %2 : block<4x8 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(optimized) == expected_ssa
    SSAVerifier(block_size=32).verify(optimized)


def test_dce_removes_unused_dot() -> None:
    lhs = tl.zeros((4, 16), tl.float32)
    rhs = tl.zeros((16, 8), tl.float32)
    result = tl.dot(lhs, rhs)

    lowering = SSALowering()
    lowering.lower_expr(result.expr)

    optimized = DCEPass().run(lowering.ops)

    assert optimized == []
