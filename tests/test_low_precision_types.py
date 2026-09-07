import inspect
from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.block_shapes import CudaKernelLayout
from mytriton.cuda_codegen import (
    CudaRegisterTileRef,
    SSACUDACodegen,
)
from mytriton.cuda_dot_staging import (
    CudaDotSharedBuffers,
    CudaSharedBuffer,
    cuda_scalar_nbytes,
)
from mytriton.optim import CSEPass, DCEPass
from mytriton.ssa import (
    SSAItem,
    SSALowering,
    SSAOp,
    SSAPrinter,
    SSAValue,
)
from mytriton.ssa_verification import CompileError, SSAVerifier
from mytriton.trace import (
    BF16,
    BOOL,
    F16,
    F32,
    I32,
    PTR_BF16,
    PTR_F16,
    PTR_F32,
    BlockType,
    Cast,
    Const,
    Param,
    PointerType,
    ScalarType,
    Type,
    make_runtime_params,
)
from mytriton.type_inference import TypeInference


def runtime_array_kernel(value: object) -> None:
    del value


def runtime_array_param(value: object) -> Param:
    signature = inspect.signature(runtime_array_kernel)
    bound = signature.bind(value)

    params = make_runtime_params(
        signature,
        bound.arguments,
    )

    assert len(params) == 1
    return params[0]


@triton.jit
def cast_f16_to_f32_kernel(
    source,
    destination,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(source + offsets)
    converted = tl.cast(values, tl.float32)
    tl.store(destination + offsets, converted)


@triton.jit
def low_precision_dot_kernel(
    a,
    b,
    out,
    M,
    N,
    K,
    k_base,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    offsets_n = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
    offsets_k = tl.arange(0, BK)

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

    result = tl.dot(a_values, b_values)

    output_pointers = out + offsets_m * N + offsets_n
    output_mask = (offsets_m < M) & (offsets_n < N)
    tl.store(output_pointers, result, mask=output_mask)


def test_low_precision_dtypes_are_public() -> None:
    assert tl.float16 is F16
    assert tl.bfloat16 is BF16
    assert str(tl.float16) == "f16"
    assert str(tl.bfloat16) == "bf16"


def test_block_constructors_accept_low_precision_dtypes() -> None:
    f16_value = tl.zeros((4, 8), tl.float16)
    bf16_value = tl.zeros((4, 8), tl.bfloat16)

    assert TypeInference().infer(f16_value.expr) == BlockType((4, 8), F16)
    assert TypeInference().infer(bf16_value.expr) == BlockType((4, 8), BF16)


@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype", "expected_dtype"),
    [
        (F16, F16, F16),
        (BF16, BF16, BF16),
        (F16, F32, F32),
        (F32, F16, F32),
        (BF16, F32, F32),
        (F32, BF16, F32),
        (F16, BF16, F32),
        (BF16, F16, F32),
        (I32, F16, F16),
        (F16, I32, F16),
        (I32, BF16, BF16),
        (BF16, I32, BF16),
    ],
)
def test_low_precision_numeric_promotion(
    lhs_dtype: ScalarType,
    rhs_dtype: ScalarType,
    expected_dtype: ScalarType,
) -> None:
    lhs = tl.zeros((8,), lhs_dtype)
    rhs = tl.zeros((8,), rhs_dtype)

    result = lhs + rhs

    assert TypeInference().infer(result.expr) == BlockType(
        (8,),
        expected_dtype,
    )


def test_low_precision_arithmetic_lowers_to_verified_ssa() -> None:
    lhs = tl.zeros((8,), tl.float16)
    rhs = tl.zeros((8,), tl.bfloat16)
    result = lhs + rhs

    lowering = SSALowering()
    ssa_result = lowering.lower_expr(result.expr)

    assert ssa_result.ty == BlockType((8,), F32)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(8,), dtype=f16} : vector<8 x f16>
        %1 = zeros {shape=(8,), dtype=bf16} : vector<8 x bf16>
        %2 = add %0, %1 : vector<8 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(lowering.ops) == expected_ssa
    assert SSAVerifier(block_size=8).verify(lowering.ops) == lowering.ops


def test_cast_builds_expression_tree_node() -> None:
    source = tl.zeros((8,), tl.float16)

    result = tl.cast(source, tl.float32)

    assert isinstance(result.expr, Cast)
    assert result.expr.value is source.expr
    assert result.expr.dtype is F32


@pytest.mark.parametrize(
    ("source_dtype", "destination_dtype"),
    [
        (F16, F32),
        (BF16, F32),
        (F32, F16),
        (F32, BF16),
        (F16, BF16),
        (BF16, F16),
        (I32, F16),
        (F16, I32),
    ],
)
def test_cast_preserves_shape_and_changes_element_type(
    source_dtype: ScalarType,
    destination_dtype: ScalarType,
) -> None:
    source = tl.zeros((4, 8), source_dtype)

    result = tl.cast(source, destination_dtype)

    assert TypeInference().infer(result.expr) == BlockType(
        (4, 8),
        destination_dtype,
    )


def test_cast_rejects_boolean_destination() -> None:
    source = tl.zeros((8,), tl.float32)

    with pytest.raises(
        TypeError,
        match="cast dtype must be numeric",
    ):
        tl.cast(source, tl.int1)


def test_cast_lowers_to_verified_ssa() -> None:
    source = tl.zeros((8,), tl.float16)
    result = tl.cast(source, tl.float32)

    lowering = SSALowering()
    ssa_result = lowering.lower_expr(result.expr)

    assert ssa_result.ty == BlockType((8,), F32)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(8,), dtype=f16} : vector<8 x f16>
        %1 = cast %0 {dtype=f32} : vector<8 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(lowering.ops) == expected_ssa
    assert SSAVerifier(block_size=8).verify(lowering.ops) == lowering.ops


def test_cast_verifier_rejects_wrong_result_type() -> None:
    source = SSAValue(0, BlockType((8,), F16))
    wrong_result = SSAValue(1, BlockType((8,), BF16))

    ops: list[SSAItem] = [
        SSAOp(
            opcode="zeros",
            result=source,
            attrs={
                "shape": (8,),
                "dtype": F16,
            },
        ),
        SSAOp(
            opcode="cast",
            operands=(source,),
            result=wrong_result,
            attrs={"dtype": F32},
        ),
    ]

    with pytest.raises(
        CompileError,
        match=r"expected vector<8 x f32>, got vector<8 x bf16>",
    ):
        SSAVerifier(block_size=8).verify(ops)


def test_cast_verifier_rejects_boolean_source() -> None:
    source = SSAValue(0, BlockType((8,), BOOL))
    result = SSAValue(1, BlockType((8,), F16))

    ops: list[SSAItem] = [
        SSAOp(
            opcode="zeros",
            result=source,
            attrs={
                "shape": (8,),
                "dtype": BOOL,
            },
        ),
        SSAOp(
            opcode="cast",
            operands=(source,),
            result=result,
            attrs={"dtype": F16},
        ),
    ]

    with pytest.raises(
        CompileError,
        match=r"cannot cast vector<8 x bool> to f16",
    ):
        SSAVerifier(block_size=8).verify(ops)


def test_cast_verifier_rejects_boolean_destination() -> None:
    source = SSAValue(0, BlockType((8,), F32))
    result = SSAValue(1, BlockType((8,), BOOL))

    ops: list[SSAItem] = [
        SSAOp(
            opcode="zeros",
            result=source,
            attrs={
                "shape": (8,),
                "dtype": F32,
            },
        ),
        SSAOp(
            opcode="cast",
            operands=(source,),
            result=result,
            attrs={"dtype": BOOL},
        ),
    ]

    with pytest.raises(
        CompileError,
        match="invalid cast dtype bool",
    ):
        SSAVerifier(block_size=8).verify(ops)


def test_cse_reuses_duplicate_cast() -> None:
    source = tl.zeros((8,), tl.float16)

    first = tl.cast(source, tl.float32)
    second = tl.cast(source, tl.float32)
    result = first + second

    lowering = SSALowering()
    lowering.lower_expr(result.expr)

    optimized = CSEPass().run(lowering.ops)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(8,), dtype=f16} : vector<8 x f16>
        %1 = cast %0 {dtype=f32} : vector<8 x f32>
        %3 = add %1, %1 : vector<8 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(optimized) == expected_ssa
    assert SSAVerifier(block_size=8).verify(optimized) == optimized


def test_dce_removes_unused_cast() -> None:
    source = tl.zeros((8,), tl.float16)
    result = tl.cast(source, tl.float32)

    lowering = SSALowering()
    lowering.lower_expr(result.expr)

    optimized = DCEPass().run(lowering.ops)

    assert optimized == []


@pytest.mark.parametrize("dtype", [F16, BF16])
def test_low_precision_dot_accumulates_in_f32(
    dtype: ScalarType,
) -> None:
    lhs = tl.zeros((4, 16), dtype)
    rhs = tl.zeros((16, 8), dtype)
    result = tl.dot(lhs, rhs)

    assert TypeInference().infer(result.expr) == BlockType((4, 8), F32)

    lowering = SSALowering()
    ssa_result = lowering.lower_expr(result.expr)

    assert ssa_result.ty == BlockType((4, 8), F32)

    expected_ssa = dedent(
        f"""\
        %0 = zeros {{shape=(4, 16), dtype={dtype}}} : block<4x16 x {dtype}>
        %1 = zeros {{shape=(16, 8), dtype={dtype}}} : block<16x8 x {dtype}>
        %2 = dot %0, %1 : block<4x8 x f32>
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(lowering.ops) == expected_ssa
    assert SSAVerifier(block_size=32).verify(lowering.ops) == lowering.ops


def test_dot_rejects_mixed_low_precision_operand_types() -> None:
    lhs = tl.zeros((4, 16), tl.float16)
    rhs = tl.zeros((16, 8), tl.bfloat16)

    with pytest.raises(
        TypeError,
        match="dot operand element types must match",
    ):
        TypeInference().infer(tl.dot(lhs, rhs).expr)


def test_dot_verifier_rejects_mixed_low_precision_operands() -> None:
    lhs = SSAValue(0, BlockType((4, 16), F16))
    rhs = SSAValue(1, BlockType((16, 8), BF16))
    result = SSAValue(2, BlockType((4, 8), F32))

    ops: list[SSAItem] = [
        SSAOp(
            opcode="zeros",
            result=lhs,
            attrs={
                "shape": (4, 16),
                "dtype": F16,
            },
        ),
        SSAOp(
            opcode="zeros",
            result=rhs,
            attrs={
                "shape": (16, 8),
                "dtype": BF16,
            },
        ),
        SSAOp(
            opcode="dot",
            operands=(lhs, rhs),
            result=result,
        ),
    ]

    with pytest.raises(
        CompileError,
        match="dot operand element types must match",
    ):
        SSAVerifier(block_size=32).verify(ops)


def test_numpy_float16_array_becomes_f16_pointer() -> None:
    value = np.zeros(8, dtype=np.float16)

    assert runtime_array_param(value) == Param("value", PTR_F16)


@pytest.mark.parametrize(
    ("torch_dtype_name", "expected_pointer_type"),
    [
        ("float16", PTR_F16),
        ("bfloat16", PTR_BF16),
    ],
)
def test_torch_low_precision_tensor_becomes_typed_pointer(
    torch_dtype_name: str,
    expected_pointer_type: PointerType,
) -> None:
    torch = pytest.importorskip("torch")
    value = torch.zeros(
        8,
        dtype=getattr(torch, torch_dtype_name),
    )

    assert runtime_array_param(value) == Param(
        "value",
        expected_pointer_type,
    )


def test_runtime_array_rejects_unsupported_dtype() -> None:
    value = np.zeros(8, dtype=np.int32)

    with pytest.raises(
        TypeError,
        match="only float16, bfloat16, or float32 arrays are supported",
    ):
        runtime_array_param(value)


@pytest.mark.parametrize(
    ("ty", "expected_cuda_type"),
    [
        (F16, "__half"),
        (BF16, "__nv_bfloat16"),
        (PTR_F16, "__half*"),
        (PTR_BF16, "__nv_bfloat16*"),
        (BlockType((8,), F16), "__half"),
        (BlockType((8,), BF16), "__nv_bfloat16"),
    ],
)
def test_cuda_codegen_names_low_precision_types(
    ty: Type,
    expected_cuda_type: str,
) -> None:
    codegen = SSACUDACodegen()

    assert codegen.cuda_type(ty) == expected_cuda_type


@pytest.mark.parametrize(
    ("pointer_ty", "expected_header", "unexpected_header"),
    [
        (
            PTR_F16,
            "#include <cuda_fp16.h>",
            "#include <cuda_bf16.h>",
        ),
        (
            PTR_BF16,
            "#include <cuda_bf16.h>",
            "#include <cuda_fp16.h>",
        ),
    ],
)
def test_cuda_codegen_emits_required_low_precision_header(
    pointer_ty: PointerType,
    expected_header: str,
    unexpected_header: str,
) -> None:
    codegen = SSACUDACodegen()

    cuda_src = codegen.generate(
        kernel_name="typed_kernel",
        ssa_ops=[],
        params=[Param("value", pointer_ty)],
    )

    assert cuda_src.startswith(f"{expected_header}\n\n")
    assert unexpected_header not in cuda_src


def test_cuda_codegen_does_not_add_headers_to_f32_kernel() -> None:
    codegen = SSACUDACodegen()

    cuda_src = codegen.generate(
        kernel_name="f32_kernel",
        ssa_ops=[],
        params=[Param("value", PTR_F32)],
    )

    assert "#include <cuda_fp16.h>" not in cuda_src
    assert "#include <cuda_bf16.h>" not in cuda_src
    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void f32_kernel(float* value) {
        }
        """
    ).rstrip("\n")


def test_cuda_codegen_resets_required_headers_between_kernels() -> None:
    codegen = SSACUDACodegen()

    codegen.generate(
        kernel_name="f16_kernel",
        ssa_ops=[],
        params=[Param("value", PTR_F16)],
    )
    f32_cuda_src = codegen.generate(
        kernel_name="f32_kernel",
        ssa_ops=[],
        params=[Param("value", PTR_F32)],
    )

    assert "#include" not in f32_cuda_src


@pytest.mark.parametrize(
    ("source_ty", "destination_ty", "expected_line"),
    [
        (
            F16,
            F32,
            "    float v0 = __half2float(value);",
        ),
        (
            BF16,
            F32,
            "    float v0 = __bfloat162float(value);",
        ),
        (
            F32,
            F16,
            "    __half v0 = __float2half_rn(value);",
        ),
        (
            F32,
            BF16,
            "    __nv_bfloat16 v0 = __float2bfloat16_rn(value);",
        ),
        (
            F16,
            BF16,
            "    __nv_bfloat16 v0 = __float2bfloat16_rn(__half2float(value));",
        ),
        (
            BF16,
            F16,
            "    __half v0 = __float2half_rn(__bfloat162float(value));",
        ),
        (
            I32,
            F32,
            "    float v0 = static_cast<float>(value);",
        ),
        (
            F32,
            I32,
            "    int v0 = static_cast<int>(value);",
        ),
        (
            I32,
            F16,
            "    __half v0 = __float2half_rn(static_cast<float>(value));",
        ),
        (
            F16,
            I32,
            "    int v0 = static_cast<int>(__half2float(value));",
        ),
        (
            I32,
            BF16,
            "    __nv_bfloat16 v0 = __float2bfloat16_rn(static_cast<float>(value));",
        ),
        (
            BF16,
            I32,
            "    int v0 = static_cast<int>(__bfloat162float(value));",
        ),
    ],
)
def test_cuda_codegen_lowers_numeric_cast(
    source_ty: ScalarType,
    destination_ty: ScalarType,
    expected_line: str,
) -> None:
    source = Param("value", source_ty)
    result = SSAValue(0, destination_ty)
    op = SSAOp(
        opcode="cast",
        operands=(source,),
        result=result,
        attrs={"dtype": destination_ty},
    )

    cuda_src = SSACUDACodegen().generate(
        kernel_name="cast_kernel",
        ssa_ops=[op],
        params=[source],
    )

    assert expected_line in cuda_src


@pytest.mark.codegen
def test_f16_cast_kernel_lowers_through_full_cuda_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    source = np.zeros(8, dtype=np.float16)
    destination = np.zeros(8, dtype=np.float32)

    cast_f16_to_f32_kernel.clear_cache()
    _, ssa_ops, cuda_src = cast_f16_to_f32_kernel[(1,)](
        source,
        destination,
        BLOCK_SIZE=8,
    )

    expected_ssa = dedent(
        """\
        %0 = arange {start=0, end=8} : vector<8 x i32>
        %1 = addptr source, %0 : vector<8 x ptr<f16>>
        %2 = load %1, none, none : vector<8 x f16>
        %3 = cast %2 {dtype=f32} : vector<8 x f32>
        %4 = addptr destination, %0 : vector<8 x ptr<f32>>
        store %4, %3, none
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        #include <cuda_fp16.h>

        extern "C" __global__
        void cast_f16_to_f32_kernel(__half* source, float* destination) {
            int v0 = threadIdx.x;
            __half v2 = (true ? source[v0] : __float2half_rn(0.0f));
            float v3 = __half2float(v2);
            destination[v0] = v3;
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.execution
def test_f16_cast_kernel_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    source = cp.arange(8, dtype=cp.float16)
    destination = cp.empty(8, dtype=cp.float32)

    cast_f16_to_f32_kernel.clear_cache()
    cast_f16_to_f32_kernel[(1,)](
        source,
        destination,
        BLOCK_SIZE=8,
    )
    cp.cuda.Stream.null.synchronize()

    cp.testing.assert_array_equal(
        destination,
        source.astype(cp.float32),
    )


def test_cuda_store_explicitly_converts_to_low_precision_pointer() -> None:
    destination = Param("destination", PTR_F16)
    value = Param("value", F32)

    cuda_src = SSACUDACodegen().generate(
        kernel_name="store_f16_kernel",
        ssa_ops=[
            SSAOp(
                opcode="store",
                operands=(destination, value, None),
            )
        ],
        params=[destination, value],
    )

    assert "    destination[0] = __float2half_rn(value);" in cuda_src


@pytest.mark.parametrize("dtype", [F16, BF16])
def test_low_precision_shared_buffer_uses_two_byte_elements(
    dtype: ScalarType,
) -> None:
    assert cuda_scalar_nbytes(dtype) == 2

    buffer = CudaSharedBuffer(
        name="tile",
        logical_shape=(4, 8),
        element_ty=dtype,
        stage_count=2,
    )

    assert buffer.stage_size == 32
    assert buffer.size == 64
    assert buffer.nbytes == 128


@pytest.mark.parametrize(
    ("operand_dtype", "to_float"),
    [
        (F16, "__half2float"),
        (BF16, "__bfloat162float"),
    ],
)
def test_cuda_core_dot_converts_low_precision_operands_to_f32(
    operand_dtype: ScalarType,
    to_float: str,
) -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    result = SSAValue(
        id=7,
        ty=BlockType((4, 8), F32),
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(4, 16),
            element_ty=operand_dtype,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 8),
            element_ty=operand_dtype,
        ),
    )

    codegen.emit_dot_from_shared_memory(result, buffers)

    expected_fma = (
        f"        v7 += "
        f"{to_float}(dot_lhs_7[(tile_i) * 16 + (dot_k_7)]) * "
        f"{to_float}(dot_rhs_7[(dot_k_7) * 8 + (tile_j)]);"
    )

    assert codegen.lines == [
        "    float v7 = 0.0f;",
        "    #pragma unroll",
        "    for (int dot_k_7 = 0; dot_k_7 < 16; ++dot_k_7) {",
        expected_fma,
        "    }",
        "    __syncthreads();",
    ]
    assert codegen.values[result.id] == "v7"


@pytest.mark.codegen
def test_f16_dot_lowers_through_shared_memory_to_f32_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 4, 8, 16
    BM, BK, BN = 4, 16, 8

    a = np.zeros((M, K), dtype=np.float16)
    b = np.zeros((K, N), dtype=np.float16)
    out = np.zeros((M, N), dtype=np.float32)

    low_precision_dot_kernel.clear_cache()
    _, ssa_ops, cuda_src = low_precision_dot_kernel[(1, 1)](
        a,
        b,
        out,
        M,
        N,
        K,
        0,
        BM=BM,
        BK=BK,
        BN=BN,
    )

    ssa_src = SSAPrinter().print_ops(ssa_ops)

    assert "%14 = load %10, %13, 0.0 : block<4x16 x f16>" in ssa_src
    assert "%28 = load %24, %27, 0.0 : block<16x8 x f16>" in ssa_src
    assert "%29 = dot %14, %28 : block<4x8 x f32>" in ssa_src

    assert cuda_src.startswith("#include <cuda_fp16.h>\n\n")
    assert (
        "void low_precision_dot_kernel("
        "__half* a, __half* b, float* out, "
        "int M, int N, int K, int k_base)"
    ) in cuda_src

    assert "__shared__ __half dot_lhs_29[" in cuda_src
    assert "__shared__ __half dot_rhs_29[" in cuda_src
    assert "float v29 = 0.0f;" in cuda_src

    assert (
        "v29 += "
        "__half2float(dot_lhs_29[(tile_i) * 16 + (dot_k_29)]) * "
        "__half2float(dot_rhs_29[(dot_k_29) * 8 + (tile_j)]);"
    ) in cuda_src


@pytest.mark.execution
def test_f16_dot_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 4, 8, 16
    BM, BK, BN = 4, 16, 8

    a_host = (
        np.linspace(
            -1.0,
            1.0,
            M * K,
            dtype=np.float32,
        )
        .reshape(M, K)
        .astype(np.float16)
    )

    b_host = (
        np.linspace(
            0.75,
            -0.5,
            K * N,
            dtype=np.float32,
        )
        .reshape(K, N)
        .astype(np.float16)
    )

    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    out = cp.zeros((M, N), dtype=cp.float32)

    low_precision_dot_kernel.clear_cache()
    low_precision_dot_kernel[(1, 1)](
        a,
        b,
        out,
        M,
        N,
        K,
        0,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    cp.cuda.Stream.null.synchronize()

    expected = a_host.astype(np.float32) @ b_host.astype(np.float32)

    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.codegen
def test_bf16_dot_lowers_through_shared_memory_to_f32_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 4, 8, 16
    BM, BK, BN = 4, 16, 8

    a = torch.zeros(
        (M, K),
        dtype=torch.bfloat16,
    )
    b = torch.zeros(
        (K, N),
        dtype=torch.bfloat16,
    )
    out = torch.zeros(
        (M, N),
        dtype=torch.float32,
    )

    low_precision_dot_kernel.clear_cache()
    _, ssa_ops, cuda_src = low_precision_dot_kernel[(1, 1)](
        a,
        b,
        out,
        M,
        N,
        K,
        0,
        BM=BM,
        BK=BK,
        BN=BN,
    )

    ssa_src = SSAPrinter().print_ops(ssa_ops)

    assert "%14 = load %10, %13, 0.0 : block<4x16 x bf16>" in ssa_src
    assert "%28 = load %24, %27, 0.0 : block<16x8 x bf16>" in ssa_src
    assert "%29 = dot %14, %28 : block<4x8 x f32>" in ssa_src

    assert cuda_src.startswith("#include <cuda_bf16.h>\n\n")
    assert "#include <cuda_fp16.h>" not in cuda_src

    assert (
        "void low_precision_dot_kernel("
        "__nv_bfloat16* a, __nv_bfloat16* b, float* out, "
        "int M, int N, int K, int k_base)"
    ) in cuda_src

    assert "__shared__ __nv_bfloat16 dot_lhs_29[" in cuda_src
    assert "__shared__ __nv_bfloat16 dot_rhs_29[" in cuda_src

    assert (
        "dot_lhs_29_in_bounds ? a[dot_lhs_29_source_index] : __float2bfloat16_rn(0.0f);"
    ) in cuda_src

    assert (
        "v29 += "
        "__bfloat162float(dot_lhs_29[(tile_i) * 16 + (dot_k_29)]) * "
        "__bfloat162float(dot_rhs_29[(dot_k_29) * 8 + (tile_j)]);"
    ) in cuda_src


@pytest.mark.execution
def test_bf16_dot_executes_on_supported_cuda_gpu(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cp

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        pytest.skip("bfloat16 CUDA execution requires compute capability 8.0 or newer")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 4, 8, 16
    BM, BK, BN = 4, 16, 8

    a_host = (
        torch.linspace(
            -1.0,
            1.0,
            M * K,
            dtype=torch.float32,
        )
        .reshape(M, K)
        .to(torch.bfloat16)
    )
    b_host = (
        torch.linspace(
            0.75,
            -0.5,
            K * N,
            dtype=torch.float32,
        )
        .reshape(K, N)
        .to(torch.bfloat16)
    )

    a = a_host.cuda()
    b = b_host.cuda()
    out = torch.zeros(
        (M, N),
        device="cuda",
        dtype=torch.float32,
    )

    low_precision_dot_kernel.clear_cache()
    low_precision_dot_kernel[(1, 1)](
        a,
        b,
        out,
        M,
        N,
        K,
        0,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    torch.cuda.synchronize()

    expected = a_host.to(torch.float32) @ b_host.to(torch.float32)

    torch.testing.assert_close(
        out.cpu(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cuda_codegen_casts_every_register_tile_element() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    register_layout = codegen.layout.register_tile_layout()

    source = SSAValue(
        id=0,
        ty=BlockType((4, 8), F16),
    )
    result = SSAValue(
        id=1,
        ty=BlockType((4, 8), F32),
    )

    codegen.values[source.id] = CudaRegisterTileRef(
        base="v0",
        layout=register_layout,
    )

    op = SSAOp(
        opcode="cast",
        operands=(source,),
        result=result,
        attrs={"dtype": F32},
    )

    codegen.emit_cast(op, result)

    assert codegen.lines == [
        "    float v1_0_0 = __half2float(v0_0_0);",
        "    float v1_1_0 = __half2float(v0_1_0);",
    ]

    result_ref = codegen.values[result.id]
    assert isinstance(result_ref, CudaRegisterTileRef)
    assert result_ref.layout == register_layout
    assert result_ref.element((0, 0)) == "v1_0_0"
    assert result_ref.element((1, 0)) == "v1_1_0"


def test_cuda_codegen_promotes_mixed_low_precision_binary_operands() -> None:
    lhs = Param("lhs", F16)
    rhs = Param("rhs", BF16)
    result = SSAValue(
        id=0,
        ty=F32,
    )

    op = SSAOp(
        opcode="add",
        operands=(lhs, rhs),
        result=result,
    )

    cuda_src = SSACUDACodegen().generate(
        kernel_name="mixed_add_kernel",
        ssa_ops=[op],
        params=[lhs, rhs],
    )

    assert "#include <cuda_fp16.h>" in cuda_src
    assert "#include <cuda_bf16.h>" in cuda_src
    assert ("    float v0 = (__half2float(lhs) + __bfloat162float(rhs));") in cuda_src


def test_cuda_codegen_promotes_mixed_low_precision_comparison_operands() -> None:
    lhs = Param("lhs", F16)
    rhs = Param("rhs", BF16)
    result = SSAValue(
        id=0,
        ty=BOOL,
    )

    op = SSAOp(
        opcode="cmp_lt",
        operands=(lhs, rhs),
        result=result,
    )

    cuda_src = SSACUDACodegen().generate(
        kernel_name="mixed_compare_kernel",
        ssa_ops=[op],
        params=[lhs, rhs],
    )

    assert ("    bool v0 = (__half2float(lhs) < __bfloat162float(rhs));") in cuda_src


def test_cuda_codegen_promotes_mixed_low_precision_select_arms() -> None:
    condition = Param("condition", BOOL)
    true_value = Param("true_value", F16)
    false_value = Param("false_value", BF16)
    result = SSAValue(id=0, ty=F32)

    cuda_src = SSACUDACodegen().generate(
        kernel_name="mixed_select_kernel",
        ssa_ops=[
            SSAOp(
                opcode="select",
                operands=(condition, true_value, false_value),
                result=result,
            )
        ],
        params=[condition, true_value, false_value],
    )

    assert (
        "    float v0 = (condition ? __half2float(true_value) : "
        "__bfloat162float(false_value));"
    ) in cuda_src


@pytest.mark.parametrize(
    ("opcode", "symbol"),
    [
        ("minimum", "<"),
        ("maximum", ">"),
    ],
)
def test_cuda_codegen_promotes_mixed_low_precision_extremum_operands(
    opcode: str,
    symbol: str,
) -> None:
    lhs = Param("lhs", F16)
    rhs = Param("rhs", BF16)
    result = SSAValue(id=0, ty=F32)

    cuda_src = SSACUDACodegen().generate(
        kernel_name="mixed_extremum_kernel",
        ssa_ops=[
            SSAOp(
                opcode=opcode,
                operands=(lhs, rhs),
                result=result,
            )
        ],
        params=[lhs, rhs],
    )

    converted_lhs = "__half2float(lhs)"
    converted_rhs = "__bfloat162float(rhs)"

    assert f"isnan({converted_lhs})" in cuda_src
    assert f"isnan({converted_rhs})" in cuda_src
    assert (
        f"(({converted_lhs}) {symbol} ({converted_rhs}) ? "
        f"({converted_lhs}) : ({converted_rhs}))"
    ) in cuda_src


@pytest.mark.parametrize(
    ("dtype", "expected_zero", "expected_full"),
    [
        (
            F16,
            "    __half v0 = __float2half_rn(0.0f);",
            "    __half v1 = __float2half_rn(1.5f);",
        ),
        (
            BF16,
            "    __nv_bfloat16 v0 = __float2bfloat16_rn(0.0f);",
            "    __nv_bfloat16 v1 = __float2bfloat16_rn(1.5f);",
        ),
    ],
)
def test_cuda_codegen_explicitly_converts_low_precision_constructors(
    dtype: ScalarType,
    expected_zero: str,
    expected_full: str,
) -> None:
    zero = SSAValue(
        id=0,
        ty=BlockType((8,), dtype),
    )
    full = SSAValue(
        id=1,
        ty=BlockType((8,), dtype),
    )

    ops: list[SSAItem] = [
        SSAOp(
            opcode="zeros",
            result=zero,
            attrs={
                "shape": (8,),
                "dtype": dtype,
            },
        ),
        SSAOp(
            opcode="full",
            operands=(Const(1.5),),
            result=full,
            attrs={
                "shape": (8,),
                "dtype": dtype,
            },
        ),
    ]

    cuda_src = SSACUDACodegen().generate(
        kernel_name="constructors_kernel",
        ssa_ops=ops,
        params=[],
    )

    assert expected_zero in cuda_src
    assert expected_full in cuda_src
