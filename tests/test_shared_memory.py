from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ast_frontend import trace as trace_ast
from mytriton.block_shapes import (
    CudaKernelLayout,
    cuda_kernel_layout,
)
from mytriton.cuda_codegen import (
    CudaArangeRef,
    CudaRegisterTileRef,
    SSACUDACodegen,
)
from mytriton.cuda_dot_staging import (
    CudaDotDoubleBufferingPlan,
    CudaDotOperandMatcher,
    CudaDotSharedBuffers,
    CudaDotStagingAnalysis,
    CudaDotStagingAnalyzer,
    CudaDotStagingPlan,
    CudaGlobalTile,
    CudaGlobalTilePlan,
    CudaSharedBuffer,
    SSADefinitions,
    cuda_f32_shared_row_padding,
    match_cuda_dot_double_buffering,
)
from mytriton.ssa import (
    SSAForRange,
    SSAItem,
    SSALowering,
    SSAOp,
    SSAPrinter,
    SSAValue,
)
from mytriton.trace import (
    BOOL,
    F32,
    I32,
    PTR_F32,
    BlockType,
    Const,
    Param,
    Ptr,
    Value,
    make_runtime_params,
)


@triton.jit
def single_tile_dot_kernel(
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


@triton.jit
def tiled_matmul_kernel(
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
    acc = tl.zeros((BM, BN), tl.float32)

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

        acc = acc + tl.dot(a_values, b_values)

    output_pointers = out + offsets_m * N + offsets_n
    output_mask = (offsets_m < M) & (offsets_n < N)
    tl.store(output_pointers, acc, mask=output_mask)


def test_cuda_shared_buffer_represents_flat_rank2_tile() -> None:
    buffer = CudaSharedBuffer(
        name="dot_lhs_7",
        logical_shape=(4, 16),
        element_ty=F32,
    )

    assert buffer.rows == 4
    assert buffer.columns == 16
    assert buffer.size == 64
    assert buffer.nbytes == 256
    assert buffer.element("row", "column") == "dot_lhs_7[(row) * 16 + (column)]"


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (8,),
        (4, 0),
        (4, -1),
        (4, 8, 2),
    ],
)
def test_cuda_shared_buffer_rejects_invalid_shape(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="shared buffer must be a positive rank-2 tile",
    ):
        CudaSharedBuffer(
            name="tile",
            logical_shape=shape,
            element_ty=F32,
        )


def test_cuda_codegen_declares_shared_buffer() -> None:
    codegen = SSACUDACodegen()

    buffer = codegen.declare_shared_buffer(
        name="dot_lhs_7",
        logical_shape=(4, 16),
        element_ty=F32,
    )

    assert buffer == CudaSharedBuffer(
        name="dot_lhs_7",
        logical_shape=(4, 16),
        element_ty=F32,
    )
    assert codegen.shared_lines == [
        "    __shared__ float dot_lhs_7[64];",
    ]


def test_cuda_codegen_emits_cooperative_masked_load() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    target = codegen.declare_shared_buffer(
        name="dot_lhs_7",
        logical_shape=(4, 16),
        element_ty=F32,
    )
    source = CudaGlobalTile(
        base="a",
        row_offset="blockIdx.x * 4",
        column_offset="k_base",
        row_stride="K",
        row_bound="M",
        column_bound="K",
    )

    codegen.emit_cooperative_load(target, source)

    assert codegen.lines == [
        (
            "    for (int dot_lhs_7_index = threadIdx.x; "
            "dot_lhs_7_index < 64; dot_lhs_7_index += 32) {"
        ),
        "        int dot_lhs_7_row = dot_lhs_7_index / 16;",
        "        int dot_lhs_7_column = dot_lhs_7_index % 16;",
        ("        int dot_lhs_7_global_row = (blockIdx.x * 4) + dot_lhs_7_row;"),
        ("        int dot_lhs_7_global_column = (k_base) + dot_lhs_7_column;"),
        (
            "        int dot_lhs_7_source_index = "
            "dot_lhs_7_global_row * (K) + dot_lhs_7_global_column;"
        ),
        (
            "        bool dot_lhs_7_in_bounds = "
            "dot_lhs_7_global_row < (M) && "
            "dot_lhs_7_global_column < (K);"
        ),
        (
            "        dot_lhs_7[(dot_lhs_7_row) * 16 + "
            "(dot_lhs_7_column)] = dot_lhs_7_in_bounds ? "
            "a[dot_lhs_7_source_index] : 0.0f;"
        ),
        "    }",
    ]


def test_cuda_codegen_rejects_non_row_major_cooperative_load() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )
    target = CudaSharedBuffer(
        name="tile",
        logical_shape=(4, 8),
        element_ty=F32,
    )
    source = CudaGlobalTile(
        base="a",
        row_offset="0",
        column_offset="0",
        row_stride="8",
        row_bound="4",
        column_bound="8",
    )

    with pytest.raises(
        ValueError,
        match=r"cooperative dot loads require row-major order \(1, 0\)",
    ):
        codegen.emit_cooperative_load(
            target,
            source,
            order=(0, 1),
        )

    assert codegen.lines == []


def test_cuda_codegen_emits_block_barrier() -> None:
    codegen = SSACUDACodegen()

    codegen.emit_block_barrier()

    assert codegen.lines == [
        "    __syncthreads();",
    ]


def test_cuda_codegen_stages_both_dot_operands() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    buffers = codegen.emit_dot_operand_staging(
        dot_result_id=7,
        lhs_shape=(4, 16),
        rhs_shape=(16, 8),
        element_ty=F32,
        lhs_source=CudaGlobalTile(
            base="a",
            row_offset="blockIdx.x * 4",
            column_offset="k_base",
            row_stride="K",
            row_bound="M",
            column_bound="K",
        ),
        rhs_source=CudaGlobalTile(
            base="b",
            row_offset="k_base",
            column_offset="blockIdx.y * 8",
            row_stride="N",
            row_bound="K",
            column_bound="N",
        ),
    )

    assert buffers == CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(4, 16),
            element_ty=F32,
            row_padding=1,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 8),
            element_ty=F32,
        ),
    )
    assert buffers.reduction_size == 16
    assert codegen.shared_memory_bytes == (68 + 128) * 4

    assert codegen.shared_lines == [
        "    __shared__ float dot_lhs_7[68];",
        "    __shared__ float dot_rhs_7[128];",
    ]

    assert len(codegen.lines) == 19

    assert codegen.lines[0] == (
        "    for (int dot_lhs_7_index = threadIdx.x; "
        "dot_lhs_7_index < 64; dot_lhs_7_index += 32) {"
    )
    assert codegen.lines[6] == (
        "        bool dot_lhs_7_in_bounds = "
        "dot_lhs_7_global_row < (M) && "
        "dot_lhs_7_global_column < (K);"
    )
    assert codegen.lines[7] == (
        "        dot_lhs_7[(dot_lhs_7_row) * 17 + "
        "(dot_lhs_7_column)] = dot_lhs_7_in_bounds ? "
        "a[dot_lhs_7_source_index] : 0.0f;"
    )

    assert codegen.lines[9] == (
        "    for (int dot_rhs_7_index = threadIdx.x; "
        "dot_rhs_7_index < 128; dot_rhs_7_index += 32) {"
    )
    assert codegen.lines[15] == (
        "        bool dot_rhs_7_in_bounds = "
        "dot_rhs_7_global_row < (K) && "
        "dot_rhs_7_global_column < (N);"
    )
    assert codegen.lines[16] == (
        "        dot_rhs_7[(dot_rhs_7_row) * 8 + "
        "(dot_rhs_7_column)] = dot_rhs_7_in_bounds ? "
        "b[dot_rhs_7_source_index] : 0.0f;"
    )

    assert codegen.lines[-1] == "    __syncthreads();"


def test_dot_operand_staging_rejects_incompatible_shapes() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    source = CudaGlobalTile(
        base="x",
        row_offset="0",
        column_offset="0",
        row_stride="16",
        row_bound="16",
        column_bound="16",
    )

    with pytest.raises(
        ValueError,
        match="dot staging expects compatible rank-2 operands",
    ):
        codegen.emit_dot_operand_staging(
            dot_result_id=0,
            lhs_shape=(4, 16),
            rhs_shape=(8, 8),
            element_ty=F32,
            lhs_source=source,
            rhs_source=source,
        )

    assert codegen.shared_lines == []
    assert codegen.lines == []


def test_dot_operand_staging_rejects_shared_memory_over_budget() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(1, 1),
        thread_shape=(1, 1),
    )
    source = CudaGlobalTile(
        base="x",
        row_offset="0",
        column_offset="0",
        row_stride="128",
        row_bound="128",
        column_bound="128",
    )

    with pytest.raises(
        ValueError,
        match="exceeding the conservative 49152-byte limit",
    ):
        codegen.emit_dot_operand_staging(
            dot_result_id=0,
            lhs_shape=(128, 128),
            rhs_shape=(128, 1),
            element_ty=F32,
            lhs_source=source,
            rhs_source=source,
        )

    assert codegen.shared_memory_bytes == 0
    assert codegen.shared_lines == []
    assert codegen.lines == []


def make_tiled_dot_ssa() -> tuple[list[SSAItem], SSAValue]:
    BM, BK, BN = 4, 16, 8

    a = Ptr(Param("a", PTR_F32))
    b = Ptr(Param("b", PTR_F32))

    M = Value(Param("M", I32))
    N = Value(Param("N", I32))
    K = Value(Param("K", I32))
    k_base = Value(Param("k_base", I32))

    offsets_m = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    offsets_n = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
    offsets_k = tl.arange(0, BK)

    a_rows = offsets_m
    a_columns = k_base + offsets_k[None, :]
    a_pointers = a + a_rows * K + a_columns
    a_mask = (a_rows < M) & (a_columns < K)
    lhs = tl.load(a_pointers, mask=a_mask, other=0.0)

    b_rows = k_base + offsets_k[:, None]
    b_columns = offsets_n
    b_pointers = b + b_rows * N + b_columns
    b_mask = (b_rows < K) & (b_columns < N)
    rhs = tl.load(b_pointers, mask=b_mask, other=0.0)

    dot = tl.dot(lhs, rhs)

    lowering = SSALowering()
    result = lowering.lower_expr(dot.expr)

    assert isinstance(result, SSAValue)

    return lowering.ops, result


def test_ssa_definitions_find_dot_operand_loads() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    definitions = SSADefinitions(ssa_ops)

    dot = definitions.require(dot_result, "dot")
    lhs, rhs = dot.operands

    assert isinstance(lhs, SSAValue)
    assert isinstance(rhs, SSAValue)
    assert lhs.id == 14
    assert rhs.id == 28

    lhs_load = definitions.require(lhs, "load")
    rhs_load = definitions.require(rhs, "load")

    assert lhs_load.result is lhs
    assert rhs_load.result is rhs
    assert lhs_load.opcode == "load"
    assert rhs_load.opcode == "load"

    assert definitions.get(SSAValue(id=1000, ty=I32)) is None


def test_dot_operand_matcher_recovers_lhs_global_tile() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    definitions = SSADefinitions(ssa_ops)
    dot = definitions.require(dot_result, "dot")

    lhs, _ = dot.operands
    plan = CudaDotOperandMatcher(definitions).match(lhs)

    assert plan == CudaGlobalTilePlan(
        base=Param("a", PTR_F32),
        row_offset=SSAValue(id=1, ty=I32),
        column_offset=Param("k_base", I32),
        row_stride=Param("K", I32),
        row_bound=Param("M", I32),
        column_bound=Param("K", I32),
        other=Const(0.0),
    )


def test_dot_operand_matcher_recovers_rhs_global_tile() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    definitions = SSADefinitions(ssa_ops)
    dot = definitions.require(dot_result, "dot")

    _, rhs = dot.operands
    plan = CudaDotOperandMatcher(definitions).match(rhs)

    assert plan == CudaGlobalTilePlan(
        base=Param("b", PTR_F32),
        row_offset=Param("k_base", I32),
        column_offset=SSAValue(id=20, ty=I32),
        row_stride=Param("N", I32),
        row_bound=Param("K", I32),
        column_bound=Param("N", I32),
        other=Const(0.0),
    )


def test_cuda_codegen_stages_dot_operands_from_ssa() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    definitions = SSADefinitions(ssa_ops)
    dot = definitions.require(dot_result, "dot")
    analysis = CudaDotStagingAnalyzer(definitions).analyze()

    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )
    codegen.definitions = definitions
    codegen.staging_analysis = analysis

    codegen.values[1] = "v1"
    codegen.values[20] = "v20"

    buffers = codegen.emit_dot_operand_staging_from_ssa(
        dot,
        analysis.plan_for(dot_result.id),
    )

    assert buffers.reduction_size == 16
    assert buffers.lhs.row_padding == 1
    assert buffers.rhs.row_padding == 0
    assert codegen.shared_lines == [
        "    __shared__ float dot_lhs_29[68];",
        "    __shared__ float dot_rhs_29[128];",
    ]

    assert codegen.lines[3] == (
        "        int dot_lhs_29_global_row = (v1) + dot_lhs_29_row;"
    )
    assert codegen.lines[4] == (
        "        int dot_lhs_29_global_column = (k_base) + dot_lhs_29_column;"
    )
    assert codegen.lines[5] == (
        "        int dot_lhs_29_source_index = "
        "dot_lhs_29_global_row * (K) + "
        "dot_lhs_29_global_column;"
    )
    assert codegen.lines[6] == (
        "        bool dot_lhs_29_in_bounds = "
        "dot_lhs_29_global_row < (M) && "
        "dot_lhs_29_global_column < (K);"
    )

    assert codegen.lines[12] == (
        "        int dot_rhs_29_global_row = (k_base) + dot_rhs_29_row;"
    )
    assert codegen.lines[13] == (
        "        int dot_rhs_29_global_column = (v20) + dot_rhs_29_column;"
    )
    assert codegen.lines[14] == (
        "        int dot_rhs_29_source_index = "
        "dot_rhs_29_global_row * (N) + "
        "dot_rhs_29_global_column;"
    )
    assert codegen.lines[15] == (
        "        bool dot_rhs_29_in_bounds = "
        "dot_rhs_29_global_row < (K) && "
        "dot_rhs_29_global_column < (N);"
    )

    assert codegen.lines[-1] == "    __syncthreads();"


def test_dot_staging_analysis_finds_staging_only_operations() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    definitions = SSADefinitions(ssa_ops)

    analysis = CudaDotStagingAnalyzer(definitions).analyze()
    dot = definitions.require(dot_result, "dot")
    lhs, rhs = dot.operands

    assert analysis.dot_plans == {
        dot_result.id: CudaDotStagingPlan(
            lhs=CudaDotOperandMatcher(definitions).match(lhs),
            rhs=CudaDotOperandMatcher(definitions).match(rhs),
        )
    }
    assert analysis.stageable_dot_ids == frozenset({dot_result.id})
    assert analysis.staging_only_ids == frozenset(
        [
            *range(2, 19),
            *range(21, 29),
        ]
    )


def test_dot_staging_analysis_ignores_non_load_operands() -> None:
    lhs = SSAValue(
        id=0,
        ty=BlockType((4, 16), F32),
    )
    rhs = SSAValue(
        id=1,
        ty=BlockType((16, 8), F32),
    )
    result = SSAValue(
        id=2,
        ty=BlockType((4, 8), F32),
    )

    ssa_ops: list[SSAItem] = [
        SSAOp(opcode="zeros", result=lhs),
        SSAOp(opcode="zeros", result=rhs),
        SSAOp(
            opcode="dot",
            operands=(lhs, rhs),
            result=result,
        ),
    ]

    analysis = CudaDotStagingAnalyzer(SSADefinitions(ssa_ops)).analyze()

    assert analysis == CudaDotStagingAnalysis(
        dot_plans={},
        staging_only_ids=frozenset(),
    )


def test_cuda_generate_lowers_dot_to_cuda_cores() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()

    ssa_ops.append(
        SSAOp(
            opcode="store",
            operands=(
                Param("out", PTR_F32),
                dot_result,
                None,
            ),
        )
    )

    params = [
        Param("a", PTR_F32),
        Param("b", PTR_F32),
        Param("out", PTR_F32),
        Param("M", I32),
        Param("N", I32),
        Param("K", I32),
        Param("k_base", I32),
    ]

    codegen = SSACUDACodegen()
    cuda_src = codegen.generate(
        kernel_name="cuda_core_dot_kernel",
        ssa_ops=ssa_ops,
        params=params,
    )

    assert codegen.staging_analysis.stageable_dot_ids == frozenset({dot_result.id})

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void cuda_core_dot_kernel(float* a, float* b, float* out, int M, int N, int K, int k_base) {
            __shared__ float dot_lhs_29[68];
            __shared__ float dot_rhs_29[128];

            int tile_i = threadIdx.x / 8;
            int tile_j = threadIdx.x % 8;
            int v0 = blockIdx.x;
            int v1 = (v0 * 4);
            int v19 = blockIdx.y;
            int v20 = (v19 * 8);
            for (int dot_lhs_29_index = threadIdx.x; dot_lhs_29_index < 64; dot_lhs_29_index += 32) {
                int dot_lhs_29_row = dot_lhs_29_index / 16;
                int dot_lhs_29_column = dot_lhs_29_index % 16;
                int dot_lhs_29_global_row = (v1) + dot_lhs_29_row;
                int dot_lhs_29_global_column = (k_base) + dot_lhs_29_column;
                int dot_lhs_29_source_index = dot_lhs_29_global_row * (K) + dot_lhs_29_global_column;
                bool dot_lhs_29_in_bounds = dot_lhs_29_global_row < (M) && dot_lhs_29_global_column < (K);
                dot_lhs_29[(dot_lhs_29_row) * 17 + (dot_lhs_29_column)] = dot_lhs_29_in_bounds ? a[dot_lhs_29_source_index] : 0.0f;
            }
            for (int dot_rhs_29_index = threadIdx.x; dot_rhs_29_index < 128; dot_rhs_29_index += 32) {
                int dot_rhs_29_row = dot_rhs_29_index / 8;
                int dot_rhs_29_column = dot_rhs_29_index % 8;
                int dot_rhs_29_global_row = (k_base) + dot_rhs_29_row;
                int dot_rhs_29_global_column = (v20) + dot_rhs_29_column;
                int dot_rhs_29_source_index = dot_rhs_29_global_row * (N) + dot_rhs_29_global_column;
                bool dot_rhs_29_in_bounds = dot_rhs_29_global_row < (K) && dot_rhs_29_global_column < (N);
                dot_rhs_29[(dot_rhs_29_row) * 8 + (dot_rhs_29_column)] = dot_rhs_29_in_bounds ? b[dot_rhs_29_source_index] : 0.0f;
            }
            __syncthreads();
            float v29 = 0.0f;
            #pragma unroll
            for (int dot_k_29 = 0; dot_k_29 < 16; ++dot_k_29) {
                v29 += dot_lhs_29[(tile_i) * 17 + (dot_k_29)] * dot_rhs_29[(dot_k_29) * 8 + (tile_j)];
            }
            __syncthreads();
            out[0] = v29;
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


def test_ast_frontend_lowers_shared_memory_dot_to_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 4, 8, 16
    BM, BK, BN = 4, 16, 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    single_tile_dot_kernel.clear_cache()

    _, ssa_ops, cuda_src = single_tile_dot_kernel[(1, 1)](
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

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 4 : i32
        %2 = arange {start=0, end=4} : vector<4 x i32>
        %3 = expand_dims %2 {axis=1} : block<4x1 x i32>
        %4 = add %1, %3 : block<4x1 x i32>
        %5 = mul %4, K : block<4x1 x i32>
        %6 = addptr a, %5 : block<4x1 x ptr<f32>>
        %7 = arange {start=0, end=16} : vector<16 x i32>
        %8 = expand_dims %7 {axis=0} : block<1x16 x i32>
        %9 = add k_base, %8 : block<1x16 x i32>
        %10 = addptr %6, %9 : block<4x16 x ptr<f32>>
        %11 = cmp_lt %4, M : block<4x1 x bool>
        %12 = cmp_lt %9, K : block<1x16 x bool>
        %13 = and %11, %12 : block<4x16 x bool>
        %14 = load %10, %13, 0.0 : block<4x16 x f32>
        %15 = expand_dims %7 {axis=1} : block<16x1 x i32>
        %16 = add k_base, %15 : block<16x1 x i32>
        %17 = mul %16, N : block<16x1 x i32>
        %18 = addptr b, %17 : block<16x1 x ptr<f32>>
        %19 = program_id {axis=1} : i32
        %20 = mul %19, 8 : i32
        %21 = arange {start=0, end=8} : vector<8 x i32>
        %22 = expand_dims %21 {axis=0} : block<1x8 x i32>
        %23 = add %20, %22 : block<1x8 x i32>
        %24 = addptr %18, %23 : block<16x8 x ptr<f32>>
        %25 = cmp_lt %16, K : block<16x1 x bool>
        %26 = cmp_lt %23, N : block<1x8 x bool>
        %27 = and %25, %26 : block<16x8 x bool>
        %28 = load %24, %27, 0.0 : block<16x8 x f32>
        %29 = dot %14, %28 : block<4x8 x f32>
        %30 = mul %4, N : block<4x1 x i32>
        %31 = addptr out, %30 : block<4x1 x ptr<f32>>
        %32 = addptr %31, %23 : block<4x8 x ptr<f32>>
        %35 = and %11, %26 : block<4x8 x bool>
        store %32, %29, %35
        """
    ).rstrip("\n")

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void single_tile_dot_kernel(float* a, float* b, float* out, int M, int N, int K, int k_base) {
            __shared__ float dot_lhs_29[68];
            __shared__ float dot_rhs_29[128];

            int tile_i = threadIdx.x / 8;
            int tile_j = threadIdx.x % 8;
            int v0 = blockIdx.x;
            int v1 = (v0 * 4);
            int v3 = tile_i;
            int v4 = (v1 + v3);
            bool v11 = (v4 < M);
            int v19 = blockIdx.y;
            int v20 = (v19 * 8);
            int v22 = tile_j;
            int v23 = (v20 + v22);
            bool v26 = (v23 < N);
            for (int dot_lhs_29_index = threadIdx.x; dot_lhs_29_index < 64; dot_lhs_29_index += 32) {
                int dot_lhs_29_row = dot_lhs_29_index / 16;
                int dot_lhs_29_column = dot_lhs_29_index % 16;
                int dot_lhs_29_global_row = (v1) + dot_lhs_29_row;
                int dot_lhs_29_global_column = (k_base) + dot_lhs_29_column;
                int dot_lhs_29_source_index = dot_lhs_29_global_row * (K) + dot_lhs_29_global_column;
                bool dot_lhs_29_in_bounds = dot_lhs_29_global_row < (M) && dot_lhs_29_global_column < (K);
                dot_lhs_29[(dot_lhs_29_row) * 17 + (dot_lhs_29_column)] = dot_lhs_29_in_bounds ? a[dot_lhs_29_source_index] : 0.0f;
            }
            for (int dot_rhs_29_index = threadIdx.x; dot_rhs_29_index < 128; dot_rhs_29_index += 32) {
                int dot_rhs_29_row = dot_rhs_29_index / 8;
                int dot_rhs_29_column = dot_rhs_29_index % 8;
                int dot_rhs_29_global_row = (k_base) + dot_rhs_29_row;
                int dot_rhs_29_global_column = (v20) + dot_rhs_29_column;
                int dot_rhs_29_source_index = dot_rhs_29_global_row * (N) + dot_rhs_29_global_column;
                bool dot_rhs_29_in_bounds = dot_rhs_29_global_row < (K) && dot_rhs_29_global_column < (N);
                dot_rhs_29[(dot_rhs_29_row) * 8 + (dot_rhs_29_column)] = dot_rhs_29_in_bounds ? b[dot_rhs_29_source_index] : 0.0f;
            }
            __syncthreads();
            float v29 = 0.0f;
            #pragma unroll
            for (int dot_k_29 = 0; dot_k_29 < 16; ++dot_k_29) {
                v29 += dot_lhs_29[(tile_i) * 17 + (dot_k_29)] * dot_rhs_29[(dot_k_29) * 8 + (tile_j)];
            }
            __syncthreads();
            int v30 = (v4 * N);
            bool v35 = (v11 && v26);
            if (v35) {
                out[(v30 + v23)] = v29;
            }
        }
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa
    assert cuda_src == expected_cuda_src


@pytest.mark.execution
def test_shared_memory_dot_executes_single_k_tile(
    cp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 7, 13, 7
    BM, BK, BN = 4, 8, 8

    a = cp.arange(M * K, dtype=cp.float32).reshape(M, K) / K
    b = cp.arange(K * N, dtype=cp.float32).reshape(K, N) / N
    out = cp.zeros((M, N), dtype=cp.float32)

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    single_tile_dot_kernel.clear_cache()
    single_tile_dot_kernel[grid](
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

    expected = a @ b
    cp.testing.assert_allclose(
        out,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_runtime_k_loop_lowers_dot_with_reusable_shared_memory() -> None:
    M, N, K = 4, 8, 32
    BM, BK, BN = 4, 16, 8
    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    bound = tiled_matmul_kernel.signature.bind(
        a,
        b,
        out,
        M,
        N,
        K,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    runtime_params = make_runtime_params(
        tiled_matmul_kernel.signature,
        bound.arguments,
    )
    traced_ops, _ = trace_ast(
        tiled_matmul_kernel.fn,
        tiled_matmul_kernel.signature,
        bound.arguments,
        runtime_params=runtime_params,
    )
    ssa_ops = SSALowering().lower(traced_ops)

    loop = next(op for op in ssa_ops if isinstance(op, SSAForRange))
    dot = next(op for op in loop.body if isinstance(op, SSAOp) and op.opcode == "dot")
    assert dot.result is not None

    definitions = SSADefinitions(ssa_ops)
    analysis = CudaDotStagingAnalyzer(definitions).analyze()
    plan = analysis.plan_for(dot.result.id)

    assert plan.lhs.column_offset == loop.index
    assert plan.rhs.row_offset == loop.index

    accumulation = next(
        op
        for op in loop.body
        if (isinstance(op, SSAOp) and op.opcode == "add" and dot.result in op.operands)
    )
    assert accumulation.result is not None
    assert len(loop.results) == 1

    codegen = SSACUDACodegen()
    cuda_src = codegen.generate(
        kernel_name="tiled_matmul_kernel",
        ssa_ops=ssa_ops,
        params=runtime_params,
    )

    dot_id = dot.result.id
    stage_name = f"dot_stage_{dot_id}"
    lhs_stage_size = BM * (BK + 1)
    rhs_stage_size = BK * BN
    loop_index = f"v{loop.index.id}"
    accumulator = f"v{loop.results[0].id}"
    accumulation_result = f"v{accumulation.result.id}"

    outer_loop = (
        f"    for (int {loop_index} = 0; {loop_index} < K; {loop_index} += {BK}) {{"
    )
    outer_loop_position = cuda_src.index(outer_loop)
    stage_declaration = f"        int {stage_name} = ({loop_index} / {BK}) & 1;"
    stage_position = cuda_src.index(
        stage_declaration,
        outer_loop_position,
    )
    lhs_load = f"        for (int dot_lhs_{dot_id}_index = threadIdx.x; "
    rhs_load = f"        for (int dot_rhs_{dot_id}_index = threadIdx.x; "
    fma_loop = (
        f"        for (int dot_k_{dot_id} = 0; "
        f"dot_k_{dot_id} < {BK}; ++dot_k_{dot_id}) {{"
    )
    barrier = "        __syncthreads();"

    lhs_load_position = cuda_src.index(lhs_load, outer_loop_position)
    rhs_load_position = cuda_src.index(rhs_load, lhs_load_position)
    load_barrier_position = cuda_src.index(barrier, rhs_load_position)
    fma_position = cuda_src.index(fma_loop, load_barrier_position)

    expected_accumulation = (
        f"        float {accumulation_result} = "
        f"({accumulator} + v{dot_id});\n"
        f"        {accumulator} = {accumulation_result};"
    )
    accumulation_position = cuda_src.index(
        expected_accumulation,
        fma_position,
    )

    assert (
        outer_loop_position
        < stage_position
        < lhs_load_position
        < rhs_load_position
        < load_barrier_position
        < fma_position
        < accumulation_position
    )

    assert f"global_column = ({loop_index}) + dot_lhs_{dot_id}_column;" in cuda_src
    assert f"global_row = ({loop_index}) + dot_rhs_{dot_id}_row;" in cuda_src
    assert (f"dot_lhs_{dot_id}[(({stage_name}) * {lhs_stage_size}) + ") in cuda_src
    assert (f"dot_rhs_{dot_id}[(({stage_name}) * {rhs_stage_size}) + ") in cuda_src
    assert (f"    __shared__ float dot_lhs_{dot_id}[{2 * lhs_stage_size}];") in cuda_src
    assert (f"    __shared__ float dot_rhs_{dot_id}[{2 * rhs_stage_size}];") in cuda_src

    assert cuda_src.count("__syncthreads();") == 1
    assert codegen.shared_memory_bytes == 2 * (BM * (BK + 1) + BK * BN) * 4


@pytest.mark.execution
def test_shared_memory_dot_executes_multiple_k_tiles(
    cp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 7, 13, 19
    BM, BK, BN = 4, 8, 8

    a_host = np.arange(M * K, dtype=np.float32).reshape(M, K) % 9 - 4
    b_host = np.arange(K * N, dtype=np.float32).reshape(K, N) % 7 - 3

    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    out = cp.zeros((M, N), dtype=cp.float32)

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    tiled_matmul_kernel.clear_cache()
    tiled_matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    cp.cuda.Stream.null.synchronize()

    expected = a_host @ b_host
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_dot_operand_matcher_rejects_block_bound() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    definitions = SSADefinitions(ssa_ops)
    dot = definitions.require(dot_result, "dot")

    lhs, _ = dot.operands
    lhs_load = definitions.require(lhs, "load")
    mask = lhs_load.operands[1]
    conjunction = definitions.require(mask, "and")

    row_comparison_operand = conjunction.operands[0]
    row_comparison = definitions.require(
        row_comparison_operand,
        "cmp_lt",
    )

    coordinate, _ = row_comparison.operands
    block_bound = SSAValue(
        id=1000,
        ty=BlockType((4, 1), I32),
    )
    row_comparison.operands = (coordinate, block_bound)

    with pytest.raises(
        TypeError,
        match="dot staging bounds must be scalar i32",
    ):
        CudaDotOperandMatcher(definitions).match(lhs)


def test_dot_staging_analysis_preserves_external_dependencies() -> None:
    ssa_ops, dot_result = make_tiled_dot_ssa()
    initial_definitions = SSADefinitions(ssa_ops)

    rows = initial_definitions.ops[4].result
    columns = initial_definitions.ops[23].result

    assert rows is not None
    assert columns is not None

    output_offset = SSAValue(
        id=30,
        ty=BlockType((4, 8), I32),
    )
    ssa_ops.append(
        SSAOp(
            opcode="add",
            operands=(rows, columns),
            result=output_offset,
        )
    )
    ssa_ops.append(
        SSAOp(
            opcode="store",
            operands=(
                Param("out", PTR_F32),
                output_offset,
                None,
            ),
        )
    )

    analysis = CudaDotStagingAnalyzer(SSADefinitions(ssa_ops)).analyze()

    # Shared with output address: must use ordinary lowering.
    assert not {0, 1, 2, 3, 4} & analysis.staging_only_ids
    assert not {19, 20, 21, 22, 23} & analysis.staging_only_ids

    # Private pointer/mask/load operations remain staging-only.
    assert {5, 6, 10, 11, 12, 13, 14} <= analysis.staging_only_ids
    assert {17, 18, 24, 25, 26, 27, 28} <= analysis.staging_only_ids

    assert dot_result.id in analysis.stageable_dot_ids


def test_cuda_codegen_computes_dot_from_shared_memory() -> None:
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
            element_ty=F32,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 8),
            element_ty=F32,
        ),
    )

    codegen.emit_dot_from_shared_memory(result, buffers)

    assert codegen.lines == [
        "    float v7 = 0.0f;",
        "    #pragma unroll",
        "    for (int dot_k_7 = 0; dot_k_7 < 16; ++dot_k_7) {",
        (
            "        v7 += "
            "dot_lhs_7[(tile_i) * 16 + (dot_k_7)] * "
            "dot_rhs_7[(dot_k_7) * 8 + (tile_j)];"
        ),
        "    }",
        "    __syncthreads();",
    ]
    assert codegen.values[result.id] == "v7"


def test_double_buffered_dot_skips_reuse_barrier() -> None:
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
            element_ty=F32,
            row_padding=1,
            stage_count=2,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 8),
            element_ty=F32,
            stage_count=2,
        ),
    )

    codegen.emit_dot_from_shared_memory(
        result,
        buffers,
        stage="stage",
        emit_reuse_barrier=False,
    )

    assert codegen.lines[-1] == "    }"
    assert "__syncthreads();" not in codegen.lines


def test_cuda_core_dot_rejects_mismatched_reduction_dimensions() -> None:
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
            element_ty=F32,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(8, 8),
            element_ty=F32,
        ),
    )

    with pytest.raises(
        TypeError,
        match="matching reduction dimensions",
    ):
        codegen.emit_dot_from_shared_memory(
            result,
            buffers,
        )

    assert codegen.lines == []
    assert codegen.values == {}


def test_cuda_core_dot_rejects_wrong_result_shape() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 7),
        thread_shape=(4, 7),
    )

    result = SSAValue(
        id=7,
        ty=BlockType((4, 7), F32),
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(4, 16),
            element_ty=F32,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 8),
            element_ty=F32,
        ),
    )

    with pytest.raises(
        TypeError,
        match=r"expected result shape \(4, 8\), got \(4, 7\)",
    ):
        codegen.emit_dot_from_shared_memory(
            result,
            buffers,
        )

    assert codegen.lines == []
    assert codegen.values == {}


def test_cuda_core_dot_emits_multiple_accumulators_per_thread() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    result = SSAValue(
        id=7,
        ty=BlockType((4, 8), F32),
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(4, 16),
            element_ty=F32,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 8),
            element_ty=F32,
        ),
    )

    codegen.emit_dot_from_shared_memory(result, buffers)

    assert codegen.lines == [
        "    float v7_0_0 = 0.0f;",
        "    float v7_1_0 = 0.0f;",
        "    #pragma unroll",
        "    for (int dot_k_7 = 0; dot_k_7 < 16; ++dot_k_7) {",
        (
            "        v7_0_0 += "
            "dot_lhs_7[(tile_i) * 16 + (dot_k_7)] * "
            "dot_rhs_7[(dot_k_7) * 8 + (tile_j)];"
        ),
        (
            "        v7_1_0 += "
            "dot_lhs_7[(tile_i + 2) * 16 + (dot_k_7)] * "
            "dot_rhs_7[(dot_k_7) * 8 + (tile_j)];"
        ),
        "    }",
        "    __syncthreads();",
    ]

    value = codegen.values[result.id]
    assert isinstance(value, CudaRegisterTileRef)
    assert value.element((0, 0)) == "v7_0_0"
    assert value.element((1, 0)) == "v7_1_0"


def test_cuda_register_tile_ref_names_multiple_registers() -> None:
    register_tile = CudaRegisterTileRef(
        base="v7",
        layout=CudaKernelLayout(
            output_tile_shape=(8, 8),
            thread_shape=(4, 4),
        ).register_tile_layout(),
    )

    assert register_tile.element((0, 0)) == "v7_0_0"
    assert register_tile.element((0, 1)) == "v7_0_1"
    assert register_tile.element((1, 0)) == "v7_1_0"
    assert register_tile.element((1, 1)) == "v7_1_1"


def test_cuda_register_tile_ref_preserves_scalar_name() -> None:
    register_tile = CudaRegisterTileRef(
        base="v7",
        layout=CudaKernelLayout(
            output_tile_shape=(4, 8),
            thread_shape=(4, 8),
        ).register_tile_layout(),
    )

    assert register_tile.element((0, 0)) == "v7"


def test_cuda_register_tile_ref_rejects_invalid_coordinate() -> None:
    register_tile = CudaRegisterTileRef(
        base="v7",
        layout=CudaKernelLayout(
            output_tile_shape=(8, 8),
            thread_shape=(4, 4),
        ).register_tile_layout(),
    )

    with pytest.raises(ValueError, match="invalid register coordinate"):
        register_tile.element((2, 0))


def test_cuda_codegen_emits_register_wise_binary_operation() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    ty = BlockType((4, 8), F32)
    accumulator = SSAValue(id=0, ty=ty)
    dot = SSAValue(id=1, ty=ty)
    result = SSAValue(id=2, ty=ty)

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[accumulator.id] = "acc"
    codegen.values[dot.id] = CudaRegisterTileRef(
        base="v1",
        layout=register_layout,
    )

    codegen.emit(
        SSAOp(
            opcode="add",
            operands=(accumulator, dot),
            result=result,
        )
    )

    assert codegen.lines == [
        "    float v2_0_0 = (acc + v1_0_0);",
        "    float v2_1_0 = (acc + v1_1_0);",
    ]

    value = codegen.values[result.id]
    assert isinstance(value, CudaRegisterTileRef)
    assert value.element((0, 0)) == "v2_0_0"
    assert value.element((1, 0)) == "v2_1_0"


def test_cuda_for_range_carries_register_tile() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    ty = BlockType((4, 8), F32)
    initial = SSAValue(id=0, ty=ty)
    increment = SSAValue(id=1, ty=ty)
    index = SSAValue(id=2, ty=I32)
    carried_arg = SSAValue(id=3, ty=ty)
    updated = SSAValue(id=4, ty=ty)
    result = SSAValue(id=5, ty=ty)

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[initial.id] = "initial"
    codegen.values[increment.id] = CudaRegisterTileRef(
        base="increment",
        layout=register_layout,
    )

    codegen.emit_for_range(
        SSAForRange(
            index=index,
            start=Const(0),
            stop=Const(2),
            step=Const(1),
            carried_inputs=(initial,),
            carried_args=(carried_arg,),
            body=[
                SSAOp(
                    opcode="add",
                    operands=(carried_arg, increment),
                    result=updated,
                )
            ],
            yields=(updated,),
            results=(result,),
        )
    )

    assert codegen.lines == [
        "    float v5_0_0 = initial;",
        "    float v5_1_0 = initial;",
        "    for (int v2 = 0; v2 < 2; v2 += 1) {",
        "        float v4_0_0 = (v5_0_0 + increment_0_0);",
        "        float v4_1_0 = (v5_1_0 + increment_1_0);",
        "        v5_0_0 = v4_0_0;",
        "        v5_1_0 = v4_1_0;",
        "    }",
    ]

    carried_value = codegen.values[carried_arg.id]
    result_value = codegen.values[result.id]

    assert isinstance(carried_value, CudaRegisterTileRef)
    assert isinstance(result_value, CudaRegisterTileRef)
    assert carried_value == result_value
    assert result_value.element((0, 0)) == "v5_0_0"
    assert result_value.element((1, 0)) == "v5_1_0"


def test_cuda_codegen_emits_register_wise_addptr() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    offsets = SSAValue(
        id=0,
        ty=BlockType((4, 8), I32),
    )
    pointers = SSAValue(
        id=1,
        ty=BlockType((4, 8), PTR_F32),
    )

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[offsets.id] = CudaRegisterTileRef(
        base="offset",
        layout=register_layout,
    )

    codegen.emit(
        SSAOp(
            opcode="addptr",
            operands=(Param("out", PTR_F32), offsets),
            result=pointers,
        )
    )

    assert codegen.lines == [
        "    float* v1_0_0 = out + offset_0_0;",
        "    float* v1_1_0 = out + offset_1_0;",
    ]

    value = codegen.values[pointers.id]
    assert isinstance(value, CudaRegisterTileRef)
    assert value.element((0, 0)) == "v1_0_0"
    assert value.element((1, 0)) == "v1_1_0"


def test_cuda_codegen_emits_register_wise_masked_store() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    pointers = SSAValue(
        id=0,
        ty=BlockType((4, 8), PTR_F32),
    )
    values = SSAValue(
        id=1,
        ty=BlockType((4, 8), F32),
    )
    mask = SSAValue(
        id=2,
        ty=BlockType((4, 8), BOOL),
    )

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[pointers.id] = CudaRegisterTileRef(
        base="pointer",
        layout=register_layout,
    )
    codegen.values[values.id] = CudaRegisterTileRef(
        base="value",
        layout=register_layout,
    )
    codegen.values[mask.id] = CudaRegisterTileRef(
        base="mask",
        layout=register_layout,
    )

    codegen.emit(
        SSAOp(
            opcode="store",
            operands=(pointers, values, mask),
        )
    )

    assert codegen.lines == [
        "    if (mask_0_0) {",
        "        pointer_0_0[0] = value_0_0;",
        "    }",
        "    if (mask_1_0) {",
        "        pointer_1_0[0] = value_1_0;",
        "    }",
    ]


def test_cuda_codegen_rejects_store_with_incompatible_register_layouts() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(2, 8),
    )

    pointers = SSAValue(
        id=0,
        ty=BlockType((4, 8), PTR_F32),
    )
    values = SSAValue(
        id=1,
        ty=BlockType((4, 8), F32),
    )

    codegen.values[pointers.id] = CudaRegisterTileRef(
        base="pointer",
        layout=codegen.layout.register_tile_layout(),
    )
    codegen.values[values.id] = CudaRegisterTileRef(
        base="value",
        layout=CudaKernelLayout(
            output_tile_shape=(4, 8),
            thread_shape=(4, 4),
        ).register_tile_layout(),
    )

    with pytest.raises(
        TypeError,
        match="incompatible CUDA register tile layouts",
    ):
        codegen.emit(
            SSAOp(
                opcode="store",
                operands=(pointers, values, None),
            )
        )


def test_cuda_register_tile_ref_broadcasts_register_axes() -> None:
    register_layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    ).register_tile_layout()

    rows = CudaRegisterTileRef(
        base="rows",
        layout=register_layout,
        broadcast_axes=(1,),
    )
    columns = CudaRegisterTileRef(
        base="columns",
        layout=register_layout,
        broadcast_axes=(0,),
    )

    assert rows.storage_shape == (2, 1)
    assert rows.storage_coordinates() == (
        (0, 0),
        (1, 0),
    )
    assert rows.element((0, 0)) == "rows_0_0"
    assert rows.element((0, 1)) == "rows_0_0"
    assert rows.element((1, 0)) == "rows_1_0"
    assert rows.element((1, 1)) == "rows_1_0"

    assert columns.storage_shape == (1, 2)
    assert columns.storage_coordinates() == (
        (0, 0),
        (0, 1),
    )
    assert columns.element((0, 0)) == "columns_0_0"
    assert columns.element((1, 0)) == "columns_0_0"
    assert columns.element((0, 1)) == "columns_0_1"
    assert columns.element((1, 1)) == "columns_0_1"


def test_cuda_codegen_expands_aranges_across_register_tile() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    )

    row_arange = SSAValue(
        id=0,
        ty=BlockType((8,), I32),
    )
    rows = SSAValue(
        id=1,
        ty=BlockType((8, 1), I32),
    )
    column_arange = SSAValue(
        id=2,
        ty=BlockType((8,), I32),
    )
    columns = SSAValue(
        id=3,
        ty=BlockType((1, 8), I32),
    )

    codegen.values[row_arange.id] = CudaArangeRef(
        start=0,
        end=8,
    )
    codegen.values[column_arange.id] = CudaArangeRef(
        start=0,
        end=8,
    )

    codegen.emit(
        SSAOp(
            opcode="expand_dims",
            operands=(row_arange,),
            result=rows,
            attrs={"axis": 1},
        )
    )
    codegen.emit(
        SSAOp(
            opcode="expand_dims",
            operands=(column_arange,),
            result=columns,
            attrs={"axis": 0},
        )
    )

    assert codegen.lines == [
        "    int v1_0_0 = tile_i;",
        "    int v1_1_0 = tile_i + 4;",
        "    int v3_0_0 = tile_j;",
        "    int v3_0_1 = tile_j + 4;",
    ]

    row_value = codegen.values[rows.id]
    column_value = codegen.values[columns.id]

    assert isinstance(row_value, CudaRegisterTileRef)
    assert isinstance(column_value, CudaRegisterTileRef)

    assert row_value.broadcast_axes == (1,)
    assert row_value.element((1, 1)) == "v1_1_0"

    assert column_value.broadcast_axes == (0,)
    assert column_value.element((1, 1)) == "v3_0_1"


def test_cuda_binary_operations_preserve_and_expand_register_broadcasts() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    )

    rows = SSAValue(
        id=0,
        ty=BlockType((8, 1), I32),
    )
    scaled_rows = SSAValue(
        id=1,
        ty=BlockType((8, 1), I32),
    )
    columns = SSAValue(
        id=2,
        ty=BlockType((1, 8), I32),
    )
    offsets = SSAValue(
        id=3,
        ty=BlockType((8, 8), I32),
    )

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[rows.id] = CudaRegisterTileRef(
        base="rows",
        layout=register_layout,
        broadcast_axes=(1,),
    )
    codegen.values[columns.id] = CudaRegisterTileRef(
        base="columns",
        layout=register_layout,
        broadcast_axes=(0,),
    )

    codegen.emit(
        SSAOp(
            opcode="mul",
            operands=(rows, Param("stride", I32)),
            result=scaled_rows,
        )
    )
    codegen.emit(
        SSAOp(
            opcode="add",
            operands=(scaled_rows, columns),
            result=offsets,
        )
    )

    assert codegen.lines == [
        "    int v1_0_0 = (rows_0_0 * stride);",
        "    int v1_1_0 = (rows_1_0 * stride);",
        "    int v3_0_0 = (v1_0_0 + columns_0_0);",
        "    int v3_0_1 = (v1_0_0 + columns_0_1);",
        "    int v3_1_0 = (v1_1_0 + columns_0_0);",
        "    int v3_1_1 = (v1_1_0 + columns_0_1);",
    ]

    scaled_value = codegen.values[scaled_rows.id]
    offset_value = codegen.values[offsets.id]

    assert isinstance(scaled_value, CudaRegisterTileRef)
    assert isinstance(offset_value, CudaRegisterTileRef)

    assert scaled_value.broadcast_axes == (1,)
    assert scaled_value.element((1, 1)) == "v1_1_0"

    assert offset_value.broadcast_axes == ()
    assert offset_value.element((1, 1)) == "v3_1_1"


def test_cuda_binary_operations_build_register_wise_mask() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    )

    rows = SSAValue(
        id=0,
        ty=BlockType((8, 1), I32),
    )
    columns = SSAValue(
        id=1,
        ty=BlockType((1, 8), I32),
    )
    row_mask = SSAValue(
        id=2,
        ty=BlockType((8, 1), BOOL),
    )
    column_mask = SSAValue(
        id=3,
        ty=BlockType((1, 8), BOOL),
    )
    mask = SSAValue(
        id=4,
        ty=BlockType((8, 8), BOOL),
    )

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[rows.id] = CudaRegisterTileRef(
        base="rows",
        layout=register_layout,
        broadcast_axes=(1,),
    )
    codegen.values[columns.id] = CudaRegisterTileRef(
        base="columns",
        layout=register_layout,
        broadcast_axes=(0,),
    )

    codegen.emit(
        SSAOp(
            opcode="cmp_lt",
            operands=(rows, Param("M", I32)),
            result=row_mask,
        )
    )
    codegen.emit(
        SSAOp(
            opcode="cmp_lt",
            operands=(columns, Param("N", I32)),
            result=column_mask,
        )
    )
    codegen.emit(
        SSAOp(
            opcode="and",
            operands=(row_mask, column_mask),
            result=mask,
        )
    )

    assert codegen.lines == [
        "    bool v2_0_0 = (rows_0_0 < M);",
        "    bool v2_1_0 = (rows_1_0 < M);",
        "    bool v3_0_0 = (columns_0_0 < N);",
        "    bool v3_0_1 = (columns_0_1 < N);",
        "    bool v4_0_0 = (v2_0_0 && v3_0_0);",
        "    bool v4_0_1 = (v2_0_0 && v3_0_1);",
        "    bool v4_1_0 = (v2_1_0 && v3_0_0);",
        "    bool v4_1_1 = (v2_1_0 && v3_0_1);",
    ]


def test_cuda_addptr_preserves_and_expands_register_broadcasts() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    )

    row_offsets = SSAValue(
        id=0,
        ty=BlockType((8, 1), I32),
    )
    column_offsets = SSAValue(
        id=1,
        ty=BlockType((1, 8), I32),
    )
    row_pointers = SSAValue(
        id=2,
        ty=BlockType((8, 1), PTR_F32),
    )
    output_pointers = SSAValue(
        id=3,
        ty=BlockType((8, 8), PTR_F32),
    )

    register_layout = codegen.layout.register_tile_layout()
    codegen.values[row_offsets.id] = CudaRegisterTileRef(
        base="rows",
        layout=register_layout,
        broadcast_axes=(1,),
    )
    codegen.values[column_offsets.id] = CudaRegisterTileRef(
        base="columns",
        layout=register_layout,
        broadcast_axes=(0,),
    )

    codegen.emit(
        SSAOp(
            opcode="addptr",
            operands=(Param("out", PTR_F32), row_offsets),
            result=row_pointers,
        )
    )
    codegen.emit(
        SSAOp(
            opcode="addptr",
            operands=(row_pointers, column_offsets),
            result=output_pointers,
        )
    )

    assert codegen.lines == [
        "    float* v2_0_0 = out + rows_0_0;",
        "    float* v2_1_0 = out + rows_1_0;",
        "    float* v3_0_0 = v2_0_0 + columns_0_0;",
        "    float* v3_0_1 = v2_0_0 + columns_0_1;",
        "    float* v3_1_0 = v2_1_0 + columns_0_0;",
        "    float* v3_1_1 = v2_1_0 + columns_0_1;",
    ]

    row_pointer_value = codegen.values[row_pointers.id]
    output_pointer_value = codegen.values[output_pointers.id]

    assert isinstance(row_pointer_value, CudaRegisterTileRef)
    assert isinstance(output_pointer_value, CudaRegisterTileRef)

    assert row_pointer_value.broadcast_axes == (1,)
    assert row_pointer_value.element((1, 1)) == "v2_1_0"

    assert output_pointer_value.broadcast_axes == ()
    assert output_pointer_value.element((1, 1)) == "v3_1_1"


def test_ast_frontend_lowers_multi_result_dot_to_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M = N = K = 8
    BM = BK = BN = 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    single_tile_dot_kernel.clear_cache()

    _, ssa_ops, cuda_src = single_tile_dot_kernel[(1, 1)](
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

    layout = cuda_kernel_layout(ssa_ops)

    assert layout.output_tile_shape == (8, 8)
    assert layout.thread_shape == (4, 8)
    assert layout.threads_per_block == 32
    assert layout.register_tile_layout().register_shape == (2, 1)

    dot = next(op for op in ssa_ops if isinstance(op, SSAOp) and op.opcode == "dot")
    store = next(op for op in ssa_ops if isinstance(op, SSAOp) and op.opcode == "store")

    assert dot.result is not None
    dot_id = dot.result.id

    pointer = store.operands[0]
    value = store.operands[1]
    mask = store.operands[2]

    assert isinstance(pointer, SSAValue)
    assert isinstance(value, SSAValue)
    assert isinstance(mask, SSAValue)
    assert value == dot.result

    assert "    int tile_i = threadIdx.x / 8;" in cuda_src
    assert "    int tile_j = threadIdx.x % 8;" in cuda_src

    assert f"    float v{dot_id}_0_0 = 0.0f;" in cuda_src
    assert f"    float v{dot_id}_1_0 = 0.0f;" in cuda_src

    assert (
        f"        v{dot_id}_0_0 += "
        f"dot_lhs_{dot_id}[(tile_i) * 8 + (dot_k_{dot_id})] * "
        f"dot_rhs_{dot_id}[(dot_k_{dot_id}) * 8 + (tile_j)];"
    ) in cuda_src

    assert (
        f"        v{dot_id}_1_0 += "
        f"dot_lhs_{dot_id}[(tile_i + 4) * 8 + (dot_k_{dot_id})] * "
        f"dot_rhs_{dot_id}[(dot_k_{dot_id}) * 8 + (tile_j)];"
    ) in cuda_src

    assert (
        f"    if (v{mask.id}_0_0) {{\n"
        f"        v{pointer.id}_0_0[0] = v{dot_id}_0_0;\n"
        "    }"
    ) in cuda_src

    assert (
        f"    if (v{mask.id}_1_0) {{\n"
        f"        v{pointer.id}_1_0[0] = v{dot_id}_1_0;\n"
        "    }"
    ) in cuda_src


@pytest.mark.execution
def test_shared_memory_dot_executes_multiple_results_per_thread(
    cp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 13, 11, 7
    BM, BK, BN = 8, 8, 8

    a_host = np.arange(M * K, dtype=np.float32).reshape(M, K) % 9 - 4
    b_host = np.arange(K * N, dtype=np.float32).reshape(K, N) % 7 - 3

    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    out = cp.zeros((M, N), dtype=cp.float32)

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    single_tile_dot_kernel.clear_cache()
    single_tile_dot_kernel[grid](
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

    expected = a_host @ b_host
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_runtime_k_loop_carries_multiple_results_per_thread(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 8, 8, 16
    BM, BK, BN = 8, 8, 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    tiled_matmul_kernel.clear_cache()

    _, ssa_ops, cuda_src = tiled_matmul_kernel[(1, 1)](
        a,
        b,
        out,
        M,
        N,
        K,
        BM=BM,
        BK=BK,
        BN=BN,
    )

    layout = cuda_kernel_layout(ssa_ops)

    assert layout.output_tile_shape == (8, 8)
    assert layout.thread_shape == (4, 8)
    assert layout.register_tile_layout().register_shape == (2, 1)

    loop = next(op for op in ssa_ops if isinstance(op, SSAForRange))
    dot = next(op for op in loop.body if isinstance(op, SSAOp) and op.opcode == "dot")
    accumulation = next(
        op
        for op in loop.body
        if (isinstance(op, SSAOp) and op.opcode == "add" and dot.result in op.operands)
    )

    assert dot.result is not None
    assert accumulation.result is not None
    assert len(loop.carried_inputs) == 1
    assert len(loop.results) == 1

    initial = loop.carried_inputs[0]
    assert isinstance(initial, SSAValue)

    dot_id = dot.result.id
    accumulation_id = accumulation.result.id
    accumulator_id = loop.results[0].id
    index_id = loop.index.id
    stage_name = f"dot_stage_{dot_id}"
    lhs_stage_size = BM * BK
    rhs_stage_size = BK * BN

    assert (f"    float v{accumulator_id}_0_0 = v{initial.id};") in cuda_src
    assert (f"    float v{accumulator_id}_1_0 = v{initial.id};") in cuda_src

    outer_loop = (
        f"    for (int v{index_id} = 0; v{index_id} < K; v{index_id} += {BK}) {{"
    )
    assert outer_loop in cuda_src
    assert (f"        int {stage_name} = (v{index_id} / {BK}) & 1;") in cuda_src

    assert f"        float v{dot_id}_0_0 = 0.0f;" in cuda_src
    assert f"        float v{dot_id}_1_0 = 0.0f;" in cuda_src

    assert (
        f"        v{dot_id}_1_0 += "
        f"dot_lhs_{dot_id}[(({stage_name}) * "
        f"{lhs_stage_size}) + "
        f"(tile_i + 4) * {BK} + "
        f"(dot_k_{dot_id})] * "
        f"dot_rhs_{dot_id}[(({stage_name}) * "
        f"{rhs_stage_size}) + "
        f"(dot_k_{dot_id}) * {BN} + "
        "(tile_j)];"
    ) in cuda_src

    expected_update = (
        f"        float v{accumulation_id}_0_0 = "
        f"(v{accumulator_id}_0_0 + v{dot_id}_0_0);\n"
        f"        float v{accumulation_id}_1_0 = "
        f"(v{accumulator_id}_1_0 + v{dot_id}_1_0);\n"
        f"        v{accumulator_id}_0_0 = "
        f"v{accumulation_id}_0_0;\n"
        f"        v{accumulator_id}_1_0 = "
        f"v{accumulation_id}_1_0;"
    )
    assert expected_update in cuda_src

    assert cuda_src.count("__syncthreads();") == 1


@pytest.mark.execution
def test_tiled_matmul_executes_multiple_results_per_thread(
    cp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 13, 11, 19
    BM, BK, BN = 8, 8, 8

    a_host = np.arange(M * K, dtype=np.float32).reshape(M, K) % 9 - 4
    b_host = np.arange(K * N, dtype=np.float32).reshape(K, N) % 7 - 3

    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    out = cp.zeros((M, N), dtype=cp.float32)

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    tiled_matmul_kernel.clear_cache()
    tiled_matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    cp.cuda.Stream.null.synchronize()

    expected = a_host @ b_host
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "broadcast_axes",
    [
        (1, 0),
        (0, 0),
        (-1,),
        (2,),
    ],
)
def test_cuda_register_tile_ref_rejects_invalid_broadcast_axes(
    broadcast_axes: tuple[int, ...],
) -> None:
    layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    ).register_tile_layout()

    with pytest.raises(ValueError, match="broadcast axes"):
        CudaRegisterTileRef(
            base="value",
            layout=layout,
            broadcast_axes=broadcast_axes,
        )


@pytest.mark.execution
def test_tiled_matmul_executes_two_dimensional_register_tile(
    cp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYTRITON_FRONTEND", "ast")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 17, 19, 15
    BM, BK, BN = 16, 8, 16

    a_host = np.arange(M * K, dtype=np.float32).reshape(M, K) % 9 - 4
    b_host = np.arange(K * N, dtype=np.float32).reshape(K, N) % 7 - 3

    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    out = cp.zeros((M, N), dtype=cp.float32)

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    tiled_matmul_kernel.clear_cache()
    _, ssa_ops, _ = tiled_matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    cp.cuda.Stream.null.synchronize()

    layout = cuda_kernel_layout(ssa_ops)

    assert layout.output_tile_shape == (16, 16)
    assert layout.thread_shape == (4, 8)
    assert layout.register_tile_layout().register_shape == (4, 2)

    expected = a_host @ b_host
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_cuda_shared_buffer_supports_padded_rows() -> None:
    buffer = CudaSharedBuffer(
        name="dot_lhs_7",
        logical_shape=(4, 16),
        element_ty=F32,
        row_padding=1,
    )

    assert buffer.rows == 4
    assert buffer.columns == 16
    assert buffer.row_stride == 17
    assert buffer.size == 68
    assert buffer.nbytes == 272
    assert buffer.offset(3, 15) == 66
    assert buffer.element("row", "column") == ("dot_lhs_7[(row) * 17 + (column)]")


def test_padded_lhs_rows_use_distinct_shared_memory_banks() -> None:
    unpadded = CudaSharedBuffer(
        name="unpadded",
        logical_shape=(4, 16),
        element_ty=F32,
    )
    padded = CudaSharedBuffer(
        name="padded",
        logical_shape=(4, 16),
        element_ty=F32,
        row_padding=1,
    )

    unpadded_banks = tuple(unpadded.offset(row, 0) % 32 for row in range(4))
    padded_banks = tuple(padded.offset(row, 0) % 32 for row in range(4))

    assert unpadded_banks == (0, 16, 0, 16)
    assert padded_banks == (0, 17, 2, 19)
    assert len(set(padded_banks)) == 4


@pytest.mark.parametrize(
    ("columns", "simultaneous_rows", "expected_padding"),
    [
        (16, 4, 1),
        (16, 2, 0),
        (8, 4, 0),
        (8, 8, 1),
        (32, 2, 1),
        (7, 4, 0),
    ],
)
def test_cuda_f32_shared_row_padding(
    columns: int,
    simultaneous_rows: int,
    expected_padding: int,
) -> None:
    assert (
        cuda_f32_shared_row_padding(
            columns=columns,
            simultaneous_rows=simultaneous_rows,
        )
        == expected_padding
    )


@pytest.mark.parametrize(
    ("columns", "simultaneous_rows"),
    [
        (0, 4),
        (-1, 4),
        (16, 0),
        (16, 33),
    ],
)
def test_cuda_f32_shared_row_padding_rejects_invalid_arguments(
    columns: int,
    simultaneous_rows: int,
) -> None:
    with pytest.raises(ValueError):
        cuda_f32_shared_row_padding(
            columns=columns,
            simultaneous_rows=simultaneous_rows,
        )


def test_cuda_shared_buffer_supports_multiple_stages() -> None:
    buffer = CudaSharedBuffer(
        name="dot_lhs_7",
        logical_shape=(4, 16),
        element_ty=F32,
        row_padding=1,
        stage_count=2,
    )

    assert buffer.stage_size == 68
    assert buffer.size == 136
    assert buffer.nbytes == 544

    assert buffer.offset(0, 0, stage=0) == 0
    assert buffer.offset(3, 15, stage=0) == 66
    assert buffer.offset(0, 0, stage=1) == 68
    assert buffer.offset(3, 15, stage=1) == 134

    assert buffer.element(
        "row",
        "column",
        stage="stage",
    ) == ("dot_lhs_7[((stage) * 68) + (row) * 17 + (column)]")


def test_multi_stage_shared_buffer_requires_stage_expression() -> None:
    buffer = CudaSharedBuffer(
        name="tile",
        logical_shape=(4, 16),
        element_ty=F32,
        stage_count=2,
    )

    with pytest.raises(
        ValueError,
        match="requires a stage expression",
    ):
        buffer.element("row", "column")


@pytest.mark.parametrize("stage_count", [0, -1, True])
def test_cuda_shared_buffer_rejects_invalid_stage_count(
    stage_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="stage count must be a positive integer",
    ):
        CudaSharedBuffer(
            name="tile",
            logical_shape=(4, 16),
            element_ty=F32,
            stage_count=stage_count,
        )


def test_cuda_codegen_uses_same_stage_for_loads_and_dot() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    buffers = codegen.emit_dot_operand_staging(
        dot_result_id=7,
        lhs_shape=(4, 16),
        rhs_shape=(16, 8),
        element_ty=F32,
        lhs_source=CudaGlobalTile(
            base="a",
            row_offset="blockIdx.x * 4",
            column_offset="k_base",
            row_stride="K",
            row_bound="M",
            column_bound="K",
        ),
        rhs_source=CudaGlobalTile(
            base="b",
            row_offset="k_base",
            column_offset="blockIdx.y * 8",
            row_stride="N",
            row_bound="K",
            column_bound="N",
        ),
        stage_count=2,
        stage="stage",
    )

    result = SSAValue(
        id=7,
        ty=BlockType((4, 8), F32),
    )
    codegen.emit_dot_from_shared_memory(
        result,
        buffers,
        stage="stage",
    )

    assert buffers.stage_count == 2
    assert codegen.shared_memory_bytes == (136 + 256) * 4
    assert codegen.shared_lines == [
        "    __shared__ float dot_lhs_7[136];",
        "    __shared__ float dot_rhs_7[256];",
    ]

    assert codegen.lines[7] == (
        "        dot_lhs_7[((stage) * 68) + "
        "(dot_lhs_7_row) * 17 + "
        "(dot_lhs_7_column)] = "
        "dot_lhs_7_in_bounds ? "
        "a[dot_lhs_7_source_index] : 0.0f;"
    )
    assert codegen.lines[16] == (
        "        dot_rhs_7[((stage) * 128) + "
        "(dot_rhs_7_row) * 8 + "
        "(dot_rhs_7_column)] = "
        "dot_rhs_7_in_bounds ? "
        "b[dot_rhs_7_source_index] : 0.0f;"
    )
    assert (
        "        v7 += "
        "dot_lhs_7[((stage) * 68) + "
        "(tile_i) * 17 + (dot_k_7)] * "
        "dot_rhs_7[((stage) * 128) + "
        "(dot_k_7) * 8 + (tile_j)];"
    ) in codegen.lines


def test_dot_shared_buffers_reject_mismatched_stage_counts() -> None:
    with pytest.raises(
        ValueError,
        match="matching stage counts",
    ):
        CudaDotSharedBuffers(
            lhs=CudaSharedBuffer(
                name="lhs",
                logical_shape=(4, 16),
                element_ty=F32,
                stage_count=2,
            ),
            rhs=CudaSharedBuffer(
                name="rhs",
                logical_shape=(16, 8),
                element_ty=F32,
            ),
        )


def test_double_buffering_matcher_accepts_canonical_k_loop() -> None:
    M, N, K = 4, 8, 32
    BM, BK, BN = 4, 16, 8

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)

    bound = tiled_matmul_kernel.signature.bind(
        a,
        b,
        out,
        M,
        N,
        K,
        BM=BM,
        BK=BK,
        BN=BN,
    )
    runtime_params = make_runtime_params(
        tiled_matmul_kernel.signature,
        bound.arguments,
    )
    traced_ops, _ = trace_ast(
        tiled_matmul_kernel.fn,
        tiled_matmul_kernel.signature,
        bound.arguments,
        runtime_params=runtime_params,
    )
    ssa_ops = SSALowering().lower(traced_ops)

    loop = next(item for item in ssa_ops if isinstance(item, SSAForRange))
    dot = next(
        item for item in loop.body if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None

    analysis = CudaDotStagingAnalyzer(SSADefinitions(ssa_ops)).analyze()

    assert match_cuda_dot_double_buffering(
        loop,
        analysis,
    ) == CudaDotDoubleBufferingPlan(
        dot_result_id=dot.result.id,
    )

    loop.step = Const(BK // 2)

    assert (
        match_cuda_dot_double_buffering(
            loop,
            analysis,
        )
        is None
    )

    loop.step = Const(BK)
    loop.start = Const(BK)

    assert (
        match_cuda_dot_double_buffering(
            loop,
            analysis,
        )
        is None
    )

    loop.start = Const(False)

    assert (
        match_cuda_dot_double_buffering(
            loop,
            analysis,
        )
        is None
    )

    loop.start = Const(0)
    accumulation = next(
        item
        for item in loop.body
        if (
            isinstance(item, SSAOp)
            and item.opcode == "add"
            and dot.result in item.operands
        )
    )
    accumulation.opcode = "mul"

    assert (
        match_cuda_dot_double_buffering(
            loop,
            analysis,
        )
        is None
    )

    accumulation.opcode = "add"
    loop.start = Const(0)
    loop.step = Const(BK // 2)

    fallback_codegen = SSACUDACodegen()
    fallback_cuda_src = fallback_codegen.generate(
        kernel_name="fallback_matmul_kernel",
        ssa_ops=ssa_ops,
        params=runtime_params,
    )

    assert f"dot_stage_{dot.result.id}" not in fallback_cuda_src
    assert fallback_cuda_src.count("__syncthreads();") == 2
    assert fallback_codegen.shared_memory_bytes == (BM * (BK + 1) + BK * BN) * 4
    assert (
        f"    __shared__ float dot_lhs_{dot.result.id}[{BM * (BK + 1)}];"
    ) in fallback_cuda_src
    assert (
        f"    __shared__ float dot_rhs_{dot.result.id}[{BK * BN}];"
    ) in fallback_cuda_src


def test_double_buffering_rejects_shared_memory_over_budget() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(64, 64),
        thread_shape=(4, 8),
    )

    source = CudaGlobalTile(
        base="x",
        row_offset="0",
        column_offset="0",
        row_stride="64",
        row_bound="64",
        column_bound="64",
    )

    with pytest.raises(
        ValueError,
        match="exceeding the conservative 49152-byte limit",
    ):
        codegen.emit_dot_operand_staging(
            dot_result_id=7,
            lhs_shape=(64, 64),
            rhs_shape=(64, 64),
            element_ty=F32,
            lhs_source=source,
            rhs_source=source,
            stage_count=2,
            stage="stage",
        )

    assert codegen.shared_memory_bytes == 0
    assert codegen.shared_lines == []
    assert codegen.lines == []
