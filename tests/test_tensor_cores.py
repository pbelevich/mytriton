import itertools

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.block_shapes import (
    CudaKernelLayout,
    CudaMmaM16N8K8Layout,
    cuda_mma_m16n8k8_dot_result_ids,
    cuda_threads_per_block,
    is_cuda_mma_m16n8k8_dot,
)
from mytriton.cuda_codegen import (
    CudaMmaAccumulatorRef,
    CudaRegisterTileRef,
    SSACUDACodegen,
)
from mytriton.cuda_dot_staging import (
    CudaDotSharedBuffers,
    CudaSharedBuffer,
)
from mytriton.cuda_target import CudaTarget
from mytriton.ssa import SSAForRange, SSAOp, SSAValue
from mytriton.trace import (
    BF16,
    BOOL,
    F16,
    F32,
    I32,
    PTR_F32,
    BlockType,
    Const,
    Type,
)


@triton.jit
def tensor_core_dot_kernel(
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
    tl.store(
        output_pointers,
        result,
        mask=output_mask,
    )


def test_mma_m16n8k8_layout_shapes() -> None:
    layout = CudaMmaM16N8K8Layout()

    assert layout.lhs_shape == (16, 8)
    assert layout.rhs_shape == (8, 8)
    assert layout.result_shape == (16, 8)
    assert layout.threads_per_warp == 32
    assert layout.thread_shape == (4, 8)


def test_mma_m16n8k8_fragment_coordinates() -> None:
    layout = CudaMmaM16N8K8Layout()

    assert layout.lhs_coordinates(0) == (
        (0, 0),
        (0, 1),
        (8, 0),
        (8, 1),
    )
    assert layout.rhs_coordinates(0) == (
        (0, 0),
        (1, 0),
    )
    assert layout.accumulator_coordinates(0) == (
        (0, 0),
        (0, 1),
        (8, 0),
        (8, 1),
    )

    assert layout.lhs_coordinates(31) == (
        (7, 6),
        (7, 7),
        (15, 6),
        (15, 7),
    )
    assert layout.rhs_coordinates(31) == (
        (6, 7),
        (7, 7),
    )
    assert layout.accumulator_coordinates(31) == (
        (7, 6),
        (7, 7),
        (15, 6),
        (15, 7),
    )


def test_mma_m16n8k8_fragments_cover_every_matrix_element_once() -> None:
    layout = CudaMmaM16N8K8Layout()

    lhs_coordinates = [
        coordinate
        for lane in range(layout.threads_per_warp)
        for coordinate in layout.lhs_coordinates(lane)
    ]
    rhs_coordinates = [
        coordinate
        for lane in range(layout.threads_per_warp)
        for coordinate in layout.rhs_coordinates(lane)
    ]
    accumulator_coordinates = [
        coordinate
        for lane in range(layout.threads_per_warp)
        for coordinate in layout.accumulator_coordinates(lane)
    ]

    assert len(lhs_coordinates) == len(set(lhs_coordinates))
    assert set(lhs_coordinates) == set(itertools.product(range(16), range(8)))

    assert len(rhs_coordinates) == len(set(rhs_coordinates))
    assert set(rhs_coordinates) == set(itertools.product(range(8), range(8)))

    assert len(accumulator_coordinates) == len(set(accumulator_coordinates))
    assert set(accumulator_coordinates) == set(itertools.product(range(16), range(8)))


@pytest.mark.parametrize("lane", [-1, 32])
def test_mma_m16n8k8_rejects_invalid_lane(lane: int) -> None:
    layout = CudaMmaM16N8K8Layout()

    with pytest.raises(
        ValueError,
        match="MMA lane must be between 0 and 31",
    ):
        layout.accumulator_coordinates(lane)


def make_dot_op(
    lhs_ty: Type,
    rhs_ty: Type,
    result_ty: Type,
    *,
    result_id: int = 2,
) -> SSAOp:
    lhs = SSAValue(id=result_id - 2, ty=lhs_ty)
    rhs = SSAValue(id=result_id - 1, ty=rhs_ty)
    result = SSAValue(id=result_id, ty=result_ty)

    return SSAOp(
        opcode="dot",
        operands=(lhs, rhs),
        result=result,
    )


def test_recognizes_f16_m16n8k8_dot() -> None:
    op = make_dot_op(
        BlockType((16, 8), F16),
        BlockType((8, 8), F16),
        BlockType((16, 8), F32),
    )

    assert is_cuda_mma_m16n8k8_dot(op)


def test_recognizes_bf16_m16n8k8_dot_when_operand_type_is_enabled() -> None:
    op = make_dot_op(
        BlockType((16, 8), BF16),
        BlockType((8, 8), BF16),
        BlockType((16, 8), F32),
    )

    assert is_cuda_mma_m16n8k8_dot(
        op,
        operand_types=frozenset((F16, BF16)),
    )


@pytest.mark.parametrize(
    ("lhs_ty", "rhs_ty", "result_ty"),
    [
        (
            BlockType((16, 8), F32),
            BlockType((8, 8), F32),
            BlockType((16, 8), F32),
        ),
        (
            BlockType((16, 8), BF16),
            BlockType((8, 8), BF16),
            BlockType((16, 8), F32),
        ),
        (
            BlockType((8, 8), F16),
            BlockType((8, 8), F16),
            BlockType((8, 8), F32),
        ),
        (
            BlockType((16, 16), F16),
            BlockType((16, 8), F16),
            BlockType((16, 8), F32),
        ),
        (
            BlockType((16, 8), F16),
            BlockType((8, 16), F16),
            BlockType((16, 16), F32),
        ),
    ],
)
def test_rejects_non_m16n8k8_dot(
    lhs_ty: Type,
    rhs_ty: Type,
    result_ty: Type,
) -> None:
    op = make_dot_op(lhs_ty, rhs_ty, result_ty)

    assert not is_cuda_mma_m16n8k8_dot(op)


def test_collects_mma_dot_results_recursively() -> None:
    top_level_dot = make_dot_op(
        BlockType((16, 8), F16),
        BlockType((8, 8), F16),
        BlockType((16, 8), F32),
        result_id=2,
    )
    cuda_core_dot = make_dot_op(
        BlockType((16, 8), F32),
        BlockType((8, 8), F32),
        BlockType((16, 8), F32),
        result_id=5,
    )
    nested_dot = make_dot_op(
        BlockType((16, 8), F16),
        BlockType((8, 8), F16),
        BlockType((16, 8), F32),
        result_id=9,
    )

    loop = SSAForRange(
        index=SSAValue(id=6, ty=I32),
        start=Const(0),
        stop=Const(8),
        step=Const(8),
        carried_inputs=(),
        carried_args=(),
        body=[nested_dot],
        yields=(),
        results=(),
    )

    result_ids = cuda_mma_m16n8k8_dot_result_ids(
        [
            top_level_dot,
            cuda_core_dot,
            loop,
        ]
    )

    assert result_ids == frozenset({2, 9})


def test_collects_no_results_without_compatible_dot() -> None:
    cuda_core_dot = make_dot_op(
        BlockType((16, 8), F32),
        BlockType((8, 8), F32),
        BlockType((16, 8), F32),
    )

    result_ids = cuda_mma_m16n8k8_dot_result_ids([cuda_core_dot])

    assert result_ids == frozenset()


def test_collects_bf16_mma_result_when_operand_type_is_enabled() -> None:
    bf16_dot = make_dot_op(
        BlockType((16, 8), BF16),
        BlockType((8, 8), BF16),
        BlockType((16, 8), F32),
    )

    result_ids = cuda_mma_m16n8k8_dot_result_ids(
        [bf16_dot],
        operand_types=frozenset((F16, BF16)),
    )

    assert result_ids == frozenset((2,))


def test_mma_accumulator_ref_names_four_lane_registers() -> None:
    ref = CudaMmaAccumulatorRef(
        base="v7",
        layout=CudaMmaM16N8K8Layout(),
    )

    assert ref.elements_per_lane == 4
    assert ref.element(0) == "v7_0"
    assert ref.element(1) == "v7_1"
    assert ref.element(2) == "v7_2"
    assert ref.element(3) == "v7_3"
    assert ref.elements() == (
        "v7_0",
        "v7_1",
        "v7_2",
        "v7_3",
    )


@pytest.mark.parametrize("index", [-1, 4])
def test_mma_accumulator_ref_rejects_invalid_element(
    index: int,
) -> None:
    ref = CudaMmaAccumulatorRef(
        base="v7",
        layout=CudaMmaM16N8K8Layout(),
    )

    with pytest.raises(
        ValueError,
        match="MMA accumulator element must be between 0 and 3",
    ):
        ref.element(index)


def test_cuda_codegen_packs_two_f16_values_for_ptx() -> None:
    codegen = SSACUDACodegen()

    packed = codegen.pack_f16x2(
        "shared_tile[first]",
        "shared_tile[second]",
    )

    assert packed == (
        "(static_cast<unsigned>("
        "__half_as_ushort(shared_tile[first])) | "
        "(static_cast<unsigned>("
        "__half_as_ushort(shared_tile[second])) << 16))"
    )
    assert "#include <cuda_fp16.h>" in codegen.required_headers


def test_cuda_codegen_packs_two_bf16_values_for_ptx() -> None:
    codegen = SSACUDACodegen(target=CudaTarget.from_chip("sm_80"))

    packed = codegen.pack_bf16x2(
        "shared_tile[first]",
        "shared_tile[second]",
    )

    assert packed == (
        "(static_cast<unsigned>("
        "static_cast<__nv_bfloat16_raw>(shared_tile[first]).x) | "
        "(static_cast<unsigned>("
        "static_cast<__nv_bfloat16_raw>(shared_tile[second]).x) << 16))"
    )
    assert "#include <cuda_bf16.h>" in codegen.required_headers


def test_cuda_codegen_loads_mma_operand_registers_from_shared_memory() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(16, 8),
        thread_shape=(4, 8),
    )

    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(16, 8),
            element_ty=F16,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(8, 8),
            element_ty=F16,
        ),
    )

    lhs_registers, rhs_register = codegen.emit_mma_operand_registers(
        result_id=7,
        buffers=buffers,
    )

    assert lhs_registers == (
        "mma_a_7_0",
        "mma_a_7_1",
    )
    assert rhs_register == "mma_b_7_0"

    assert codegen.lines[:2] == [
        "    int mma_group_7 = threadIdx.x >> 2;",
        "    int mma_thread_7 = threadIdx.x & 3;",
    ]

    cuda = "\n".join(codegen.lines)

    assert ("dot_lhs_7[(mma_group_7) * 8 + (mma_thread_7 * 2)]") in cuda
    assert ("dot_lhs_7[(mma_group_7) * 8 + (mma_thread_7 * 2 + 1)]") in cuda
    assert ("dot_lhs_7[(mma_group_7 + 8) * 8 + (mma_thread_7 * 2)]") in cuda
    assert ("dot_lhs_7[(mma_group_7 + 8) * 8 + (mma_thread_7 * 2 + 1)]") in cuda

    assert ("dot_rhs_7[(mma_thread_7 * 2) * 8 + (mma_group_7)]") in cuda
    assert ("dot_rhs_7[(mma_thread_7 * 2 + 1) * 8 + (mma_group_7)]") in cuda

    assert cuda.count("__half_as_ushort") == 6
    assert len(codegen.lines) == 5


def test_cuda_codegen_loads_bf16_mma_operand_registers() -> None:
    codegen = SSACUDACodegen(target=CudaTarget.from_chip("sm_80"))
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(16, 8),
        thread_shape=(4, 8),
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(16, 8),
            element_ty=BF16,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(8, 8),
            element_ty=BF16,
        ),
    )

    codegen.emit_mma_operand_registers(
        result_id=7,
        buffers=buffers,
    )

    cuda = "\n".join(codegen.lines)
    assert cuda.count("static_cast<__nv_bfloat16_raw>") == 6
    assert "__half_as_ushort" not in cuda
    assert "#include <cuda_bf16.h>" in codegen.required_headers


def test_cuda_codegen_emits_mma_m16n8k8_instruction() -> None:
    codegen = SSACUDACodegen()
    result = SSAValue(
        id=7,
        ty=BlockType((16, 8), F32),
    )

    result_ref = codegen.emit_mma_m16n8k8(
        result=result,
        lhs_registers=(
            "mma_a_7_0",
            "mma_a_7_1",
        ),
        rhs_register="mma_b_7_0",
    )

    assert result_ref == CudaMmaAccumulatorRef(
        base="v7",
        layout=CudaMmaM16N8K8Layout(),
    )
    assert codegen.values[result.id] == result_ref

    assert codegen.lines == [
        "    float v7_0 = 0.0f;",
        "    float v7_1 = 0.0f;",
        "    float v7_2 = 0.0f;",
        "    float v7_3 = 0.0f;",
        "    asm volatile(",
        ('        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "'),
        '        "{%0, %1, %2, %3}, "',
        '        "{%4, %5}, "',
        '        "{%6}, "',
        '        "{%0, %1, %2, %3};"',
        ('        : "+f"(v7_0), "+f"(v7_1), "+f"(v7_2), "+f"(v7_3)'),
        ('        : "r"(mma_a_7_0), "r"(mma_a_7_1), "r"(mma_b_7_0)'),
        "    );",
    ]


def test_cuda_codegen_emits_bf16_mma_m16n8k8_instruction() -> None:
    codegen = SSACUDACodegen(target=CudaTarget.from_chip("sm_80"))
    result = SSAValue(
        id=7,
        ty=BlockType((16, 8), F32),
    )

    codegen.emit_mma_m16n8k8(
        result=result,
        lhs_registers=("mma_a_7_0", "mma_a_7_1"),
        rhs_register="mma_b_7_0",
        operand_ty=BF16,
    )

    cuda = "\n".join(codegen.lines)
    assert "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32" in cuda
    assert ".f32.f16.f16.f32" not in cuda


def test_cuda_codegen_rejects_bf16_mma_before_sm80() -> None:
    codegen = SSACUDACodegen(target=CudaTarget.from_chip("sm_75"))
    result = SSAValue(
        id=7,
        ty=BlockType((16, 8), F32),
    )

    with pytest.raises(TypeError, match=r"bf16 MMA requires sm_80\+"):
        codegen.emit_mma_m16n8k8(
            result=result,
            lhs_registers=("mma_a_7_0", "mma_a_7_1"),
            rhs_register="mma_b_7_0",
            operand_ty=BF16,
        )


def test_cuda_codegen_computes_mma_from_shared_memory() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(16, 8),
        thread_shape=(4, 8),
    )

    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(16, 8),
            element_ty=F16,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(8, 8),
            element_ty=F16,
        ),
    )
    result = SSAValue(
        id=7,
        ty=BlockType((16, 8), F32),
    )

    accumulator = CudaMmaAccumulatorRef(
        base="v3",
        layout=CudaMmaM16N8K8Layout(),
    )

    result_ref = codegen.emit_mma_from_shared_memory(
        result=result,
        buffers=buffers,
        accumulator=accumulator,
    )

    assert isinstance(result_ref, CudaMmaAccumulatorRef)
    assert codegen.values[result.id] == result_ref

    cuda = "\n".join(codegen.lines)

    assert "    float v7_0 = v3_0;" in codegen.lines
    assert "    float v7_1 = v3_1;" in codegen.lines
    assert "    float v7_2 = v3_2;" in codegen.lines
    assert "    float v7_3 = v3_3;" in codegen.lines

    operand_position = cuda.index("unsigned mma_a_7_0")
    instruction_position = cuda.index("mma.sync.aligned.m16n8k8")
    barrier_position = cuda.rindex("__syncthreads();")

    assert operand_position < instruction_position < barrier_position
    assert cuda.count("mma.sync.aligned.m16n8k8") == 1
    assert "for (int dot_k_7" not in cuda
    assert codegen.lines[-1] == "    __syncthreads();"


def test_cuda_codegen_spills_mma_accumulator_to_shared_memory() -> None:
    codegen = SSACUDACodegen()
    value = CudaMmaAccumulatorRef(
        base="v7",
        layout=CudaMmaM16N8K8Layout(),
    )

    buffer = codegen.emit_mma_accumulator_spill(
        value_id=7,
        value=value,
    )

    assert buffer == CudaSharedBuffer(
        name="mma_result_7",
        logical_shape=(16, 8),
        element_ty=F32,
    )
    assert buffer.nbytes == 16 * 8 * 4
    assert codegen.shared_memory_bytes == 16 * 8 * 4
    assert codegen.shared_lines == ["    __shared__ float mma_result_7[128];"]

    assert codegen.lines == [
        "    int mma_store_group_7 = threadIdx.x >> 2;",
        "    int mma_store_thread_7 = threadIdx.x & 3;",
        (
            "    mma_result_7[(mma_store_group_7) * 8 + "
            "(mma_store_thread_7 * 2)] = v7_0;"
        ),
        (
            "    mma_result_7[(mma_store_group_7) * 8 + "
            "(mma_store_thread_7 * 2 + 1)] = v7_1;"
        ),
        (
            "    mma_result_7[(mma_store_group_7 + 8) * 8 + "
            "(mma_store_thread_7 * 2)] = v7_2;"
        ),
        (
            "    mma_result_7[(mma_store_group_7 + 8) * 8 + "
            "(mma_store_thread_7 * 2 + 1)] = v7_3;"
        ),
        "    __syncthreads();",
    ]


def test_cuda_store_redistributes_mma_fragment_through_shared_memory() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(16, 8),
        thread_shape=(4, 8),
    )
    register_layout = codegen.layout.register_tile_layout()

    pointers = SSAValue(
        id=0,
        ty=BlockType((16, 8), PTR_F32),
    )
    value = SSAValue(
        id=1,
        ty=BlockType((16, 8), F32),
    )
    mask = SSAValue(
        id=2,
        ty=BlockType((16, 8), BOOL),
    )

    codegen.values[pointers.id] = CudaRegisterTileRef(
        base="ptr",
        layout=register_layout,
    )
    codegen.values[value.id] = CudaMmaAccumulatorRef(
        base="v1",
        layout=CudaMmaM16N8K8Layout(),
    )
    codegen.values[mask.id] = CudaRegisterTileRef(
        base="mask",
        layout=register_layout,
    )

    codegen.emit_store(
        SSAOp(
            opcode="store",
            operands=(pointers, value, mask),
        )
    )

    assert codegen.shared_lines == ["    __shared__ float mma_result_1[128];"]

    cuda = "\n".join(codegen.lines)

    expected_stores = [
        (
            "    if (mask_0_0) {\n"
            "        ptr_0_0[0] = "
            "mma_result_1[(tile_i) * 8 + (tile_j)];\n"
            "    }"
        ),
        (
            "    if (mask_1_0) {\n"
            "        ptr_1_0[0] = "
            "mma_result_1[(tile_i + 4) * 8 + (tile_j)];\n"
            "    }"
        ),
        (
            "    if (mask_2_0) {\n"
            "        ptr_2_0[0] = "
            "mma_result_1[(tile_i + 8) * 8 + (tile_j)];\n"
            "    }"
        ),
        (
            "    if (mask_3_0) {\n"
            "        ptr_3_0[0] = "
            "mma_result_1[(tile_i + 12) * 8 + (tile_j)];\n"
            "    }"
        ),
    ]

    for expected_store in expected_stores:
        assert expected_store in cuda

    assert cuda.count("if (mask_") == 4


@pytest.mark.codegen
def test_f16_m16n8k8_dot_lowers_to_tensor_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 16, 8, 8
    BM, BK, BN = 16, 8, 8

    a = np.zeros((M, K), dtype=np.float16)
    b = np.zeros((K, N), dtype=np.float16)
    out = np.zeros((M, N), dtype=np.float32)

    tensor_core_dot_kernel.clear_cache()
    _, ssa_ops, cuda_src = tensor_core_dot_kernel[(1, 1)](
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

    assert cuda_threads_per_block(ssa_ops) == 32

    assert cuda_src.startswith("#include <cuda_fp16.h>\n\n")
    assert (
        "void tensor_core_dot_kernel("
        "__half* a, __half* b, float* out, "
        "int M, int N, int K, int k_base)"
    ) in cuda_src

    assert "__shared__ __half dot_lhs_" in cuda_src
    assert "__shared__ __half dot_rhs_" in cuda_src
    assert "__shared__ float mma_result_" in cuda_src

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 1
    assert cuda_src.count("__half_as_ushort") == 6

    assert "for (int dot_k_" not in cuda_src
    assert "__half2float(dot_lhs_" not in cuda_src
    assert "__half2float(dot_rhs_" not in cuda_src


@pytest.mark.codegen
def test_bf16_m16n8k8_dot_lowers_to_tensor_core_for_sm80(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mytriton.compiler as compiler

    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    monkeypatch.setattr(
        compiler,
        "cuda_execution_required",
        lambda runtime_args, *, backend_name: True,
    )
    monkeypatch.setattr(compiler, "cuda_chip", lambda runtime_args: "sm_80")
    monkeypatch.setattr(compiler, "execute_cuda_if_needed", lambda **kwargs: None)

    M, N, K = 16, 8, 8
    BM, BK, BN = 16, 8, 8
    a = torch.zeros((M, K), dtype=torch.bfloat16)
    b = torch.zeros((K, N), dtype=torch.bfloat16)
    out = torch.zeros((M, N), dtype=torch.float32)

    tensor_core_dot_kernel.clear_cache()
    _, ssa_ops, cuda_src = tensor_core_dot_kernel[(1, 1)](
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

    assert cuda_threads_per_block(ssa_ops) == 32
    assert cuda_src.startswith("#include <cuda_bf16.h>\n\n")
    assert "#include <cuda_fp16.h>" not in cuda_src
    assert "__shared__ __nv_bfloat16 dot_lhs_" in cuda_src
    assert "__shared__ __nv_bfloat16 dot_rhs_" in cuda_src
    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32") == 1
    assert cuda_src.count("static_cast<__nv_bfloat16_raw>") == 6
    assert "for (int dot_k_" not in cuda_src
    assert "__bfloat162float(dot_lhs_" not in cuda_src


@pytest.mark.codegen
def test_bf16_m16n8k8_dot_uses_cuda_core_fallback_for_sm75(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 16, 8, 8
    BM, BK, BN = 16, 8, 8
    a = torch.zeros((M, K), dtype=torch.bfloat16)
    b = torch.zeros((K, N), dtype=torch.bfloat16)
    out = torch.zeros((M, N), dtype=torch.float32)

    tensor_core_dot_kernel.clear_cache()
    _, _, cuda_src = tensor_core_dot_kernel[(1, 1)](
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

    assert "mma.sync.aligned.m16n8k8" not in cuda_src
    assert "for (int dot_k_" in cuda_src
    assert "__bfloat162float(dot_lhs_" in cuda_src
    assert "__bfloat162float(dot_rhs_" in cuda_src


@pytest.mark.execution
def test_f16_m16n8k8_tensor_core_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = int(cp.cuda.Device().compute_capability)
    if capability < 75:
        pytest.skip("m16n8k8 f16 MMA requires compute capability 7.5+")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 16, 8, 8
    BM, BK, BN = 16, 8, 8

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
    out = cp.full(
        (M, N),
        cp.nan,
        dtype=cp.float32,
    )

    tensor_core_dot_kernel.clear_cache()
    _, _, cuda_src = tensor_core_dot_kernel[(1, 1)](
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

    assert ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") in cuda_src
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=2e-3,
        atol=2e-3,
    )


@triton.jit
def tensor_core_matmul_kernel(
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

    acc = tl.zeros(
        (BM, BN),
        tl.float32,
    )

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
    tl.store(
        output_pointers,
        acc,
        mask=output_mask,
    )


@pytest.mark.codegen
def test_f16_tiled_matmul_lowers_k_loop_to_tensor_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 16, 8, 16
    BM, BK, BN = 16, 8, 8

    a = np.zeros((M, K), dtype=np.float16)
    b = np.zeros((K, N), dtype=np.float16)
    out = np.zeros((M, N), dtype=np.float32)

    tensor_core_matmul_kernel.clear_cache()
    _, ssa_ops, cuda_src = tensor_core_matmul_kernel[(1, 1)](
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

    assert cuda_threads_per_block(ssa_ops) == 32

    loop = next(item for item in ssa_ops if isinstance(item, SSAForRange))
    dot = next(
        item for item in loop.body if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None
    assert dot.result.id in (cuda_mma_m16n8k8_dot_result_ids(ssa_ops))

    accumulation = next(
        item
        for item in loop.body
        if (
            isinstance(item, SSAOp)
            and item.opcode == "add"
            and dot.result in item.operands
        )
    )
    assert accumulation.result is not None

    assert len(loop.carried_inputs) == 1
    assert len(loop.results) == 1

    initial_accumulator = loop.carried_inputs[0]
    loop_result = loop.results[0]

    assert isinstance(initial_accumulator, SSAValue)

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 1
    assert "for (int dot_k_" not in cuda_src

    assert f"mma_a_{dot.result.id}_0" in cuda_src
    assert f"mma_b_{dot.result.id}_0" in cuda_src

    for index in range(4):
        assert (
            f"float v{loop_result.id}_{index} = v{initial_accumulator.id};"
        ) in cuda_src
        assert (
            f"float v{dot.result.id}_{index} = v{loop_result.id}_{index};"
        ) in cuda_src
        assert (f"v{loop_result.id}_{index} = v{dot.result.id}_{index};") in cuda_src

    assert f"v{accumulation.result.id}_" not in cuda_src


def test_cuda_codegen_emits_mma_with_existing_accumulator() -> None:
    codegen = SSACUDACodegen()
    result = SSAValue(
        id=7,
        ty=BlockType((16, 8), F32),
    )
    accumulator = CudaMmaAccumulatorRef(
        base="v3",
        layout=CudaMmaM16N8K8Layout(),
    )

    result_ref = codegen.emit_mma_m16n8k8(
        result=result,
        lhs_registers=(
            "mma_a_7_0",
            "mma_a_7_1",
        ),
        rhs_register="mma_b_7_0",
        accumulator=accumulator,
    )

    assert result_ref == CudaMmaAccumulatorRef(
        base="v7",
        layout=CudaMmaM16N8K8Layout(),
    )
    assert codegen.lines[:4] == [
        "    float v7_0 = v3_0;",
        "    float v7_1 = v3_1;",
        "    float v7_2 = v3_2;",
        "    float v7_3 = v3_3;",
    ]
    assert '        : "+f"(v7_0), "+f"(v7_1), "+f"(v7_2), "+f"(v7_3)' in codegen.lines


@pytest.mark.execution
def test_f16_tiled_tensor_core_matmul_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = int(cp.cuda.Device().compute_capability)
    if capability < 75:
        pytest.skip("m16n8k8 f16 MMA requires compute capability 7.5+")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 21, 13, 19
    BM, BK, BN = 16, 8, 8

    rng = np.random.default_rng(20)
    a_host = rng.normal(
        0.0,
        0.25,
        size=(M, K),
    ).astype(np.float16)
    b_host = rng.normal(
        0.0,
        0.25,
        size=(K, N),
    ).astype(np.float16)

    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    out = cp.full(
        (M, N),
        cp.nan,
        dtype=cp.float32,
    )

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    tensor_core_matmul_kernel.clear_cache()
    _, _, cuda_src = tensor_core_matmul_kernel[grid](
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

    expected = a_host.astype(np.float32) @ b_host.astype(np.float32)

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 1
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.execution
def test_bf16_tiled_tensor_core_matmul_executes_on_sm80(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del cp

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("m16n8k8 bf16 MMA requires compute capability 8.0+")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 21, 13, 19
    BM, BK, BN = 16, 8, 8
    generator = torch.Generator().manual_seed(20)
    a = (
        torch.randn((M, K), generator=generator, dtype=torch.float32)
        .mul_(0.25)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    b = (
        torch.randn((K, N), generator=generator, dtype=torch.float32)
        .mul_(0.25)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    out = torch.full(
        (M, N),
        torch.nan,
        device="cuda",
        dtype=torch.float32,
    )
    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
    )

    tensor_core_matmul_kernel.clear_cache()
    _, _, cuda_src = tensor_core_matmul_kernel[grid](
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
    torch.cuda.synchronize()

    expected = a.float() @ b.float()
    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32") == 1
    torch.testing.assert_close(
        out,
        expected,
        rtol=2e-3,
        atol=2e-3,
    )
