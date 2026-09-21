import itertools

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.block_shapes import (
    CudaKernelLayout,
    CudaMmaCtaTileLayout,
    CudaMmaM16N8K8Layout,
    CudaMmaWarpTileLayout,
    cuda_mma_cta_tile_layout,
    cuda_mma_cta_tile_layouts,
    cuda_mma_m16n8k8_dot_result_ids,
    cuda_mma_warp_tile_layout,
    cuda_mma_warp_tile_layouts,
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


def test_composable_mma_warp_tile_layout() -> None:
    layout = CudaMmaWarpTileLayout(m=32, n=16, k=16)

    assert layout.lhs_shape == (32, 16)
    assert layout.rhs_shape == (16, 16)
    assert layout.result_shape == (32, 16)

    assert layout.m_tiles == 2
    assert layout.n_tiles == 2
    assert layout.k_tiles == 2

    assert layout.instruction_count == 8
    assert layout.accumulator_fragments_per_lane == 4
    assert layout.accumulator_elements_per_lane == 16

    assert layout.instruction_coordinates() == (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )


def test_mma_cta_tile_layout_maps_four_warps() -> None:
    warp_tile = CudaMmaWarpTileLayout(
        m=32,
        n=32,
        k=16,
    )
    layout = CudaMmaCtaTileLayout(
        warp_tile=warp_tile,
        warps_m=2,
        warps_n=2,
    )

    assert layout.warp_shape == (2, 2)
    assert layout.warp_count == 4
    assert layout.threads_per_block == 128
    assert layout.thread_shape == (8, 16)

    assert layout.lhs_shape == (64, 16)
    assert layout.rhs_shape == (16, 64)
    assert layout.result_shape == (64, 64)

    assert tuple(
        layout.warp_coordinates(warp_id) for warp_id in range(layout.warp_count)
    ) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )

    assert tuple(
        layout.warp_result_offset(warp_id) for warp_id in range(layout.warp_count)
    ) == (
        (0, 0),
        (0, 32),
        (32, 0),
        (32, 32),
    )

    assert tuple(
        layout.warp_lhs_offset(warp_id) for warp_id in range(layout.warp_count)
    ) == (
        (0, 0),
        (0, 0),
        (32, 0),
        (32, 0),
    )

    assert tuple(
        layout.warp_rhs_offset(warp_id) for warp_id in range(layout.warp_count)
    ) == (
        (0, 0),
        (0, 32),
        (0, 0),
        (0, 32),
    )


def test_mma_accumulator_ref_tracks_multi_warp_cta_shape() -> None:
    cta_layout = CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )

    accumulator = CudaMmaAccumulatorRef(
        base="v7",
        layout=cta_layout.warp_tile,
        cta_layout=cta_layout,
    )

    assert accumulator.logical_shape == (64, 64)
    assert accumulator.warp_tile_layout == CudaMmaWarpTileLayout(
        m=32,
        n=32,
        k=16,
    )
    assert accumulator.elements_per_lane == 32
    assert accumulator.elements()[0] == "v7_0"
    assert accumulator.elements()[-1] == "v7_31"


@pytest.mark.parametrize(
    ("warps_m", "warps_n", "message"),
    [
        (0, 2, "warps_m must be a positive integer"),
        (-1, 2, "warps_m must be a positive integer"),
        (True, 2, "warps_m must be a positive integer"),
        (2, 0, "warps_n must be a positive integer"),
        (2, True, "warps_n must be a positive integer"),
        (33, 1, "CTA MMA tile requires at most 1024 threads"),
    ],
)
def test_mma_cta_tile_layout_rejects_invalid_warp_grid(
    warps_m: int,
    warps_n: int,
    message: str,
) -> None:
    warp_tile = CudaMmaWarpTileLayout(
        m=32,
        n=32,
        k=16,
    )

    with pytest.raises(ValueError, match=message):
        CudaMmaCtaTileLayout(
            warp_tile=warp_tile,
            warps_m=warps_m,
            warps_n=warps_n,
        )


@pytest.mark.parametrize("warp_id", [-1, 4, True])
def test_mma_cta_tile_layout_rejects_invalid_warp_id(
    warp_id: int,
) -> None:
    layout = CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )

    with pytest.raises(
        ValueError,
        match="warp ID must be between 0 and 3",
    ):
        layout.warp_coordinates(warp_id)


def test_composable_mma_warp_tile_offsets_instruction_fragments() -> None:
    layout = CudaMmaWarpTileLayout(m=32, n=16, k=16)

    assert layout.lhs_fragment_coordinates(
        lane=0,
        m_tile=1,
        k_tile=1,
    ) == (
        (16, 8),
        (16, 9),
        (24, 8),
        (24, 9),
    )

    assert layout.rhs_fragment_coordinates(
        lane=0,
        k_tile=1,
        n_tile=1,
    ) == (
        (8, 8),
        (9, 8),
    )

    assert layout.accumulator_fragment_coordinates(
        lane=0,
        m_tile=1,
        n_tile=1,
    ) == (
        (16, 8),
        (16, 9),
        (24, 8),
        (24, 9),
    )


def test_composable_mma_warp_tile_offsets_ldmatrix_addresses() -> None:
    layout = CudaMmaWarpTileLayout(m=32, n=16, k=16)

    assert layout.lhs_ldmatrix_address(
        lane=0,
        m_tile=1,
        k_tile=1,
    ) == (16, 8)

    assert layout.lhs_ldmatrix_address(
        lane=15,
        m_tile=1,
        k_tile=1,
    ) == (31, 8)

    # Lanes 16-31 duplicate the addresses supplied by lanes 0-15.
    assert layout.lhs_ldmatrix_address(
        lane=16,
        m_tile=1,
        k_tile=1,
    ) == (16, 8)

    assert layout.rhs_ldmatrix_address(
        lane=0,
        k_tile=1,
        n_tile=1,
    ) == (8, 8)

    assert layout.rhs_ldmatrix_address(
        lane=7,
        k_tile=1,
        n_tile=1,
    ) == (15, 8)

    # Every group of eight lanes supplies the same eight row addresses.
    assert layout.rhs_ldmatrix_address(
        lane=8,
        k_tile=1,
        n_tile=1,
    ) == (8, 8)


@pytest.mark.parametrize(
    ("m", "n", "k", "message"),
    [
        (0, 8, 8, "M dimension must be a positive multiple of 16"),
        (24, 8, 8, "M dimension must be a positive multiple of 16"),
        (16, 12, 8, "N dimension must be a positive multiple of 8"),
        (16, 8, 4, "K dimension must be a positive multiple of 8"),
        (True, 8, 8, "M dimension must be a positive multiple of 16"),
    ],
)
def test_composable_mma_warp_tile_rejects_invalid_shapes(
    m: int,
    n: int,
    k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CudaMmaWarpTileLayout(m=m, n=n, k=k)


def test_composable_mma_warp_tile_rejects_invalid_fragment_indices() -> None:
    layout = CudaMmaWarpTileLayout(m=32, n=16, k=16)

    with pytest.raises(
        ValueError,
        match="M tile index must be between 0 and 1",
    ):
        layout.lhs_fragment_coordinates(
            lane=0,
            m_tile=2,
            k_tile=0,
        )

    with pytest.raises(
        ValueError,
        match="K tile index must be between 0 and 1",
    ):
        layout.lhs_fragment_coordinates(
            lane=0,
            m_tile=0,
            k_tile=-1,
        )

    with pytest.raises(
        ValueError,
        match="N tile index must be between 0 and 1",
    ):
        layout.rhs_fragment_coordinates(
            lane=0,
            k_tile=0,
            n_tile=2,
        )

    with pytest.raises(
        ValueError,
        match="M tile index must be between 0 and 1",
    ):
        layout.accumulator_fragment_coordinates(
            lane=0,
            m_tile=True,
            n_tile=0,
        )


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


@pytest.mark.parametrize(
    ("lane", "expected_lhs", "expected_rhs"),
    [
        (0, (0, 0), (0, 0)),
        (7, (7, 0), (7, 0)),
        (8, (8, 0), (0, 0)),
        (15, (15, 0), (7, 0)),
        (16, (0, 0), (0, 0)),
        (31, (15, 0), (7, 0)),
    ],
)
def test_mma_m16n8k8_ldmatrix_addresses(
    lane: int,
    expected_lhs: tuple[int, int],
    expected_rhs: tuple[int, int],
) -> None:
    layout = CudaMmaM16N8K8Layout()

    assert layout.lhs_ldmatrix_address(lane) == expected_lhs
    assert layout.rhs_ldmatrix_address(lane) == expected_rhs


def test_mma_m16n8k8_ldmatrix_addresses_cover_matrix_rows() -> None:
    layout = CudaMmaM16N8K8Layout()

    lhs_addresses = {
        layout.lhs_ldmatrix_address(lane) for lane in range(layout.threads_per_warp)
    }
    rhs_addresses = {
        layout.rhs_ldmatrix_address(lane) for lane in range(layout.threads_per_warp)
    }

    assert lhs_addresses == {(row, 0) for row in range(16)}
    assert rhs_addresses == {(row, 0) for row in range(8)}


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


def test_recognizes_composable_f16_mma_warp_tile() -> None:
    op = make_dot_op(
        BlockType((32, 16), F16),
        BlockType((16, 16), F16),
        BlockType((32, 16), F32),
    )

    assert cuda_mma_warp_tile_layout(op) == CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )


def test_recognizes_four_warp_f16_mma_cta_tile() -> None:
    op = make_dot_op(
        BlockType((64, 16), F16),
        BlockType((16, 64), F16),
        BlockType((64, 64), F32),
    )

    assert cuda_mma_cta_tile_layout(op) == CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )


def test_mma_cta_tile_does_not_replace_one_warp_layout() -> None:
    op = make_dot_op(
        BlockType((32, 16), F16),
        BlockType((16, 32), F16),
        BlockType((32, 32), F32),
    )

    assert cuda_mma_cta_tile_layout(op) is None
    assert cuda_mma_warp_tile_layout(op) == CudaMmaWarpTileLayout(
        m=32,
        n=32,
        k=16,
    )


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


def test_collects_composable_mma_warp_tile_layouts_recursively() -> None:
    top_level_dot = make_dot_op(
        BlockType((16, 8), F16),
        BlockType((8, 8), F16),
        BlockType((16, 8), F32),
        result_id=2,
    )
    nested_dot = make_dot_op(
        BlockType((32, 16), F16),
        BlockType((16, 16), F16),
        BlockType((32, 16), F32),
        result_id=9,
    )

    loop = SSAForRange(
        index=SSAValue(id=6, ty=I32),
        start=Const(0),
        stop=Const(16),
        step=Const(16),
        carried_inputs=(),
        carried_args=(),
        body=[nested_dot],
        yields=(),
        results=(),
    )

    layouts = cuda_mma_warp_tile_layouts(
        [
            top_level_dot,
            loop,
        ]
    )

    assert layouts == {
        2: CudaMmaWarpTileLayout(m=16, n=8, k=8),
        9: CudaMmaWarpTileLayout(m=32, n=16, k=16),
    }


def test_collects_mma_cta_tile_layouts_recursively() -> None:
    top_level_dot = make_dot_op(
        BlockType((64, 16), F16),
        BlockType((16, 64), F16),
        BlockType((64, 64), F32),
        result_id=2,
    )
    one_warp_dot = make_dot_op(
        BlockType((32, 16), F16),
        BlockType((16, 32), F16),
        BlockType((32, 32), F32),
        result_id=5,
    )
    nested_dot = make_dot_op(
        BlockType((64, 16), F16),
        BlockType((16, 32), F16),
        BlockType((64, 32), F32),
        result_id=9,
    )

    loop = SSAForRange(
        index=SSAValue(id=6, ty=I32),
        start=Const(0),
        stop=Const(16),
        step=Const(16),
        carried_inputs=(),
        carried_args=(),
        body=[nested_dot],
        yields=(),
        results=(),
    )

    layouts = cuda_mma_cta_tile_layouts(
        [
            top_level_dot,
            one_warp_dot,
            loop,
        ]
    )

    assert layouts == {
        2: CudaMmaCtaTileLayout(
            warp_tile=CudaMmaWarpTileLayout(
                m=32,
                n=32,
                k=16,
            ),
            warps_m=2,
            warps_n=2,
        ),
        9: CudaMmaCtaTileLayout(
            warp_tile=CudaMmaWarpTileLayout(
                m=32,
                n=32,
                k=16,
            ),
            warps_m=2,
            warps_n=1,
        ),
    }


def test_mma_cta_tile_determines_cuda_threads_per_block() -> None:
    dot = make_dot_op(
        BlockType((64, 16), F16),
        BlockType((16, 64), F16),
        BlockType((64, 64), F32),
        result_id=2,
    )
    assert dot.result is not None

    pointers = SSAValue(
        id=3,
        ty=BlockType((64, 64), PTR_F32),
    )
    store = SSAOp(
        opcode="store",
        operands=(pointers, dot.result, None),
    )

    assert cuda_threads_per_block([dot, store]) == 128


def test_cuda_codegen_emits_multi_warp_cta_prologue() -> None:
    cta_layout = CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )

    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(64, 64),
        thread_shape=cta_layout.thread_shape,
    )
    codegen.mma_cta_dot_layouts = {
        7: cta_layout,
    }

    codegen.emit_rank2_prologue()

    assert codegen.lines == [
        "    int mma_warp_id = threadIdx.x >> 5;",
        "    int mma_lane_id = threadIdx.x & 31;",
        "    int mma_warp_m = mma_warp_id / 2;",
        "    int mma_warp_n = mma_warp_id % 2;",
        "    int tile_i = mma_warp_m * 4 + mma_lane_id / 8;",
        "    int tile_j = mma_warp_n * 8 + mma_lane_id % 8;",
    ]


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


def test_mma_accumulator_ref_names_composable_lane_registers() -> None:
    ref = CudaMmaAccumulatorRef(
        base="v7",
        layout=CudaMmaWarpTileLayout(m=32, n=16, k=16),
    )

    assert ref.elements_per_lane == 16

    assert ref.fragment_elements(
        m_tile=0,
        n_tile=0,
    ) == (
        "v7_0",
        "v7_1",
        "v7_2",
        "v7_3",
    )
    assert ref.fragment_elements(
        m_tile=0,
        n_tile=1,
    ) == (
        "v7_4",
        "v7_5",
        "v7_6",
        "v7_7",
    )
    assert ref.fragment_elements(
        m_tile=1,
        n_tile=0,
    ) == (
        "v7_8",
        "v7_9",
        "v7_10",
        "v7_11",
    )
    assert ref.fragment_elements(
        m_tile=1,
        n_tile=1,
    ) == (
        "v7_12",
        "v7_13",
        "v7_14",
        "v7_15",
    )

    assert ref.elements() == tuple(f"v7_{index}" for index in range(16))


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


def test_cuda_codegen_emits_u32_shared_memory_address() -> None:
    codegen = SSACUDACodegen()

    address = codegen.emit_shared_u32_address(
        name="mma_a_7_address",
        element="dot_lhs_7[(row) * 24 + (column)]",
    )

    assert address == "mma_a_7_address"
    assert codegen.lines == [
        (
            "    unsigned mma_a_7_address = static_cast<unsigned>("
            "__cvta_generic_to_shared("
            "&dot_lhs_7[(row) * 24 + (column)]));"
        ),
    ]


def test_cuda_codegen_emits_ldmatrix_m8n8_x2() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )

    registers = codegen.emit_ldmatrix_m8n8_x2(
        result_prefix="mma_a_7",
        address="mma_a_7_address",
    )

    assert registers == (
        "mma_a_7_0",
        "mma_a_7_1",
    )
    assert codegen.lines == [
        "    unsigned mma_a_7_0;",
        "    unsigned mma_a_7_1;",
        "    asm volatile(",
        '        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "',
        '        "{%0, %1}, [%2];"',
        '        : "=r"(mma_a_7_0), "=r"(mma_a_7_1)',
        '        : "r"(mma_a_7_address)',
        '        : "memory"',
        "    );",
    ]


def test_cuda_codegen_rejects_ldmatrix_before_sm75() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_70"),
    )

    with pytest.raises(
        TypeError,
        match=r"ldmatrix requires sm_75\+",
    ):
        codegen.emit_ldmatrix_m8n8_x2(
            result_prefix="mma_a_7",
            address="mma_a_7_address",
        )

    assert codegen.lines == []


def test_cuda_codegen_emits_ldmatrix_m8n8_x1_trans() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )

    register = codegen.emit_ldmatrix_m8n8_x1_trans(
        result_prefix="mma_b_7",
        address="mma_b_7_address",
    )

    assert register == "mma_b_7_0"
    assert codegen.lines == [
        "    unsigned mma_b_7_0;",
        "    asm volatile(",
        '        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "',
        '        "{%0}, [%1];"',
        '        : "=r"(mma_b_7_0)',
        '        : "r"(mma_b_7_address)',
        '        : "memory"',
        "    );",
    ]


def test_cuda_codegen_emits_ldmatrix_m8n8_x4_trans() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )

    registers = codegen.emit_ldmatrix_m8n8(
        result_prefix="mma_b_7",
        address="mma_b_7_address",
        matrix_count=4,
        transpose=True,
    )

    assert registers == (
        "mma_b_7_0",
        "mma_b_7_1",
        "mma_b_7_2",
        "mma_b_7_3",
    )
    assert codegen.lines == [
        "    unsigned mma_b_7_0;",
        "    unsigned mma_b_7_1;",
        "    unsigned mma_b_7_2;",
        "    unsigned mma_b_7_3;",
        "    asm volatile(",
        '        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "',
        '        "{%0, %1, %2, %3}, [%4];"',
        (
            '        : "=r"(mma_b_7_0), "=r"(mma_b_7_1), '
            '"=r"(mma_b_7_2), "=r"(mma_b_7_3)'
        ),
        '        : "r"(mma_b_7_address)',
        '        : "memory"',
        "    );",
    ]


@pytest.mark.parametrize("matrix_count", [0, 3, 8, True])
def test_cuda_codegen_rejects_invalid_ldmatrix_matrix_count(
    matrix_count: int,
) -> None:
    codegen = SSACUDACodegen()

    with pytest.raises(
        ValueError,
        match="matrix count must be one of 1, 2, or 4",
    ):
        codegen.emit_ldmatrix_m8n8(
            result_prefix="mma_a_7",
            address="mma_a_7_address",
            matrix_count=matrix_count,
        )

    assert codegen.lines == []


def test_cuda_codegen_loads_composable_mma_operands_with_ldmatrix() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(32, 16),
        thread_shape=(4, 8),
    )
    layout = CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(32, 16),
            element_ty=F16,
            row_padding=8,
            alignment=16,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 16),
            element_ty=F16,
            row_padding=8,
            alignment=16,
        ),
    )

    instruction_operands = codegen.emit_ldmatrix_mma_warp_tile_operand_registers(
        result_id=7,
        buffers=buffers,
        layout=layout,
    )

    assert len(instruction_operands) == layout.instruction_count == 8

    assert instruction_operands[0] == (
        (
            "mma_a_7_0_0_0",
            "mma_a_7_0_0_1",
        ),
        "mma_b_7_0_0_0",
    )
    assert instruction_operands[-1] == (
        (
            "mma_a_7_0_1_2",
            "mma_a_7_0_1_3",
        ),
        "mma_b_7_1_0_1",
    )

    # A[m, k] is reused across different N atoms.
    assert instruction_operands[0][0] == instruction_operands[2][0]

    # B[k, n] is reused across different M atoms.
    assert instruction_operands[0][1] == instruction_operands[4][1]

    cuda = "\n".join(codegen.lines)

    assert cuda.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x2.shared.b16" not in cuda
    assert cuda.count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16" not in cuda

    assert ("__cvta_generic_to_shared(&dot_lhs_7[(mma_lane_7) * 24 + (8)])") in cuda
    assert (
        "__cvta_generic_to_shared("
        "&dot_rhs_7[((mma_lane_7 & 7) + 8) * 24 "
        "+ (((mma_lane_7 & 15) >> 3) * 8)])"
    ) in cuda

    assert "__half_as_ushort" not in cuda


def test_cuda_codegen_ldmatrix_uses_x2_for_unpaired_lhs_m_tile() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(48, 8),
        thread_shape=(4, 8),
    )
    layout = CudaMmaWarpTileLayout(
        m=48,
        n=8,
        k=8,
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="lhs",
            logical_shape=(48, 8),
            element_ty=F16,
            alignment=16,
        ),
        rhs=CudaSharedBuffer(
            name="rhs",
            logical_shape=(8, 8),
            element_ty=F16,
            alignment=16,
        ),
    )

    instruction_operands = codegen.emit_ldmatrix_mma_warp_tile_operand_registers(
        result_id=7,
        buffers=buffers,
        layout=layout,
    )

    assert instruction_operands == (
        (
            ("mma_a_7_0_0_0", "mma_a_7_0_0_1"),
            "mma_b_7_0_0_0",
        ),
        (
            ("mma_a_7_0_0_2", "mma_a_7_0_0_3"),
            "mma_b_7_0_0_0",
        ),
        (
            ("mma_a_7_2_0_0", "mma_a_7_2_0_1"),
            "mma_b_7_0_0_0",
        ),
    )

    cuda = "\n".join(codegen.lines)

    assert cuda.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 1
    assert cuda.count("ldmatrix.sync.aligned.m8n8.x2.shared.b16") == 1
    assert cuda.count("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16") == 1


def test_cuda_codegen_ldmatrix_groups_rhs_n_tiles() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(16, 56),
        thread_shape=(4, 8),
    )
    layout = CudaMmaWarpTileLayout(
        m=16,
        n=56,
        k=8,
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="lhs",
            logical_shape=(16, 8),
            element_ty=F16,
            alignment=16,
        ),
        rhs=CudaSharedBuffer(
            name="rhs",
            logical_shape=(8, 56),
            element_ty=F16,
            alignment=16,
        ),
    )

    instruction_operands = codegen.emit_ldmatrix_mma_warp_tile_operand_registers(
        result_id=7,
        buffers=buffers,
        layout=layout,
    )

    rhs_registers = tuple(
        rhs_register for _lhs_registers, rhs_register in instruction_operands
    )

    assert rhs_registers == (
        "mma_b_7_0_0_0",
        "mma_b_7_0_0_1",
        "mma_b_7_0_0_2",
        "mma_b_7_0_0_3",
        "mma_b_7_0_4_0",
        "mma_b_7_0_4_1",
        "mma_b_7_0_6_0",
    )

    cuda = "\n".join(codegen.lines)

    assert cuda.count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") == 1
    assert cuda.count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") == 1
    assert cuda.count("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16") == 1


def test_cuda_codegen_rejects_unaligned_ldmatrix_operands() -> None:
    codegen = SSACUDACodegen()
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(16, 8),
        thread_shape=(4, 8),
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="lhs",
            logical_shape=(16, 8),
            element_ty=F16,
        ),
        rhs=CudaSharedBuffer(
            name="rhs",
            logical_shape=(8, 8),
            element_ty=F16,
        ),
    )

    with pytest.raises(
        TypeError,
        match="16-byte-aligned shared buffers",
    ):
        codegen.emit_ldmatrix_mma_warp_tile_operand_registers(
            result_id=7,
            buffers=buffers,
            layout=CudaMmaWarpTileLayout(
                m=16,
                n=8,
                k=8,
            ),
        )

    assert codegen.lines == []


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


def test_cuda_codegen_emits_mma_into_selected_accumulator_fragment() -> None:
    codegen = SSACUDACodegen()

    codegen.emit_mma_m16n8k8_instruction(
        accumulator_registers=(
            "v7_4",
            "v7_5",
            "v7_6",
            "v7_7",
        ),
        lhs_registers=(
            "mma_a_7_0",
            "mma_a_7_1",
        ),
        rhs_register="mma_b_7_0",
        operand_ty=F16,
    )

    cuda = "\n".join(codegen.lines)

    assert codegen.lines[0] == "    asm volatile("
    assert codegen.lines[-1] == "    );"

    assert (': "+f"(v7_4), "+f"(v7_5), "+f"(v7_6), "+f"(v7_7)') in cuda
    assert (': "r"(mma_a_7_0), "r"(mma_a_7_1), "r"(mma_b_7_0)') in cuda

    assert "float v7_4" not in cuda
    assert cuda.count("mma.sync.aligned.m16n8k8") == 1


def test_cuda_codegen_composes_warp_tile_from_mma_instructions() -> None:
    codegen = SSACUDACodegen()
    layout = CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )
    result = SSAValue(
        id=7,
        ty=BlockType((32, 16), F32),
    )

    instruction_operands = tuple(
        (
            (
                f"mma_a_7_{m_tile}_{n_tile}_{k_tile}_0",
                f"mma_a_7_{m_tile}_{n_tile}_{k_tile}_1",
            ),
            f"mma_b_7_{m_tile}_{n_tile}_{k_tile}_0",
        )
        for m_tile, n_tile, k_tile in layout.instruction_coordinates()
    )

    result_ref = codegen.emit_mma_warp_tile(
        result=result,
        layout=layout,
        instruction_operands=instruction_operands,
        operand_ty=F16,
    )

    assert result_ref == CudaMmaAccumulatorRef(
        base="v7",
        layout=layout,
    )
    assert codegen.values[result.id] == result_ref

    assert codegen.lines[:16] == [
        f"    float v7_{index} = 0.0f;" for index in range(16)
    ]

    cuda = "\n".join(codegen.lines)

    assert cuda.count("mma.sync.aligned.m16n8k8") == layout.instruction_count == 8

    for m_tile in range(layout.m_tiles):
        for n_tile in range(layout.n_tiles):
            fragment = result_ref.fragment_elements(
                m_tile=m_tile,
                n_tile=n_tile,
            )
            output_constraints = (
                f': "+f"({fragment[0]}), '
                f'"+f"({fragment[1]}), '
                f'"+f"({fragment[2]}), '
                f'"+f"({fragment[3]})'
            )
            assert cuda.count(output_constraints) == layout.k_tiles

    assert (
        ': "r"(mma_a_7_0_0_0_0), "r"(mma_a_7_0_0_0_1), "r"(mma_b_7_0_0_0_0)'
    ) in cuda
    assert (
        ': "r"(mma_a_7_1_1_1_0), "r"(mma_a_7_1_1_1_1), "r"(mma_b_7_1_1_1_0)'
    ) in cuda


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


def test_cuda_codegen_computes_composable_mma_warp_tile_from_shared_memory() -> None:
    codegen = SSACUDACodegen(
        target=CudaTarget.from_chip("sm_75"),
    )
    codegen.layout = CudaKernelLayout(
        output_tile_shape=(32, 16),
        thread_shape=(4, 8),
    )
    layout = CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )
    buffers = CudaDotSharedBuffers(
        lhs=CudaSharedBuffer(
            name="dot_lhs_7",
            logical_shape=(32, 16),
            element_ty=F16,
            row_padding=8,
            alignment=16,
        ),
        rhs=CudaSharedBuffer(
            name="dot_rhs_7",
            logical_shape=(16, 16),
            element_ty=F16,
            row_padding=8,
            alignment=16,
        ),
    )
    result = SSAValue(
        id=7,
        ty=BlockType((32, 16), F32),
    )
    accumulator = CudaMmaAccumulatorRef(
        base="v3",
        layout=layout,
    )

    result_ref = codegen.emit_mma_warp_tile_from_shared_memory(
        result=result,
        buffers=buffers,
        layout=layout,
        accumulator=accumulator,
    )

    assert result_ref == CudaMmaAccumulatorRef(
        base="v7",
        layout=layout,
    )
    assert codegen.values[result.id] == result_ref

    cuda = "\n".join(codegen.lines)

    assert "    float v7_0 = v3_0;" in codegen.lines
    assert "    float v7_15 = v3_15;" in codegen.lines

    assert cuda.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x2.shared.b16" not in cuda
    assert cuda.count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16" not in cuda
    assert cuda.count("mma.sync.aligned.m16n8k8") == 8

    ldmatrix_position = cuda.index("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
    mma_position = cuda.index("mma.sync.aligned.m16n8k8")
    barrier_position = cuda.rindex("__syncthreads();")

    assert ldmatrix_position < mma_position < barrier_position
    assert "__half_as_ushort" not in cuda
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


def test_cuda_codegen_spills_composable_mma_accumulator_to_shared_memory() -> None:
    codegen = SSACUDACodegen()
    layout = CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )
    value = CudaMmaAccumulatorRef(
        base="v7",
        layout=layout,
    )

    buffer = codegen.emit_mma_accumulator_spill(
        value_id=7,
        value=value,
    )

    assert buffer == CudaSharedBuffer(
        name="mma_result_7",
        logical_shape=(32, 16),
        element_ty=F32,
    )
    assert buffer.nbytes == 32 * 16 * 4
    assert codegen.shared_memory_bytes == 32 * 16 * 4
    assert codegen.shared_lines == ["    __shared__ float mma_result_7[512];"]

    assert codegen.lines[:2] == [
        "    int mma_store_group_7 = threadIdx.x >> 2;",
        "    int mma_store_thread_7 = threadIdx.x & 3;",
    ]

    assert (
        "    mma_result_7[(mma_store_group_7) * 16 + (mma_store_thread_7 * 2)] = v7_0;"
    ) in codegen.lines
    assert (
        "    mma_result_7[(mma_store_group_7) * 16 + "
        "(mma_store_thread_7 * 2 + 8)] = v7_4;"
    ) in codegen.lines
    assert (
        "    mma_result_7[(mma_store_group_7 + 16) * 16 + "
        "(mma_store_thread_7 * 2)] = v7_8;"
    ) in codegen.lines
    assert (
        "    mma_result_7[(mma_store_group_7 + 24) * 16 + "
        "(mma_store_thread_7 * 2 + 9)] = v7_15;"
    ) in codegen.lines

    assert len(codegen.lines) == 19
    assert codegen.lines[-1] == "    __syncthreads();"


def test_cuda_codegen_spills_multi_warp_cta_accumulator() -> None:
    cta_layout = CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )
    codegen = SSACUDACodegen()
    value = CudaMmaAccumulatorRef(
        base="v7",
        layout=cta_layout.warp_tile,
        cta_layout=cta_layout,
    )

    buffer = codegen.emit_mma_accumulator_spill(
        value_id=7,
        value=value,
    )

    assert buffer == CudaSharedBuffer(
        name="mma_result_7",
        logical_shape=(64, 64),
        element_ty=F32,
    )
    assert buffer.nbytes == 64 * 64 * 4
    assert codegen.shared_memory_bytes == 64 * 64 * 4
    assert codegen.shared_lines == ["    __shared__ float mma_result_7[4096];"]

    assert codegen.lines[:2] == [
        "    int mma_store_group_7 = mma_lane_id >> 2;",
        "    int mma_store_thread_7 = mma_lane_id & 3;",
    ]

    assert (
        "    mma_result_7["
        "(mma_warp_m * 32 + mma_store_group_7) * 64 + "
        "(mma_warp_n * 32 + mma_store_thread_7 * 2)"
        "] = v7_0;"
    ) in codegen.lines

    assert (
        "    mma_result_7["
        "(mma_warp_m * 32 + mma_store_group_7 + 24) * 64 + "
        "(mma_warp_n * 32 + mma_store_thread_7 * 2 + 25)"
        "] = v7_31;"
    ) in codegen.lines

    assert len(codegen.lines) == 35
    assert codegen.lines[-1] == "    __syncthreads();"


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

    assert "__align__(16) __shared__ __half dot_lhs_" in cuda_src
    assert "__align__(16) __shared__ __half dot_rhs_" in cuda_src
    assert "__shared__ float mma_result_" in cuda_src

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 1
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x2.shared.b16") == 1
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16") == 1
    assert "__half_as_ushort" not in cuda_src

    assert "for (int dot_k_" not in cuda_src
    assert "__half2float(dot_lhs_" not in cuda_src
    assert "__half2float(dot_rhs_" not in cuda_src


@pytest.mark.codegen
def test_f16_composable_warp_tile_dot_lowers_to_tensor_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 32, 16, 16
    BM, BK, BN = 32, 16, 16

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

    dot = next(
        item for item in ssa_ops if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None

    assert cuda_mma_warp_tile_layouts(ssa_ops)[dot.result.id] == (
        CudaMmaWarpTileLayout(
            m=32,
            n=16,
            k=16,
        )
    )

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 8
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x2.shared.b16" not in cuda_src
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16" not in cuda_src
    assert "__half_as_ushort" not in cuda_src

    assert f"mma_a_{dot.result.id}_0_0_0" in cuda_src
    assert f"mma_b_{dot.result.id}_1_0_1" in cuda_src

    assert "__shared__ float mma_result_" in cuda_src
    assert "for (int dot_k_" not in cuda_src
    assert "__half2float(dot_lhs_" not in cuda_src
    assert "__half2float(dot_rhs_" not in cuda_src


@pytest.mark.codegen
def test_f16_multi_warp_cta_dot_lowers_to_tensor_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 64, 64, 16
    BM, BK, BN = 64, 16, 64

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

    assert cuda_threads_per_block(ssa_ops) == 128

    dot = next(
        item for item in ssa_ops if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None

    assert cuda_mma_cta_tile_layouts(ssa_ops)[dot.result.id] == (
        CudaMmaCtaTileLayout(
            warp_tile=CudaMmaWarpTileLayout(
                m=32,
                n=32,
                k=16,
            ),
            warps_m=2,
            warps_n=2,
        )
    )

    assert "int mma_warp_id = threadIdx.x >> 5;" in cuda_src
    assert "int mma_lane_id = threadIdx.x & 31;" in cuda_src
    assert "int mma_warp_m = mma_warp_id / 2;" in cuda_src
    assert "int mma_warp_n = mma_warp_id % 2;" in cuda_src

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 16
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") == 2
    assert "__half_as_ushort" not in cuda_src

    result_id = dot.result.id

    assert (
        f"dot_lhs_{result_id}[(mma_warp_m * 32 + mma_lane_id) * 24 + (0)]"
    ) in cuda_src

    assert (
        f"dot_rhs_{result_id}["
        "((mma_lane_id & 7)) * 72 + "
        "(mma_warp_n * 32 + (mma_lane_id >> 3) * 8)]"
    ) in cuda_src

    assert (
        f"mma_result_{result_id}["
        f"(mma_warp_m * 32 + mma_store_group_{result_id}) * 64 + "
        f"(mma_warp_n * 32 + mma_store_thread_{result_id} * 2)"
        f"] = v{result_id}_0;"
    ) in cuda_src

    assert "__shared__ __half dot_lhs_" in cuda_src
    assert "__shared__ __half dot_rhs_" in cuda_src
    assert "__shared__ float mma_result_" in cuda_src


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
    assert "__align__(16) __shared__ __nv_bfloat16 dot_lhs_" in cuda_src
    assert "__align__(16) __shared__ __nv_bfloat16 dot_rhs_" in cuda_src
    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32") == 1
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x2.shared.b16") == 1
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16") == 1
    assert "static_cast<__nv_bfloat16_raw>" not in cuda_src
    assert "for (int dot_k_" not in cuda_src
    assert "__bfloat162float(dot_lhs_" not in cuda_src


@pytest.mark.codegen
def test_bf16_composable_warp_tile_lowers_to_tensor_core_for_sm80(
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
    monkeypatch.setattr(
        compiler,
        "cuda_chip",
        lambda runtime_args: "sm_80",
    )
    monkeypatch.setattr(
        compiler,
        "execute_cuda_if_needed",
        lambda **kwargs: None,
    )

    M, N, K = 32, 16, 16
    BM, BK, BN = 32, 16, 16

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

    dot = next(
        item for item in ssa_ops if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None

    assert cuda_mma_warp_tile_layouts(
        ssa_ops,
        operand_types=frozenset((F16, BF16)),
    )[dot.result.id] == CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )

    assert cuda_src.startswith("#include <cuda_bf16.h>\n\n")
    assert "#include <cuda_fp16.h>" not in cuda_src

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32") == 8
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x2.shared.b16" not in cuda_src
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") == 2
    assert "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16" not in cuda_src
    assert "static_cast<__nv_bfloat16_raw>" not in cuda_src

    assert "mma.sync.aligned.m16n8k8.row.col.f32.f16" not in cuda_src
    assert "__half_as_ushort" not in cuda_src
    assert "for (int dot_k_" not in cuda_src
    assert "__bfloat162float(dot_lhs_" not in cuda_src


@pytest.mark.codegen
def test_bf16_multi_warp_cta_lowers_k_loop_for_sm80(
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
    monkeypatch.setattr(
        compiler,
        "cuda_chip",
        lambda runtime_args: "sm_80",
    )
    monkeypatch.setattr(
        compiler,
        "execute_cuda_if_needed",
        lambda **kwargs: None,
    )

    M, N, K = 64, 64, 32
    BM, BK, BN = 64, 16, 64

    a = torch.zeros((M, K), dtype=torch.bfloat16)
    b = torch.zeros((K, N), dtype=torch.bfloat16)
    out = torch.zeros((M, N), dtype=torch.float32)

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

    loop = next(item for item in ssa_ops if isinstance(item, SSAForRange))
    dot = next(
        item for item in loop.body if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None

    cta_layout = CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )

    assert (
        cuda_mma_cta_tile_layouts(
            ssa_ops,
            operand_types=frozenset((F16, BF16)),
        )[dot.result.id]
        == cta_layout
    )
    assert (
        cuda_threads_per_block(
            ssa_ops,
            mma_operand_types=frozenset((F16, BF16)),
        )
        == 128
    )

    assert "int mma_warp_id = threadIdx.x >> 5;" in cuda_src
    assert "int mma_lane_id = threadIdx.x & 31;" in cuda_src

    assert (
        cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32")
        == cta_layout.warp_tile.instruction_count
        == 16
    )
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") == 2
    assert "static_cast<__nv_bfloat16_raw>" not in cuda_src

    assert "#include <cuda_bf16.h>" in cuda_src
    assert "#include <cuda_fp16.h>" not in cuda_src
    assert "mma.sync.aligned.m16n8k8.row.col.f32.f16" not in cuda_src


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


@pytest.mark.codegen
def test_f16_composable_mma_lowers_k_loop_to_tensor_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 32, 16, 32
    BM, BK, BN = 32, 16, 16

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

    layout = CudaMmaWarpTileLayout(
        m=32,
        n=16,
        k=16,
    )
    assert cuda_mma_warp_tile_layouts(ssa_ops)[dot.result.id] == layout

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

    assert (
        cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32")
        == layout.instruction_count
        == 8
    )
    assert "for (int dot_k_" not in cuda_src

    assert f"mma_a_{dot.result.id}_0_0_0" in cuda_src
    assert f"mma_b_{dot.result.id}_1_0_1" in cuda_src

    for index in range(layout.accumulator_elements_per_lane):
        assert (
            f"float v{loop_result.id}_{index} = v{initial_accumulator.id};"
        ) in cuda_src
        assert (
            f"float v{dot.result.id}_{index} = v{loop_result.id}_{index};"
        ) in cuda_src
        assert (f"v{loop_result.id}_{index} = v{dot.result.id}_{index};") in cuda_src

    assert f"v{accumulation.result.id}_" not in cuda_src

    assert (
        cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
        == (layout.m_tiles // 2) * layout.k_tiles
    )
    assert (
        cuda_src.count("ldmatrix.sync.aligned.m8n8.x2.shared.b16")
        == (layout.m_tiles % 2) * layout.k_tiles
    )

    assert (
        cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
        == (layout.n_tiles // 4) * layout.k_tiles
    )
    assert (
        cuda_src.count("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16")
        == ((layout.n_tiles % 4) // 2) * layout.k_tiles
    )
    assert (
        cuda_src.count("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16")
        == (layout.n_tiles % 2) * layout.k_tiles
    )

    assert "__half_as_ushort" not in cuda_src


@pytest.mark.codegen
def test_f16_multi_warp_cta_lowers_k_loop_to_tensor_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 64, 64, 32
    BM, BK, BN = 64, 16, 64

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

    assert cuda_threads_per_block(ssa_ops) == 128

    loop = next(item for item in ssa_ops if isinstance(item, SSAForRange))
    dot = next(
        item for item in loop.body if isinstance(item, SSAOp) and item.opcode == "dot"
    )
    assert dot.result is not None

    cta_layout = CudaMmaCtaTileLayout(
        warp_tile=CudaMmaWarpTileLayout(
            m=32,
            n=32,
            k=16,
        ),
        warps_m=2,
        warps_n=2,
    )
    assert cuda_mma_cta_tile_layouts(ssa_ops)[dot.result.id] == (cta_layout)

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

    assert (
        cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32")
        == cta_layout.warp_tile.instruction_count
        == 16
    )
    assert "for (int dot_k_" not in cuda_src

    assert (
        f"__align__(16) __shared__ __half dot_lhs_{dot.result.id}[3072];"
    ) in cuda_src
    assert (
        f"__align__(16) __shared__ __half dot_rhs_{dot.result.id}[2304];"
    ) in cuda_src
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == 2
    assert cuda_src.count("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") == 2

    for index in range(cta_layout.warp_tile.accumulator_elements_per_lane):
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
def test_f16_composable_tensor_core_matmul_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = int(cp.cuda.Device().compute_capability)
    if capability < 75:
        pytest.skip("m16n8k8 f16 MMA requires compute capability 7.5+")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 45, 23, 37
    BM, BK, BN = 32, 16, 16

    rng = np.random.default_rng(21)
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

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 8
    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=3e-3,
        atol=3e-3,
    )


@pytest.mark.execution
def test_f16_multi_warp_cta_tensor_core_matmul_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = int(cp.cuda.Device().compute_capability)
    if capability < 75:
        pytest.skip("m16n8k8 f16 MMA requires compute capability 7.5+")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 95, 79, 37
    BM, BK, BN = 64, 16, 64

    rng = np.random.default_rng(22)
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
    assert grid == (2, 2)

    tensor_core_matmul_kernel.clear_cache()
    _, ssa_ops, cuda_src = tensor_core_matmul_kernel[grid](
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

    assert cuda_threads_per_block(ssa_ops) == 128
    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32") == 16

    cp.testing.assert_allclose(
        out,
        cp.asarray(expected),
        rtol=3e-3,
        atol=3e-3,
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


@pytest.mark.execution
def test_bf16_composable_tensor_core_matmul_executes_on_sm80(
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

    M, N, K = 45, 23, 37
    BM, BK, BN = 32, 16, 16

    generator = torch.Generator().manual_seed(21)
    a = (
        torch.randn(
            (M, K),
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    b = (
        torch.randn(
            (K, N),
            generator=generator,
            dtype=torch.float32,
        )
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

    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32") == 8
    torch.testing.assert_close(
        out,
        expected,
        rtol=3e-3,
        atol=3e-3,
    )


@pytest.mark.execution
def test_bf16_multi_warp_cta_tensor_core_matmul_executes_on_sm80(
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

    M, N, K = 95, 79, 37
    BM, BK, BN = 64, 16, 64

    generator = torch.Generator().manual_seed(22)
    a = (
        torch.randn(
            (M, K),
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    b = (
        torch.randn(
            (K, N),
            generator=generator,
            dtype=torch.float32,
        )
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
    assert grid == (2, 2)

    tensor_core_matmul_kernel.clear_cache()
    _, ssa_ops, cuda_src = tensor_core_matmul_kernel[grid](
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

    assert (
        cuda_threads_per_block(
            ssa_ops,
            mma_operand_types=frozenset((F16, BF16)),
        )
        == 128
    )
    assert cuda_src.count("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32") == 16

    torch.testing.assert_close(
        out,
        expected,
        rtol=3e-3,
        atol=3e-3,
    )
