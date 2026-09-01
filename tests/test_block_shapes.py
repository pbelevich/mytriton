import pytest

from mytriton.block_shapes import (
    CudaCooperativeTileLayout,
    CudaKernelLayout,
    CudaRegisterTileLayout,
    CudaTileLayout,
    cuda_kernel_layout,
)
from mytriton.ssa import SSAItem, SSAOp, SSAValue
from mytriton.ssa_verification import SSAVerifier
from mytriton.trace import BOOL, F32, I32, PTR_F32, BlockType, Param


@pytest.mark.parametrize(
    ("tile_shape", "rank", "threads_per_block", "is_rank2"),
    [
        ((32,), 1, 32, False),
        ((4, 8), 2, 32, True),
        ((16, 16), 2, 256, True),
    ],
)
def test_cuda_kernel_layout(
    tile_shape: tuple[int, ...],
    rank: int,
    threads_per_block: int,
    is_rank2: bool,
) -> None:
    layout = CudaKernelLayout(
        output_tile_shape=tile_shape,
        thread_shape=tile_shape,
    )

    assert layout.output_tile_shape == tile_shape
    assert layout.thread_shape == tile_shape
    assert layout.rank == rank
    assert layout.threads_per_block == threads_per_block
    assert layout.is_rank2 is is_rank2


@pytest.mark.parametrize(
    "tile_shape",
    [
        (),
        (0,),
        (-1,),
        (4, 0),
        (2, 3, 4),
    ],
)
def test_cuda_kernel_layout_rejects_invalid_shape(
    tile_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        CudaKernelLayout(
            output_tile_shape=tile_shape,
            thread_shape=tile_shape,
        )


def test_cuda_kernel_layout_is_determined_by_store_operands() -> None:
    internal_value = SSAValue(
        id=0,
        ty=BlockType((2, 3), F32),
    )
    output_pointer = SSAValue(
        id=1,
        ty=BlockType((4, 8), PTR_F32),
    )
    output_value = SSAValue(
        id=2,
        ty=BlockType((4, 8), F32),
    )
    output_mask = SSAValue(
        id=3,
        ty=BlockType((4, 8), BOOL),
    )

    ssa_ops: list[SSAItem] = [
        SSAOp(
            opcode="zeros",
            result=internal_value,
        ),
        SSAOp(
            opcode="store",
            operands=(output_pointer, output_value, output_mask),
        ),
    ]

    layout = cuda_kernel_layout(ssa_ops)

    assert layout.output_tile_shape == (4, 8)
    assert layout.thread_shape == (4, 8)
    assert layout.threads_per_block == 32


def test_reduction_input_determines_cuda_thread_shape() -> None:
    offsets = SSAValue(id=0, ty=BlockType((4,), I32))
    pointers = SSAValue(id=1, ty=BlockType((4,), PTR_F32))
    values = SSAValue(id=2, ty=BlockType((4,), F32))
    total = SSAValue(id=3, ty=F32)

    ssa_ops: list[SSAItem] = [
        SSAOp(
            opcode="arange",
            result=offsets,
            attrs={"start": 0, "end": 4},
        ),
        SSAOp(
            opcode="addptr",
            operands=(Param("x", PTR_F32), offsets),
            result=pointers,
        ),
        SSAOp(
            opcode="load",
            operands=(pointers, None, None),
            result=values,
        ),
        SSAOp(
            opcode="sum",
            operands=(values,),
            result=total,
        ),
        SSAOp(
            opcode="store",
            operands=(Param("out", PTR_F32), total, None),
        ),
    ]

    layout = cuda_kernel_layout(ssa_ops)

    assert layout.output_tile_shape == (1,)
    assert layout.thread_shape == (4,)
    SSAVerifier(layout.threads_per_block).verify(ssa_ops)


def test_cuda_kernel_layout_rejects_mixed_rank_store_domains() -> None:
    rank2_pointer = SSAValue(id=0, ty=BlockType((4, 8), PTR_F32))
    rank2_value = SSAValue(id=1, ty=BlockType((4, 8), F32))
    rank1_pointer = SSAValue(id=2, ty=BlockType((4,), PTR_F32))
    rank1_value = SSAValue(id=3, ty=BlockType((4,), F32))

    ssa_ops: list[SSAItem] = [
        SSAOp(
            opcode="store",
            operands=(rank2_pointer, rank2_value, None),
        ),
        SSAOp(
            opcode="store",
            operands=(rank1_pointer, rank1_value, None),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="mixed rank-1/rank-2 store domains",
    ):
        cuda_kernel_layout(ssa_ops)


def test_cuda_kernel_layout_maps_logical_tiles_to_thread_axes() -> None:
    layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    assert layout.tile_layout((4, 8)) == CudaTileLayout(
        logical_shape=(4, 8),
        thread_axes=(0, 1),
    )
    assert layout.tile_layout(
        (4, 1),
        broadcast_axes=(1,),
    ) == CudaTileLayout(
        logical_shape=(4, 1),
        thread_axes=(0, None),
    )
    assert layout.tile_layout(
        (1, 8),
        broadcast_axes=(0,),
    ) == CudaTileLayout(
        logical_shape=(1, 8),
        thread_axes=(None, 1),
    )


def test_cuda_kernel_layout_preserves_degenerate_thread_axis() -> None:
    layout = CudaKernelLayout(
        output_tile_shape=(1, 8),
        thread_shape=(1, 8),
    )

    assert layout.tile_layout(
        (1, 1),
        broadcast_axes=(1,),
    ) == CudaTileLayout(
        logical_shape=(1, 1),
        thread_axes=(0, None),
    )


@pytest.mark.parametrize(
    "logical_shape",
    [
        (4, 3),
        (2, 8),
        (4,),
        (4, 8, 1),
    ],
)
def test_cuda_kernel_layout_rejects_unmappable_logical_tiles(
    logical_shape: tuple[int, ...],
) -> None:
    layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    with pytest.raises(ValueError):
        layout.tile_layout(logical_shape)


def test_cuda_kernel_layout_separates_output_tile_from_threads() -> None:
    layout = CudaKernelLayout(
        output_tile_shape=(64, 64),
        thread_shape=(8, 32),
    )

    assert layout.output_tile_shape == (64, 64)
    assert layout.thread_shape == (8, 32)
    assert layout.threads_per_block == 256


def test_register_tile_layout_maps_multiple_results_per_thread() -> None:
    layout = CudaRegisterTileLayout(
        logical_shape=(8, 8),
        thread_shape=(4, 4),
    )

    assert layout.register_shape == (2, 2)
    assert layout.registers_per_thread == 4

    assert layout.logical_coordinate(
        thread_coordinate=(2, 3),
        register_coordinate=(0, 0),
    ) == (2, 3)
    assert layout.logical_coordinate(
        thread_coordinate=(2, 3),
        register_coordinate=(0, 1),
    ) == (2, 7)
    assert layout.logical_coordinate(
        thread_coordinate=(2, 3),
        register_coordinate=(1, 0),
    ) == (6, 3)
    assert layout.logical_coordinate(
        thread_coordinate=(2, 3),
        register_coordinate=(1, 1),
    ) == (6, 7)

    logical_coordinates = {
        layout.logical_coordinate(
            thread_coordinate=(thread_row, thread_column),
            register_coordinate=(register_row, register_column),
        )
        for thread_row in range(layout.thread_shape[0])
        for thread_column in range(layout.thread_shape[1])
        for register_row in range(layout.register_shape[0])
        for register_column in range(layout.register_shape[1])
    }

    assert logical_coordinates == {
        (row, column)
        for row in range(layout.logical_shape[0])
        for column in range(layout.logical_shape[1])
    }


def test_cooperative_layout_distributes_a_tile_across_threads() -> None:
    kernel_layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    with pytest.raises(ValueError):
        kernel_layout.tile_layout((4, 16))

    a_layout = kernel_layout.cooperative_tile_layout((4, 16))

    assert a_layout == CudaCooperativeTileLayout(
        logical_shape=(4, 16),
        threads_per_block=32,
        order=(1, 0),
    )
    assert a_layout.size == 64
    assert a_layout.iterations_per_thread == 2

    assert a_layout.linear_index(thread_index=0, iteration=0) == 0
    assert a_layout.linear_index(thread_index=31, iteration=0) == 31
    assert a_layout.linear_index(thread_index=0, iteration=1) == 32
    assert a_layout.linear_index(thread_index=31, iteration=1) == 63

    assert a_layout.coordinates(0) == (0, 0)
    assert a_layout.coordinates(15) == (0, 15)
    assert a_layout.coordinates(16) == (1, 0)
    assert a_layout.coordinates(63) == (3, 15)


def test_cooperative_layout_distributes_b_tile_across_threads() -> None:
    kernel_layout = CudaKernelLayout(
        output_tile_shape=(4, 8),
        thread_shape=(4, 8),
    )

    b_layout = kernel_layout.cooperative_tile_layout((16, 8))

    assert b_layout.logical_shape == (16, 8)
    assert b_layout.threads_per_block == 32
    assert b_layout.size == 128
    assert b_layout.iterations_per_thread == 4

    assert b_layout.coordinates(b_layout.linear_index(thread_index=0, iteration=0)) == (
        0,
        0,
    )
    assert b_layout.coordinates(b_layout.linear_index(thread_index=0, iteration=1)) == (
        4,
        0,
    )
    assert b_layout.coordinates(b_layout.linear_index(thread_index=0, iteration=3)) == (
        12,
        0,
    )


def test_cooperative_layout_marks_tail_threads_inactive() -> None:
    kernel_layout = CudaKernelLayout(
        output_tile_shape=(8, 32),
        thread_shape=(8, 32),
    )
    tile_layout = kernel_layout.cooperative_tile_layout((10, 10))

    assert tile_layout.threads_per_block == 256
    assert tile_layout.size == 100
    assert tile_layout.iterations_per_thread == 1

    assert tile_layout.contains(tile_layout.linear_index(thread_index=99, iteration=0))
    assert not tile_layout.contains(
        tile_layout.linear_index(thread_index=100, iteration=0)
    )


@pytest.mark.parametrize(
    ("logical_shape", "thread_axes"),
    [
        ((), ()),
        ((4, 0), (0, 1)),
        ((4, 8), (0,)),
        ((4, 8), (0, 0)),
        ((4, 8), (0, 2)),
        ((4, 8, 1), (0, 1, None)),
    ],
)
def test_cuda_tile_layout_rejects_invalid_mapping(
    logical_shape: tuple[int, ...],
    thread_axes: tuple[int | None, ...],
) -> None:
    with pytest.raises(ValueError):
        CudaTileLayout(
            logical_shape=logical_shape,
            thread_axes=thread_axes,
        )


@pytest.mark.parametrize(
    "order",
    [
        (),
        (0,),
        (0, 0),
        (0, 2),
        (1, 2),
    ],
)
def test_cooperative_layout_rejects_invalid_order(
    order: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="order must be a permutation"):
        CudaCooperativeTileLayout(
            logical_shape=(4, 8),
            threads_per_block=32,
            order=order,
        )


def test_cooperative_layout_rejects_invalid_thread_or_iteration() -> None:
    layout = CudaCooperativeTileLayout(
        logical_shape=(4, 16),
        threads_per_block=32,
        order=(1, 0),
    )

    with pytest.raises(ValueError, match="invalid thread index"):
        layout.linear_index(thread_index=32, iteration=0)

    with pytest.raises(ValueError, match="invalid cooperative iteration"):
        layout.linear_index(thread_index=0, iteration=2)

    with pytest.raises(ValueError, match="outside tile"):
        layout.coordinates(64)


def test_cuda_kernel_layout_rejects_different_ranks() -> None:
    with pytest.raises(
        ValueError,
        match="output tile and CUDA thread shapes must have the same rank",
    ):
        CudaKernelLayout(
            output_tile_shape=(64, 64),
            thread_shape=(256,),
        )


def test_cuda_layouts_reject_more_than_1024_threads() -> None:
    with pytest.raises(
        ValueError,
        match="CUDA threads per block must be between 1 and 1024",
    ):
        CudaKernelLayout(
            output_tile_shape=(64, 64),
            thread_shape=(32, 33),
        )

    with pytest.raises(
        ValueError,
        match="threads per block must be an integer between 1 and 1024",
    ):
        CudaCooperativeTileLayout(
            logical_shape=(64, 64),
            threads_per_block=1025,
            order=(1, 0),
        )


@pytest.mark.parametrize(
    ("logical_shape", "thread_shape"),
    [
        ((8,), (4,)),
        ((8, 8), (4,)),
        ((8, 0), (4, 4)),
        ((8, 8), (0, 4)),
        ((7, 8), (4, 4)),
        ((4, 8), (8, 4)),
    ],
)
def test_register_tile_layout_rejects_invalid_shapes(
    logical_shape: tuple[int, ...],
    thread_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        CudaRegisterTileLayout(logical_shape, thread_shape)


def test_register_tile_layout_rejects_invalid_coordinates() -> None:
    layout = CudaRegisterTileLayout(
        logical_shape=(8, 8),
        thread_shape=(4, 4),
    )

    with pytest.raises(ValueError, match="invalid thread coordinate"):
        layout.logical_coordinate(
            thread_coordinate=(4, 0),
            register_coordinate=(0, 0),
        )

    with pytest.raises(ValueError, match="invalid thread coordinate"):
        layout.logical_coordinate(
            thread_coordinate=(-1, 0),
            register_coordinate=(0, 0),
        )

    with pytest.raises(ValueError, match="invalid register coordinate"):
        layout.logical_coordinate(
            thread_coordinate=(0, 0),
            register_coordinate=(2, 0),
        )

    with pytest.raises(ValueError, match="invalid register coordinate"):
        layout.logical_coordinate(
            thread_coordinate=(0, 0),
            register_coordinate=(0, -1),
        )


def test_cuda_kernel_layout_builds_output_register_tile_layout() -> None:
    kernel_layout = CudaKernelLayout(
        output_tile_shape=(8, 8),
        thread_shape=(4, 4),
    )

    assert kernel_layout.register_tile_layout() == CudaRegisterTileLayout(
        logical_shape=(8, 8),
        thread_shape=(4, 4),
    )


@pytest.mark.parametrize(
    ("output_shape", "expected_thread_shape"),
    [
        ((4, 8), (4, 8)),
        ((8, 8), (4, 8)),
        ((16, 32), (4, 8)),
    ],
)
def test_dot_kernel_layout_uses_one_warp_register_tile(
    output_shape: tuple[int, ...],
    expected_thread_shape: tuple[int, ...],
) -> None:
    rows, columns = output_shape

    lhs = SSAValue(
        id=0,
        ty=BlockType((rows, 16), F32),
    )
    rhs = SSAValue(
        id=1,
        ty=BlockType((16, columns), F32),
    )
    dot = SSAValue(
        id=2,
        ty=BlockType(output_shape, F32),
    )
    pointers = SSAValue(
        id=3,
        ty=BlockType(output_shape, PTR_F32),
    )

    ssa_ops: list[SSAItem] = [
        SSAOp(
            opcode="dot",
            operands=(lhs, rhs),
            result=dot,
        ),
        SSAOp(
            opcode="store",
            operands=(pointers, dot, None),
        ),
    ]

    layout = cuda_kernel_layout(ssa_ops)

    assert layout.output_tile_shape == output_shape
    assert layout.thread_shape == expected_thread_shape
    assert layout.threads_per_block <= 32

    register_layout = layout.register_tile_layout()
    assert register_layout.logical_shape == output_shape
    assert register_layout.thread_shape == expected_thread_shape


def test_non_dot_kernel_keeps_one_thread_per_result() -> None:
    shape = (8, 8)
    pointers = SSAValue(
        id=0,
        ty=BlockType(shape, PTR_F32),
    )
    values = SSAValue(
        id=1,
        ty=BlockType(shape, F32),
    )

    layout = cuda_kernel_layout(
        [
            SSAOp(
                opcode="store",
                operands=(pointers, values, None),
            )
        ]
    )

    assert layout.output_tile_shape == shape
    assert layout.thread_shape == shape
    assert layout.register_tile_layout().register_shape == (1, 1)
