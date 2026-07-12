from __future__ import annotations

from typing import TYPE_CHECKING

from .trace import BlockType

if TYPE_CHECKING:
    from .ssa import SSAOp


def prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()

    max_rank = max(len(shape) for shape in shapes)
    padded = [(1,) * (max_rank - len(shape)) + shape for shape in shapes]

    dims = []
    for dim_values in zip(*padded, strict=True):
        non_ones = {dim for dim in dim_values if dim != 1}
        if len(non_ones) > 1:
            rendered = ", ".join(
                "x".join(str(dim) for dim in shape) for shape in shapes
            )
            raise ValueError(f"cannot broadcast shapes: {rendered}")

        dims.append(next(iter(non_ones), 1))

    return tuple(dims)


def result_block_shapes(ssa_ops: list[SSAOp]) -> list[tuple[int, ...]]:
    return [
        op.result.ty.shape
        for op in ssa_ops
        if op.result is not None and isinstance(op.result.ty, BlockType)
    ]


def operand_block_shapes(operands) -> list[tuple[int, ...]]:
    shapes = []
    for operand in operands:
        ty = getattr(operand, "ty", None)
        if isinstance(ty, BlockType):
            shapes.append(ty.shape)
    return shapes


def store_block_shapes(ssa_ops: list[SSAOp]) -> list[tuple[int, ...]]:
    shapes = []
    for op in ssa_ops:
        if op.opcode == "store":
            shapes.extend(operand_block_shapes(op.operands))
    return shapes


def cuda_kernel_block_shape(ssa_ops: list[SSAOp]) -> tuple[int, ...]:
    store_shapes = store_block_shapes(ssa_ops)
    store_rank2_shapes = [shape for shape in store_shapes if len(shape) == 2]

    if store_rank2_shapes:
        block_shape = broadcast_shapes(*store_rank2_shapes)

        if len(block_shape) != 2:
            raise ValueError(f"expected rank-2 CUDA block shape, got {block_shape}")

        return block_shape

    shapes = result_block_shapes(ssa_ops)

    if not shapes:
        return (1,)

    if any(len(shape) > 2 for shape in shapes):
        rendered = ", ".join(str(shape) for shape in shapes)
        raise ValueError(
            f"CUDA lowering supports only rank-1/rank-2 blocks, got {rendered}"
        )

    rank2_shapes = [shape for shape in shapes if len(shape) == 2]
    rank1_shapes = [shape for shape in shapes if len(shape) == 1]

    if rank2_shapes:
        block_shape = broadcast_shapes(*rank2_shapes)

        if len(block_shape) != 2:
            raise ValueError(f"expected rank-2 CUDA block shape, got {block_shape}")

        return block_shape

    widths = {shape[0] for shape in rank1_shapes}
    if len(widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(widths))
        raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")

    return (next(iter(widths)),)


def cuda_threads_per_block(ssa_ops: list[SSAOp]) -> int:
    block_shape = cuda_kernel_block_shape(ssa_ops)
    threads_per_block = prod(block_shape)

    if not 1 <= threads_per_block <= 1024:
        raise ValueError(
            "CUDA threads per block must be between 1 and 1024, "
            f"got {threads_per_block}"
        )

    return threads_per_block
