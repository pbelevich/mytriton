from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .trace import BlockType

if TYPE_CHECKING:
    from .ssa import SSAItem


def prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


@dataclass(frozen=True)
class CudaTileLayout:
    """Mapping from logical tile dimensions to CUDA thread dimensions."""

    logical_shape: tuple[int, ...]
    thread_axes: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.logical_shape) <= 2:
            raise ValueError(
                f"CUDA tile layouts support rank 1 or 2, got {self.logical_shape}"
            )

        if any(type(dim) is not int or dim <= 0 for dim in self.logical_shape):
            raise ValueError(
                f"logical tile dimensions must be positive integers, "
                f"got {self.logical_shape}"
            )

        if len(self.logical_shape) != len(self.thread_axes):
            raise ValueError(
                "logical shape and thread axes must have the same rank, "
                f"got {self.logical_shape} and {self.thread_axes}"
            )

        mapped_axes = [axis for axis in self.thread_axes if axis is not None]

        if any(axis < 0 or axis >= len(self.logical_shape) for axis in mapped_axes):
            raise ValueError(
                f"CUDA thread axes must be valid logical axes, got {self.thread_axes}"
            )

        if len(mapped_axes) != len(set(mapped_axes)):
            raise ValueError(f"CUDA thread axes must be unique, got {self.thread_axes}")


@dataclass(frozen=True)
class CudaCooperativeTileLayout:
    """Distribution of a logical tile across all threads in a CUDA block."""

    logical_shape: tuple[int, ...]
    threads_per_block: int
    order: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.logical_shape) <= 2:
            raise ValueError(
                f"cooperative CUDA tiles support rank 1 or 2, got {self.logical_shape}"
            )

        if any(type(dim) is not int or dim <= 0 for dim in self.logical_shape):
            raise ValueError(
                f"logical tile dimensions must be positive integers, "
                f"got {self.logical_shape}"
            )

        if (
            type(self.threads_per_block) is not int
            or not 1 <= self.threads_per_block <= 1024
        ):
            raise ValueError(
                f"threads per block must be an integer between 1 and 1024, "
                f"got {self.threads_per_block}"
            )

        expected_axes = tuple(range(len(self.logical_shape)))
        if tuple(sorted(self.order)) != expected_axes:
            raise ValueError(
                f"order must be a permutation of {expected_axes}, got {self.order}"
            )

    @property
    def size(self) -> int:
        return prod(self.logical_shape)

    @property
    def iterations_per_thread(self) -> int:
        return (self.size + self.threads_per_block - 1) // self.threads_per_block

    def linear_index(self, thread_index: int, iteration: int) -> int:
        if not 0 <= thread_index < self.threads_per_block:
            raise ValueError(
                f"invalid thread index {thread_index} for "
                f"{self.threads_per_block} threads"
            )

        if not 0 <= iteration < self.iterations_per_thread:
            raise ValueError(
                f"invalid cooperative iteration {iteration}; expected "
                f"0 <= iteration < {self.iterations_per_thread}"
            )

        return thread_index + iteration * self.threads_per_block

    def contains(self, linear_index: int) -> bool:
        return 0 <= linear_index < self.size

    def coordinates(self, linear_index: int) -> tuple[int, ...]:
        if not self.contains(linear_index):
            raise ValueError(
                f"linear index {linear_index} is outside tile {self.logical_shape}"
            )

        coordinates = [0] * len(self.logical_shape)
        remaining = linear_index

        for axis in self.order:
            coordinates[axis] = remaining % self.logical_shape[axis]
            remaining //= self.logical_shape[axis]

        return tuple(coordinates)


@dataclass(frozen=True)
class CudaKernelLayout:
    """Logical output tile and physical CUDA thread organization."""

    output_tile_shape: tuple[int, ...]
    thread_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, shape in (
            ("output tile", self.output_tile_shape),
            ("CUDA thread", self.thread_shape),
        ):
            if not shape:
                raise ValueError(f"{name} shape must have at least one dimension")

            if len(shape) > 2:
                raise ValueError(f"{name} shape supports rank 1 or 2, got {shape}")

            if any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError(
                    f"{name} dimensions must be positive integers, got {shape}"
                )

        if len(self.output_tile_shape) != len(self.thread_shape):
            raise ValueError(
                "output tile and CUDA thread shapes must have the same rank, "
                f"got {self.output_tile_shape} and {self.thread_shape}"
            )

        threads_per_block = prod(self.thread_shape)
        if threads_per_block > 1024:
            raise ValueError(
                "CUDA threads per block must be between 1 and 1024, "
                f"got {threads_per_block}"
            )

    @property
    def rank(self) -> int:
        return len(self.thread_shape)

    @property
    def is_rank2(self) -> bool:
        return self.rank == 2

    @property
    def threads_per_block(self) -> int:
        return prod(self.thread_shape)

    def tile_layout(
        self,
        logical_shape: tuple[int, ...],
        *,
        broadcast_axes: tuple[int, ...] = (),
    ) -> CudaTileLayout:
        if len(logical_shape) != self.rank:
            raise ValueError(
                f"logical tile rank must match CUDA thread rank: "
                f"{logical_shape} vs {self.thread_shape}"
            )

        if any(axis < 0 or axis >= self.rank for axis in broadcast_axes):
            raise ValueError(
                f"invalid broadcast axes {broadcast_axes} for {logical_shape}"
            )

        thread_axes: list[int | None] = []

        for axis, (logical_dim, thread_dim) in enumerate(
            zip(logical_shape, self.thread_shape, strict=True)
        ):
            if axis in broadcast_axes:
                if logical_dim != 1:
                    raise ValueError(
                        f"broadcast axis {axis} must have size 1, got {logical_shape}"
                    )
                thread_axes.append(None)
            elif logical_dim == thread_dim:
                thread_axes.append(axis)
            elif logical_dim == 1:
                thread_axes.append(None)
            else:
                raise ValueError(
                    f"cannot project logical tile {logical_shape} onto "
                    f"CUDA thread shape {self.thread_shape}"
                )

        return CudaTileLayout(
            logical_shape=logical_shape,
            thread_axes=tuple(thread_axes),
        )

    def cooperative_tile_layout(
        self,
        logical_shape: tuple[int, ...],
        *,
        order: tuple[int, ...] | None = None,
    ) -> CudaCooperativeTileLayout:
        if len(logical_shape) != self.rank:
            raise ValueError(
                f"cooperative tile rank must match CUDA kernel rank: "
                f"{logical_shape} vs {self.thread_shape}"
            )

        if order is None:
            order = tuple(reversed(range(len(logical_shape))))

        return CudaCooperativeTileLayout(
            logical_shape=logical_shape,
            threads_per_block=self.threads_per_block,
            order=order,
        )


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


def store_block_shapes(ssa_ops: list[SSAItem]) -> list[tuple[int, ...]]:
    """Collect block shapes that participate in observable memory writes."""

    from .ssa import SSAForRange, SSAValue

    shapes = []

    for op in ssa_ops:
        if isinstance(op, SSAForRange):
            shapes.extend(store_block_shapes(op.body))
            continue

        if op.opcode != "store":
            continue

        for operand in op.operands:
            if isinstance(operand, SSAValue) and isinstance(operand.ty, BlockType):
                shapes.append(operand.ty.shape)

    return shapes


def reduction_block_shapes(ssa_ops: list[SSAItem]) -> list[tuple[int, ...]]:
    """Collect input shapes of reductions that require cooperative threads."""

    from .ssa import SSAForRange, SSAValue

    shapes = []

    for op in ssa_ops:
        if isinstance(op, SSAForRange):
            shapes.extend(reduction_block_shapes(op.body))
            continue

        if op.opcode not in {"sum", "max", "min"}:
            continue

        operand = op.operands[0]
        if isinstance(operand, SSAValue) and isinstance(operand.ty, BlockType):
            shapes.append(operand.ty.shape)

    return shapes


def _infer_cuda_kernel_tile_shape(ssa_ops: list[SSAItem]) -> tuple[int, ...]:
    shapes = store_block_shapes(ssa_ops)

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
        if rank1_shapes:
            rendered = ", ".join(str(shape) for shape in shapes)
            raise ValueError(
                "CUDA lowering does not support mixed rank-1/rank-2 "
                f"store domains, got {rendered}"
            )

        block_shape = broadcast_shapes(*rank2_shapes)

        if len(block_shape) != 2:
            raise ValueError(f"expected rank-2 CUDA block shape, got {block_shape}")

        return block_shape

    widths = {shape[0] for shape in rank1_shapes}
    if len(widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(widths))
        raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")

    return (next(iter(widths)),)


def _infer_cuda_thread_shape(
    output_tile_shape: tuple[int, ...],
    reduction_shapes: list[tuple[int, ...]],
) -> tuple[int, ...]:
    if not reduction_shapes:
        return output_tile_shape

    if any(len(shape) != 1 for shape in reduction_shapes):
        rendered = ", ".join(str(shape) for shape in reduction_shapes)
        raise ValueError(f"CUDA reductions require rank-1 inputs, got {rendered}")

    reduction_widths = {shape[0] for shape in reduction_shapes}
    if len(reduction_widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(reduction_widths))
        raise ValueError(f"CUDA reductions require one thread width, got: {rendered}")

    reduction_width = next(iter(reduction_widths))

    if len(output_tile_shape) == 1:
        output_width = output_tile_shape[0]
        if output_width not in (1, reduction_width):
            raise ValueError(
                f"reduction width {reduction_width} does not match "
                f"output tile width {output_width}"
            )
        return (reduction_width,)

    if prod(output_tile_shape) != reduction_width:
        raise ValueError(
            f"reduction width {reduction_width} does not match "
            f"output tile size {prod(output_tile_shape)}"
        )

    return output_tile_shape


def cuda_kernel_layout(ssa_ops: list[SSAItem]) -> CudaKernelLayout:
    output_tile_shape = _infer_cuda_kernel_tile_shape(ssa_ops)
    thread_shape = _infer_cuda_thread_shape(
        output_tile_shape,
        reduction_block_shapes(ssa_ops),
    )

    return CudaKernelLayout(
        output_tile_shape=output_tile_shape,
        thread_shape=thread_shape,
    )


def cuda_threads_per_block(ssa_ops: list[SSAItem]) -> int:
    layout = cuda_kernel_layout(ssa_ops)
    threads = layout.threads_per_block
    if not 1 <= threads <= 1024:
        raise ValueError(
            f"CUDA threads per block must be between 1 and 1024, got {threads}"
        )
    return threads
