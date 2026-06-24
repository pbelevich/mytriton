import inspect
from dataclasses import dataclass
from typing import Any

from .cuda_codegen import SSACUDACodegen
from .cuda_utils import CudaKernelCache, execute_cuda_if_needed
from .ssa import SSALowering, SSAOp
from .trace import VectorType, constexpr, make_runtime_params, trace


def _cuda_threads_per_block(ssa_ops: list[SSAOp]) -> int:
    vector_widths = {
        op.result.ty.size
        for op in ssa_ops
        if op.result is not None and isinstance(op.result.ty, VectorType)
    }

    if not vector_widths:
        return 1

    if len(vector_widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(vector_widths))
        raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")

    threads_per_block = next(iter(vector_widths))
    if not 1 <= threads_per_block <= 1024:
        raise ValueError(
            "CUDA threads per block must be between 1 and 1024, "
            f"got {threads_per_block}"
        )

    return threads_per_block


def _constexpr_key(meta: dict[str, object]) -> tuple[tuple[str, object], ...]:
    supported = (bool, int, float, str)
    for name, value in meta.items():
        if not isinstance(value, supported):
            raise TypeError(f"{name}: constexpr value must be bool, int, float, or str")
    return tuple(meta.items())


@dataclass(frozen=True)
class _CompilationArtifact:
    ops: tuple[Any, ...]
    ssa_ops: tuple[SSAOp, ...]
    cuda_src: str
    threads_per_block: int


class CompiledKernel:
    def __init__(self, fn):
        self.fn = fn
        signature = inspect.signature(fn)
        annotations = inspect.get_annotations(fn, eval_str=True)
        parameters = [
            parameter.replace(
                annotation=annotations.get(name, parameter.annotation),
            )
            for name, parameter in signature.parameters.items()
        ]
        self.signature = signature.replace(parameters=parameters)
        self.compilation_cache: dict[tuple[object, ...], _CompilationArtifact] = {}
        self.cuda_cache: CudaKernelCache = {}

    def clear_cache(self) -> None:
        self.compilation_cache.clear()
        self.cuda_cache.clear()

    def __getitem__(self, grid):

        def launch(*args, **kwargs):
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()

            meta = {
                name: bound.arguments[name]
                for name, parameter in self.signature.parameters.items()
                if parameter.annotation is constexpr
            }

            launch_grid = grid(meta) if callable(grid) else grid
            if isinstance(launch_grid, int):
                launch_grid = (launch_grid,)
            launch_grid = tuple(launch_grid)

            if not 1 <= len(launch_grid) <= 3 or any(
                type(size) is not int or size <= 0 for size in launch_grid
            ):
                raise ValueError(f"invalid launch grid: {launch_grid}")

            runtime_args = tuple(
                bound.arguments[name]
                for name, parameter in self.signature.parameters.items()
                if parameter.annotation is not constexpr
            )
            params = make_runtime_params(self.signature, bound.arguments)
            cache_key = (
                tuple(param.ty for param in params),
                _constexpr_key(meta),
            )

            artifact = self.compilation_cache.get(cache_key)
            if artifact is None:
                ops, params = trace(
                    self.fn,
                    self.signature,
                    bound.arguments,
                    runtime_params=params,
                )
                ssa_ops = SSALowering().lower(ops)
                threads_per_block = _cuda_threads_per_block(ssa_ops)
                cuda_src = SSACUDACodegen().generate(
                    self.fn.__name__,
                    ssa_ops,
                    params,
                )
                artifact = _CompilationArtifact(
                    ops=tuple(ops),
                    ssa_ops=tuple(ssa_ops),
                    cuda_src=cuda_src,
                    threads_per_block=threads_per_block,
                )
                self.compilation_cache[cache_key] = artifact

            execute_cuda_if_needed(
                kernel_cache=self.cuda_cache,
                cuda_src=artifact.cuda_src,
                kernel_name=self.fn.__name__,
                launch_grid=launch_grid,
                threads_per_block=artifact.threads_per_block,
                runtime_args=runtime_args,
            )

            return list(artifact.ops), list(artifact.ssa_ops), artifact.cuda_src

        return launch


def jit(fn):
    return CompiledKernel(fn)
