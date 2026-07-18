import inspect
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, ParamSpec, TypeAlias, cast

from .ast_frontend import trace
from .block_shapes import cuda_threads_per_block
from .cuda_codegen import SSACUDACodegen
from .cuda_utils import (
    CudaKernelCache,
    CudaUnavailableError,
    cuda_chip,
    cuda_execution_required,
    execute_cuda_if_needed,
    execute_mlir_cubin_if_needed,
)
from .mlir_codegen import MLIRCodegen, compile_mlir_source_to_cubin
from .optim import ConstantFoldPass, CSEPass, DCEPass, PassManager
from .ssa import SSALowering, SSAOp
from .ssa_verification import SSAVerifier
from .trace import (
    is_constexpr_annotation,
    make_runtime_params,
)

P = ParamSpec("P")
Meta: TypeAlias = dict[str, Any]
LaunchDimensions: TypeAlias = int | Sequence[int]
Grid: TypeAlias = LaunchDimensions | Callable[[Meta], LaunchDimensions]
CompilationResult: TypeAlias = tuple[list[Any], list[SSAOp], str]


Backend: TypeAlias = Literal["cuda", "mlir"]


def _constexpr_key(
    meta: dict[str, object],
) -> tuple[tuple[str, type[object], object], ...]:
    supported = (bool, int, float, str)
    entries = []
    for name, value in meta.items():
        if not isinstance(value, supported):
            raise TypeError(f"{name}: constexpr value must be bool, int, float, or str")

        normalized = float(value).hex() if isinstance(value, float) else value
        entries.append((name, type(value), normalized))

    return tuple(entries)


@dataclass(frozen=True)
class _CompilationArtifact:
    ops: tuple[Any, ...]
    ssa_ops: tuple[SSAOp, ...]
    src: str
    threads_per_block: int
    cubin: bytes | None = None


class CompiledKernel(Generic[P]):
    def __init__(self, fn: Callable[P, Any]) -> None:
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

    def _resolve_backend(self) -> Backend:
        selected = os.environ.get("MYTRITON_BACKEND", "cuda")
        if selected not in {"cuda", "mlir"}:
            raise ValueError(
                f"unsupported backend {selected!r}; expected 'cuda' or 'mlir'"
            )
        return cast(Backend, selected)

    def __getitem__(self, grid: Grid) -> Callable[P, CompilationResult]:

        def launch(*args: P.args, **kwargs: P.kwargs) -> CompilationResult:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()

            meta: Meta = {
                name: bound.arguments[name]
                for name, parameter in self.signature.parameters.items()
                if is_constexpr_annotation(parameter.annotation)
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
                if not is_constexpr_annotation(parameter.annotation)
            )
            params = make_runtime_params(self.signature, bound.arguments)
            backend = self._resolve_backend()
            chip = None
            if backend == "mlir":
                try:
                    chip = cuda_chip()
                except CudaUnavailableError:
                    chip = "sm_80"

            cache_key = (
                backend,
                chip,
                tuple(param.ty for param in params),
                _constexpr_key(meta),
            )

            artifact = self.compilation_cache.get(cache_key)
            if artifact is None:
                trace_fn = trace
                ops, params = trace_fn(
                    self.fn,
                    self.signature,
                    bound.arguments,
                    runtime_params=params,
                )
                ssa_ops = SSALowering().lower(ops)
                threads_per_block = cuda_threads_per_block(ssa_ops)

                # The optimizer assumes lowering produced valid SSA.
                verifier = SSAVerifier(threads_per_block)
                verifier.verify(ssa_ops)

                # PassManager re-runs the verifier after each rewrite pass, so
                # a broken optimization fails before CUDA codegen sees it.
                ssa_ops = PassManager(
                    passes=[
                        ConstantFoldPass(),
                        CSEPass(),
                        DCEPass(),
                    ],
                    verifier=verifier,
                ).run(ssa_ops)

                # Recompute after optimization: DCE/CSE may remove the vector
                # ops that originally determined the CUDA block size.
                threads_per_block = cuda_threads_per_block(ssa_ops)
                SSAVerifier(threads_per_block).verify(ssa_ops)

                cubin = None
                if backend == "mlir":
                    src = MLIRCodegen().generate(self.fn.__name__, ssa_ops, params)
                else:
                    src = SSACUDACodegen().generate(
                        self.fn.__name__,
                        ssa_ops,
                        params,
                    )

                artifact = _CompilationArtifact(
                    ops=tuple(ops),
                    ssa_ops=tuple(ssa_ops),
                    src=src,
                    threads_per_block=threads_per_block,
                    cubin=cubin,
                )
                self.compilation_cache[cache_key] = artifact

            if backend == "mlir":
                needs_execution = cuda_execution_required(
                    runtime_args, backend_name="MLIR"
                )
                if artifact.cubin is None and needs_execution:
                    assert chip is not None
                    artifact = replace(
                        artifact,
                        cubin=compile_mlir_source_to_cubin(artifact.src, chip=chip),
                    )
                    self.compilation_cache[cache_key] = artifact

                if needs_execution:
                    assert artifact.cubin is not None
                    execute_mlir_cubin_if_needed(
                        kernel_cache=self.cuda_cache,
                        cubin=artifact.cubin,
                        kernel_name=self.fn.__name__,
                        launch_grid=launch_grid,
                        threads_per_block=artifact.threads_per_block,
                        runtime_args=runtime_args,
                    )
            else:
                execute_cuda_if_needed(
                    kernel_cache=self.cuda_cache,
                    cuda_src=artifact.src,
                    kernel_name=self.fn.__name__,
                    launch_grid=launch_grid,
                    threads_per_block=artifact.threads_per_block,
                    runtime_args=runtime_args,
                )

            return (
                list(deepcopy(artifact.ops)),
                list(deepcopy(artifact.ssa_ops)),
                artifact.src,
            )

        return launch


def jit(fn: Callable[P, Any]) -> CompiledKernel[P]:
    return CompiledKernel(fn)
