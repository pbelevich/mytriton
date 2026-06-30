from __future__ import annotations

from dataclasses import dataclass

FUNC_CLEANUP_PIPELINE = "builtin.module(func.func(cse,canonicalize))"

GPU_CLEANUP_PIPELINE = "builtin.module(gpu.module(gpu.func(cse,canonicalize)))"

GENERIC_CLEANUP_PIPELINE = "builtin.module(any(cse,canonicalize))"


class MLIRUnavailableError(RuntimeError):
    pass


class MLIRPassPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class MLIRPipelineResult:
    ok: bool
    output: str
    error: str | None = None


def mlir_available() -> bool:
    try:
        import mlir.ir
        import mlir.passmanager  # noqa: F401
    except ImportError:
        return False

    return True


def require_mlir():
    try:
        # Importing these modules usually registers dialects in MLIR Python builds.
        from mlir.dialects import arith, func, gpu, memref, scf  # noqa: F401
        from mlir.ir import Context, Location, Module
        from mlir.passmanager import PassManager
    except ImportError as error:
        raise MLIRUnavailableError(
            "MLIR Python bindings are not installed or common dialect bindings "
            "are not importable."
        ) from error

    return Context, Location, Module, PassManager


def parse_mlir_module(mlir_text: str) -> str:
    Context, Location, Module, _ = require_mlir()

    with Context() as _, Location.unknown():
        module = Module.parse(mlir_text)
        return str(module)


def run_mlir_pass_pipeline(mlir_text: str, pipeline: str) -> str:
    Context, Location, Module, PassManager = require_mlir()

    try:
        with Context() as _, Location.unknown():
            module = Module.parse(mlir_text)
            pm = PassManager.parse(pipeline)
            pm.run(module.operation)
            return str(module)
    except Exception as error:
        raise MLIRPassPipelineError(
            f"failed to run MLIR pass pipeline {pipeline!r}: {error}"
        ) from error


def try_run_mlir_pass_pipeline(mlir_text: str, pipeline: str) -> MLIRPipelineResult:
    try:
        return MLIRPipelineResult(
            ok=True,
            output=run_mlir_pass_pipeline(mlir_text, pipeline),
        )
    except Exception as error:
        return MLIRPipelineResult(
            ok=False,
            output=mlir_text,
            error=str(error),
        )
