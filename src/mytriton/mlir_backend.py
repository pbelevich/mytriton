from __future__ import annotations


class MLIRUnavailableError(RuntimeError):
    pass


def mlir_available() -> bool:
    try:
        import mlir.ir
        import mlir.passmanager  # noqa: F401
    except ImportError:
        return False

    return True


def require_mlir():
    try:
        # Importing these modules registers the dialects in typical MLIR builds.
        from mlir.dialects import arith, func, memref, scf  # noqa: F401
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

    with Context() as _, Location.unknown():
        module = Module.parse(mlir_text)
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        return str(module)
