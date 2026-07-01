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


def _require_mlir():
    try:
        from mlir.ir import Context, Location, Module
        from mlir.passmanager import PassManager
    except ImportError as error:
        raise MLIRUnavailableError("MLIR Python bindings are not available") from error

    return Context, Location, Module, PassManager


def run_mlir_pass_pipeline(mlir_text: str, pipeline: str) -> str:
    Context, Location, Module, PassManager = _require_mlir()

    with Context() as _, Location.unknown():
        module = Module.parse(mlir_text)
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        return str(module)


def run_pipeline(mlir_text: str, stages: list[tuple[str, str]]) -> str:
    current = mlir_text
    for name, pipeline in stages:
        try:
            current = run_mlir_pass_pipeline(current, pipeline)
        except Exception as error:
            raise RuntimeError(f"MLIR stage {name!r} failed:\n{error}") from error
    return current


def gpu_to_cubin_stages(*, chip: str = "sm_80", opt_level: int = 3):
    return [
        (
            "cleanup",
            "builtin.module(any(cse,canonicalize))",
        ),
        (
            "attach-nvvm-target",
            "builtin.module("
            f"nvvm-attach-target{{module=.* chip={chip} O={opt_level}}}"
            ")",
        ),
        (
            "convert-scf-to-cf",
            "builtin.module(gpu.module(gpu.func(convert-scf-to-cf)))",
        ),
        (
            "convert-gpu-to-nvvm",
            "builtin.module(gpu.module(convert-gpu-to-nvvm))",
        ),
        (
            "gpu-module-to-binary",
            "builtin.module(gpu-module-to-binary)",
        ),
    ]


def extract_gpu_binary(mlir_text: str, *, symbol_name: str = "kernels") -> bytes:
    marker = f"gpu.binary @{symbol_name}"
    start = mlir_text.find(marker)
    if start < 0:
        raise ValueError(f"MLIR module does not contain {marker}")

    object_start = mlir_text.find(', "', start)
    if object_start < 0:
        raise ValueError(f"MLIR gpu.binary @{symbol_name} does not contain an object")

    return _decode_mlir_byte_string(mlir_text, object_start + 3)


def _decode_mlir_byte_string(text: str, start: int) -> bytes:
    hex_digits = set("0123456789abcdefABCDEF")
    output = bytearray()
    index = start

    while index < len(text):
        char = text[index]
        if char == '"':
            return bytes(output)

        if (
            char == "\\"
            and index + 2 < len(text)
            and text[index + 1] in hex_digits
            and text[index + 2] in hex_digits
        ):
            output.append(int(text[index + 1 : index + 3], 16))
            index += 3
            continue

        if char == "\\" and index + 1 < len(text):
            output.append(ord(text[index + 1]))
            index += 2
            continue

        output.extend(char.encode("utf-8"))
        index += 1

    raise ValueError("unterminated MLIR byte string")
