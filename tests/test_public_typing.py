import inspect

import mytriton as triton
import mytriton.language as tl


def _mypy_contract() -> None:
    tl.load(1)  # type: ignore[arg-type]
    tl.store(1, 2.0)  # type: ignore[arg-type]
    tl.arange(0, 1.0)  # type: ignore[arg-type]
    tl.where(1, 1.0, 2.0)  # type: ignore[arg-type]
    tl.where(True, "negative", 2.0)  # type: ignore[arg-type]
    triton.cdiv(1.0, 2)  # type: ignore[arg-type]

    @triton.jit
    def typed_kernel(value: int) -> None:
        del value

    typed_kernel[(1,)]("not an int")  # type: ignore[arg-type]


def test_public_dsl_functions_have_annotations():
    for function in (
        tl.program_id,
        tl.arange,
        tl.load,
        tl.store,
        tl.minimum,
        tl.maximum,
        tl.exp,
        tl.where,
    ):
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(
            parameter.annotation is not inspect.Signature.empty
            for parameter in signature.parameters.values()
        )
