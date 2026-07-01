import pytest

from mytriton.mlir_codegen import MLIRCodegen
from mytriton.ssa import SSAOp, SSAValue
from mytriton.trace import I32, VectorType


def test_mlir_codegen_rejects_nonzero_program_id_axis():
    ops = [
        SSAOp(
            opcode="program_id",
            result=SSAValue(id=0, ty=I32),
            attrs={"axis": 1},
        )
    ]

    with pytest.raises(TypeError, match=r"MLIR MVP supports only program_id\(0\)"):
        MLIRCodegen().generate("kernel", ops, [])


def test_mlir_codegen_rejects_nonzero_arange_start():
    ops = [
        SSAOp(
            opcode="arange",
            result=SSAValue(id=0, ty=VectorType(size=4, element=I32)),
            attrs={"start": 4, "end": 8},
        )
    ]

    with pytest.raises(TypeError, match=r"MLIR MVP supports only arange\(0, N\)"):
        MLIRCodegen().generate("kernel", ops, [])
