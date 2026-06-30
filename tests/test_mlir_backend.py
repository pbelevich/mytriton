from textwrap import dedent

import pytest

from mytriton.mlir_backend import (
    mlir_available,
    parse_mlir_module,
    run_mlir_pass_pipeline,
)

pytestmark = pytest.mark.skipif(
    not mlir_available(),
    reason="MLIR Python bindings are not installed",
)


def test_parse_mlir_module():
    mlir_text = dedent(
        """\
        module {
          func.func @identity(%arg0: i32) -> i32 {
            return %arg0 : i32
          }
        }
        """
    )

    rendered = parse_mlir_module(mlir_text)

    assert "module" in rendered
    assert "func.func @identity" in rendered
    assert "return" in rendered


def test_run_mlir_pass_pipeline():
    mlir_text = dedent(
        """\
        module {
          func.func @add_zero(%arg0: i32) -> i32 {
            %c0 = arith.constant 0 : i32
            %0 = arith.addi %arg0, %c0 : i32
            return %0 : i32
          }
        }
        """
    )

    rendered = run_mlir_pass_pipeline(
        mlir_text,
        "builtin.module(func.func(canonicalize,cse))",
    )

    assert "func.func @add_zero" in rendered
