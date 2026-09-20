import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ast_frontend import ASTFrontendError


@triton.jit
def pass_kernel(out):
    pass


@triton.jit
def bare_return_kernel(out):
    return


@triton.jit
def return_after_store_kernel(out):
    tl.store(out, 1.0)
    return


@triton.jit
def return_before_store_kernel(out):
    return
    tl.store(out, 1.0)


@triton.jit
def return_value_kernel(out):
    return 1


@triton.jit
def runtime_loop_return_kernel(out, n):
    for _ in range(n):
        return


def test_pass_kernel_compiles_to_empty_body():
    out = np.empty(1, dtype=np.float32)

    expression_ops, ssa_ops, cuda_src = pass_kernel[(1,)](out)

    assert expression_ops == []
    assert ssa_ops == []
    assert "void pass_kernel(float* out) {\n}" in cuda_src


def test_bare_return_kernel_compiles_to_empty_body():
    out = np.empty(1, dtype=np.float32)

    expression_ops, ssa_ops, cuda_src = bare_return_kernel[(1,)](out)

    assert expression_ops == []
    assert ssa_ops == []
    assert "void bare_return_kernel(float* out) {\n}" in cuda_src


def test_bare_return_preserves_preceding_operations():
    out = np.empty(1, dtype=np.float32)

    _, _, cuda_src = return_after_store_kernel[(1,)](out)

    assert "out[0] = 1.0f;" in cuda_src


def test_bare_return_skips_following_operations():
    out = np.empty(1, dtype=np.float32)

    expression_ops, ssa_ops, cuda_src = return_before_store_kernel[(1,)](out)

    assert expression_ops == []
    assert ssa_ops == []
    assert "out[0] = 1.0f;" not in cuda_src


def test_return_value_is_rejected():
    out = np.empty(1, dtype=np.float32)

    with pytest.raises(ASTFrontendError, match="kernel return values"):
        return_value_kernel[(1,)](out)


def test_return_inside_runtime_loop_is_rejected():
    out = np.empty(1, dtype=np.float32)

    with pytest.raises(ASTFrontendError, match="return inside a runtime for loop"):
        runtime_loop_return_kernel[(1,)](out, 1)
