from __future__ import annotations

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.cuda_codegen import SSACUDACodegen
from mytriton.trace import Const


@triton.jit
def inconsistent_width_kernel(out, n):
    short_offsets = tl.arange(0, 128)
    long_offsets = tl.arange(0, 256)
    tl.store(out + short_offsets, 1.0, mask=short_offsets < n)
    tl.store(out + long_offsets, 2.0, mask=long_offsets < n)


@triton.jit
def tile_kernel(out, n, TILE_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    tl.store(out + offsets, 1.0, mask=offsets < n)


@triton.jit
def scalar_copy_kernel(source, out):
    value = tl.load(source)
    tl.store(out, value)


@triton.jit
def nested_pointer_kernel(out):
    tl.store(out + 1 + 2, 1.0)


@triton.jit
def nonzero_range_kernel(out):
    offsets = tl.arange(4, 8)
    tl.store(out + offsets, 1.0)


@triton.jit
def compile_only_kernel(x, y, out, n, TILE: tl.constexpr):
    offsets = tl.arange(0, TILE)
    mask = offsets < n
    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, x_values + y_values, mask=mask)


def test_rejects_inconsistent_vector_widths():
    n = 256
    out = np.empty(n, dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="CUDA lowering requires one vector width, got: 128, 256",
    ):
        inconsistent_width_kernel[(1,)](out, n)


def test_cuda_block_size_comes_from_vector_width():
    n = 32
    out = np.empty(n, dtype=np.float32)

    _, _, cuda_src = tile_kernel[(1,)](out, n, TILE_SIZE=32)

    assert "threadIdx.x" in cuda_src


def test_scalar_unmasked_load_and_store():
    source = np.array([3.0], dtype=np.float32)
    out = np.empty_like(source)

    _, _, cuda_src = scalar_copy_kernel[(1,)](source, out)

    assert "float v0 = source[0];" in cuda_src
    assert "out[0] = v0;" in cuda_src


def test_nested_pointer_arithmetic():
    out = np.empty(4, dtype=np.float32)

    _, _, cuda_src = nested_pointer_kernel[(1,)](out)

    assert "out[(1 + 2)] = 1.0f;" in cuda_src


def test_nonzero_arange_start():
    out = np.empty(8, dtype=np.float32)

    _, _, cuda_src = nonzero_range_kernel[(1,)](out)

    assert "int v0 = (4 + threadIdx.x);" in cuda_src


def test_cuda_float_literals():
    codegen = SSACUDACodegen()

    assert codegen.literal(float("nan")) == "__int_as_float(0x7fc00000)"
    assert codegen.literal(float("inf")) == "__int_as_float(0x7f800000)"
    assert codegen.literal(float("-inf")) == "(-__int_as_float(0x7f800000))"


def test_rejects_boolean_grid_dimension():
    out = np.empty(1, dtype=np.float32)

    with pytest.raises(ValueError, match="invalid launch grid"):
        scalar_copy_kernel[(True,)](out, out)


def test_numpy_compilation_does_not_load_cupy(monkeypatch):
    import mytriton.cuda_utils as cuda_utils

    def fail():
        raise AssertionError("CuPy must not be loaded for NumPy arguments")

    monkeypatch.setattr(cuda_utils, "_cupy", fail)

    n = 16
    x = np.ones(n, dtype=np.float32)
    y = np.ones(n, dtype=np.float32)
    out = np.empty_like(x)

    compile_only_kernel[(1,)](x, y, out, n, TILE=16)


def test_compilation_is_cached(monkeypatch):
    import mytriton.compiler as compiler

    trace_calls = 0
    original_trace = compiler.trace

    def counting_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(compiler, "trace", counting_trace)

    @triton.jit
    def cached_kernel(out, n, TILE: tl.constexpr):
        offsets = tl.arange(0, TILE)
        tl.store(out + offsets, 1.0, mask=offsets < n)

    out = np.empty(16, dtype=np.float32)
    cached_kernel[(1,)](out, 16, TILE=16)
    cached_kernel[(1,)](out, 16, TILE=16)

    assert trace_calls == 1


def test_cache_hit_still_validates_runtime_arguments():
    @triton.jit
    def validation_kernel(source, out):
        tl.store(out, tl.load(source))

    source = np.ones(2, dtype=np.float32)
    out = np.empty_like(source)
    validation_kernel[(1,)](source, out)

    matrix = np.ones((2, 2), dtype=np.float32)
    non_contiguous = matrix[:, 0]

    with pytest.raises(TypeError, match="only C-contiguous arrays are supported"):
        validation_kernel[(1,)](non_contiguous, out)


def test_different_constexpr_values_miss_compilation_cache(monkeypatch):
    import mytriton.compiler as compiler

    trace_calls = 0
    original_trace = compiler.trace

    def counting_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(compiler, "trace", counting_trace)

    @triton.jit
    def constexpr_kernel(out, n, TILE: tl.constexpr):
        offsets = tl.arange(0, TILE)
        tl.store(out + offsets, 1.0, mask=offsets < n)

    out = np.empty(32, dtype=np.float32)
    constexpr_kernel[(1,)](out, 32, TILE=16)
    constexpr_kernel[(1,)](out, 32, TILE=32)

    assert trace_calls == 2


def test_different_runtime_types_miss_compilation_cache(monkeypatch):
    import mytriton.compiler as compiler

    trace_calls = 0
    original_trace = compiler.trace

    def counting_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(compiler, "trace", counting_trace)

    @triton.jit
    def scalar_kernel(out, value):
        tl.store(out, value)

    out = np.empty(1, dtype=np.float32)
    scalar_kernel[(1,)](out, 1)
    scalar_kernel[(1,)](out, 1.0)

    assert trace_calls == 2


def test_different_grids_hit_compilation_cache(monkeypatch):
    import mytriton.compiler as compiler

    trace_calls = 0
    original_trace = compiler.trace

    def counting_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(compiler, "trace", counting_trace)

    @triton.jit
    def grid_kernel(out):
        tl.store(out, 1.0)

    out = np.empty(1, dtype=np.float32)
    grid_kernel[(1,)](out)
    grid_kernel[(2,)](out)

    assert trace_calls == 1


def test_cached_results_cannot_mutate_artifact():
    @triton.jit
    def immutable_kernel(out):
        tl.store(out, 1.0)

    out = np.empty(1, dtype=np.float32)
    ops, ssa_ops, _ = immutable_kernel[(1,)](out)
    ops[0].value = Const(9.0)
    ssa_ops[-1].operands = ()
    ops.clear()
    ssa_ops.clear()

    cached_ops, cached_ssa_ops, _ = immutable_kernel[(1,)](out)

    assert cached_ops[0].value == Const(1.0)
    assert cached_ssa_ops[-1].operands


def test_constexpr_cache_distinguishes_python_types():
    @triton.jit
    def type_sensitive_kernel(out, VALUE: tl.constexpr):
        tl.store(out, 1.0 if type(VALUE) is bool else 2.0)

    out = np.empty(1, dtype=np.float32)
    _, _, bool_src = type_sensitive_kernel[(1,)](out, VALUE=True)
    _, _, int_src = type_sensitive_kernel[(1,)](out, VALUE=1)

    assert "out[0] = 1.0f;" in bool_src
    assert "out[0] = 2.0f;" in int_src


def test_constexpr_cache_preserves_signed_zero():
    @triton.jit
    def signed_zero_kernel(out, VALUE: tl.constexpr):
        tl.store(out, VALUE)

    out = np.empty(1, dtype=np.float32)
    _, _, positive_src = signed_zero_kernel[(1,)](out, VALUE=0.0)
    _, _, negative_src = signed_zero_kernel[(1,)](out, VALUE=-0.0)

    assert "out[0] = 0.0f;" in positive_src
    assert "out[0] = -0.0f;" in negative_src


def test_clear_cache_forces_recompilation(monkeypatch):
    import mytriton.compiler as compiler

    trace_calls = 0
    original_trace = compiler.trace

    def counting_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(compiler, "trace", counting_trace)

    @triton.jit
    def clearable_kernel(out):
        tl.store(out, 1.0)

    out = np.empty(1, dtype=np.float32)
    clearable_kernel[(1,)](out)
    clearable_kernel[(1,)](out)
    clearable_kernel.clear_cache()
    clearable_kernel[(1,)](out)

    assert trace_calls == 2


def test_rejects_unhashable_constexpr_value():
    @triton.jit
    def constexpr_kernel(out, META: tl.constexpr):
        tl.store(out, 1.0)

    out = np.empty(1, dtype=np.float32)

    with pytest.raises(
        TypeError,
        match="META: constexpr value must be bool, int, float, or str",
    ):
        constexpr_kernel[(1,)](out, META=[])


def test_triton_style_constexpr_annotation_is_supported():
    @triton.jit
    def legacy_kernel(out, ENABLED: tl.constexpr):
        tl.store(out, 1.0 if ENABLED else 0.0)

    out = np.empty(1, dtype=np.float32)
    _, _, cuda_src = legacy_kernel[(1,)](out, ENABLED=True)

    assert "out[0] = 1.0f;" in cuda_src
