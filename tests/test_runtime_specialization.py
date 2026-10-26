import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.runtime_specialization import (
    RuntimeSpecialization,
    runtime_specialization,
)


def aligned_and_shifted_float32_views() -> tuple[np.ndarray, np.ndarray]:
    storage = np.empty(16, dtype=np.float32)
    aligned_offset = (-storage.ctypes.data % 16) // storage.itemsize
    aligned = storage[aligned_offset:]
    return aligned, aligned[1:]


def test_integer_specialization_caps_power_of_two_divisibility() -> None:
    assert runtime_specialization(1) == RuntimeSpecialization(divisibility=1)
    assert runtime_specialization(96) == RuntimeSpecialization(divisibility=32)
    assert runtime_specialization(4096) == RuntimeSpecialization(divisibility=128)
    assert runtime_specialization(0) == RuntimeSpecialization(divisibility=128)


def test_array_specialization_records_pointer_alignment() -> None:
    aligned_array, shifted_array = aligned_and_shifted_float32_views()
    aligned = runtime_specialization(aligned_array)
    shifted = runtime_specialization(shifted_array)

    assert aligned.alignment == 16
    assert shifted.alignment == 4


def test_equal_specialization_properties_share_compilation_cache() -> None:
    @triton.jit
    def fill_kernel(out, n, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.store(out + offsets, 1.0, mask=offsets < n)

    out = np.empty(128, dtype=np.float32)
    fill_kernel[(1,)](out, 32, BLOCK=128)
    fill_kernel[(1,)](out, 96, BLOCK=128)
    fill_kernel[(1,)](out, 64, BLOCK=128)

    assert len(fill_kernel.compilation_cache) == 2


def test_pointer_alignment_separates_compilation_cache() -> None:
    @triton.jit
    def copy_kernel(source, out):
        tl.store(out, tl.load(source))

    source, shifted_source = aligned_and_shifted_float32_views()
    output, shifted_output = aligned_and_shifted_float32_views()
    source.fill(1.0)

    copy_kernel[(1,)](source, output)
    copy_kernel[(1,)](shifted_source, shifted_output)

    assert len(copy_kernel.compilation_cache) == 2
