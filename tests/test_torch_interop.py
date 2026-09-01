from typing import Any

import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.cuda_utils import (
    CudaKernelCache,
    cuda_module,
    execute_cuda_if_needed,
)


@triton.jit
def torch_add_kernel(
    x,
    y,
    out,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.load(y + offsets, mask=mask)

    tl.store(
        out + offsets,
        x_values + y_values,
        mask=mask,
    )


def torch_module() -> Any:
    return pytest.importorskip("torch")


def test_torch_cpu_tensors_are_compile_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = torch_module()

    import mytriton.cuda_utils as cuda_utils

    def fail():
        raise AssertionError("CuPy must not be loaded for CPU Torch tensors")

    monkeypatch.setattr(cuda_utils, "_cupy", fail)

    n = 16
    x = torch.ones(n, dtype=torch.float32)
    y = torch.ones(n, dtype=torch.float32)
    out = torch.empty(n, dtype=torch.float32)

    torch_add_kernel.clear_cache()

    _, _, cuda_src = torch_add_kernel[(1,)](
        x,
        y,
        out,
        n,
        BLOCK_SIZE=16,
    )

    assert "void torch_add_kernel" in cuda_src


@pytest.mark.execution
def test_torch_cuda_tensors_execute(
    backend: str,
) -> None:
    torch = torch_module()

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    assert backend in {"cuda", "mlir"}

    n = 37
    block_size = 32

    x = torch.arange(
        n,
        device="cuda",
        dtype=torch.float32,
    )
    y = (
        torch.arange(
            n,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.5
    )
    out = torch.full(
        (n,),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )

    grid = ((n + block_size - 1) // block_size,)

    torch_add_kernel.clear_cache()
    torch_add_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK_SIZE=block_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out,
        x + y,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.execution
def test_torch_cuda_tensor_requiring_grad_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = torch_module()

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    n = 32
    x = torch.arange(
        n,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.ones_like(x)
    out = torch.empty_like(x, requires_grad=False)

    torch_add_kernel.clear_cache()
    torch_add_kernel[(1,)](
        x,
        y,
        out,
        n,
        BLOCK_SIZE=n,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out, x.detach() + y)
    assert not out.requires_grad


@pytest.mark.execution
def test_torch_launch_uses_current_torch_stream() -> None:
    torch = torch_module()

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    cp = cuda_module()
    tensor = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )
    torch_stream = torch.cuda.Stream(device=tensor.device)

    observed_streams: list[int] = []

    def probe_kernel(
        launch_grid: tuple[int, ...],
        block: tuple[int, ...],
        args: tuple[object, ...],
    ) -> None:
        del launch_grid, block, args
        observed_streams.append(cp.cuda.get_current_stream().ptr)

    cuda_src = "unused"
    kernel_name = "stream_probe"
    device_index = tensor.device.index
    assert device_index is not None
    cache_key = (cuda_src, kernel_name, device_index)
    kernel_cache: CudaKernelCache = {
        cache_key: probe_kernel,
    }

    with torch.cuda.stream(torch_stream):
        execute_cuda_if_needed(
            kernel_cache=kernel_cache,
            cuda_src=cuda_src,
            kernel_name=kernel_name,
            launch_grid=(1,),
            threads_per_block=1,
            runtime_args=(tensor,),
        )

    assert observed_streams == [torch_stream.cuda_stream]


@pytest.mark.execution
def test_torch_cuda_tensors_execute_on_non_default_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = torch_module()

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    n = 4097
    block_size = 256
    torch_stream = torch.cuda.Stream()

    with torch.cuda.stream(torch_stream):
        x = torch.arange(
            n,
            device="cuda",
            dtype=torch.float32,
        )
        y = x * 0.25
        out = torch.full(
            (n,),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )

        grid = ((n + block_size - 1) // block_size,)

        torch_add_kernel.clear_cache()
        torch_add_kernel[grid](
            x,
            y,
            out,
            n,
            BLOCK_SIZE=block_size,
        )

        actual = out.clone()
        expected = x + y

    torch_stream.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
