import pytest

from mytriton.cuda_utils import cuda_execution_required
from mytriton.runtime_args import RuntimeArrayInfo, array_arg_info


class FakeDType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class FakeDevice:
    def __init__(self, device_type: str, index: int | None) -> None:
        self.type = device_type
        self.index = index
        self.id = index


class FakeFlags:
    def __init__(self, *, c_contiguous: bool) -> None:
        self.c_contiguous = c_contiguous


class FakeTorchTensor:
    __module__ = "torch"

    def __init__(
        self,
        *,
        device: str,
        device_index: int | None,
        c_contiguous: bool = True,
    ) -> None:
        self.dtype = FakeDType("torch.float32")
        self.device = FakeDevice(device, device_index)
        self._c_contiguous = c_contiguous

    def is_contiguous(self) -> bool:
        return self._c_contiguous


class FakeCupyArray:
    __module__ = "cupy"

    def __init__(self, *, device_index: int) -> None:
        self.dtype = FakeDType("float32")
        self.device = FakeDevice("cuda", device_index)
        self.flags = FakeFlags(c_contiguous=True)


def test_array_arg_info_recognizes_torch_cpu_tensor() -> None:
    tensor = FakeTorchTensor(
        device="cpu",
        device_index=None,
        c_contiguous=False,
    )

    assert array_arg_info(tensor) == RuntimeArrayInfo(
        framework="torch",
        device="cpu",
        device_index=None,
        dtype_name="float32",
        c_contiguous=False,
    )


def test_array_arg_info_recognizes_torch_cuda_tensor() -> None:
    tensor = FakeTorchTensor(
        device="cuda",
        device_index=2,
    )

    assert array_arg_info(tensor) == RuntimeArrayInfo(
        framework="torch",
        device="cuda",
        device_index=2,
        dtype_name="float32",
        c_contiguous=True,
    )


def test_torch_cpu_arrays_are_compilation_only() -> None:
    tensor = FakeTorchTensor(
        device="cpu",
        device_index=None,
    )

    assert not cuda_execution_required((tensor,), backend_name="CUDA")


def test_torch_cuda_arrays_require_execution() -> None:
    tensor = FakeTorchTensor(
        device="cuda",
        device_index=0,
    )

    assert cuda_execution_required((tensor,), backend_name="CUDA")


def test_execution_rejects_mixed_cpu_and_cuda_arrays() -> None:
    cpu = FakeTorchTensor(
        device="cpu",
        device_index=None,
    )
    cuda = FakeTorchTensor(
        device="cuda",
        device_index=0,
    )

    with pytest.raises(
        TypeError,
        match="does not support mixed CPU and CUDA arrays",
    ):
        cuda_execution_required((cpu, cuda), backend_name="CUDA")


def test_execution_rejects_mixed_cuda_frameworks() -> None:
    torch_tensor = FakeTorchTensor(
        device="cuda",
        device_index=0,
    )
    cupy_array = FakeCupyArray(device_index=0)

    with pytest.raises(
        TypeError,
        match="mixed CUDA array frameworks: cupy, torch",
    ):
        cuda_execution_required(
            (torch_tensor, cupy_array),
            backend_name="CUDA",
        )


def test_execution_rejects_multiple_cuda_devices() -> None:
    first = FakeTorchTensor(
        device="cuda",
        device_index=0,
    )
    second = FakeTorchTensor(
        device="cuda",
        device_index=1,
    )

    with pytest.raises(
        TypeError,
        match="requires one CUDA device, got: 0, 1",
    ):
        cuda_execution_required(
            (first, second),
            backend_name="CUDA",
        )
