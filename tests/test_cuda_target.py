import pytest

from mytriton.cuda_target import DEFAULT_CUDA_TARGET, CudaTarget


@pytest.mark.parametrize(
    ("chip", "major", "minor"),
    [
        ("sm_75", 7, 5),
        ("sm_80", 8, 0),
    ],
)
def test_cuda_target_round_trips_chip(
    chip: str,
    major: int,
    minor: int,
) -> None:
    target = CudaTarget.from_chip(chip)

    assert target.major == major
    assert target.minor == minor
    assert target.chip == chip


@pytest.mark.parametrize("chip", ["80", "compute_80", "sm_8x", "sm_8"])
def test_cuda_target_rejects_invalid_chip(chip: str) -> None:
    with pytest.raises(ValueError, match="invalid CUDA chip"):
        CudaTarget.from_chip(chip)


def test_cuda_target_reports_mma_features() -> None:
    sm_70 = CudaTarget.from_chip("sm_70")
    sm_75 = CudaTarget.from_chip("sm_75")
    sm_80 = CudaTarget.from_chip("sm_80")

    assert not sm_70.supports_f16_mma_m16n8k8
    assert not sm_70.supports_bf16_mma_m16n8k8
    assert sm_75.supports_f16_mma_m16n8k8
    assert not sm_75.supports_bf16_mma_m16n8k8
    assert sm_80.supports_f16_mma_m16n8k8
    assert sm_80.supports_bf16_mma_m16n8k8


def test_default_cuda_target_is_sm75() -> None:
    assert CudaTarget.from_chip("sm_75") == DEFAULT_CUDA_TARGET
