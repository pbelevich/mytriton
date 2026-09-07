from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaTarget:
    """CUDA architecture features needed by source-level lowering."""

    major: int
    minor: int

    def __post_init__(self) -> None:
        if (
            type(self.major) is not int
            or type(self.minor) is not int
            or self.major < 0
            or not 0 <= self.minor <= 9
        ):
            raise ValueError(
                f"invalid CUDA compute capability {self.major}.{self.minor}"
            )

    @classmethod
    def from_chip(cls, chip: str) -> CudaTarget:
        prefix = "sm_"
        capability = chip.removeprefix(prefix)

        if (
            not chip.startswith(prefix)
            or not capability.isdigit()
            or len(capability) < 2
        ):
            raise ValueError(f"invalid CUDA chip {chip!r}")

        return cls(
            major=int(capability[:-1]),
            minor=int(capability[-1]),
        )

    @property
    def chip(self) -> str:
        return f"sm_{self.major}{self.minor}"

    @property
    def compute_capability(self) -> int:
        return self.major * 10 + self.minor

    @property
    def supports_f16_mma_m16n8k8(self) -> bool:
        return self.compute_capability >= 75

    @property
    def supports_bf16_mma_m16n8k8(self) -> bool:
        return self.compute_capability >= 80


# Compilation-only calls do not select a CUDA device. sm_75 is the oldest target
# supported by the current f16 tensor-core lowering and remains a conservative
# default for source inspection on CPU arrays.
DEFAULT_CUDA_TARGET = CudaTarget(major=7, minor=5)
