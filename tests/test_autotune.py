from __future__ import annotations

import numpy as np
import pytest

import mytriton as triton
import mytriton.autotuner as autotuner_module
import mytriton.language as tl
from mytriton.autotuner import AutotunedKernel

AUTOTUNE_CONFIGS = (
    triton.Config({"BM": 16, "BK": 8, "BN": 8}, num_warps=1),
    triton.Config({"BM": 32, "BK": 8, "BN": 16}, num_warps=1),
)


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["K"], warmup=0, rep=1)
@triton.jit
def autotuned_matmul_kernel(
    a,
    b,
    output,
    M,
    N,
    K,
    k_base,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    columns = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
    inner = tl.arange(0, BK)
    a_columns = k_base + inner[None, :]
    b_rows = k_base + inner[:, None]
    a_values = tl.load(
        a + rows * K + a_columns,
        mask=(rows < M) & (a_columns < K),
        other=0.0,
    )
    b_values = tl.load(
        b + b_rows * N + columns,
        mask=(b_rows < K) & (columns < N),
        other=0.0,
    )
    result = tl.dot(a_values, b_values)
    tl.store(
        output + rows * N + columns,
        result,
        mask=(rows < M) & (columns < N),
    )


def test_config_copies_kwargs_and_exposes_launch_options() -> None:
    kwargs = {"BLOCK": 32}
    config = triton.Config(kwargs, num_warps=2)
    kwargs["BLOCK"] = 64

    assert dict(config.kwargs) == {"BLOCK": 32}
    assert config.all_kwargs() == {"BLOCK": 32, "num_warps": 2}
    assert repr(config) == "Config(kwargs={'BLOCK': 32}, num_warps=2)"
    with pytest.raises(TypeError):
        config.kwargs["BLOCK"] = 64  # type: ignore[index]


@pytest.mark.parametrize("num_warps", [0, 3, 64, True])
def test_config_rejects_invalid_num_warps(num_warps: int) -> None:
    with pytest.raises(ValueError, match="power of two between 1 and 32"):
        triton.Config({}, num_warps=num_warps)


def test_config_rejects_non_compile_time_value() -> None:
    with pytest.raises(TypeError, match="compile-time values"):
        triton.Config({"BLOCK": [32]})


def test_autotune_rejects_empty_and_duplicate_configs() -> None:
    @triton.jit
    def kernel(BLOCK: tl.constexpr):
        del BLOCK

    with pytest.raises(ValueError, match="at least one Config"):
        triton.autotune(configs=[], key=[])(kernel)

    config = triton.Config({"BLOCK": 32})
    with pytest.raises(ValueError, match="configs must be unique"):
        triton.autotune(configs=[config, config], key=[])(kernel)

    with pytest.raises(TypeError, match="sequence of argument names"):
        triton.autotune(configs=[config], key="BLOCK")(kernel)


def test_autotune_requires_jit_kernel() -> None:
    with pytest.raises(TypeError, match="must wrap an @jit kernel"):

        @triton.autotune(configs=[triton.Config({})], key=[])  # type: ignore[arg-type]
        def undecorated_kernel() -> None:
            pass


def test_autotune_validates_config_and_key_arguments() -> None:
    @triton.jit
    def kernel(value, BLOCK: tl.constexpr):
        del value, BLOCK

    with pytest.raises(ValueError, match="unknown kernel argument 'missing'"):
        triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=["missing"])(kernel)

    with pytest.raises(ValueError, match=r"must be annotated tl\.constexpr"):
        triton.autotune(configs=[triton.Config({"value": 32})], key=[])(kernel)


def test_compile_only_launch_uses_first_config() -> None:
    autotuned_matmul_kernel.clear_cache()
    a, b, output = _matmul_arguments()

    _, _, cuda_source = autotuned_matmul_kernel[(1, 1)](
        a,
        b,
        output,
        32,
        16,
        8,
        0,
    )

    assert autotuned_matmul_kernel.best_config == AUTOTUNE_CONFIGS[0]
    assert len(autotuned_matmul_kernel.cache) == 1
    assert "mma.sync.aligned" in cuda_source


def _matmul_arguments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((32, 8), dtype=np.float16),
        np.zeros((8, 16), dtype=np.float16),
        np.empty((32, 16), dtype=np.float32),
    )


def test_autotune_benchmarks_once_and_caches_fastest_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autotuned_matmul_kernel.clear_cache()
    a, b, output = _matmul_arguments()
    measured = iter((2.0, 1.0, 4.0, 3.0))
    benchmark_calls = 0

    monkeypatch.setattr(
        autotuner_module, "cuda_execution_required", lambda *_a, **_k: True
    )
    monkeypatch.setattr(autotuner_module, "cuda_chip", lambda _args: "sm_80")

    def fake_benchmark(self, **kwargs) -> float:
        nonlocal benchmark_calls
        del self, kwargs
        benchmark_calls += 1
        return next(measured)

    monkeypatch.setattr(AutotunedKernel, "_benchmark_cuda", fake_benchmark)

    autotuned_matmul_kernel[(1, 1)](a, b, output, 32, 16, 8, 0)
    assert autotuned_matmul_kernel.best_config == AUTOTUNE_CONFIGS[1]
    assert benchmark_calls == 2
    assert len(autotuned_matmul_kernel.compilation_cache) == 2

    autotuned_matmul_kernel[(1, 1)](a, b, output, 32, 16, 8, 0)
    assert benchmark_calls == 2

    autotuned_matmul_kernel[(1, 1)](a, b, output, 32, 16, 9, 0)
    assert benchmark_calls == 4
    assert len(autotuned_matmul_kernel.cache) == 2


def test_autotune_rejects_launch_override() -> None:
    autotuned_matmul_kernel.clear_cache()
    a, b, output = _matmul_arguments()

    with pytest.raises(
        ValueError,
        match="overrides autotuned meta-parameter 'BM'",
    ):
        autotuned_matmul_kernel[(1, 1)](
            a,
            b,
            output,
            32,
            16,
            8,
            0,
            BM=32,
        )


@pytest.mark.execution
def test_autotuned_tensor_core_kernel_executes_on_cuda(
    cp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = int(cp.cuda.Device().compute_capability)
    if capability < 75:
        pytest.skip("tensor-core autotune requires compute capability 7.5+")

    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    autotuned_matmul_kernel.clear_cache()
    rng = np.random.default_rng(28)
    a_host = rng.normal(0.0, 0.25, size=(16, 8)).astype(np.float16)
    b_host = rng.normal(0.0, 0.25, size=(8, 8)).astype(np.float16)
    a = cp.asarray(a_host)
    b = cp.asarray(b_host)
    output = cp.full((16, 8), cp.nan, dtype=cp.float32)

    autotuned_matmul_kernel[
        lambda meta: (
            triton.cdiv(16, meta["BM"]),
            triton.cdiv(8, meta["BN"]),
        )
    ](a, b, output, 16, 8, 8, 0)
    cp.cuda.Stream.null.synchronize()

    expected = a_host.astype(np.float32) @ b_host.astype(np.float32)
    cp.testing.assert_allclose(output, cp.asarray(expected), rtol=2e-3, atol=2e-3)
    assert autotuned_matmul_kernel.best_config in AUTOTUNE_CONFIGS
    assert len(autotuned_matmul_kernel.cache) == 1
    assert len(autotuned_matmul_kernel.timings) == 1
