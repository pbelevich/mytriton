from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Generic, ParamSpec, Protocol, cast

from .compiler import CompilationResult, CompiledKernel, Grid
from .cuda_utils import (
    benchmark_cuda_kernel,
    cuda_chip,
    cuda_execution_required,
)
from .runtime_args import array_arg_info
from .trace import is_constexpr_annotation, make_runtime_params

P = ParamSpec("P")


def _validate_num_warps(num_warps: int) -> None:
    if (
        type(num_warps) is not int
        or num_warps <= 0
        or num_warps & (num_warps - 1)
        or num_warps > 32
    ):
        raise ValueError(
            f"num_warps must be a power of two between 1 and 32, got {num_warps}"
        )


@dataclass(frozen=True, eq=False)
class Config:
    """One set of compile-time arguments considered by :func:`autotune`."""

    kwargs: Mapping[str, object]
    num_warps: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.kwargs, Mapping):
            raise TypeError("Config kwargs must be a mapping")
        if any(not isinstance(name, str) for name in self.kwargs):
            raise TypeError("Config kwargs keys must be strings")
        if any(
            not isinstance(value, (bool, int, float, str))
            for value in self.kwargs.values()
        ):
            raise TypeError(
                "Config values must be bool, int, float, or str compile-time values"
            )
        _validate_num_warps(self.num_warps)
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))

    def all_kwargs(self) -> dict[str, object]:
        """Return meta-parameters and launch options as a new dictionary."""

        return {**self.kwargs, "num_warps": self.num_warps}

    def __repr__(self) -> str:
        return f"Config(kwargs={dict(self.kwargs)!r}, num_warps={self.num_warps})"

    def __hash__(self) -> int:
        return hash(_config_identity(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Config) and _config_identity(self) == _config_identity(
            other
        )


class AutotuneLauncher(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> CompilationResult: ...


def _config_identity(config: Config) -> tuple[object, ...]:
    entries = tuple(
        (name, type(value), repr(value))
        for name, value in sorted(config.kwargs.items())
    )
    return entries, config.num_warps


def _key_value(value: object) -> tuple[type[object], object]:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(
            f"autotune key values must be hashable, got {type(value).__name__}"
        ) from error
    normalized = float(value).hex() if isinstance(value, float) else value
    return type(value), normalized


class AutotunedKernel(Generic[P]):
    """A JIT kernel that selects and caches the fastest CUDA configuration."""

    def __init__(
        self,
        kernel: CompiledKernel[P],
        *,
        configs: Sequence[Config],
        key: Sequence[str],
        warmup: int,
        rep: int,
    ) -> None:
        if not isinstance(kernel, CompiledKernel):
            raise TypeError("@autotune must wrap an @jit kernel")
        if not configs:
            raise ValueError("autotune requires at least one Config")
        if any(not isinstance(config, Config) for config in configs):
            raise TypeError("autotune configs must contain only Config objects")
        if type(warmup) is not int or warmup < 0:
            raise ValueError(f"warmup must be a non-negative integer, got {warmup}")
        if type(rep) is not int or rep <= 0:
            raise ValueError(f"rep must be a positive integer, got {rep}")

        names = tuple(kernel.signature.parameters)
        if isinstance(key, (str, bytes)):
            raise TypeError("autotune key must be a sequence of argument names")
        key_tuple = tuple(key)
        if len(set(key_tuple)) != len(key_tuple):
            raise ValueError("autotune key names must be unique")
        unknown_keys = [name for name in key_tuple if name not in names]
        if unknown_keys:
            raise ValueError(
                f"autotune key contains unknown kernel argument {unknown_keys[0]!r}"
            )

        identities = [_config_identity(config) for config in configs]
        if len(set(identities)) != len(identities):
            raise ValueError("autotune configs must be unique")

        for config in configs:
            for name in config.kwargs:
                parameter = kernel.signature.parameters.get(name)
                if parameter is None:
                    raise ValueError(
                        f"Config contains unknown kernel argument {name!r}"
                    )
                if not is_constexpr_annotation(parameter.annotation):
                    raise ValueError(
                        f"Config argument {name!r} must be annotated tl.constexpr"
                    )
                if name in key_tuple:
                    raise ValueError(
                        f"autotune key argument {name!r} cannot also be configured"
                    )

        self.kernel = kernel
        self.configs = tuple(configs)
        self.key = key_tuple
        self.warmup = warmup
        self.rep = rep
        self.cache: dict[tuple[object, ...], Config] = {}
        self.timings: dict[tuple[object, ...], dict[Config, float]] = {}
        self.best_config: Config | None = None
        self._lock = RLock()

    @property
    def fn(self) -> Callable[P, Any]:
        return self.kernel.fn

    @property
    def signature(self):
        return self.kernel.signature

    @property
    def compilation_cache(self):
        return self.kernel.compilation_cache

    @property
    def cuda_cache(self):
        return self.kernel.cuda_cache

    def clear_cache(self) -> None:
        self.cache.clear()
        self.timings.clear()
        self.best_config = None
        self.kernel.clear_cache()

    def _configured_kwargs(
        self,
        kwargs: Mapping[str, object],
        config: Config,
    ) -> dict[str, object]:
        conflicts = set(kwargs).intersection(config.kwargs)
        if conflicts:
            name = min(conflicts)
            raise ValueError(
                f"kernel launch overrides autotuned meta-parameter {name!r}"
            )
        if "num_warps" in kwargs:
            raise ValueError("kernel launch overrides autotuned option 'num_warps'")
        return {**kwargs, **config.kwargs, "num_warps": config.num_warps}

    def _bound_arguments(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        config: Config,
    ):
        configured = self._configured_kwargs(kwargs, config)
        launch_options = {"num_warps"}
        kernel_kwargs = {
            name: value
            for name, value in configured.items()
            if name not in launch_options
        }
        bound = self.signature.bind(*args, **kernel_kwargs)
        bound.apply_defaults()
        return bound

    def _runtime_args(self, bound) -> tuple[object, ...]:
        return tuple(
            bound.arguments[name]
            for name, parameter in self.signature.parameters.items()
            if not is_constexpr_annotation(parameter.annotation)
        )

    def _cache_key(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> tuple[object, ...]:
        probe = self.configs[0]
        bound = self._bound_arguments(args, kwargs, probe)
        runtime_args = self._runtime_args(bound)
        params = make_runtime_params(self.signature, bound.arguments)
        backend = self.kernel._resolve_backend()
        executes = cuda_execution_required(runtime_args, backend_name="autotune")
        if executes and backend != "cuda":
            raise ValueError("autotune execution currently supports only CUDA")
        chip = cuda_chip(runtime_args) if executes else "compile-only"
        device_index = None
        if executes:
            device_index = next(
                (
                    info.device_index
                    for value in runtime_args
                    if ((info := array_arg_info(value)) is not None and info.is_cuda)
                ),
                None,
            )
        keyed_values = tuple(
            (name, *_key_value(bound.arguments[name])) for name in self.key
        )
        return (
            backend,
            chip,
            device_index,
            tuple(param.ty for param in params),
            keyed_values,
        )

    def _benchmark_cuda(
        self,
        *,
        cuda_source: str,
        launch_grid: tuple[int, ...],
        threads_per_block: int,
        runtime_args: tuple[object, ...],
    ) -> float:
        return benchmark_cuda_kernel(
            kernel_cache=self.cuda_cache,
            cuda_src=cuda_source,
            kernel_name=self.fn.__name__,
            launch_grid=launch_grid,
            threads_per_block=threads_per_block,
            runtime_args=runtime_args,
            warmup=self.warmup,
            rep=self.rep,
        )

    def _cuda_launch(
        self,
        *,
        grid: Grid,
        bound,
        cuda_source: str,
    ) -> tuple[tuple[int, ...], int]:
        meta = {
            name: bound.arguments[name]
            for name, parameter in self.signature.parameters.items()
            if is_constexpr_annotation(parameter.annotation)
        }
        launch_grid = grid(meta) if callable(grid) else grid
        if isinstance(launch_grid, int):
            launch_grid = (launch_grid,)
        normalized_grid = tuple(launch_grid)

        artifact = next(
            (
                artifact
                for artifact in self.compilation_cache.values()
                if artifact.src == cuda_source
            ),
            None,
        )
        if artifact is None:
            raise RuntimeError("autotune could not find the compiled CUDA artifact")

        return normalized_grid, artifact.threads_per_block

    def _select_config(
        self,
        grid: Grid,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        cache_key: tuple[object, ...],
    ) -> Config:
        probe_bound = self._bound_arguments(args, kwargs, self.configs[0])
        runtime_args = self._runtime_args(probe_bound)
        if not cuda_execution_required(runtime_args, backend_name="autotune"):
            selected = self.configs[0]
            self.timings[cache_key] = {}
            return selected

        measurements: dict[Config, float] = {}
        tuning_start = time.perf_counter()
        for config in self.configs:
            configured = self._configured_kwargs(kwargs, config)
            bound = self._bound_arguments(args, kwargs, config)
            launcher = cast(
                Callable[..., CompilationResult],
                self.kernel[grid],
            )
            _, _, cuda_source = launcher(*args, **configured)
            launch_grid, threads_per_block = self._cuda_launch(
                grid=grid,
                bound=bound,
                cuda_source=cuda_source,
            )
            measurements[config] = self._benchmark_cuda(
                cuda_source=cuda_source,
                launch_grid=launch_grid,
                threads_per_block=threads_per_block,
                runtime_args=runtime_args,
            )

        selected = min(measurements, key=measurements.__getitem__)
        self.timings[cache_key] = measurements
        if os.environ.get("MYTRITON_PRINT_AUTOTUNING") == "1":
            elapsed_ms = (time.perf_counter() - tuning_start) * 1.0e3
            print(
                f"mytriton autotune: {self.fn.__name__} selected {selected} "
                f"in {elapsed_ms:.2f} ms"
            )
        return selected

    def __getitem__(self, grid: Grid) -> AutotuneLauncher:
        def launch(*args: Any, **kwargs: Any) -> CompilationResult:
            positional = cast(tuple[object, ...], args)
            named = cast(dict[str, object], kwargs)
            cache_key = self._cache_key(positional, named)

            with self._lock:
                config = self.cache.get(cache_key)
                if config is None:
                    config = self._select_config(
                        grid,
                        positional,
                        named,
                        cache_key,
                    )
                    self.cache[cache_key] = config
                self.best_config = config

            configured = self._configured_kwargs(named, config)
            launcher = cast(
                Callable[..., CompilationResult],
                self.kernel[grid],
            )
            return launcher(*args, **configured)

        return launch


def autotune(
    *,
    configs: Sequence[Config],
    key: Sequence[str],
    warmup: int = 2,
    rep: int = 5,
) -> Callable[[CompiledKernel[P]], AutotunedKernel[P]]:
    """Tune a JIT kernel and cache the fastest config for each runtime key."""

    def decorate(kernel: CompiledKernel[P]) -> AutotunedKernel[P]:
        return AutotunedKernel(
            kernel,
            configs=configs,
            key=key,
            warmup=warmup,
            rep=rep,
        )

    return decorate
