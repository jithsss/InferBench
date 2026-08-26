from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    description: str
    module: str
    function: str = "main"

    def run(self) -> None:
        module = import_module(self.module)
        entrypoint: Callable[[], None] = getattr(
            module,
            self.function,
        )
        entrypoint()


_REGISTRY: dict[str, BenchmarkSpec] = {}


def register_benchmark(
    name: str,
    description: str,
    module: str,
    function: str = "main",
) -> None:
    _REGISTRY[name] = BenchmarkSpec(
        name=name,
        description=description,
        module=module,
        function=function,
    )


def get_benchmark(name: str) -> BenchmarkSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))

        raise KeyError(
            f"Unknown benchmark '{name}'. "
            f"Available: {available}"
        ) from exc


def list_benchmarks() -> list[BenchmarkSpec]:
    return sorted(
        _REGISTRY.values(),
        key=lambda item: item.name,
    )