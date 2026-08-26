from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    model: str
    model_type: str
    runtime: str
    execution_provider: str
    precision: str
    batch_size: int

    average_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None

    throughput: float | None = None
    throughput_unit: str | None = None

    ttft_ms: float | None = None
    tokens_per_second: float | None = None

    peak_memory_mb: float | None = None

    accuracy_metric: str | None = None
    accuracy_value: float | None = None

    notes: str | None = None


def save_result(
    result: BenchmarkResult,
    output_path: str,
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )


def load_result(
    path: str,
) -> BenchmarkResult:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return BenchmarkResult(**data)