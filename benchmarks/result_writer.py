from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.result_schema import BenchmarkResult


HISTORY_PATH = Path(
    "results/history/benchmarks.jsonl"
)


def save_benchmark_result(
    result: BenchmarkResult,
    path: str,
) -> None:
    save_result(result, path)
    save_history(result)


def save_result(
    result: BenchmarkResult,
    path: str,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )


def save_history(
    result: BenchmarkResult,
) -> None:
    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = asdict(result)

    record["timestamp"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    with HISTORY_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(record)
            + "\n"
        )

    print(
        f"History saved to: "
        f"{HISTORY_PATH}"
    )


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []

    records = []

    with HISTORY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError:
                continue

    return records