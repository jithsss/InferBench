from benchmarks.result_schema import (
    BenchmarkResult,
    save_result,
)


def save_benchmark_result(
    result: BenchmarkResult,
    path: str,
) -> None:
    save_result(
        result,
        path,
    )

    print(
        f"Result saved to: {path}"
    )