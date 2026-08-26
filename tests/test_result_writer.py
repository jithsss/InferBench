from benchmarks.result_schema import BenchmarkResult
from benchmarks.result_writer import save_benchmark_result


def main() -> None:
    result = BenchmarkResult(
        model="TestModel",
        model_type="vision",
        runtime="TestRuntime",
        execution_provider="CPU",
        precision="FP32",
        batch_size=1,
        average_latency_ms=10.0,
        p50_latency_ms=9.5,
        p95_latency_ms=11.0,
        p99_latency_ms=12.0,
        throughput=100.0,
        throughput_unit="FPS",
        notes="Result writer test.",
    )

    save_benchmark_result(
        result,
        "results/test_result.json",
    )


if __name__ == "__main__":
    main()