from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    iterations: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float


def calculate_statistics(latencies_ms: list[float]) -> BenchmarkResult:
    if not latencies_ms:
        raise ValueError("No latency measurements were provided.")

    latencies = np.array(latencies_ms, dtype=np.float64)

    average_latency_ms = float(np.mean(latencies))
    p50_latency_ms = float(np.percentile(latencies, 50))
    p95_latency_ms = float(np.percentile(latencies, 95))
    p99_latency_ms = float(np.percentile(latencies, 99))

    # Throughput based on the average per-inference latency.
    throughput_fps = 1000.0 / average_latency_ms

    return BenchmarkResult(
        iterations=len(latencies_ms),
        average_latency_ms=average_latency_ms,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        throughput_fps=throughput_fps,
    )


def print_results(result: BenchmarkResult) -> None:
    print("\n--- Benchmark Results ---")
    print(f"Iterations:       {result.iterations}")
    print(f"Average latency:  {result.average_latency_ms:.3f} ms")
    print(f"P50 latency:      {result.p50_latency_ms:.3f} ms")
    print(f"P95 latency:      {result.p95_latency_ms:.3f} ms")
    print(f"P99 latency:      {result.p99_latency_ms:.3f} ms")
    print(f"Throughput:       {result.throughput_fps:.2f} FPS")