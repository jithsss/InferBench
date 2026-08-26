from pathlib import Path
from typing import Iterable

from benchmarks.result_schema import BenchmarkResult, load_result


RESULTS_DIR = Path("results")


def load_all_results() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for path in RESULTS_DIR.rglob("*.json"):
        try:
            result = load_result(str(path))
            results.append(result)
        except Exception as exc:
            print(
                f"Skipping {path}: {exc}"
            )

    return results


def format_latency(
    value: float | None,
) -> str:
    if value is None:
        return "-"

    return f"{value:.3f} ms"


def format_throughput(
    result: BenchmarkResult,
) -> str:
    if result.model_type == "vision":
        if result.throughput is None:
            return "-"

        return f"{result.throughput:.2f} FPS"

    if result.model_type == "llm":
        if result.tokens_per_second is None:
            return "-"

        return (
            f"{result.tokens_per_second:.2f} tok/s"
        )

    return "-"


def print_vision_results(
    results: Iterable[BenchmarkResult],
) -> None:
    vision_results = [
        result
        for result in results
        if result.model_type == "vision"
    ]

    if not vision_results:
        return

    print("\n=== VISION RESULTS ===")

    print(
        f"{'Model':<14}"
        f"{'Runtime':<18}"
        f"{'Precision':<10}"
        f"{'Latency':>12}"
        f"{'Throughput':>16}"
    )

    print("-" * 70)

    for result in vision_results:
        print(
            f"{result.model:<14}"
            f"{result.runtime:<18}"
            f"{result.precision:<10}"
            f"{format_latency(result.average_latency_ms):>12}"
            f"{format_throughput(result):>16}"
        )


def print_llm_results(
    results: Iterable[BenchmarkResult],
) -> None:
    llm_results = [
        result
        for result in results
        if result.model_type == "llm"
    ]

    if not llm_results:
        return

    print("\n=== LLM RESULTS ===")

    print(
        f"{'Model':<18}"
        f"{'Runtime':<22}"
        f"{'Precision':<10}"
        f"{'TTFT':>12}"
        f"{'Tokens/sec':>16}"
        f"{'Latency':>14}"
    )

    print("-" * 92)

    for result in llm_results:
        ttft = (
            f"{result.ttft_ms:.2f} ms"
            if result.ttft_ms is not None
            else "-"
        )

        tok_s = (
            f"{result.tokens_per_second:.2f}"
            if result.tokens_per_second is not None
            else "-"
        )

        latency = format_latency(
            result.average_latency_ms
        )

        print(
            f"{result.model:<18}"
            f"{result.runtime:<22}"
            f"{result.precision:<10}"
            f"{ttft:>12}"
            f"{tok_s:>16}"
            f"{latency:>14}"
        )


def print_vision_analysis(
    results: Iterable[BenchmarkResult],
) -> None:
    vision_results = [
        result
        for result in results
        if result.model_type == "vision"
        and result.throughput is not None
    ]

    if not vision_results:
        return

    print("\n=== VISION ANALYSIS ===")

    for model in sorted(
        {result.model for result in vision_results}
    ):
        model_results = [
            result
            for result in vision_results
            if result.model == model
        ]

        best = max(
            model_results,
            key=lambda result: result.throughput,
        )

        print(
            f"{model}: "
            f"best throughput = "
            f"{best.throughput:.2f} "
            f"{best.throughput_unit}"
            f" ({best.precision})"
        )

        fp32_results = [
            result
            for result in model_results
            if result.precision == "FP32"
        ]

        if fp32_results:
            baseline = max(
                fp32_results,
                key=lambda result: result.throughput,
            )

            speedup = (
                best.throughput
                / baseline.throughput
            )

            latency_reduction = None

            if (
                baseline.average_latency_ms
                and best.average_latency_ms
            ):
                latency_reduction = (
                    1
                    - (
                        best.average_latency_ms
                        / baseline.average_latency_ms
                    )
                ) * 100

            print(
                f"  Speedup vs FP32: "
                f"{speedup:.2f}x"
            )

            if latency_reduction is not None:
                print(
                    f"  Latency reduction: "
                    f"{latency_reduction:.1f}%"
                )


def print_llm_analysis(
    results: Iterable[BenchmarkResult],
) -> None:
    llm_results = [
        result
        for result in results
        if result.model_type == "llm"
        and result.tokens_per_second is not None
    ]

    if not llm_results:
        return

    print("\n=== LLM ANALYSIS ===")

    for model in sorted(
        {result.model for result in llm_results}
    ):
        model_results = [
            result
            for result in llm_results
            if result.model == model
        ]

        best = max(
            model_results,
            key=lambda result: result.tokens_per_second,
        )

        print(
            f"{model}: "
            f"best generation throughput = "
            f"{best.tokens_per_second:.2f} tok/s"
        )


def main() -> None:
    results = load_all_results()

    if not results:
        print(
            "No benchmark result JSON files found."
        )
        return

    print(
        f"Loaded {len(results)} benchmark result(s)."
    )

    print_vision_results(results)
    print_llm_results(results)

    print_vision_analysis(results)
    print_llm_analysis(results)


if __name__ == "__main__":
    main()