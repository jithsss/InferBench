import statistics
import time

import onnxruntime_genai as og

from benchmarks.result_schema import BenchmarkResult
from benchmarks.result_writer import save_benchmark_result


MODEL_PATH = "models/qwen3-0.6b"

PROMPT = (
    "Explain what model quantization is and "
    "why INT8 can improve inference performance."
)

MAX_NEW_TOKENS = 128
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

RESULT_PATH = (
    "results/llm/qwen3_0.6b_baseline.json"
)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p / 100.0

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def create_generator(
    model: og.Model,
    input_tokens,
) -> og.Generator:
    params = og.GeneratorParams(model)

    params.set_search_options(
        max_length=len(input_tokens) + MAX_NEW_TOKENS,
        do_sample=False,
    )

    generator = og.Generator(model, params)

    return generator


def run_generation(
    model: og.Model,
    input_tokens,
) -> tuple[float, float, float, int]:
    generator = create_generator(
        model,
        input_tokens,
    )

    # Prompt processing
    prompt_start = time.perf_counter()

    generator.append_tokens(input_tokens)

    prompt_end = time.perf_counter()

    prompt_processing_ms = (
        prompt_end - prompt_start
    ) * 1000.0

    prompt_token_count = generator.token_count()

    # First token
    sampling_start = time.perf_counter()

    generator.generate_next_token()

    sampling_end = time.perf_counter()

    sampling_ms = (
        sampling_end - sampling_start
    ) * 1000.0

    ttft_ms = (
        prompt_processing_ms + sampling_ms
    )

    # Remaining tokens
    token_times = []

    while not generator.is_done():
        token_start = time.perf_counter()

        generator.generate_next_token()

        token_end = time.perf_counter()

        token_times.append(
            token_end - token_start
        )

    sequence = generator.get_sequence(0)

    generated_tokens = (
        len(sequence) - prompt_token_count
    )

    if generated_tokens <= 0:
        raise RuntimeError(
            "No generated tokens."
        )

    subsequent_tokens = max(
        generated_tokens - 1,
        1,
    )

    generation_time_s = sum(token_times)

    tokens_per_second = (
        subsequent_tokens / generation_time_s
        if generation_time_s > 0
        else 0.0
    )

    total_latency_ms = (
        prompt_processing_ms
        + sampling_ms
        + generation_time_s * 1000.0
    )

    return (
        ttft_ms,
        tokens_per_second,
        total_latency_ms,
        generated_tokens,
    )


def main() -> None:
    print("Loading Qwen3-0.6B...")

    config = og.Config(MODEL_PATH)

    config.clear_providers()
    config.append_provider("cuda")

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)

    print("Model loaded successfully.")
    print("Execution provider: CUDA")

    input_tokens = tokenizer.encode(PROMPT)

    print(
        f"Prompt tokens: {len(input_tokens)}"
    )

    # Warm-up
    print(
        f"Running {WARMUP_RUNS} warm-up runs..."
    )

    for _ in range(WARMUP_RUNS):
        run_generation(
            model,
            input_tokens,
        )

    # Benchmark
    print(
        f"Running {BENCHMARK_RUNS} benchmark runs..."
    )

    ttft_values: list[float] = []
    throughput_values: list[float] = []
    latency_values: list[float] = []
    token_counts: list[int] = []

    for run_index in range(BENCHMARK_RUNS):
        (
            ttft_ms,
            tokens_per_second,
            total_latency_ms,
            generated_tokens,
        ) = run_generation(
            model,
            input_tokens,
        )

        ttft_values.append(ttft_ms)
        throughput_values.append(tokens_per_second)
        latency_values.append(total_latency_ms)
        token_counts.append(generated_tokens)

        print(
            f"Run {run_index + 1}: "
            f"TTFT={ttft_ms:.2f} ms, "
            f"Tokens/sec={tokens_per_second:.2f}, "
            f"Total={total_latency_ms:.2f} ms"
        )

    average_ttft = statistics.mean(ttft_values)
    average_tokens_per_second = statistics.mean(
        throughput_values
    )
    average_latency = statistics.mean(
        latency_values
    )
    average_generated_tokens = statistics.mean(
        token_counts
    )

    print("\n=== Qwen3-0.6B Benchmark ===")

    print(
        f"Prompt tokens: "
        f"{len(input_tokens)}"
    )

    print(
        f"Generated tokens: "
        f"{average_generated_tokens:.0f}"
    )

    print("\nTime to First Token")
    print("-------------------")
    print(
        f"Average: {average_ttft:.2f} ms"
    )
    print(
        f"P50:     {percentile(ttft_values, 50):.2f} ms"
    )
    print(
        f"P95:     {percentile(ttft_values, 95):.2f} ms"
    )
    print(
        f"P99:     {percentile(ttft_values, 99):.2f} ms"
    )

    print("\nGeneration Throughput")
    print("---------------------")
    print(
        f"Average: "
        f"{average_tokens_per_second:.2f} tokens/sec"
    )
    print(
        f"P50: "
        f"{percentile(throughput_values, 50):.2f} tokens/sec"
    )
    print(
        f"P95: "
        f"{percentile(throughput_values, 95):.2f} tokens/sec"
    )
    print(
        f"P99: "
        f"{percentile(throughput_values, 99):.2f} tokens/sec"
    )

    print("\nTotal Generation Latency")
    print("------------------------")
    print(
        f"Average: {average_latency:.2f} ms"
    )
    print(
        f"P50:     "
        f"{percentile(latency_values, 50):.2f} ms"
    )
    print(
        f"P95:     "
        f"{percentile(latency_values, 95):.2f} ms"
    )
    print(
        f"P99:     "
        f"{percentile(latency_values, 99):.2f} ms"
    )

    # Save unified result
    benchmark_result = BenchmarkResult(
        model="Qwen3-0.6B",
        model_type="llm",
        runtime="ONNX Runtime GenAI",
        execution_provider="CUDA",
        precision="Q4F16",
        batch_size=1,
        ttft_ms=average_ttft,
        tokens_per_second=average_tokens_per_second,
        average_latency_ms=average_latency,
        p50_latency_ms=percentile(
            latency_values,
            50,
        ),
        p95_latency_ms=percentile(
            latency_values,
            95,
        ),
        p99_latency_ms=percentile(
            latency_values,
            99,
        ),
        notes=(
            "16-token prompt, 128 generated tokens, "
            "2 warm-up runs, 5 benchmark runs."
        ),
    )

    save_benchmark_result(
        benchmark_result,
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()