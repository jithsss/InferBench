import statistics
import time

import onnxruntime_genai as og


MODEL_PATH = "models/qwen3-0.6b"

PROMPT = (
    "Explain what model quantization is and "
    "why INT8 can improve inference performance."
)

MAX_NEW_TOKENS = 128
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


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

    # ---------------------------------------------------------
    # Prompt processing
    # ---------------------------------------------------------

    prompt_start = time.perf_counter()

    generator.append_tokens(input_tokens)

    prompt_end = time.perf_counter()

    prompt_processing_ms = (
        prompt_end - prompt_start
    ) * 1000.0

    prompt_token_count = generator.token_count()

    # ---------------------------------------------------------
    # First token / sampling
    # ---------------------------------------------------------

    sampling_start = time.perf_counter()

    generator.generate_next_token()

    sampling_end = time.perf_counter()

    sampling_ms = (
        sampling_end - sampling_start
    ) * 1000.0

    ttft_ms = (
        prompt_processing_ms + sampling_ms
    )

    # ---------------------------------------------------------
    # Remaining token generation
    # ---------------------------------------------------------

    token_times = []

    while not generator.is_done():
        token_start = time.perf_counter()

        generator.generate_next_token()

        token_end = time.perf_counter()

        token_times.append(
            token_end - token_start
        )

    sequence = generator.get_sequence(0)

    total_tokens = len(sequence)

    generated_tokens = (
        total_tokens - prompt_token_count
    )

    if generated_tokens <= 0:
        raise RuntimeError(
            "No generated tokens."
        )

    # The first token was generated during sampling.
    # Subsequent token timings are stored in token_times.
    subsequent_tokens = max(
        generated_tokens - 1,
        1,
    )

    generation_time_s = sum(token_times)

    generation_tokens_per_sec = (
        subsequent_tokens / generation_time_s
        if generation_time_s > 0
        else 0.0
    )

    total_start_to_finish_ms = (
        prompt_processing_ms
        + sampling_ms
        + generation_time_s * 1000.0
    )

    return (
        ttft_ms,
        generation_tokens_per_sec,
        total_start_to_finish_ms,
        generated_tokens,
    )


def print_stats(
    name: str,
    values: list[float],
    unit: str,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Average: {statistics.mean(values):.2f} {unit}"
    )

    print(
        f"P50:     {percentile(values, 50):.2f} {unit}"
    )

    print(
        f"P95:     {percentile(values, 95):.2f} {unit}"
    )

    print(
        f"P99:     {percentile(values, 99):.2f} {unit}"
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

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print(
        f"Running {WARMUP_RUNS} warm-up runs..."
    )

    for _ in range(WARMUP_RUNS):
        run_generation(
            model,
            input_tokens,
        )

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    print(
        f"Running {BENCHMARK_RUNS} benchmark runs..."
    )

    ttft_values = []
    throughput_values = []
    total_latency_values = []
    generated_token_counts = []

    for run_index in range(
        BENCHMARK_RUNS
    ):
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
        throughput_values.append(
            tokens_per_second
        )
        total_latency_values.append(
            total_latency_ms
        )
        generated_token_counts.append(
            generated_tokens
        )

        print(
            f"Run {run_index + 1}: "
            f"TTFT={ttft_ms:.2f} ms, "
            f"Tokens/sec={tokens_per_second:.2f}, "
            f"Total={total_latency_ms:.2f} ms"
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n=== Qwen3-0.6B Benchmark ===")

    print(
        f"Prompt tokens: "
        f"{len(input_tokens)}"
    )

    print(
        f"Generated tokens: "
        f"{statistics.mean(generated_token_counts):.0f}"
    )

    print_stats(
        "Time to First Token",
        ttft_values,
        "ms",
    )

    print_stats(
        "Generation Throughput",
        throughput_values,
        "tokens/sec",
    )

    print_stats(
        "Total Generation Latency",
        total_latency_values,
        "ms",
    )


if __name__ == "__main__":
    main()