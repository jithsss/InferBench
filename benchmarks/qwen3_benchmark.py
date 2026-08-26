import time

import onnxruntime_genai as og


MODEL_PATH = "models/qwen3-0.6b"

PROMPT = (
    "Explain what model quantization is and "
    "why INT8 can improve inference performance."
)

MAX_NEW_TOKENS = 128


def main() -> None:
    print("Loading Qwen3-0.6B...")

    config = og.Config(MODEL_PATH)

    # Explicitly select CUDA.
    config.clear_providers()
    config.append_provider("cuda")

    print("Execution provider: CUDA")

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)

    print("Model loaded successfully.")

    # Tokenize prompt.
    input_tokens = tokenizer.encode(PROMPT)

    params = og.GeneratorParams(model)

    params.set_search_options(
        max_length=len(input_tokens) + MAX_NEW_TOKENS,
        do_sample=False,
    )

    generator = og.Generator(
        model,
        params,
    )

    # Add the prompt to the generator.
    generator.append_tokens(input_tokens)

    prompt_token_count = generator.token_count()

    print(
        f"Prompt tokens: {prompt_token_count}"
    )

    # ---------------------------------------------------------
    # First token / TTFT
    # ---------------------------------------------------------

    start_time = time.perf_counter()

    generator.generate_next_token()

    first_token_time = time.perf_counter()

    first_token = generator.get_next_tokens()[0]

    # ---------------------------------------------------------
    # Generate the remaining tokens
    # ---------------------------------------------------------

    while not generator.is_done():
        generator.generate_next_token()

    end_time = time.perf_counter()

    # Total sequence includes prompt + generated tokens.
    total_tokens = len(
        generator.get_sequence(0)
    )

    generated_tokens = (
        total_tokens - prompt_token_count
    )

    if generated_tokens <= 0:
        raise RuntimeError(
            "No generated tokens were produced."
        )

    ttft_ms = (
        first_token_time - start_time
    ) * 1000

    generation_time_s = (
        end_time - first_token_time
    )

    # We already generated the first token before
    # measuring generation_time_s, so include it.
    tokens_per_second = (
        generated_tokens
        / (end_time - start_time)
    )

    total_latency_ms = (
        end_time - start_time
    ) * 1000

    # ---------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------

    output_tokens = generator.get_sequence(0)

    output_text = tokenizer.decode(
        output_tokens
    )

    print("\n--- Qwen3-0.6B Benchmark ---")

    print(
        f"Prompt tokens:       {prompt_token_count}"
    )

    print(
        f"Generated tokens:    {generated_tokens}"
    )

    print(
        f"First token ID:      {first_token}"
    )

    print(
        f"TTFT:                {ttft_ms:.2f} ms"
    )

    print(
        f"Total generation:    {generation_time_s:.3f} s"
    )

    print(
        f"Tokens/sec:          {tokens_per_second:.2f}"
    )

    print(
        f"Total latency:       {total_latency_ms:.2f} ms"
    )

    print("\n--- Output ---")
    print(output_text)


if __name__ == "__main__":
    main()