import time
import os

class DynamicLLMBenchmark:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        import onnxruntime_genai as og
        config = og.Config(self.model_dir)
        config.clear_providers()
        try:
            config.append_provider('cuda')
        except Exception:
            pass
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)

    def run_benchmark(self, prompt: str, max_new_tokens: int = 128) -> dict:
        import onnxruntime_genai as og
        input_tokens = self.tokenizer.encode(prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + max_new_tokens, do_sample=False)
        generator = og.Generator(self.model, params)
        
        prompt_start = time.perf_counter()
        generator.append_tokens(input_tokens)
        prompt_end = time.perf_counter()
        prompt_processing_ms = (prompt_end - prompt_start) * 1000.0
        prompt_token_count = generator.token_count()
        
        sampling_start = time.perf_counter()
        generator.generate_next_token()
        sampling_end = time.perf_counter()
        sampling_ms = (sampling_end - sampling_start) * 1000.0
        ttft_ms = prompt_processing_ms + sampling_ms
        
        token_times = []
        while not generator.is_done():
            token_start = time.perf_counter()
            generator.generate_next_token()
            token_end = time.perf_counter()
            token_times.append(token_end - token_start)
            
        sequence = generator.get_sequence(0)
        generated_tokens = len(sequence) - prompt_token_count
        subsequent_tokens = max(generated_tokens - 1, 1)
        generation_time_s = sum(token_times)
        tokens_per_second = subsequent_tokens / generation_time_s if generation_time_s > 0 else 0.0
        total_latency_ms = prompt_processing_ms + sampling_ms + (generation_time_s * 1000.0)
        
        output_text = self.tokenizer.decode(sequence[prompt_token_count:])
        
        return {
            'ttft_ms': ttft_ms,
            'tokens_per_second': tokens_per_second,
            'total_latency_ms': total_latency_ms,
            'generated_tokens': generated_tokens,
            'prompt_tokens': len(input_tokens),
            'output_text': output_text
        }
