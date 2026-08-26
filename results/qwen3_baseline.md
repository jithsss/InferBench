# Qwen3-0.6B Baseline

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Runtime: ONNX Runtime GenAI
- Execution Provider: CUDA
- Model: Qwen3-0.6B Q4F16
- Prompt tokens: 16
- Maximum new tokens: 128

## Results

| Metric           | Result    |
|------------------|-----------|
| Prompt tokens    | 16        |
| Generated tokens | 128       |
| TTFT             | 3.71 ms   |
| Generation time  | 0.586 s   |
| Tokens/sec       | 217.15    |
| Total latency    | 589.45 ms |

## Notes

This is the initial single-run Qwen3 benchmark.
A repeated benchmark will be added for more stable performance statistics.


# Qwen3-0.6B Baseline

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Runtime: ONNX Runtime GenAI
- Execution Provider: CUDA
- Model: Qwen3-0.6B Q4F16
- Prompt tokens: 16
- Generated tokens: 128
- Warm-up runs: 2
- Benchmark runs: 5

## Results

| Metric                | Average      | P50          | P95          | P99          |
|-----------------------|--------------|--------------|--------------|--------------|
| TTFT                  | 6.35 ms      | 6.39 ms      | 6.43 ms      | 6.44 ms      |
| Generation throughput | 240.37 tok/s | 240.44 tok/s | 246.52 tok/s | 246.90 tok/s |
| Total latency         | 534.95 ms    | 534.47 ms    | 550.68 ms    | 552.61 ms    |

## Notes

This is the initial steady-state Qwen3 benchmark after warm-up.