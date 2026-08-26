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