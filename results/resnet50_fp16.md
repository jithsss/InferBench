# ResNet50 FP16 Optimization

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Runtime: ONNX Runtime
- Execution Provider: CUDAExecutionProvider
- Batch size: 1
- Input: 1 × 3 × 224 × 224
- Benchmark iterations: 100
- Warm-up: 20

## Results

| Metric          | FP32       | FP16       |
|-----------------|------------|------------|
| Average latency | 3.390 ms   | 2.767 ms   |
| P50 latency     | 3.313 ms   | 2.761 ms   |
| P95 latency     | 3.861 ms   | 3.186 ms   |
| P99 latency     | 4.188 ms   | 3.233 ms   |
| Throughput      | 294.99 FPS | 361.36 FPS |

## Optimization Results

- Average latency improvement: 18.4%
- Throughput improvement: 22.5%

## Correctness

FP32 vs FP16:

- Maximum absolute difference: 0.00396371
- Mean absolute difference: 0.00065461
- Predicted class: 21 for both models
- Predicted class match: True