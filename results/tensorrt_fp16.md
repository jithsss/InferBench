# ResNet50 TensorRT FP16 Benchmark

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Runtime: ONNX Runtime + TensorRT Execution Provider
- Precision: FP16
- Batch size: 1
- Input shape: 1 × 3 × 224 × 224
- Warm-up runs: 20
- Benchmark iterations: 100

## Results

| Metric          | Result     |
|-----------------|------------|
| Average latency | 1.189 ms   |
| P50 latency     | 1.102 ms   |
| P95 latency     | 1.713 ms   |
| P99 latency     | 2.200 ms   |
| Throughput      | 841.06 FPS |

## TensorRT FP32 Comparison

| Metric          | FP32       | FP16       |
|-----------------|------------|------------|
| Average latency | 1.995 ms   | 1.189 ms   |
| P50 latency     | 1.961 ms   | 1.102 ms   |
| P95 latency     | 2.032 ms   | 1.713 ms   |
| P99 latency     | 2.716 ms   | 2.200 ms   |
| Throughput      | 501.26 FPS | 841.06 FPS |

## Optimization Result

- Average latency reduction: 40.4%
- Throughput improvement: 67.8%

## Notes

TensorRT FP16 was enabled through the TensorRT execution provider.
The benchmark measures steady-state inference after warm-up and does
not include TensorRT engine build time.