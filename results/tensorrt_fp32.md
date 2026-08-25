# ResNet50 TensorRT FP32 Benchmark

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Runtime: ONNX Runtime + TensorRT Execution Provider
- Precision: FP32
- Batch size: 1
- Input shape: 1 × 3 × 224 × 224
- Warm-up runs: 20
- Benchmark iterations: 100

## Results

| Metric          | Result     |
|------------------------------|
| Average latency | 1.995 ms   |
| P50 latency     | 1.961 ms   |
| P95 latency     | 2.032 ms   |
| P99 latency     | 2.716 ms   |
| Throughput      | 501.26 FPS |

## Comparison with PyTorch

| Runtime             | Precision | Average Latency | Throughput |
|---------------------|-----------|-----------------|------------|     
| PyTorch             | FP32      | 4.665 ms        | 214.38 FPS |
| ONNX Runtime + CUDA | FP32      | 3.390 ms        | 294.99 FPS |
| TensorRT            | FP32      | 1.995 ms        | 501.26 FPS |

## Observation

TensorRT FP32 achieved substantially lower latency and higher throughput
than the PyTorch and ONNX Runtime CUDA baselines in the batch-1
benchmark.

Compared with the PyTorch FP32 baseline, average latency decreased
from 4.665 ms to 1.995 ms, while throughput increased from 214.38 FPS
to 501.26 FPS.

## Notes

The benchmark measures steady-state inference performance after warm-up.
TensorRT engine build/initialization time is not included in the reported
latency.