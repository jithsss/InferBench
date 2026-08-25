# ResNet50 ONNX Runtime GPU Benchmark

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Precision: FP32
- Batch size: 1
- Input shape: 1 × 3 × 224 × 224
- Iterations: 100
- Warm-up runs: 20
- Execution Provider: CUDAExecutionProvider

## Results

| Metric          | Result   |  
|---|---:--------------------|        
| Average latency | 3.390 ms |
| P50 latency     | 3.313 ms |
| P95 latency     | 3.861 ms |
| P99 latency     | 4.188 ms |
| Throughput      | 294.99 FPS |

## PyTorch Comparison

| Runtime           | Average Latency | Throughput |
|-------------------|-----------------|------------|
| PyTorch FP32      | 4.665 ms        |214.38 FPS  |
| ONNX Runtime FP32 | 3.390 ms        | 294.99 FPS |

## Observation

ONNX Runtime achieved lower latency and higher throughput than the
PyTorch FP32 baseline in this batch-1 benchmark.