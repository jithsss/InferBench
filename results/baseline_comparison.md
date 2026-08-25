# ResNet50 FP32 Runtime Comparison

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Input: 1 × 3 × 224 × 224
- Precision: FP32
- Iterations: 100
- Warm-up: 20

## Results

| Runtime     | Device   | Average  | P50      | P95      | P99      | Throughput |
|-------------|---:------|---:------|---:------|---:------|---:------|------------|
| PyTorch     | RTX 3060 | 4.665 ms | 4.156 ms | 6.464 ms | 8.670 ms | 214.38 FPS |
| ONNX Runtime| RTX 3060 | 3.228 ms | 3.134 ms | 3.572 ms | 3.801 ms | 309.75 FPS |

## Observation

ONNX Runtime achieved lower latency and higher throughput than the
PyTorch baseline in this batch-1 FP32 benchmark.

Average latency decreased by approximately 30.8%, while throughput
increased by approximately 44.5%.