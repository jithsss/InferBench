# ResNet50 FP32 Runtime Comparison

Hardware:
- GPU: NVIDIA GeForce RTX 3060

| Runtime | Device | Precision | Latency | Throughput |
|---|---|---|---:|---:|
| PyTorch | RTX 3060 | FP32 | 4.726 ms | 211.58 FPS |
| ONNX Runtime | RTX 3060 | FP32 | 25.213 ms | 39.66 FPS |
| ONNX Runtime | CPU | FP32 | 23.952 ms | 41.75 FPS |

## Observation

In the current batch-1 benchmark, PyTorch achieved substantially
higher throughput than ONNX Runtime. This indicates that the current
benchmark includes runtime/input overhead and that ONNX Runtime
requires further profiling and optimization before drawing conclusions
about deployment performance.