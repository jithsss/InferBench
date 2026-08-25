# ResNet50 INT8 Optimization

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- Runtime: ONNX Runtime
- Execution Provider: CUDAExecutionProvider
- Batch size: 1
- Precision: INT8
- Iterations: 100
- Warm-up runs: 20

## Results

| Metric          | INT8       |
|-----------------|------------|
| Average latency | 4.981 ms   |
| P50 latency     | 4.828 ms   |
| P95 latency     | 5.809 ms   |
| P99 latency     | 6.687 ms   |
| Throughput      | 200.78 FPS |

## Correctness

- Maximum absolute difference: 1.28025126
- Mean absolute difference: 0.13526142
- FP32 predicted class: 21
- INT8 predicted class: 21
- Predicted class match: True

## Observation

The current INT8 configuration is slower than both FP32 and FP16
on the RTX 3060 batch-1 benchmark.

The ONNX Runtime CUDA execution provider reports 54 Memcpy nodes
in the graph, which may contribute to the observed performance overhead.