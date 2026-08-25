# ResNet50 TensorRT INT8 Benchmark

## Hardware

- GPU: NVIDIA GeForce RTX 3060
- TensorRT: 10.4.0
- Precision: INT8
- Batch size: 1
- Input shape: 1 × 3 × 224 × 224
- Calibration images: 50
- Warm-up runs: 20
- Benchmark iterations: 100

## Results

| Metric          | Result      |
|-----------------|-------------|
| Average latency | 0.624 ms    |
| P50 latency     | 0.594 ms    |
| P95 latency     | 0.831 ms    |
| P99 latency     | 0.923 ms    |
| Throughput      | 1602.92 FPS |

## TensorRT Comparison

| Precision | Average Latency | Throughput  |
|-----------|-----------------|-------------|          
| FP32      | 1.995 ms        | 501.26 FPS  |
| FP16      | 1.189 ms        | 841.06 FPS  |
| INT8      | 0.624 ms        | 1602.92 FPS |

## Optimization

Compared with TensorRT FP16:

- Latency reduction: 47.5%
- Throughput improvement: 90.6%

Compared with the original PyTorch FP32 baseline:

- Latency reduction: 86.6%
- Throughput improvement: 647.7%
- Throughput multiplier: 7.48×

## Prediction Agreement

A 50-image evaluation was performed by comparing TensorRT INT8
top-1 predictions against the original PyTorch FP32 model.

- Evaluation images: 50
- Matching predictions: 44
- Prediction agreement: 88.00%

This is prediction agreement with the FP32 reference, not
classification accuracy, because ground-truth labels were not
available for these images.

## Observed mismatches

| Image   | FP32 | INT8 |
|---------|------|------|      
| 002.jpg | 398  | 860  |
| 342.jpg | 301  | 125  |
| 343.jpg | 301  | 616  |
| 350.jpg | 301  | 304  |
| 354.jpg | 301  | 304  |
| 355.jpg | 301  | 304  |