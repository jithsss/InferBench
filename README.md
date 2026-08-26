# InferBench

InferBench is an AI inference benchmarking and profiling framework. It evaluates, compares, and profiles the performance of various machine learning models across different runtimes (such as TensorRT and ONNX Runtime) and precisions (FP32, FP16, INT8).

## Features

- **Benchmarking**: Measure average, P50, P95, and P99 latencies, along with throughput (Tokens/sec or FPS) for Large Language Models (LLMs), Image Classification, and Object Detection (YOLO).
- **YOLO Quality Validation**: Object detection quality is measured via a robust prediction agreement metric comparing FP32 vs. optimized TensorRT outputs, ensuring high-fidelity INT8 engines without requiring massive labeled datasets.
- **Profiling**: Profile ONNX models to identify performance bottlenecks, such as implicit CPU fallbacks and inefficient host-to-device memory copies (`Memcpy`).
- **Standardized Reporting**: Save and compare benchmark results using a unified JSON schema.

## Project Structure

- `inferbench/`: Main application package and command-line interface entry point.
- `benchmarks/`: Benchmark implementations for specific models and runtime combinations.
- `profiling/`: Tools for tracing and analyzing ONNX Runtime execution (`ort_profile.py`).
- `export/`: Utilities for exporting standard models (e.g., PyTorch to ONNX).
- `runtimes/`: Infrastructure for runtime environment validation and benchmark registration.
- `results/`: Standardized output directory for benchmark JSON reports.

## Installation

Ensure you have Python installed (and appropriate NVIDIA drivers/CUDA), then install the dependencies:

```bash
pip install -r requirements.txt
```

*(Note: Certain benchmarks, such as TensorRT, may require additional dependencies like `tensorrt` or `onnxruntime-genai` depending on what you are benchmarking. You can find extra requirements in files like `requirements-resnet-tensorrt.txt`).*

## Usage

InferBench provides a command-line interface via the `inferbench` module. You can run it from the root of the repository.

### List Benchmarks
List all registered benchmarks available to run:
```bash
python -m inferbench list
```

### Run a Benchmark
Execute a specific benchmark. For example, to run the Qwen3 0.6B LLM benchmark:
```bash
python -m inferbench run qwen3-0.6b
```

To run a vision model with TensorRT INT8:
```bash
python -m inferbench run resnet50-tensorrt-int8
```

### Compare Results
Compare previously saved benchmark results located in the `results/` folder:
```bash
python -m inferbench compare
```

### Profile an ONNX Model
Profile an ONNX model to generate an execution trace and automatically analyze it for CPU fallback and Memcpy overhead:
```bash
python -m inferbench profile-onnx export/resnet50_int8.onnx --output-dir profiling/
```
