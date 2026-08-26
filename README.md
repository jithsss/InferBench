# 🚀 InferBench v1.1

InferBench is a professional AI inference **optimization, benchmarking, and profiling framework**. It provides an end-to-end pipeline to aggressively compress neural networks, validate their accuracy post-optimization, and definitively prove their performance gains across different runtimes (such as TensorRT, ONNX Runtime, and PyTorch).

## 🔄 The End-to-End Pipeline

InferBench is not just a measurement tool; it is a complete deployment lab that splits into four phases:
1. **Optimization:** Uses dedicated calibrators to crush models from FP32 down to FP16 or INT8, fusing layers and building highly-optimized TensorRT engines.
2. **Benchmarking & Validation:** Runs the optimized models to measure exact throughput/latency, while strictly checking that the optimizations didn't destroy the model's intelligence (via Prediction Agreement or WER).
3. **Profiling:** Traces the execution on the hardware level to hunt for hidden bottlenecks (like unexpected `Memcpy` overheads or CPU fallbacks).
4. **Visualization:** Presents the final business value—proving exact speedups and quality retention—in a native Streamlit web dashboard.

## ✨ Key Features

- **Aggressive Quantization**: Built-in scripts to calibrate and compile INT8 TensorRT engines for complex models like YOLOv8n and ResNet50.
- **Multi-Modal Benchmarking**: Accurately measure end-to-end latencies (Avg, P50, P95, P99) and throughput (Tokens/sec, FPS, or RTF) for:
  - **Large Language Models** (e.g., Qwen3)
  - **Image Classification** (e.g., ResNet50)
  - **Object Detection** (e.g., YOLOv8n)
  - **Speech Recognition** (e.g., Whisper Tiny/Base/Small)
- **Quality Validation**: Measure Whisper speech recognition quality using Word Error Rate (WER) and Character Error Rate (CER). Object detection quality is validated via prediction agreement metrics to ensure INT8 calibration succeeded.
- **Advanced Profiling**: Profile ONNX and PyTorch models to automatically detect performance bottlenecks, such as implicit CPU fallbacks and inefficient host-to-device memory copies (`Memcpy`).
- **Interactive Pro Dashboard**: Visualize your profiling results and optimization scaling side-by-side via a modern Streamlit web interface.

## 📁 Architecture

- `quantization/`: INT8 entropy calibrators, precision converters, and TensorRT engine builders.
- `inferbench/`: CLI application entry point (`python -m inferbench`).
- `benchmarks/`: Benchmark implementations (ResNet, YOLO, Qwen, Whisper) and adapter logic.
- `profiling/`: ONNX and PyTorch hardware trace profiling and diagnostics logic.
- `dashboard/`: Native Streamlit performance dashboard (`app.py`).
- `export/`: Model ingestion (PyTorch to ONNX).
- `runtimes/`: Infrastructure for runtime environment validation and dynamic DLL loading.
- `configs/`: User configuration (`runtime.json`) for NVIDIA environments.

## ⚙️ Installation & Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **Configure TensorRT (Windows):**
If you have TensorRT installed locally but it's not in your system PATH, you can dynamically link it by editing `configs/runtime.json`:
```json
{
  "nvidia": {
    "cuda_path": "",
    "tensorrt_path": "B:/download/TensorRT-10.4.0.26"
  }
}
```

## 🚀 Usage Guide

InferBench operates primarily via a CLI, guiding you through the pipeline.

### Phase 1: Model Optimization
Before running benchmarks, compile your optimized engines using the `quantization/` suite. For example, to build an INT8 TensorRT engine:
```bash
python quantization/build_yolov8_tensorrt.py
```
*(This will process the calibration dataset and compile the `.engine` file).*

### Phase 2: Benchmarking & Validation
See all registered models and runtimes:
```bash
python -m inferbench list
```

Execute a specific benchmark to measure performance and validate accuracy. Metrics will be written to `results/`.
```bash
# Vision Baseline vs Optimized (TensorRT)
python -m inferbench run yolov8n-tensorrt-fp32
python -m inferbench run yolov8n-tensorrt-int8

# Speech Recognition Baseline vs Optimized (PyTorch FP16)
python -m inferbench run whisper-tiny-baseline
python -m inferbench run whisper-tiny-fp16

# Language Model (ONNX Runtime GenAI)
python -m inferbench run qwen3-0.6b
```

### Phase 3: Profiling & Diagnostics
If an optimized model isn't hitting expected targets, trace its execution to detect CPU fallback nodes or memory bottlenecks:
```bash
# Profile an ONNX graph
python -m inferbench profile-onnx export/yolov8n.onnx

# Profile a PyTorch model
python -m inferbench profile-whisper tiny
```

### Phase 4: Interactive Dashboard
Compare results, view speedup metrics across precisions, and read automated diagnostic alerts in the browser:
```bash
streamlit run dashboard/app.py
```

---

### Speech Metrics Guide
- **RTF (Real-Time Factor)**: `inference_time / audio_duration`. A value < 1 means inference is faster than real-time. Lower is better.
- **WER/CER**: Word Error Rate and Character Error Rate measure transcription quality against a ground-truth reference. Lower is better.
