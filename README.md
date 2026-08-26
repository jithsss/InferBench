# ⚡ InferBench v1.1

InferBench is a professional AI inference benchmarking, profiling, and optimization framework. It evaluates and compares the performance of machine learning models across different runtimes (such as TensorRT, ONNX Runtime, and ONNX Runtime GenAI) and precision formats (FP32, FP16, INT8).

## ✨ Key Features

- **Multi-Modal Benchmarking**: Accurately measure end-to-end latencies (Avg, P50, P95, P99) and throughput (Tokens/sec or FPS) for:
  - **Large Language Models** (e.g., Qwen3)
  - **Image Classification** (e.g., ResNet50)
  - **Object Detection** (e.g., YOLOv8n)
- **Interactive Pro Dashboard**: Visualize your profiling results and optimization scaling via a modern, native Streamlit web interface.
- **YOLO Quality Validation**: Object detection quality is validated via a robust prediction agreement metric (cosine similarity between FP32 ONNX and INT8 TRT), ensuring high-fidelity models without requiring large evaluation datasets locally.
- **Advanced Profiling**: Profile ONNX models to automatically detect performance bottlenecks, such as implicit CPU fallbacks and inefficient host-to-device memory copies (`Memcpy`).

## 📁 Architecture

- `inferbench/`: CLI application entry point (`python -m inferbench`)
- `dashboard/`: Native Streamlit performance dashboard (`app.py`)
- `benchmarks/`: Benchmark implementations (ResNet, YOLO, Qwen) and adapter logic
- `profiling/`: ONNX profiling extraction and diagnostics logic
- `quantization/`: INT8 entropy calibrators and TensorRT engine builders
- `export/`: Model ingestion (PyTorch to ONNX)
- `runtimes/`: Infrastructure for runtime environment validation and dynamic DLL loading
- `configs/`: User configuration (`runtime.json`) for NVIDIA environments

## 🚀 Installation & Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Refer to `requirements-resnet-tensorrt.txt` if you are explicitly testing TRT optimizations).*

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

## 💻 Usage

InferBench operates primarily via a CLI.

### 1. List Available Benchmarks
See all registered models and runtimes:
```bash
python -m inferbench list
```

### 2. Run Benchmarks
Execute a specific benchmark. Metrics will be written as JSON to `results/`.
```bash
# Language Model (ONNX Runtime GenAI)
python -m inferbench run qwen3-0.6b

# Object Detection (TensorRT Engine)
python -m inferbench run yolov8n-tensorrt-int8
```

### 3. Profile ONNX Graphs
Trace ONNX model execution to detect CPU fallback nodes:
```bash
python -m inferbench profile-onnx export/yolov8n.onnx
```

### 4. Interactive Dashboard
Compare results, view speedup metrics, and read diagnostic alerts in the browser:
```bash
streamlit run dashboard/app.py
```
