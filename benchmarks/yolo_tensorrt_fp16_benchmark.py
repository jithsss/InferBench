import time
from pathlib import Path
import torch
import tensorrt as trt

from benchmarks.benchmark_utils import calculate_statistics, print_results
from benchmarks.result_schema import BenchmarkResult
from benchmarks.result_writer import save_benchmark_result
from benchmarks.yolov8_adapter import compute_prediction_agreement
from export.export_yolov8 import main as export_yolov8
from quantization.build_yolov8_tensorrt import build_engine

ONNX_PATH = Path("export/yolov8n.onnx")
ENGINE_PATH = Path("export/yolov8n_tensorrt_fp16.engine")
RESULT_PATH = "results/vision/yolov8n_tensorrt_fp16.json"

def main() -> None:
    if not ONNX_PATH.exists():
        export_yolov8()
    
    if not ENGINE_PATH.exists():
        build_engine("fp16")
        
    print("TensorRT:", trt.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)

    engine_data = ENGINE_PATH.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT FP16 engine.")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    batch_size = 1
    input_shape = (batch_size, 3, 640, 640)
    context.set_input_shape(input_name, input_shape)
    output_shape = context.get_tensor_shape(output_name)

    print("Engine:", ENGINE_PATH)
    print("Input:", input_name, "shape:", input_shape)
    print("Output:", output_name, "shape:", tuple(output_shape))

    input_tensor = torch.randn(input_shape, dtype=torch.float32, device="cuda")
    output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream

    for _ in range(20):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000)

    result = calculate_statistics(latencies_ms)
    print_results(result)
    
    print("Computing FP32-vs-FP16 prediction agreement...")
    agreement = compute_prediction_agreement(str(ENGINE_PATH), str(ONNX_PATH))
    print(f"Prediction agreement: {agreement:.2f}%")

    benchmark_result = BenchmarkResult(
        model="YOLOv8n",
        model_type="vision",
        runtime="TensorRT",
        execution_provider="TensorRT",
        precision="FP16",
        batch_size=batch_size,
        average_latency_ms=result.average_latency_ms,
        p50_latency_ms=result.p50_latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        p99_latency_ms=result.p99_latency_ms,
        throughput=result.throughput_fps,
        throughput_unit="FPS",
        task="object_detection",
        input_resolution="640x640",
        prediction_agreement=agreement,
        notes="YOLOv8n FP16 TensorRT benchmark.",
    )

    save_benchmark_result(benchmark_result, RESULT_PATH)

if __name__ == "__main__":
    main()
