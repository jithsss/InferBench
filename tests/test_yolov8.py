import json
from pathlib import Path
from benchmarks.result_schema import BenchmarkResult, save_result, load_result
from runtimes.registry import get_benchmark, list_benchmarks
import runtimes.benchmarks_registry

def test_yolo_schema(tmp_path: Path):
    result = BenchmarkResult(
        model="YOLOv8n",
        model_type="vision",
        runtime="TensorRT",
        execution_provider="TensorRT",
        precision="INT8",
        batch_size=1,
        average_latency_ms=5.0,
        throughput=200.0,
        task="object_detection",
        input_resolution="640x640",
        prediction_agreement=99.5
    )
    
    file_path = tmp_path / "test_yolo.json"
    save_result(result, str(file_path))
    
    loaded = load_result(str(file_path))
    assert loaded.model == "YOLOv8n"
    assert loaded.task == "object_detection"
    assert loaded.prediction_agreement == 99.5

def test_yolo_registry():
    benchmarks = [b.name for b in list_benchmarks()]
    assert "yolov8n-tensorrt-fp32" in benchmarks
    assert "yolov8n-tensorrt-fp16" in benchmarks
    assert "yolov8n-tensorrt-int8" in benchmarks
    
    fp32_bench = get_benchmark("yolov8n-tensorrt-fp32")
    assert fp32_bench.name == "yolov8n-tensorrt-fp32"
    assert "YOLO" in fp32_bench.description
