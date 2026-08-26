from runtimes.registry import register_benchmark


register_benchmark(
    "resnet50-tensorrt-fp32",
    "ResNet50 TensorRT FP32 benchmark",
    "benchmarks.tensorrt_fp32_benchmark",
)

register_benchmark(
    "resnet50-tensorrt-fp16",
    "ResNet50 TensorRT FP16 benchmark",
    "benchmarks.tensorrt_fp16_benchmark",
)

register_benchmark(
    "resnet50-tensorrt-int8",
    "ResNet50 TensorRT INT8 benchmark",
    "benchmarks.tensorrt_int8_benchmark",
)

register_benchmark(
    "qwen3-0.6b",
    "Qwen3-0.6B ONNX Runtime GenAI benchmark",
    "benchmarks.qwen3_benchmark",
)

register_benchmark(
    "yolov8n-tensorrt-fp32",
    "YOLOv8n TensorRT FP32 benchmark",
    "benchmarks.yolo_tensorrt_fp32_benchmark",
)

register_benchmark(
    "yolov8n-tensorrt-fp16",
    "YOLOv8n TensorRT FP16 benchmark",
    "benchmarks.yolo_tensorrt_fp16_benchmark",
)

register_benchmark(
    "yolov8n-tensorrt-int8",
    "YOLOv8n TensorRT INT8 benchmark",
    "benchmarks.yolo_tensorrt_int8_benchmark",
)