import time

import numpy as np
import torch
import onnxruntime as ort

from benchmarks.benchmark_utils import (
    calculate_statistics,
    print_results,
)
from benchmarks.result_schema import BenchmarkResult
from benchmarks.result_writer import save_benchmark_result


MODEL_PATH = "export/resnet50_fp32.onnx"
RESULT_PATH = "results/vision/resnet50_tensorrt_fp16.json"


def main() -> None:
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    tensorrt_options = {
        "trt_fp16_enable": True,
    }

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            (
                "TensorrtExecutionProvider",
                tensorrt_options,
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Model:", MODEL_PATH)
    print("Active providers:", session.get_providers())
    print("TensorRT FP16: enabled")

    batch_size = 1

    np.random.seed(42)

    input_data = np.random.randn(
        batch_size,
        3,
        224,
        224,
    ).astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warm-up
    for _ in range(20):
        session.run(
            [output_name],
            {input_name: input_data},
        )

    latencies_ms: list[float] = []

    for _ in range(100):
        start = time.perf_counter()

        session.run(
            [output_name],
            {input_name: input_data},
        )

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000)

    result = calculate_statistics(
        latencies_ms
    )

    print_results(result)

    benchmark_result = BenchmarkResult(
        model="ResNet50",
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
        notes=(
            "TensorRT FP16 enabled through the "
            "ONNX Runtime TensorRT Execution Provider. "
            "20 warm-up runs and 100 measured runs."
        ),
    )

    save_benchmark_result(
        benchmark_result,
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()