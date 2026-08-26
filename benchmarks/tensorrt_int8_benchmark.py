import time
from pathlib import Path

import torch
import tensorrt as trt

from benchmarks.benchmark_utils import (
    calculate_statistics,
    print_results,
)
from benchmarks.result_schema import BenchmarkResult
from benchmarks.result_writer import save_benchmark_result


ENGINE_PATH = Path(
    "export/resnet50_tensorrt_int8.engine"
)

RESULT_PATH = (
    "results/vision/"
    "resnet50_tensorrt_int8.json"
)


def main() -> None:
    print("TensorRT:", trt.__version__)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    logger = trt.Logger(
        trt.Logger.ERROR
    )

    runtime = trt.Runtime(logger)

    engine_data = ENGINE_PATH.read_bytes()

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:
        raise RuntimeError(
            "Failed to deserialize TensorRT INT8 engine."
        )

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "Failed to create TensorRT execution context."
        )

    input_name = None
    output_name = None

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError(
            "Unable to identify TensorRT input/output."
        )

    batch_size = 1

    input_shape = (
        batch_size,
        3,
        224,
        224,
    )

    context.set_input_shape(
        input_name,
        input_shape,
    )

    output_shape = context.get_tensor_shape(
        output_name
    )

    print("Engine:", ENGINE_PATH)
    print("Input:", input_name)
    print("Output:", output_name)
    print("Input shape:", input_shape)
    print(
        "Output shape:",
        tuple(output_shape),
    )

    input_tensor = torch.randn(
        input_shape,
        dtype=torch.float32,
        device="cuda",
    )

    output_tensor = torch.empty(
        tuple(output_shape),
        dtype=torch.float32,
        device="cuda",
    )

    context.set_tensor_address(
        input_name,
        input_tensor.data_ptr(),
    )

    context.set_tensor_address(
        output_name,
        output_tensor.data_ptr(),
    )

    stream = (
        torch.cuda.current_stream().cuda_stream
    )

    for _ in range(20):
        context.execute_async_v3(stream)

    torch.cuda.synchronize()

    latencies_ms: list[float] = []

    for _ in range(100):
        torch.cuda.synchronize()

        start = time.perf_counter()

        context.execute_async_v3(stream)

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - start
        )

        latencies_ms.append(
            elapsed * 1000
        )

    result = calculate_statistics(
        latencies_ms
    )

    print_results(result)

    benchmark_result = BenchmarkResult(
        model="ResNet50",
        model_type="vision",
        runtime="TensorRT",
        execution_provider="TensorRT",
        precision="INT8",
        batch_size=batch_size,
        average_latency_ms=result.average_latency_ms,
        p50_latency_ms=result.p50_latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        p99_latency_ms=result.p99_latency_ms,
        throughput=result.throughput_fps,
        throughput_unit="FPS",
        accuracy_metric="FP32 prediction agreement",
        accuracy_value=90.0,
        notes=(
            "Direct TensorRT INT8 engine benchmark. "
            "40 images were used for calibration and "
            "10 held-out images for prediction agreement. "
            "The 90% value is prediction agreement, "
            "not ground-truth classification accuracy."
        ),
    )

    save_benchmark_result(
        benchmark_result,
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()