import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

from benchmarks.benchmark_utils import (
    calculate_statistics,
    print_results,
)


ENGINE_PATH = Path(
    "export/resnet50_tensorrt_int8.engine"
)


def main() -> None:
    logger = trt.Logger(trt.Logger.ERROR)

    print("TensorRT:", trt.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    # Load serialized TensorRT engine.
    runtime = trt.Runtime(logger)

    engine_data = ENGINE_PATH.read_bytes()

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:
        raise RuntimeError(
            "Failed to deserialize TensorRT engine."
        )

    print("Engine:", ENGINE_PATH)

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "Failed to create TensorRT execution context."
        )

    # Find input/output tensors.
    input_name = None
    output_name = None

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError(
            "Could not identify engine input/output."
        )

    print("Input:", input_name)
    print("Output:", output_name)

    # Batch 1 benchmark shape.
    input_shape = (1, 3, 224, 224)

    context.set_input_shape(
        input_name,
        input_shape,
    )

    # TensorRT tells us the actual output shape.
    output_shape = context.get_tensor_shape(
        output_name
    )

    print("Input shape:", input_shape)
    print("Output shape:", tuple(output_shape))

    # Allocate GPU tensors using PyTorch.
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

    # Tell TensorRT where tensors live.
    context.set_tensor_address(
        input_name,
        input_tensor.data_ptr(),
    )

    context.set_tensor_address(
        output_name,
        output_tensor.data_ptr(),
    )

    stream = torch.cuda.current_stream().cuda_stream

    # Warm-up.
    for _ in range(20):
        context.execute_async_v3(stream)

    torch.cuda.synchronize()

    # Benchmark.
    latencies_ms: list[float] = []

    for _ in range(100):
        torch.cuda.synchronize()

        start = time.perf_counter()

        context.execute_async_v3(stream)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        latencies_ms.append(
            elapsed * 1000
        )

    result = calculate_statistics(
        latencies_ms
    )

    print_results(result)

    # Report memory allocated by PyTorch.
    print(
        f"Peak GPU memory:  "
        f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
    )


if __name__ == "__main__":
    main()