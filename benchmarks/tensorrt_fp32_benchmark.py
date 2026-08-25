import time

import numpy as np
import torch
import onnxruntime as ort

from benchmarks.benchmark_utils import (
    calculate_statistics,
    print_results,
)


MODEL_PATH = "export/resnet50_fp32.onnx"


def main() -> None:
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Model:", MODEL_PATH)
    print("Active providers:", session.get_providers())

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

    # Benchmark
    for _ in range(100):
        start = time.perf_counter()

        session.run(
            [output_name],
            {input_name: input_data},
        )

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000)

    result = calculate_statistics(latencies_ms)

    print_results(result)


if __name__ == "__main__":
    main()