import time

import torch
import numpy as np
import onnxruntime as ort

from benchmarks.benchmark_utils import (
    calculate_statistics,
    print_results,
)


MODEL_PATH = "export/resnet50_fp32.onnx"


def benchmark_onnx(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    input_data: np.ndarray,
    warmup_runs: int = 20,
    benchmark_runs: int = 100,
) -> list[float]:

    # Warm-up
    for _ in range(warmup_runs):
        session.run(
            [output_name],
            {input_name: input_data},
        )

    latencies_ms: list[float] = []

    # Benchmark
    for _ in range(benchmark_runs):

        start = time.perf_counter()

        session.run(
            [output_name],
            {input_name: input_data},
        )

        elapsed = time.perf_counter() - start

        latencies_ms.append(elapsed * 1000)

    return latencies_ms


def main() -> None:
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Model:", MODEL_PATH)
    print("Active providers:", session.get_providers())

    np.random.seed(42)

    input_data = np.random.randn(
        1, 3, 224, 224
    ).astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    latencies_ms = benchmark_onnx(
        session=session,
        input_name=input_name,
        output_name=output_name,
        input_data=input_data,
    )

    result = calculate_statistics(latencies_ms)

    print_results(result)


if __name__ == "__main__":
    main()