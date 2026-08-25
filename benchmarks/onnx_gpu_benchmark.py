import time

import numpy as np
import torch
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

    for _ in range(benchmark_runs):
        start = time.perf_counter()

        session.run(
            [output_name],
            {input_name: input_data},
        )

        # Synchronize CUDA work before stopping timer.
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000)

    return latencies_ms


def main() -> None:
    batch_size = 1

    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    # Reset PyTorch CUDA memory statistics.
    torch.cuda.reset_peak_memory_stats()

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Model:", MODEL_PATH)
    print("Batch size:", batch_size)
    print("Active providers:", session.get_providers())

    np.random.seed(42)

    input_data = np.random.randn(
        batch_size,
        3,
        224,
        224,
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

    # PyTorch CUDA allocator statistics are not a reliable measurement
    # of ONNX Runtime's own allocator, so we don't label this as
    # "ONNX memory usage."
    #
    # We will add proper provider-level memory profiling later.


if __name__ == "__main__":
    main()