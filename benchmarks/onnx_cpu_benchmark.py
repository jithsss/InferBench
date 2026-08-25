import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_PATH = Path("export/resnet50_fp32.onnx")


def main() -> None:
    print(f"Model: {MODEL_PATH}")

    # Create an ONNX Runtime session using the CPU execution provider.
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print(f"Execution provider: {session.get_providers()[0]}")

    # Create the same input shape used by the PyTorch benchmark.
    np.random.seed(42)
    input_data = np.random.randn(
        1, 3, 224, 224
    ).astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warm-up
    for _ in range(20):
        session.run(
            [output_name],
            {input_name: input_data},
        )

    # Benchmark
    iterations = 100

    start = time.perf_counter()

    for _ in range(iterations):
        session.run(
            [output_name],
            {input_name: input_data},
        )

    elapsed = time.perf_counter() - start

    average_latency_ms = (
        elapsed / iterations
    ) * 1000

    throughput_fps = iterations / elapsed

    print("\n--- ONNX Runtime CPU Baseline ---")
    print(f"Iterations:       {iterations}")
    print(f"Average latency:  {average_latency_ms:.3f} ms")
    print(f"Throughput:       {throughput_fps:.2f} FPS")


if __name__ == "__main__":
    main()