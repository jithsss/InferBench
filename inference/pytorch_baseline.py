import time

import torch
import torchvision.models as models

from benchmarks.benchmark_utils import (
    calculate_statistics,
    print_results,
)


def benchmark_pytorch(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 20,
    benchmark_runs: int = 100,
) -> list[float]:
    model.eval()

    # Warm-up
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    latencies_ms: list[float] = []

    # Individual measurements
    for _ in range(benchmark_runs):
        if input_tensor.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            _ = model(input_tensor)

        if input_tensor.is_cuda:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000)

    return latencies_ms


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        torch.cuda.reset_peak_memory_stats()

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    latencies_ms = benchmark_pytorch(
        model=model,
        input_tensor=input_tensor,
    )

    result = calculate_statistics(latencies_ms)

    print_results(result)

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            f"Peak GPU memory:  {peak_memory_mb:.2f} MB"
        )


if __name__ == "__main__":
    main()