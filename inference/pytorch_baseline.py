import time

import torch
import torchvision.models as models

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load pretrained ResNet50
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    # Create a dummy input
    input_tensor = torch.randn(1, 3, 224, 224, device=device)

    # Warm-up runs
    with torch.inference_mode():
        for _ in range(20):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    average_latency_ms = (elapsed / iterations) * 1000
    throughput_fps = iterations / elapsed

    print("\n--- PyTorch Baseline ---")
    print(f"Iterations:       {iterations}")
    print(f"Average latency:  {average_latency_ms:.3f} ms")
    print(f"Throughput:       {throughput_fps:.2f} FPS")


if __name__ == "__main__":
    main()
