from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models


def main() -> None:
    onnx_path = Path("export/resnet50_fp32.onnx")

    # Load the same pretrained PyTorch model
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    # Use one fixed input for both runtimes
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 3, 224, 224)

    # PyTorch inference
    with torch.inference_mode():
        pytorch_output = model(input_tensor)

    pytorch_output = pytorch_output.cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["output"],
        {"input": input_tensor.numpy()},
    )[0]

    # Compare outputs
    max_difference = np.max(
        np.abs(pytorch_output - onnx_output)
    )

    mean_difference = np.mean(
        np.abs(pytorch_output - onnx_output)
    )

    print("--- ONNX Correctness Test ---")
    print(f"Maximum absolute difference: {max_difference:.8f}")
    print(f"Mean absolute difference:    {mean_difference:.8f}")

    # Numerical tolerance
    is_correct = np.allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-3,
        atol=1e-5,
    )

    print(f"Outputs match:               {is_correct}")

    if not is_correct:
        raise RuntimeError("PyTorch and ONNX outputs differ beyond tolerance.")

    print("ONNX correctness: OK")


if __name__ == "__main__":
    main()