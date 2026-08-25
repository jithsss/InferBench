from pathlib import Path

import torch
import torchvision.models as models
import onnx


def main() -> None:
    output_path = Path("export/resnet50_fp32.onnx")

    # Load pretrained ResNet50
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Inference mode
    model.eval()

    # Example input
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Exporting ResNet50 to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX model saved to: {output_path}")

    # Validate the ONNX file
    print("Checking ONNX model...")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print("ONNX model validation: OK")


if __name__ == "__main__":
    main()