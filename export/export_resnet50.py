from pathlib import Path

import onnx
import torch
import torchvision.models as models


def main() -> None:
    output_path = Path("export/resnet50_fp32_trt.onnx")

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
    )

    print("Exporting ResNet50 for TensorRT...")

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
        external_data=False,
    )

    print(
        f"ONNX model saved to: {output_path}"
    )

    print("Checking ONNX model...")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print("ONNX model validation: OK")


if __name__ == "__main__":
    main()