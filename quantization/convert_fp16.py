from pathlib import Path

import onnx
from onnxconverter_common import float16


INPUT_MODEL = Path("export/resnet50_fp32.onnx")
OUTPUT_MODEL = Path("export/resnet50_fp16.onnx")


def main() -> None:
    print(f"Loading: {INPUT_MODEL}")

    model = onnx.load(INPUT_MODEL)

    print("Converting FP32 → FP16...")

    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
    )

    onnx.save(fp16_model, OUTPUT_MODEL)

    print(f"Saved FP16 model: {OUTPUT_MODEL}")

    # Validate the converted model.
    print("Validating FP16 model...")

    loaded_model = onnx.load(OUTPUT_MODEL)
    onnx.checker.check_model(loaded_model)

    print("FP16 model validation: OK")


if __name__ == "__main__":
    main()