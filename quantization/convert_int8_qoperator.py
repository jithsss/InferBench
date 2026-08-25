from pathlib import Path

import onnx
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from quantization.calibration.data_reader import (
    ResNetCalibrationDataReader,
)


INPUT_MODEL = Path("export/resnet50_fp32.onnx")
OUTPUT_MODEL = Path("export/resnet50_int8_qoperator.onnx")
IMAGE_DIR = "quantization/calibration/images"


def main() -> None:
    model = onnx.load(INPUT_MODEL)

    input_name = model.graph.input[0].name

    print("Model input:", input_name)
    print("Starting INT8 QOperator quantization...")

    calibration_reader = ResNetCalibrationDataReader(
        image_dir=IMAGE_DIR,
        input_name=input_name,
        max_images=50,
    )

    quantize_static(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    print(f"Saved INT8 model: {OUTPUT_MODEL}")

    print("Validating INT8 model...")

    quantized_model = onnx.load(OUTPUT_MODEL)
    onnx.checker.check_model(quantized_model)

    print("INT8 QOperator model validation: OK")


if __name__ == "__main__":
    main()