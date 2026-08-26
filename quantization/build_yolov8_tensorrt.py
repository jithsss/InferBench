import argparse
from pathlib import Path
import tensorrt as trt
from quantization.tensorrt_yolov8_calibrator import YoloEntropyCalibrator

ONNX_MODEL = Path("export/yolov8n.onnx")
CALIBRATION_IMAGES = "quantization/calibration/calibration_set"
CALIBRATION_CACHE = "quantization/calibration/yolov8n_int8_40cal.cache"

def build_engine(precision: str) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Loading ONNX model: {ONNX_MODEL}")
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")

    with ONNX_MODEL.open("rb") as file:
        if not parser.parse(file.read()):
            for index in range(parser.num_errors):
                print(parser.get_error(index))
            raise RuntimeError("Unable to parse ONNX model.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    # Usually YOLOv8 uses (-1, 3, 640, 640) or (1, 3, 640, 640)
    opt_shape = (1, 3, 640, 640)
    if input_shape[2] != -1 and input_shape[3] != -1:
        opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, min=opt_shape, opt=opt_shape, max=opt_shape)
    config.add_optimization_profile(profile)

    engine_path = Path(f"export/yolov8n_tensorrt_{precision}.engine")

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16) # Fallback if layer not supported in INT8
        
        calibrator = YoloEntropyCalibrator(
            image_dir=CALIBRATION_IMAGES,
            input_shape=opt_shape,
            cache_file=CALIBRATION_CACHE,
            max_images=40,
        )
        config.int8_calibrator = calibrator

    print(f"Building TensorRT {precision.upper()} engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError(f"TensorRT failed to build the {precision.upper()} engine.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    print(f"TensorRT {precision.upper()} engine saved to: {engine_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    args = parser.parse_args()
    build_engine(args.precision)
