from pathlib import Path

import tensorrt as trt

from quantization.tensorrt_int8_calibrator import (
    ResNetEntropyCalibrator,
)


ONNX_MODEL = Path(
    "export/resnet50_fp32_trt.onnx"
)

ENGINE_PATH = Path(
    "export/resnet50_tensorrt_int8.engine"
)

CALIBRATION_IMAGES = (
    "quantization/calibration/images"
)

CALIBRATION_CACHE = (
    "quantization/calibration/"
    "resnet50_int8.cache"
)


def main() -> None:
    logger = trt.Logger(trt.Logger.INFO)

    print("TensorRT version:", trt.__version__)

    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
    )

    parser = trt.OnnxParser(
        network,
        logger,
    )

    print(f"Loading ONNX model: {ONNX_MODEL}")

    with ONNX_MODEL.open("rb") as file:
        model_data = file.read()

    if not parser.parse(model_data):
        print("TensorRT ONNX parsing failed.")

        for index in range(parser.num_errors):
            print(parser.get_error(index))

        raise RuntimeError(
            "Unable to parse ONNX model."
        )

    print("ONNX parsing: OK")

    config = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        2 * 1024 * 1024 * 1024,
    )

    # Enable INT8.
    config.set_flag(trt.BuilderFlag.INT8)

    # ---------------------------------------------------------
    # Optimization profile for the dynamic batch dimension.
    #
    # We currently benchmark batch size 1, so optimize the
    # engine specifically around batch size 1.
    # ---------------------------------------------------------

    input_tensor = network.get_input(0)

    print(
        "Network input:",
        input_tensor.name,
    )

    print(
        "Network input shape:",
        input_tensor.shape,
    )

    profile = builder.create_optimization_profile()

    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 224, 224),
        opt=(1, 3, 224, 224),
        max=(1, 3, 224, 224),
    )

    config.add_optimization_profile(profile)

    print("Optimization profile:")
    print("  MIN:", (1, 3, 224, 224))
    print("  OPT:", (1, 3, 224, 224))
    print("  MAX:", (1, 3, 224, 224))

    # ---------------------------------------------------------
    # INT8 calibration
    # ---------------------------------------------------------

    calibrator = ResNetEntropyCalibrator(
        image_dir=CALIBRATION_IMAGES,
        input_shape=(1, 3, 224, 224),
        cache_file=CALIBRATION_CACHE,
        max_images=50,
    )

    config.int8_calibrator = calibrator

    print(
        "Building TensorRT INT8 engine..."
    )

    serialized_engine = (
        builder.build_serialized_network(
            network,
            config,
        )
    )

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT failed to build "
            "the INT8 engine."
        )

    ENGINE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ENGINE_PATH.write_bytes(
        serialized_engine
    )

    print(
        f"TensorRT INT8 engine saved to: "
        f"{ENGINE_PATH}"
    )


if __name__ == "__main__":
    main()