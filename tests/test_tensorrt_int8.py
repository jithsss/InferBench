import numpy as np
import torch
import tensorrt as trt
from PIL import Image


FP32_ENGINE_PATH = "export/resnet50_tensorrt_int8.engine"
IMAGE_DIR = "quantization/calibration/images"


def preprocess(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image_array = (image_array - mean) / std

    image_array = np.transpose(
        image_array,
        (2, 0, 1),
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return np.ascontiguousarray(
        image_array,
        dtype=np.float32,
    )


def create_engine_context():
    logger = trt.Logger(trt.Logger.ERROR)

    runtime = trt.Runtime(logger)

    with open(FP32_ENGINE_PATH, "rb") as file:
        engine_data = file.read()

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:
        raise RuntimeError(
            "Failed to load TensorRT INT8 engine."
        )

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "Failed to create TensorRT execution context."
        )

    return engine, context


def main() -> None:
    print("TensorRT:", trt.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    engine, context = create_engine_context()

    input_name = None
    output_name = None

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError(
            "Unable to find engine input/output."
        )

    image_paths = sorted(
        list(
            __import__("pathlib")
            .Path(IMAGE_DIR)
            .glob("*.jpg")
        )
    )[:50]

    if not image_paths:
        raise RuntimeError(
            f"No JPG images found in {IMAGE_DIR}"
        )

    predictions = []

    print(
        f"Evaluating {len(image_paths)} images..."
    )

    for image_path in image_paths:
        input_data = preprocess(
            str(image_path)
        )

        input_tensor = torch.from_numpy(
            input_data
        ).cuda()

        context.set_input_shape(
            input_name,
            tuple(input_tensor.shape),
        )

        output_shape = context.get_tensor_shape(
            output_name
        )

        output_tensor = torch.empty(
            tuple(output_shape),
            dtype=torch.float32,
            device="cuda",
        )

        context.set_tensor_address(
            input_name,
            input_tensor.data_ptr(),
        )

        context.set_tensor_address(
            output_name,
            output_tensor.data_ptr(),
        )

        stream = (
            torch.cuda.current_stream().cuda_stream
        )

        context.execute_async_v3(stream)

        torch.cuda.synchronize()

        output = (
            output_tensor
            .detach()
            .cpu()
            .numpy()
        )

        predicted_class = int(
            np.argmax(output, axis=1)[0]
        )

        predictions.append(
            {
                "image": image_path.name,
                "class": predicted_class,
                "output": output,
            }
        )

    print("\n--- TensorRT INT8 Predictions ---")

    for prediction in predictions[:10]:
        print(
            f"{prediction['image']}: "
            f"class {prediction['class']}"
        )

    print(
        f"\nTotal evaluated images: "
        f"{len(predictions)}"
    )


if __name__ == "__main__":
    main()