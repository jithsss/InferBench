from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import tensorrt as trt


ENGINE_PATH = Path(
    "export/resnet50_tensorrt_int8_40cal.engine"
)

IMAGE_DIR = Path(
    "quantization/calibration/evaluation"
)


def preprocess(image_path: Path) -> torch.Tensor:
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

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1),
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return torch.from_numpy(
        np.ascontiguousarray(
            image_array,
            dtype=np.float32,
        )
    )


def load_tensorrt_engine():
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)

    engine_data = ENGINE_PATH.read_bytes()

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
            "Failed to create TensorRT context."
        )

    input_name = None
    output_name = None

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError(
            "Could not find TensorRT input/output."
        )

    return context, input_name, output_name


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Original FP32 ResNet50.
    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(
        weights=weights
    ).to(device)

    model.eval()

    # TensorRT INT8 engine.
    (
        trt_context,
        trt_input_name,
        trt_output_name,
    ) = load_tensorrt_engine()

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )[:50]

    if not image_paths:
        raise RuntimeError(
            f"No JPG images found in {IMAGE_DIR}"
        )

    agreements = 0

    print(
        f"Comparing {len(image_paths)} images..."
    )

    for image_path in image_paths:

        input_tensor = preprocess(
            image_path
        ).to(device)

        # --------------------------------------------------
        # PyTorch FP32
        # --------------------------------------------------

        with torch.inference_mode():
            pytorch_output = model(
                input_tensor
            )

        pytorch_class = int(
            torch.argmax(
                pytorch_output,
                dim=1,
            ).item()
        )

        # --------------------------------------------------
        # TensorRT INT8
        # --------------------------------------------------

        trt_input = input_tensor.contiguous()

        trt_context.set_input_shape(
            trt_input_name,
            tuple(trt_input.shape),
        )

        output_shape = (
            trt_context.get_tensor_shape(
                trt_output_name
            )
        )

        trt_output = torch.empty(
            tuple(output_shape),
            dtype=torch.float32,
            device="cuda",
        )

        trt_context.set_tensor_address(
            trt_input_name,
            trt_input.data_ptr(),
        )

        trt_context.set_tensor_address(
            trt_output_name,
            trt_output.data_ptr(),
        )

        stream = (
            torch.cuda.current_stream().cuda_stream
        )

        trt_context.execute_async_v3(
            stream
        )

        torch.cuda.synchronize()

        trt_class = int(
            torch.argmax(
                trt_output,
                dim=1,
            ).item()
        )

        matches = (
            pytorch_class == trt_class
        )

        if matches:
            agreements += 1

        print(
            f"{image_path.name}: "
            f"FP32={pytorch_class}, "
            f"INT8={trt_class}, "
            f"match={matches}"
        )

    agreement_rate = (
        agreements / len(image_paths)
    ) * 100

    print("\n--- Prediction Agreement ---")
    print(
        f"Images:             {len(image_paths)}"
    )
    print(
        f"Matching predictions: {agreements}"
    )
    print(
        f"Agreement rate:     {agreement_rate:.2f}%"
    )


if __name__ == "__main__":
    main()