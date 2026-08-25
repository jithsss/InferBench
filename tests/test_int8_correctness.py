import numpy as np
import torch
import onnxruntime as ort


FP32_MODEL = "export/resnet50_fp32.onnx"
INT8_MODEL = "export/resnet50_int8.onnx"


def create_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        model_path,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )


def main() -> None:
    torch.manual_seed(42)

    # Use the same input for both models.
    input_data = torch.randn(
        1, 3, 224, 224
    ).numpy().astype(np.float32)

    # Importing torch before creating ORT sessions
    # allows the CUDA provider to use the compatible
    # CUDA/cuDNN libraries in this environment.
    fp32_session = create_session(FP32_MODEL)
    int8_session = create_session(INT8_MODEL)

    print("FP32 providers:", fp32_session.get_providers())
    print("INT8 providers:", int8_session.get_providers())

    fp32_input_name = fp32_session.get_inputs()[0].name
    fp32_output_name = fp32_session.get_outputs()[0].name

    int8_input_name = int8_session.get_inputs()[0].name
    int8_output_name = int8_session.get_outputs()[0].name

    fp32_output = fp32_session.run(
        [fp32_output_name],
        {fp32_input_name: input_data},
    )[0]

    int8_output = int8_session.run(
        [int8_output_name],
        {int8_input_name: input_data},
    )[0]

    difference = np.abs(
        fp32_output - int8_output
    )

    max_difference = float(np.max(difference))
    mean_difference = float(np.mean(difference))

    fp32_class = int(
        np.argmax(fp32_output, axis=1)[0]
    )

    int8_class = int(
        np.argmax(int8_output, axis=1)[0]
    )

    print("\n--- FP32 vs INT8 Correctness ---")
    print(
        f"Maximum absolute difference: "
        f"{max_difference:.8f}"
    )
    print(
        f"Mean absolute difference:    "
        f"{mean_difference:.8f}"
    )
    print(f"FP32 predicted class:        {fp32_class}")
    print(f"INT8 predicted class:        {int8_class}")
    print(
        f"Predicted class matches:     "
        f"{fp32_class == int8_class}"
    )


if __name__ == "__main__":
    main()