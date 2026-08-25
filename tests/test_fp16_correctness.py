import numpy as np
import torch
import onnxruntime as ort


FP32_MODEL = "export/resnet50_fp32.onnx"
FP16_MODEL = "export/resnet50_fp16.onnx"


def main() -> None:
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    np.random.seed(42)

    input_data = np.random.randn(
        1, 3, 224, 224
    ).astype(np.float32)

    # FP32 session
    fp32_session = ort.InferenceSession(
        FP32_MODEL,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # FP16 session
    fp16_session = ort.InferenceSession(
        FP16_MODEL,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input_name_fp32 = fp32_session.get_inputs()[0].name
    output_name_fp32 = fp32_session.get_outputs()[0].name

    input_name_fp16 = fp16_session.get_inputs()[0].name
    output_name_fp16 = fp16_session.get_outputs()[0].name

    print("FP32 providers:", fp32_session.get_providers())
    print("FP16 providers:", fp16_session.get_providers())

    # Run both models with the same input.
    fp32_output = fp32_session.run(
        [output_name_fp32],
        {input_name_fp32: input_data},
    )[0]

    fp16_output = fp16_session.run(
        [output_name_fp16],
        {input_name_fp16: input_data},
    )[0]

    # Compare outputs.
    difference = np.abs(
        fp32_output - fp16_output
    )

    max_difference = float(np.max(difference))
    mean_difference = float(np.mean(difference))

    # Compare predicted class.
    fp32_class = int(np.argmax(fp32_output, axis=1)[0])
    fp16_class = int(np.argmax(fp16_output, axis=1)[0])

    print("\n--- FP32 vs FP16 Correctness ---")
    print(
        f"Maximum absolute difference: "
        f"{max_difference:.8f}"
    )
    print(
        f"Mean absolute difference:    "
        f"{mean_difference:.8f}"
    )
    print(f"FP32 predicted class:        {fp32_class}")
    print(f"FP16 predicted class:        {fp16_class}")
    print(
        f"Predicted class matches:     "
        f"{fp32_class == fp16_class}"
    )


if __name__ == "__main__":
    main()