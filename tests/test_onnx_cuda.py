import torch
import onnxruntime as ort

# Importing torch first makes its CUDA/cuDNN DLLs available.
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

session = ort.InferenceSession(
    "export/resnet50_fp32.onnx",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

print("Available providers:", ort.get_available_providers())
print("Active providers:", session.get_providers())