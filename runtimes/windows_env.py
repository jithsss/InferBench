import os
from pathlib import Path


TENSORRT_ROOT = Path(
    r"B:\download\TensorRT-10.4.0.26"
)

CUDA_ROOT = Path(
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
)


def configure_nvidia_runtime() -> None:
    paths = [
        str(TENSORRT_ROOT / "lib"),
        str(TENSORRT_ROOT / "bin"),
        str(CUDA_ROOT / "bin"),
    ]

    current_path = os.environ.get("PATH", "")

    for path in reversed(paths):
        if path not in current_path:
            current_path = path + os.pathsep + current_path

    os.environ["PATH"] = current_path

    # Also help Python locate DLL directories explicitly.
    for path in paths:
        if Path(path).exists():
            try:
                os.add_dll_directory(path)
            except (AttributeError, OSError):
                pass