import importlib
import json
import os
from pathlib import Path


CONFIG_PATH = Path("configs/runtime.json")


def load_runtime_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_runtime_paths() -> tuple[Path | None, Path | None]:
    config = load_runtime_config()

    nvidia = config.get("nvidia", {})

    cuda_path = (
        nvidia.get("cuda_path")
        or os.environ.get("INFERBENCH_CUDA_PATH")
    )

    tensorrt_path = (
        nvidia.get("tensorrt_path")
        or os.environ.get("INFERBENCH_TENSORRT_PATH")
    )

    cuda_root = (
        Path(cuda_path)
        if cuda_path
        else None
    )

    tensorrt_root = (
        Path(tensorrt_path)
        if tensorrt_path
        else None
    )

    return cuda_root, tensorrt_root


def configure_nvidia_runtime() -> None:
    cuda_root, tensorrt_root = (
        get_runtime_paths()
    )

    paths: list[Path] = []

    if cuda_root:
        paths.append(
            cuda_root / "bin"
        )

    if tensorrt_root:
        paths.append(
            tensorrt_root / "lib"
        )

        paths.append(
            tensorrt_root / "bin"
        )

    current_path = os.environ.get(
        "PATH",
        "",
    )

    for path in reversed(paths):
        path_str = str(path)

        if path_str not in current_path:
            current_path = (
                path_str
                + os.pathsep
                + current_path
            )

        if path.exists():
            try:
                os.add_dll_directory(
                    path_str
                )
            except (
                AttributeError,
                OSError,
            ):
                pass

    os.environ["PATH"] = current_path

    if cuda_root:
        os.environ["CUDA_PATH"] = str(
            cuda_root
        )


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def check_cuda() -> bool:
    try:
        configure_nvidia_runtime()

        import torch

        return bool(
            torch.cuda.is_available()
        )
    except Exception:
        return False


def check_onnxruntime() -> bool:
    try:
        configure_nvidia_runtime()

        import onnxruntime as ort

        return (
            "CUDAExecutionProvider"
            in ort.get_available_providers()
        )
    except Exception:
        return False


def check_tensor_rt() -> bool:
    try:
        configure_nvidia_runtime()

        import tensorrt as trt

        return trt.__version__.startswith(
            "10."
        )
    except Exception:
        return False


def check_genai() -> bool:
    try:
        import onnxruntime_genai

        return True
    except Exception:
        return False


def environment_report(
    benchmark_name: str,
) -> dict[str, bool]:

    if benchmark_name.startswith("resnet50-tensorrt") or benchmark_name.startswith("yolov8n-tensorrt"):
        return {
            "cuda": check_cuda(),
            "onnxruntime": check_onnxruntime(),
            "tensorrt": check_tensor_rt(),
        }

    if benchmark_name == "qwen3-0.6b":
        return {
            "cuda": check_cuda(),
            "onnxruntime": check_onnxruntime(),
            "onnxruntime_genai": check_genai(),
        }

    return {
        "cuda": check_cuda(),
        "onnxruntime": check_onnxruntime(),
    }


def print_environment_report(
    benchmark_name: str,
) -> bool:

    report = environment_report(
        benchmark_name
    )

    print(
        f"\nEnvironment check: "
        f"{benchmark_name}"
    )

    for name, available in report.items():
        status = (
            "OK"
            if available
            else "MISSING"
        )

        print(
            f"  {name:<22} {status}"
        )

    return all(report.values())