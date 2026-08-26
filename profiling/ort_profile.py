import argparse
import json
import time
from pathlib import Path

import numpy as np

from runtimes.environment import configure_nvidia_runtime

# Configure CUDA/TensorRT DLL paths before importing ONNX Runtime.
configure_nvidia_runtime()

import onnxruntime as ort


DEFAULT_MODEL = "export/resnet50_int8.onnx"
DEFAULT_OUTPUT_DIR = Path("profiling")


def create_session(
    model_path: str,
) -> ort.InferenceSession:
    session_options = ort.SessionOptions()

    # Enable ONNX Runtime profiling.
    session_options.enable_profiling = True

    # Reduce log noise while keeping warnings/errors.
    session_options.log_severity_level = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    active_providers = session.get_providers()

    print(
        "Active providers:",
        active_providers,
    )

    # Never silently profile on CPU when this tool is intended
    # to profile CUDA execution.
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. "
            "Refusing to produce a GPU profile."
        )

    return session


def get_input_shape(
    input_meta,
) -> list[int]:
    shape: list[int] = []

    for dim in input_meta.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            # ResNet50 benchmark default.
            shape.append(1)

    if len(shape) != 4:
        raise RuntimeError(
            f"Expected a 4D image input, "
            f"but received shape {shape}"
        )

    return shape


def summarize_profile(
    profile_path: Path,
) -> dict:
    with profile_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        events = json.load(file)

    memcpy_events = []
    cpu_events = []
    cuda_events = []

    total_event_time_us = 0.0

    for event in events:
        name = str(
            event.get("name", "")
        )

        category = str(
            event.get("cat", "")
        )

        args = event.get(
            "args",
            {},
        )

        provider = str(
            args.get(
                "provider",
                "",
            )
        )

        duration = event.get(
            "dur",
            0,
        )

        if isinstance(
            duration,
            (int, float),
        ):
            total_event_time_us += float(
                duration
            )

        if "Memcpy" in name:
            memcpy_events.append(event)

        if "CPUExecutionProvider" in provider:
            cpu_events.append(event)

        if "CUDAExecutionProvider" in provider:
            cuda_events.append(event)

        # Some ORT profiles expose provider information
        # through categories instead.
        if "cpu" in category.lower():
            cpu_events.append(event)

        if "cuda" in category.lower():
            cuda_events.append(event)

    memcpy_total_us = sum(
        float(
            event.get(
                "dur",
                0,
            )
        )
        for event in memcpy_events
        if isinstance(
            event.get(
                "dur",
                0,
            ),
            (int, float),
        )
    )

    return {
        "total_events": len(events),
        "memcpy_events": len(memcpy_events),
        "memcpy_total_us": memcpy_total_us,
        "cpu_events": len(cpu_events),
        "cuda_events": len(cuda_events),
        "total_event_time_us": total_event_time_us,
    }


def print_summary(
    summary: dict,
) -> None:
    print("\n=== ONNX Runtime Profile ===")

    print(
        f"Total profile events: "
        f"{summary['total_events']}"
    )

    print(
        f"Memcpy events:        "
        f"{summary['memcpy_events']}"
    )

    print(
        f"Memcpy time:          "
        f"{summary['memcpy_total_us']:.2f} us"
    )

    print(
        f"CPU-related events:   "
        f"{summary['cpu_events']}"
    )

    print(
        f"CUDA-related events:  "
        f"{summary['cuda_events']}"
    )

    print(
        f"Total event time:     "
        f"{summary['total_event_time_us']:.2f} us"
    )

    if summary["memcpy_events"] > 0:
        print(
            "\n⚠ Host/device copy activity detected."
        )

    if summary["cpu_events"] > 0:
        print(
            "⚠ CPU execution activity detected."
        )

    if (
        summary["memcpy_events"] == 0
        and summary["cpu_events"] == 0
    ):
        print(
            "\n✓ No obvious CPU fallback or "
            "Memcpy activity detected."
        )


def profile_model(
    model_path: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Model: {model_path}"
    )

    session = create_session(
        model_path
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    output_name = output_meta.name

    input_shape = get_input_shape(
        input_meta
    )

    print(
        f"Input: {input_name}"
    )

    print(
        f"Input shape: {input_shape}"
    )

    print(
        f"Output: {output_name}"
    )

    rng = np.random.default_rng(42)

    input_data = rng.random(
        input_shape,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print("Running warm-up...")

    for _ in range(5):
        session.run(
            [output_name],
            {
                input_name: input_data,
            },
        )

    # ---------------------------------------------------------
    # Profiled inference
    # ---------------------------------------------------------

    print(
        "Running profiled inference..."
    )

    start = time.perf_counter()

    session.run(
        [output_name],
        {
            input_name: input_data,
        },
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    # Finalize profiling and get the generated JSON path.
    profile_path_str = (
        session.end_profiling()
    )

    source_profile = Path(
        profile_path_str
    )

    destination = (
        output_dir
        / source_profile.name
    )

    if (
        source_profile.resolve()
        != destination.resolve()
    ):
        destination.write_bytes(
            source_profile.read_bytes()
        )

    summary = summarize_profile(
        destination
    )

    print(
        f"\nProfiled inference latency: "
        f"{elapsed_ms:.3f} ms"
    )

    print_summary(summary)

    summary_path = (
        destination.with_suffix(
            ".summary.json"
        )
    )

    summary_data = {
        "model": model_path,
        "active_providers": session.get_providers(),
        "input_name": input_name,
        "input_shape": input_shape,
        "output_name": output_name,
        "inference_latency_ms": elapsed_ms,
        "profile": str(destination),
        **summary,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_data,
            file,
            indent=2,
        )

    print(
        f"\nProfile: {destination}"
    )

    print(
        f"Summary: {summary_path}"
    )

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile an ONNX Runtime CUDA model "
            "and detect CPU/copy activity."
        )
    )

    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=(
            "Path to ONNX model "
            f"(default: {DEFAULT_MODEL})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for profiling output",
    )

    args = parser.parse_args()

    try:
        profile_model(
            args.model,
            Path(args.output_dir),
        )
    except KeyboardInterrupt:
        print(
            "\nProfiling interrupted."
        )
        return 130
    except Exception as exc:
        print(
            f"\nProfiling failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )