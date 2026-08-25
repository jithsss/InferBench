import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort


MODEL_PATH = "export/resnet50_int8.onnx"
PROFILE_DIR = Path("profiling")


def main() -> None:
    PROFILE_DIR.mkdir(exist_ok=True)

    # Importing torch before ONNX Runtime allows the CUDA/cuDNN
    # libraries bundled with PyTorch to be available to ONNX Runtime.
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    session_options = ort.SessionOptions()

    # Enable ONNX Runtime profiling.
    session_options.enable_profiling = True

    # Show useful ONNX Runtime initialization messages.
    session_options.log_severity_level = 1

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=session_options,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Model:", MODEL_PATH)
    print("Active providers:", session.get_providers())

    # Create deterministic test input.
    np.random.seed(42)

    input_data = np.random.randn(
        1,
        3,
        224,
        224,
    ).astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warm-up
    print("\nRunning warm-up...")

    for _ in range(20):
        session.run(
            [output_name],
            {input_name: input_data},
        )

    # Profile several inference runs.
    print("Running profiled inference...")

    for _ in range(20):
        session.run(
            [output_name],
            {input_name: input_data},
        )

    # Finish profiling.
    profile_path = session.end_profiling()

    print("\nProfile generated:")
    print(profile_path)

    profile_file = Path(profile_path)

    # Load profile JSON.
    with profile_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        profile = json.load(file)

    # ONNX Runtime normally produces a list of trace events.
    if isinstance(profile, list):
        events = profile
    else:
        events = profile.get(
            "traceEvents",
            [],
        )

    print(f"Total profile events: {len(events)}")

    # ------------------------------------------------------------------
    # Find all memcpy/copy-related events.
    # ------------------------------------------------------------------

    memcpy_events = []

    for event in events:
        name = str(
            event.get("name", "")
        )

        name_lower = name.lower()

        if (
            "memcpy" in name_lower
            or "copy" in name_lower
        ):
            memcpy_events.append(event)

    print(
        f"Memcpy/copy-related events: "
        f"{len(memcpy_events)}"
    )

    # ------------------------------------------------------------------
    # Count actual memcpy kernel events.
    #
    # A single transfer usually creates multiple trace events:
    #
    #   fence_before
    #   kernel_time
    #   fence_after
    #
    # We only count kernel_time here so we don't count
    # the fences as separate memory-copy operations.
    # ------------------------------------------------------------------

    memcpy_kernel_events = []

    for event in events:
        name = str(
            event.get("name", "")
        )

        if (
            "Memcpy" in name
            and "kernel_time" in name
        ):
            duration_us = float(
                event.get("dur", 0)
            )

            memcpy_kernel_events.append(
                {
                    "name": name,
                    "duration_us": duration_us,
                }
            )

    total_memcpy_time_us = sum(
        event["duration_us"]
        for event in memcpy_kernel_events
    )

    average_memcpy_time_us = (
        total_memcpy_time_us
        / len(memcpy_kernel_events)
        if memcpy_kernel_events
        else 0.0
    )

    # ------------------------------------------------------------------
    # Print memcpy analysis.
    # ------------------------------------------------------------------

    print("\n--- Memcpy Analysis ---")

    print(
        "Actual memcpy kernel events: "
        f"{len(memcpy_kernel_events)}"
    )

    print(
        "Total memcpy kernel time: "
        f"{total_memcpy_time_us:.2f} us"
    )

    print(
        "Average memcpy time: "
        f"{average_memcpy_time_us:.2f} us"
    )

    # ------------------------------------------------------------------
    # Display slowest memcpy operations.
    # ------------------------------------------------------------------

    print(
        "\n--- Slowest Memcpy Operations ---"
    )

    sorted_memcpy_events = sorted(
        memcpy_kernel_events,
        key=lambda item: item["duration_us"],
        reverse=True,
    )

    for event in sorted_memcpy_events[:20]:
        print(
            f"{event['duration_us']:.2f} us - "
            f"{event['name']}"
        )


if __name__ == "__main__":
    main()