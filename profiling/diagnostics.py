import json
from pathlib import Path


PROFILE_DIR = Path("profiling")


def load_latest_summary() -> dict | None:
    summaries = sorted(
        PROFILE_DIR.glob("*.summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not summaries:
        return None

    with summaries[0].open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def diagnose(summary: dict) -> list[str]:
    findings = []

    memcpy_events = summary.get(
        "memcpy_events",
        0,
    )

    memcpy_time_us = summary.get(
        "memcpy_total_us",
        0,
    )

    cpu_events = summary.get(
        "cpu_events",
        0,
    )

    cuda_events = summary.get(
        "cuda_events",
        0,
    )

    if memcpy_events > 0:
        findings.append(
            "Host/device copy activity detected."
        )

    if cpu_events > 0:
        findings.append(
            "CPU execution activity detected."
        )

    if cuda_events > 0:
        findings.append(
            "CUDA execution is active."
        )

    if memcpy_time_us > 5000:
        findings.append(
            "Memcpy overhead exceeds 5 ms "
            "in the profiled inference."
        )

    return findings