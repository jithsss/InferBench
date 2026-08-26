import argparse
from pathlib import Path

from benchmarks import compare
from profiling import ort_profile
from runtimes.environment import (
    print_environment_report,
)
from runtimes.registry import (
    get_benchmark,
    list_benchmarks,
)

# Import registration metadata only.
# Actual benchmark modules are loaded lazily.
import runtimes.benchmarks_registry  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferbench",
        description=(
            "InferBench AI inference "
            "benchmarking framework"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ---------------------------------------------------------
    # list
    # ---------------------------------------------------------

    subparsers.add_parser(
        "list",
        help="List available benchmarks",
    )

    # ---------------------------------------------------------
    # compare
    # ---------------------------------------------------------

    subparsers.add_parser(
        "compare",
        help="Compare saved benchmark results",
    )

    # ---------------------------------------------------------
    # run
    # ---------------------------------------------------------

    run_parser = subparsers.add_parser(
        "run",
        help="Run a registered benchmark",
    )

    run_parser.add_argument(
        "benchmark",
        help="Name of the benchmark to run",
    )

    # ---------------------------------------------------------
    # profile-onnx
    # ---------------------------------------------------------

    profile_parser = subparsers.add_parser(
        "profile-onnx",
        help="Profile an ONNX Runtime CUDA model",
    )

    profile_parser.add_argument(
        "model",
        nargs="?",
        default="export/resnet50_int8.onnx",
        help=(
            "Path to ONNX model "
            "(default: export/resnet50_int8.onnx)"
        ),
    )

    profile_parser.add_argument(
        "--output-dir",
        default="profiling",
        help="Directory for profiling output",
    )

    return parser


def handle_list() -> int:
    benchmarks = list_benchmarks()

    if not benchmarks:
        print(
            "No benchmarks registered."
        )
        return 0

    print(
        "Available benchmarks:\n"
    )

    for benchmark in benchmarks:
        print(
            f"{benchmark.name:<32} "
            f"{benchmark.description}"
        )

    return 0


def handle_compare() -> int:
    compare.main()
    return 0


def handle_run(
    benchmark_name: str,
) -> int:
    try:
        benchmark = get_benchmark(
            benchmark_name
        )
    except KeyError as exc:
        print(
            f"Error: {exc}"
        )
        return 1

    environment_ok = (
        print_environment_report(
            benchmark_name
        )
    )

    if not environment_ok:
        print(
            "\nEnvironment requirements are "
            "not satisfied."
        )

        print(
            "Benchmark was not started."
        )

        return 1

    print(
        f"\nRunning benchmark: "
        f"{benchmark.name}"
    )

    try:
        benchmark.run()

    except KeyboardInterrupt:
        print(
            "\nBenchmark interrupted."
        )
        return 130

    except Exception as exc:
        print(
            "\nBenchmark failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    return 0


def handle_profile_onnx(
    model: str,
    output_dir: str,
) -> int:
    print(
        f"Profiling ONNX model: {model}"
    )

    try:
        ort_profile.profile_model(
            model,
            Path(output_dir),
        )

    except KeyboardInterrupt:
        print(
            "\nProfiling interrupted."
        )
        return 130

    except Exception as exc:
        print(
            "\nProfiling failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    return 0


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "list":
        return handle_list()

    if args.command == "compare":
        return handle_compare()

    if args.command == "run":
        return handle_run(
            args.benchmark
        )

    if args.command == "profile-onnx":
        return handle_profile_onnx(
            args.model,
            args.output_dir,
        )

    parser.error(
        f"Unknown command: {args.command}"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )