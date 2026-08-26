import pytest
from benchmarks.result_schema import BenchmarkResult
from runtimes.registry import get_benchmark

def test_whisper_schema_fields():
    result = BenchmarkResult(
        model="whisper-tiny",
        model_type="speech",
        runtime="PyTorch",
        execution_provider="CUDA",
        precision="FP16",
        batch_size=1,
        audio_duration_seconds=5.0,
        real_time_factor=0.5,
        wer=10.0,
        cer=5.0
    )
    
    assert result.model_type == "speech"
    assert result.audio_duration_seconds == 5.0
    assert result.real_time_factor == 0.5
    assert result.wer == 10.0
    assert result.cer == 5.0

def test_whisper_lazy_registration():
    # Make sure we can retrieve them without importing whisper module logic directly
    import runtimes.benchmarks_registry  # noqa: F401
    
    benchmark = get_benchmark("whisper-tiny-baseline")
    assert benchmark.name == "whisper-tiny-baseline"
    assert benchmark.module == "benchmarks.whisper_benchmark"
    assert benchmark.function == "run_tiny_baseline"

    benchmark_fp16 = get_benchmark("whisper-tiny-fp16")
    assert benchmark_fp16.name == "whisper-tiny-fp16"
    assert benchmark_fp16.module == "benchmarks.whisper_benchmark"
    assert benchmark_fp16.function == "run_tiny_fp16"
