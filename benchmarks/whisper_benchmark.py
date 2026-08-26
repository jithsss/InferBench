import time
import torch
import numpy as np

from benchmarks.whisper_adapter import WhisperAdapter
from benchmarks.result_schema import BenchmarkResult
from benchmarks.result_writer import save_benchmark_result


def run_whisper_benchmark(size: str, precision: str) -> None:
    print(f"Loading Whisper {size} ({precision}) on CUDA...")

    adapter = WhisperAdapter(
        model_name=size,
        precision=precision,
        device="cuda"
    )

    print(f"Model loaded in {adapter.load_time_ms:.1f} ms")

    input_features = adapter.load_and_preprocess()
    print(f"Audio preprocessed in {adapter.prep_time_ms:.1f} ms. Duration: {adapter.audio_duration:.2f} s")

    print("Running warmup...")
    for _ in range(3):
        adapter.evaluate(input_features)

    print("Running evaluation loop...")
    latencies = []
    
    transcription, wer, cer = "", 0.0, 0.0
    for i in range(10):
        t0 = time.time()
        transcription, wer, cer = adapter.evaluate(input_features)
        torch.cuda.synchronize()
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    rtf = (avg_latency / 1000.0) / adapter.audio_duration

    print(f"\n--- Whisper {size} {precision} Results ---")
    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"RTF             : {rtf:.3f}")
    print(f"WER             : {wer:.2f}%")
    print(f"CER             : {cer:.2f}%")
    print(f"Transcription   : {transcription}\n")

    result = BenchmarkResult(
        model=f"whisper-{size}",
        model_type="speech",
        runtime="PyTorch",
        execution_provider="CUDAExecutionProvider",
        precision=precision,
        batch_size=1,
        average_latency_ms=avg_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        task="speech_recognition",
        audio_duration_seconds=adapter.audio_duration,
        real_time_factor=rtf,
        wer=wer,
        cer=cer,
        language="en",
        preprocessing_latency_ms=adapter.prep_time_ms,
        model_load_time_ms=adapter.load_time_ms,
    )

    save_benchmark_result(result, f"results/speech/whisper_{size}_{precision.lower()}.json")


def run_tiny_baseline() -> None:
    run_whisper_benchmark("tiny", "FP32")

def run_tiny_fp16() -> None:
    run_whisper_benchmark("tiny", "FP16")

def run_base_baseline() -> None:
    run_whisper_benchmark("base", "FP32")

def run_base_fp16() -> None:
    run_whisper_benchmark("base", "FP16")

def run_small_baseline() -> None:
    run_whisper_benchmark("small", "FP32")

def run_small_fp16() -> None:
    run_whisper_benchmark("small", "FP16")
