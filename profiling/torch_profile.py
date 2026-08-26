import json
import time
from pathlib import Path
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from benchmarks.whisper_adapter import WhisperAdapter

PROFILE_DIR = Path("profiling")

def profile_whisper(model_size: str = "tiny") -> None:
    print(f"Profiling Whisper {model_size} (FP16) via PyTorch Profiler...")
    
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    adapter = WhisperAdapter(model_size, "FP16")
    input_features = adapter.load_and_preprocess()

    print("Running warmup...")
    for _ in range(2):
        adapter.evaluate(input_features)

    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("model_inference"):
            start = time.perf_counter()
            adapter.evaluate(input_features)
            torch.cuda.synchronize()
            inference_time = (time.perf_counter() - start) * 1000

    print("Analyzing trace...")
    events = prof.key_averages()
    
    cpu_events = 0
    cuda_events = 0
    memcpy_events = 0
    memcpy_total_us = 0.0

    for evt in events:
        name = evt.key.lower()
        if "to" in name or "copy" in name or "memcpy" in name:
            memcpy_events += evt.count
            memcpy_total_us += evt.cpu_time_total + evt.cuda_time_total
        elif evt.cuda_time_total > 0:
            cuda_events += evt.count
        else:
            cpu_events += evt.count

    summary = {
        "_file": f"torch_profile_whisper_{model_size}.json",
        "model": f"whisper-{model_size}",
        "memcpy_events": memcpy_events,
        "cpu_events": cpu_events,
        "cuda_events": cuda_events,
        "memcpy_total_us": memcpy_total_us,
        "inference_latency_ms": inference_time,
        "total_events": len(events),
        "total_event_time_us": sum(e.cpu_time_total + e.cuda_time_total for e in events),
        "active_providers": ["CUDAExecutionProvider"],
        "input_name": "input_features",
        "input_shape": str(list(input_features.shape)),
        "output_name": "transcription_tokens"
    }

    timestamp = int(time.time())
    out_path = PROFILE_DIR / f"torch_profile_whisper_{model_size}_{timestamp}.summary.json"
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Profile summary saved to: {out_path}")
    print(f"  CPU Events:    {cpu_events}")
    print(f"  CUDA Events:   {cuda_events}")
    print(f"  Memcpy Events: {memcpy_events} ({memcpy_total_us:.2f} us)")
    print(f"  Latency:       {inference_time:.2f} ms")
