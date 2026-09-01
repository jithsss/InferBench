import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import streamlit as st
from runtimes.environment import configure_nvidia_runtime

configure_nvidia_runtime()

from benchmarks.result_writer import load_history
from profiling.diagnostics import diagnose, load_latest_summary

RESULTS_DIR = PROJECT_ROOT / "results"
PROFILING_DIR = PROJECT_ROOT / "profiling"

st.set_page_config(
    page_title="InferBench Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_json_file(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None

@st.cache_data(ttl=5)
def load_results() -> list[dict]:
    results = []
    if not RESULTS_DIR.exists(): return results
    for path in RESULTS_DIR.rglob("*.json"):
        if path.name == "test_result.json": continue
        data = load_json_file(path)
        if data is None or "model" not in data: continue
        data["_file"] = str(path.relative_to(PROJECT_ROOT))
        results.append(data)
    return results

@st.cache_data(ttl=5)
def load_profile_summaries() -> list[dict]:
    summaries = []
    if not PROFILING_DIR.exists(): return summaries
    for path in PROFILING_DIR.glob("*.summary.json"):
        data = load_json_file(path)
        if data is None: continue
        data["_file"] = str(path.relative_to(PROJECT_ROOT))
        summaries.append(data)
    summaries.sort(
        key=lambda item: Path(item["_file"]).stat().st_mtime if Path(PROJECT_ROOT / item["_file"]).exists() else 0,
        reverse=True,
    )
    return summaries

def clear_cache() -> None:
    load_results.clear()
    load_profile_summaries.clear()

results = load_results()
if not results:
    st.title("⚡ InferBench Pro")
    st.info("No benchmark results found in results/.")
    st.stop()

df = pd.DataFrame(results)
expected_columns = [
    "model", "model_type", "runtime", "execution_provider", "precision", 
    "batch_size", "average_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", 
    "throughput", "throughput_unit", "ttft_ms", "tokens_per_second", "peak_memory_mb", 
    "accuracy_metric", "accuracy_value", "task", "input_resolution", "prediction_agreement", "notes"
]
for col in expected_columns:
    if col not in df.columns:
        df[col] = None

# Sidebar
st.sidebar.title("⚡ InferBench")
st.sidebar.caption("Inference Performance Lab")
st.sidebar.divider()

st.sidebar.subheader("Filters")
models = sorted(df["model"].dropna().astype(str).unique())
runtimes = sorted(df["runtime"].dropna().astype(str).unique())
precisions = sorted(df["precision"].dropna().astype(str).unique())

model_filter = st.sidebar.selectbox("Model", ["All", *models])
runtime_filter = st.sidebar.selectbox("Runtime", ["All", *runtimes])
precision_filter = st.sidebar.selectbox("Precision", ["All", *precisions])

st.sidebar.divider()
if st.sidebar.button("↻ Refresh Data", use_container_width=True):
    clear_cache()
    st.rerun()

st.sidebar.caption("Data loaded from `results/` and `profiling/` directories.")

# Apply filters
filtered_df = df.copy()
if model_filter != "All": filtered_df = filtered_df[filtered_df["model"] == model_filter]
if runtime_filter != "All": filtered_df = filtered_df[filtered_df["runtime"] == runtime_filter]
if precision_filter != "All": filtered_df = filtered_df[filtered_df["precision"] == precision_filter]

# Hero
st.title("Inference Performance Lab")
st.markdown("Benchmark, optimize, profile, and compare AI inference workloads across architectures.")

# Tabs
tab_overview, tab_classification, tab_detection, tab_speech, tab_llm, tab_diagnostics, tab_upload = st.tabs([
    "📊 Overview", 
    "🖼️ Classification", 
    "🔍 Object Detection", 
    "🎙️ Speech",
    "💬 LLM", 
    "🛠️ Diagnostics",
    "📤 Custom Upload"
])

with tab_overview:
    st.subheader("System Overview")
    
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("Total Benchmarks", len(filtered_df))
    cc2.metric("Unique Models", filtered_df["model"].nunique())
    cc3.metric("Vision Runs", len(filtered_df[filtered_df["model_type"] == "vision"]))
    cc4.metric("LLM Runs", len(filtered_df[filtered_df["model_type"] == "llm"]))
    cc5.metric("Speech Runs", len(filtered_df[filtered_df["model_type"] == "speech"]))

    st.subheader("🏆 Top Performance")
    
    c_df = filtered_df[(filtered_df["model_type"] == "vision") & (filtered_df["task"] != "object_detection")].dropna(subset=["throughput"])
    d_df = filtered_df[(filtered_df["model_type"] == "vision") & (filtered_df["task"] == "object_detection")].dropna(subset=["throughput"])
    
    l_df = filtered_df[filtered_df["model_type"] == "llm"].dropna(subset=["tokens_per_second"])
    
    s_df = filtered_df[filtered_df["model_type"] == "speech"].dropna(subset=["real_time_factor"])

    active_sections = []
    
    if not c_df.empty:
        active_sections.append(("Classification", c_df, "throughput", "Fastest Classification", " FPS", True))
    if not d_df.empty:
        active_sections.append(("Detection", d_df, "throughput", "Fastest Object Detection", " FPS", True))
    if not l_df.empty:
        active_sections.append(("LLM", l_df, "tokens_per_second", "Fastest LLM Generation", " tok/s", True))
    if not s_df.empty:
        active_sections.append(("Speech", s_df, "real_time_factor", "Fastest Speech (RTF)", " RTF", False))

    if active_sections:
        cols = st.columns(len(active_sections))
        for col, (sec_name, sec_df, sec_metric, sec_title, sec_unit, higher_is_better) in zip(cols, active_sections):
            with col:
                if higher_is_better:
                    best = sec_df.loc[sec_df[sec_metric].idxmax()]
                else:
                    best = sec_df.loc[sec_df[sec_metric].idxmin()]
                st.metric(sec_title, f'{best[sec_metric]:.3f}{sec_unit}', f'{best["model"]} ({best["precision"]})')
    else:
        st.info("No performance data available for the current filters.")

    history = load_history()
    if history:
        st.subheader("⏱️ Recent Runs")
        recent_df = pd.DataFrame(history).tail(8).copy()
        
        display_cols = {
            "timestamp": "Time", 
            "model": "Model", 
            "model_type": "Type",
            "runtime": "Runtime", 
            "precision": "Precision", 
            "average_latency_ms": "Latency (ms)",
            "throughput": "FPS", 
            "tokens_per_second": "Tok/s", 
            "real_time_factor": "RTF"
        }
        
        cols_to_keep = [c for c in display_cols.keys() if c in recent_df.columns]
        recent_df = recent_df[cols_to_keep].rename(columns=display_cols)
        
        st.dataframe(
            recent_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "FPS": st.column_config.NumberColumn(format="%.1f"),
                "Tok/s": st.column_config.NumberColumn(format="%.1f"),
                "RTF": st.column_config.NumberColumn(format="%.3f"),
                "Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
                "Time": st.column_config.DatetimeColumn(format="h:mm a - MMM D, YYYY")
            }
        )

def render_vision_tab(df_var, title, is_detection):
    if df_var.empty:
        st.info(f"No {title.lower()} results match the current filters.")
        return
        
    st.subheader(f"⚡ {title} Optimization")
    measurable = df_var.dropna(subset=["throughput"])
    best = measurable.loc[measurable["throughput"].idxmax()] if not measurable.empty else None
    
    baseline_candidates = df_var[df_var["precision"] == "FP32"].dropna(subset=["throughput"])
    baseline = baseline_candidates.loc[baseline_candidates["throughput"].idxmax()] if not baseline_candidates.empty else None
    
    best_latency = df_var.dropna(subset=["average_latency_ms"])
    best_latency_row = best_latency.loc[best_latency["average_latency_ms"].idxmin()] if not best_latency.empty else None

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Peak Throughput", f'{best["throughput"]:.1f} FPS' if best is not None else "—", best["precision"] if best is not None else None)
    with mc2:
        st.metric("Lowest Latency", f'{best_latency_row["average_latency_ms"]:.2f} ms' if best_latency_row is not None else "—", best_latency_row["precision"] if best_latency_row is not None else None, delta_color="inverse")
    
    speedup = float(best["throughput"]) / float(baseline["throughput"]) if (best is not None and baseline is not None and float(baseline["throughput"]) != 0) else None
    with mc3:
        st.metric("Speedup vs FP32", f"{speedup:.2f}×" if speedup else "—")
        
    lat_red = (1 - (float(best["average_latency_ms"]) / float(baseline["average_latency_ms"]))) * 100 if (best is not None and baseline is not None and float(baseline["average_latency_ms"]) != 0) else None
    with mc4:
        st.metric("Latency Reduction", f"-{lat_red:.1f}%" if lat_red else "—")

    st.subheader("📊 Configuration Comparison")
    
    if is_detection:
        table = df_var[["model", "runtime", "precision", "input_resolution", "average_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput", "prediction_agreement"]].copy()
        table.columns = ["Model", "Runtime", "Precision", "Resolution", "Avg Latency (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Throughput (FPS)", "Agreement (%)"]
        
        st.dataframe(
            table, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Avg Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
                "P50 (ms)": st.column_config.NumberColumn(format="%.2f"),
                "P95 (ms)": st.column_config.NumberColumn(format="%.2f"),
                "P99 (ms)": st.column_config.NumberColumn(format="%.2f"),
                "Throughput (FPS)": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(table["Throughput (FPS)"].max())),
                "Agreement (%)": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
    else:
        table = df_var[["model", "runtime", "precision", "average_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput", "accuracy_metric", "accuracy_value"]].copy()
        table.columns = ["Model", "Runtime", "Precision", "Avg Latency (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Throughput (FPS)", "Quality Metric", "Quality (%)"]
        
        st.dataframe(
            table, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Avg Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
                "P50 (ms)": st.column_config.NumberColumn(format="%.2f"),
                "P95 (ms)": st.column_config.NumberColumn(format="%.2f"),
                "P99 (ms)": st.column_config.NumberColumn(format="%.2f"),
                "Throughput (FPS)": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(table["Throughput (FPS)"].max())),
            }
        )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Throughput by Precision (Higher is better)**")
        chart_df = df_var.groupby("precision", as_index=True)["throughput"].max()
        if not chart_df.empty: st.bar_chart(chart_df, color="#3b82f6")
    with cc2:
        st.markdown("**Latency by Precision (Lower is better)**")
        lat_df = df_var.groupby("precision", as_index=True)["average_latency_ms"].min()
        if not lat_df.empty: st.bar_chart(lat_df, color="#ef4444")

with tab_classification:
    render_vision_tab(filtered_df[(filtered_df["model_type"] == "vision") & (filtered_df["task"] != "object_detection")].copy(), "Image Classification", False)

with tab_detection:
    render_vision_tab(filtered_df[(filtered_df["model_type"] == "vision") & (filtered_df["task"] == "object_detection")].copy(), "Object Detection", True)

with tab_llm:
    llm_df = filtered_df[filtered_df["model_type"] == "llm"].copy()
    if llm_df.empty:
        st.info("No LLM results match the current filters.")
    else:
        st.subheader("💬 LLM Inference")
        measurable = llm_df.dropna(subset=["tokens_per_second"])
        best = measurable.loc[measurable["tokens_per_second"].idxmax()] if not measurable.empty else None
        
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric("Max Generation", f'{best["tokens_per_second"]:.1f} tok/s' if best is not None else "—", best["precision"] if best is not None else None)
        with lc2:
            st.metric("Fastest TTFT", f'{best["ttft_ms"]:.1f} ms' if best is not None else "—")
        with lc3:
            st.metric("Total Latency", f'{best["average_latency_ms"]:.1f} ms' if best is not None else "—")
        with lc4:
            st.metric("Active Format", str(best["precision"]) if best is not None else "—")

        st.subheader("📊 Configuration Comparison")
        table = llm_df[["model", "runtime", "precision", "ttft_ms", "tokens_per_second", "average_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]].copy()
        table.columns = ["Model", "Runtime", "Precision", "TTFT (ms)", "Tokens/sec", "Total Latency (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)"]
        
        st.dataframe(
            table, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "TTFT (ms)": st.column_config.NumberColumn(format="%.2f"),
                "Total Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
                "Tokens/sec": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(table["Tokens/sec"].max())),
            }
        )

        st.markdown("**Throughput by Precision**")
        throughput_df = llm_df.groupby("precision", as_index=True)["tokens_per_second"].max()
        if not throughput_df.empty: st.bar_chart(throughput_df, color="#10b981")

with tab_speech:
    speech_df = filtered_df[filtered_df["model_type"] == "speech"].copy()
    if speech_df.empty:
        st.info("No Speech recognition results match the current filters.")
    else:
        st.subheader("🎙️ Speech Recognition")
        measurable = speech_df.dropna(subset=["real_time_factor"])
        best = measurable.loc[measurable["real_time_factor"].idxmin()] if not measurable.empty else None
        
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric("Fastest RTF", f'{best["real_time_factor"]:.3f}' if best is not None else "N/A", best["precision"] if best is not None else None, delta_color="inverse")
        with lc2:
            st.metric("Audio Duration", f'{best["audio_duration_seconds"]:.1f} s' if best is not None else "N/A")
        with lc3:
            st.metric("Best WER", f'{best["wer"]:.2f}%' if best is not None else "N/A", delta_color="inverse")
        with lc4:
            st.metric("Total Latency", f'{best["average_latency_ms"]:.1f} ms' if best is not None else "N/A", delta_color="inverse")

        st.subheader("🔍 Configuration Comparison")
        table = speech_df[["model", "runtime", "precision", "average_latency_ms", "real_time_factor", "wer", "cer", "audio_duration_seconds"]].copy()
        table.columns = ["Model", "Runtime", "Precision", "Latency (ms)", "RTF", "WER (%)", "CER (%)", "Audio Duration (s)"]
        
        st.dataframe(
            table, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
                "RTF": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=float(table["RTF"].max())),
                "WER (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "CER (%)": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )

        st.markdown("**Real Time Factor by Precision (Lower is better)**")
        rtf_df = speech_df.groupby("precision", as_index=True)["real_time_factor"].min()
        if not rtf_df.empty: st.bar_chart(rtf_df, color="#8b5cf6")

with tab_diagnostics:
    st.subheader("🛠️ Profiling Diagnostics")
    summaries = load_profile_summaries()
    
    if not summaries:
        st.info("No profiling summaries found. Run `python -m inferbench profile-onnx <model.onnx>` to generate them.")
    else:
        for idx, summary in enumerate(summaries):
            file_name = Path(summary.get('_file', '')).stem.replace('.summary', '')
            with st.expander(f"Profile: {file_name}", expanded=(idx == 0)):
                
                # Header info
                st.markdown(f"**Model:** `{summary.get('model', 'Unknown')}`")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", f"{summary.get('inference_latency_ms', 0):.2f} ms")
                col2.metric("Total Events", f"{summary.get('total_events', 0)}")
                col3.metric("Total Event Time", f"{summary.get('total_event_time_us', 0) / 1000:.2f} ms")
                
                st.markdown("### Execution Providers")
                
                # Check for CPU Fallback or Memcpy issues
                memcpy = summary.get('memcpy_events', 0)
                cpu_events = summary.get('cpu_events', 0)
                cuda_events = summary.get('cuda_events', 0)
                
                if memcpy == 0 and cpu_events == 0:
                    st.success("✓ Graph executing natively on GPU without CPU fallback or memory transfers.")
                else:
                    if memcpy > 0:
                        st.warning(f"⚠️ Detected {memcpy} Memcpy events. The model is transferring memory between CPU and GPU mid-inference, which hurts performance.")
                    if cpu_events > 0:
                        st.error(f"❌ Detected {cpu_events} CPU events! Layers are falling back to the CPU.")

                c1, c2, c3 = st.columns(3)
                c1.metric("CUDA Events", cuda_events)
                c2.metric("CPU Events", cpu_events, delta_color="inverse")
                c3.metric("Memcpy Events", memcpy, delta_color="inverse")

                st.markdown("### Graph Details")
                providers = ", ".join(summary.get('active_providers', []))
                st.text(f"Providers: {providers}")
                st.text(f"Input: {summary.get('input_name')} {summary.get('input_shape')}")
                st.text(f"Output: {summary.get('output_name')}")

with tab_upload:
    st.subheader('Dynamic Model Benchmark')
    import os
    import tempfile
    import zipfile
    import shutil
    from benchmarks.dynamic_benchmark import DynamicBenchmark
    from benchmarks.dynamic_llm_benchmark import DynamicLLMBenchmark

    workload_type = st.radio("Workload Type", ["Standard (Vision/Audio)", "Large Language Model (LLM)"], horizontal=True)

    if workload_type == "Standard (Vision/Audio)":
        model_file = st.file_uploader('Upload Model (.onnx, .engine)', type=['onnx', 'engine', 'plan'])
        
        data_mode = st.radio('Input Data Source', ['Auto-Generate Dummy Data', 'Upload Custom Tensor (.npy)', 'Upload Media File'])
        
        data_file = None
        if data_mode == 'Upload Custom Tensor (.npy)':
            data_file = st.file_uploader('Upload Input Tensor', type=['npy'])
        elif data_mode == 'Upload Media File':
            data_file = st.file_uploader('Upload Media', type=['png', 'jpg', 'jpeg', 'mp4', 'wav', 'opus'])

        if st.button('Run Benchmark') and model_file:
            with st.spinner('Running Dynamic Benchmark...'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.' + model_file.name.split('.')[-1]) as tmp_m:
                    tmp_m.write(model_file.getvalue())
                    tmp_m_path = tmp_m.name
                
                try:
                    db = DynamicBenchmark(tmp_m_path)
                    
                    input_data = None
                    if data_mode == 'Auto-Generate Dummy Data':
                        input_data = db.generate_dummy_data()
                    elif data_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + data_file.name.split('.')[-1]) as tmp_d:
                            tmp_d.write(data_file.getvalue())
                            tmp_d_path = tmp_d.name
                        
                        if data_mode == 'Upload Custom Tensor (.npy)':
                            input_data = db.load_npy(tmp_d_path)
                        else:
                            input_data = db.process_media(tmp_d_path)
                        os.unlink(tmp_d_path)
                    else:
                        st.warning('No data provided, falling back to dummy data.')
                        input_data = db.generate_dummy_data()

                    res = db.run_benchmark(input_data)
                    
                    st.success('Benchmark Complete!')
                    st.write(f'**Inferred Input Shape:** `{res["input_shape"]}` (`{res["input_dtype"]}`)')
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric('Avg Latency', f'{res["avg_latency"]:.2f} ms')
                    c2.metric('P50', f'{res["p50"]:.2f} ms')
                    c3.metric('P99', f'{res["p99"]:.2f} ms')
                    c4.metric('Throughput', f'{res["fps"]:.2f} FPS')
                except Exception as e:
                    import traceback
                    st.error(f'Error: {e}')
                    st.code(traceback.format_exc())
                finally:
                    if os.path.exists(tmp_m_path):
                        os.unlink(tmp_m_path)
    else:
        model_zip = st.file_uploader('Upload ONNX GenAI Model Folder (.zip)', type=['zip'])
        prompt = st.text_area("Input Prompt", value="Explain what model quantization is and why INT8 can improve inference performance.")
        max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=2048, value=128)
        
        if st.button('Run LLM Benchmark') and model_zip:
            with st.spinner('Extracting Model and Running Benchmark...'):
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "model.zip")
                extract_path = os.path.join(temp_dir, "model_extracted")
                
                with open(zip_path, "wb") as f:
                    f.write(model_zip.getvalue())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                # Zip might contain a single top-level folder
                model_dir_to_use = extract_path
                contents = os.listdir(extract_path)
                if len(contents) == 1 and os.path.isdir(os.path.join(extract_path, contents[0])):
                    model_dir_to_use = os.path.join(extract_path, contents[0])
                
                try:
                    llm_db = DynamicLLMBenchmark(model_dir_to_use)
                    res = llm_db.run_benchmark(prompt, max_new_tokens)
                    
                    st.success('LLM Benchmark Complete!')
                    st.write(f"**Prompt Tokens:** {res['prompt_tokens']} | **Generated Tokens:** {res['generated_tokens']}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric('TTFT', f"{res['ttft_ms']:.2f} ms")
                    c2.metric('Throughput', f"{res['tokens_per_second']:.2f} tok/s")
                    c3.metric('Total Latency', f"{res['total_latency_ms']:.2f} ms")
                    
                    st.text_area("Model Output", value=res['output_text'], height=200, disabled=True)
                except Exception as e:
                    import traceback
                    st.error(f'Error: {e}')
                    st.code(traceback.format_exc())
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
