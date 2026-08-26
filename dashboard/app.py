import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
tab_overview, tab_classification, tab_detection, tab_llm, tab_diagnostics = st.tabs(
    ["📊 Overview", "🖼️ Classification", "🔍 Object Detection", "💬 LLM", "🛠️ Diagnostics"]
)

with tab_overview:
    st.subheader("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Benchmarks", len(filtered_df))
    with col2:
        st.metric("Unique Models", filtered_df["model"].nunique())
    with col3:
        st.metric("Vision Runs", len(filtered_df[filtered_df["model_type"] == "vision"]))
    with col4:
        st.metric("LLM Runs", len(filtered_df[filtered_df["model_type"] == "llm"]))

    st.subheader("🏆 Top Performance")
    
    c_df = filtered_df[filtered_df["task"] != "object_detection"]
    c_df = c_df[c_df["model_type"] == "vision"].dropna(subset=["throughput"])
    
    d_df = filtered_df[filtered_df["task"] == "object_detection"].dropna(subset=["throughput"])
    
    l_df = filtered_df[filtered_df["model_type"] == "llm"].dropna(subset=["tokens_per_second"])

    active_sections = []
    
    if not c_df.empty:
        active_sections.append(("Classification", c_df, "throughput", "Fastest Classification", " FPS"))
    if not d_df.empty:
        active_sections.append(("Detection", d_df, "throughput", "Fastest Object Detection", " FPS"))
    if not l_df.empty:
        active_sections.append(("LLM", l_df, "tokens_per_second", "Fastest LLM Generation", " tok/s"))

    if active_sections:
        cols = st.columns(len(active_sections))
        for col, (sec_name, sec_df, sec_metric, sec_title, sec_unit) in zip(cols, active_sections):
            with col:
                best = sec_df.loc[sec_df[sec_metric].idxmax()]
                st.metric(sec_title, f'{best[sec_metric]:.1f}{sec_unit}', f'{best["model"]} ({best["precision"]})')
    else:
        st.info("No performance data available for the current filters.")

    history = load_history()
    if history:
        st.subheader("⏱️ Recent Runs")
        recent_df = pd.DataFrame(history).tail(8).copy()
        
        display_cols = {"timestamp": "Time", "model": "Model", "runtime": "Runtime", "precision": "Precision", "throughput": "FPS", "tokens_per_second": "Tok/s", "average_latency_ms": "Latency (ms)"}
        recent_df = recent_df.rename(columns={k:v for k,v in display_cols.items() if k in recent_df.columns})
        
        st.dataframe(
            recent_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "FPS": st.column_config.NumberColumn(format="%.1f"),
                "Tok/s": st.column_config.NumberColumn(format="%.1f"),
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