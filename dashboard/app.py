import json
import sys
from pathlib import Path

# ------------------------------------------------------------
# Make project-root packages importable when Streamlit runs
# dashboard/app.py as a script.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from benchmarks.result_writer import load_history
from profiling.diagnostics import (
    diagnose,
    load_latest_summary,
)


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"
PROFILING_DIR = PROJECT_ROOT / "profiling"


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="InferBench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Global styling
# ============================================================

st.markdown(
    """
    <style>
    /* ----------------------------------------------------- */
    /* Global */
    /* ----------------------------------------------------- */

    .stApp {
        background: #0b0f14;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: rgba(11, 15, 20, 0.85);
    }

    /* ----------------------------------------------------- */
    /* Sidebar */
    /* ----------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #0f141b;
        border-right: 1px solid #202933;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ----------------------------------------------------- */
    /* Typography */
    /* ----------------------------------------------------- */

    .brand {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #f3f4f6;
        margin-bottom: 0;
    }

    .brand-accent {
        color: #60a5fa;
    }

    .tagline {
        color: #8b95a1;
        font-size: 0.95rem;
        margin-top: 0.15rem;
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #f3f4f6;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #8b95a1;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        letter-spacing: -0.02em;
        color: #f3f4f6;
        margin-top: 1.8rem;
        margin-bottom: 0.9rem;
    }

    .small-label {
        color: #8b95a1;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    /* ----------------------------------------------------- */
    /* Cards */
    /* ----------------------------------------------------- */

    .metric-card {
        background: #111821;
        border: 1px solid #202933;
        border-radius: 14px;
        padding: 1.15rem 1.2rem;
        min-height: 115px;
    }

    .metric-label {
        color: #8b95a1;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        color: #f3f4f6;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.35rem;
        letter-spacing: -0.03em;
    }

    .metric-caption {
        color: #6f7b88;
        font-size: 0.8rem;
        margin-top: 0.15rem;
    }

    .diagnostic-card {
        background: #111821;
        border: 1px solid #202933;
        border-radius: 14px;
        padding: 1.15rem;
        height: 100%;
    }

    .diagnostic-title {
        color: #f3f4f6;
        font-weight: 700;
        font-size: 1rem;
    }

    .diagnostic-value {
        color: #f3f4f6;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .diagnostic-muted {
        color: #8b95a1;
        font-size: 0.82rem;
    }

    .status-good {
        color: #34d399;
        font-weight: 750;
    }

    .status-warn {
        color: #fbbf24;
        font-weight: 750;
    }

    .status-bad {
        color: #f87171;
        font-weight: 750;
    }

    /* ----------------------------------------------------- */
    /* Tables */
    /* ----------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #202933;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ----------------------------------------------------- */
    /* Tabs */
    /* ----------------------------------------------------- */

    button[data-baseweb="tab"] {
        color: #8b95a1;
        font-weight: 650;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa;
    }

    /* ----------------------------------------------------- */
    /* Expanders */
    /* ----------------------------------------------------- */

    details {
        border: 1px solid #202933 !important;
        border-radius: 12px !important;
        background: #111821 !important;
    }

    /* ----------------------------------------------------- */
    /* Footer */
    /* ----------------------------------------------------- */

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #202933;
        color: #596575;
        font-size: 0.78rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def is_missing(value) -> bool:
    return value is None or pd.isna(value)


def fmt_ms(value) -> str:
    if is_missing(value):
        return "—"

    return f"{float(value):.3f} ms"


def fmt_fps(value) -> str:
    if is_missing(value):
        return "—"

    return f"{float(value):,.2f} FPS"


def fmt_tok_s(value) -> str:
    if is_missing(value):
        return "—"

    return f"{float(value):,.2f} tok/s"


def fmt_percent(value) -> str:
    if is_missing(value):
        return "—"

    return f"{float(value):.1f}%"


def load_json_file(path: Path) -> dict | None:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return None


# ============================================================
# Result loading
# ============================================================

@st.cache_data(ttl=5)
def load_results() -> list[dict]:
    results = []

    if not RESULTS_DIR.exists():
        return results

    for path in RESULTS_DIR.rglob("*.json"):
        # Ignore test files.
        if path.name == "test_result.json":
            continue

        data = load_json_file(path)

        if data is None:
            continue

        # Require benchmark-result structure.
        if "model" not in data:
            continue

        data["_file"] = str(
            path.relative_to(PROJECT_ROOT)
        )

        results.append(data)

    return results


@st.cache_data(ttl=5)
def load_profile_summaries() -> list[dict]:
    summaries = []

    if not PROFILING_DIR.exists():
        return summaries

    for path in PROFILING_DIR.glob(
        "*.summary.json"
    ):
        data = load_json_file(path)

        if data is None:
            continue

        data["_file"] = str(
            path.relative_to(PROJECT_ROOT)
        )

        summaries.append(data)

    summaries.sort(
        key=lambda item: Path(
            item["_file"]
        ).stat().st_mtime
        if Path(
            PROJECT_ROOT / item["_file"]
        ).exists()
        else 0,
        reverse=True,
    )

    return summaries


def clear_cache() -> None:
    load_results.clear()
    load_profile_summaries.clear()


# ============================================================
# Normalize dataframe
# ============================================================

results = load_results()

if not results:
    st.markdown(
        '<div class="brand">'
        '<span class="brand-accent">⚡</span> InferBench'
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "No benchmark results found in results/."
    )

    st.stop()


df = pd.DataFrame(results)

expected_columns = [
    "model",
    "model_type",
    "runtime",
    "execution_provider",
    "precision",
    "batch_size",
    "average_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "throughput",
    "throughput_unit",
    "ttft_ms",
    "tokens_per_second",
    "peak_memory_mb",
    "accuracy_metric",
    "accuracy_value",
    "notes",
]

for column in expected_columns:
    if column not in df.columns:
        df[column] = None


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown(
    '<div class="brand">'
    '<span class="brand-accent">⚡</span> InferBench'
    "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="tagline">'
    "Inference Performance Lab"
    "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="small-label">Filters</div>',
    unsafe_allow_html=True,
)

models = sorted(
    df["model"]
    .dropna()
    .astype(str)
    .unique()
)

runtimes = sorted(
    df["runtime"]
    .dropna()
    .astype(str)
    .unique()
)

precisions = sorted(
    df["precision"]
    .dropna()
    .astype(str)
    .unique()
)

model_filter = st.sidebar.selectbox(
    "Model",
    ["All", *models],
)

runtime_filter = st.sidebar.selectbox(
    "Runtime",
    ["All", *runtimes],
)

precision_filter = st.sidebar.selectbox(
    "Precision",
    ["All", *precisions],
)

st.sidebar.divider()

if st.sidebar.button(
    "↻ Refresh data",
    use_container_width=True,
):
    clear_cache()
    st.rerun()

st.sidebar.markdown(
    """
    <div style="
        color:#596575;
        font-size:0.75rem;
        line-height:1.6;
        margin-top:1rem;
    ">
    Results are loaded from<br>
    <b>results/</b><br><br>
    Profiling summaries are loaded from<br>
    <b>profiling/</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Apply filters
# ============================================================

filtered_df = df.copy()

if model_filter != "All":
    filtered_df = filtered_df[
        filtered_df["model"] == model_filter
    ]

if runtime_filter != "All":
    filtered_df = filtered_df[
        filtered_df["runtime"] == runtime_filter
    ]

if precision_filter != "All":
    filtered_df = filtered_df[
        filtered_df["precision"] == precision_filter
    ]


# ============================================================
# Hero
# ============================================================

st.markdown(
    '<div class="hero-title">'
    "Inference Performance Lab"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    "Benchmark, optimize, profile, and compare AI inference workloads."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Main tabs
# ============================================================

tab_overview, tab_vision, tab_llm, tab_diagnostics = st.tabs(
    [
        "Overview",
        "Vision",
        "LLM",
        "Diagnostics",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    total_results = len(
        filtered_df
    )

    total_models = filtered_df[
        "model"
    ].nunique()

    vision_count = len(
        filtered_df[
            filtered_df["model_type"] == "vision"
        ]
    )

    llm_count = len(
        filtered_df[
            filtered_df["model_type"] == "llm"
        ]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Benchmark Results</div>
                <div class="metric-value">{total_results}</div>
                <div class="metric-caption">Saved measurements</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Models</div>
                <div class="metric-value">{total_models}</div>
                <div class="metric-caption">Unique workloads</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Vision</div>
                <div class="metric-value">{vision_count}</div>
                <div class="metric-caption">Vision benchmark results</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">LLM</div>
                <div class="metric-value">{llm_count}</div>
                <div class="metric-caption">LLM benchmark results</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Best results
    # --------------------------------------------------------

    vision_df = filtered_df[
        filtered_df["model_type"] == "vision"
    ].dropna(
        subset=["throughput"]
    )

    llm_df = filtered_df[
        filtered_df["model_type"] == "llm"
    ].dropna(
        subset=["tokens_per_second"]
    )

    st.markdown(
        '<div class="section-title">Top Performance</div>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)

    with p1:
        if not vision_df.empty:
            best = vision_df.loc[
                vision_df["throughput"].idxmax()
            ]

            st.markdown(
                f"""
                <div class="diagnostic-card">
                    <div class="small-label">
                        Best Vision Throughput
                    </div>
                    <div class="metric-value">
                        {fmt_fps(best["throughput"])}
                    </div>
                    <div class="diagnostic-title">
                        {best["model"]}
                    </div>
                    <div class="diagnostic-muted">
                        {best["runtime"]} · {best["precision"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No vision throughput result available."
            )

    with p2:
        if not llm_df.empty:
            best = llm_df.loc[
                llm_df["tokens_per_second"].idxmax()
            ]

            st.markdown(
                f"""
                <div class="diagnostic-card">
                    <div class="small-label">
                        Best LLM Throughput
                    </div>
                    <div class="metric-value">
                        {fmt_tok_s(best["tokens_per_second"])}
                    </div>
                    <div class="diagnostic-title">
                        {best["model"]}
                    </div>
                    <div class="diagnostic-muted">
                        {best["runtime"]} · {best["precision"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No LLM throughput result available."
            )

    # --------------------------------------------------------
    # Recent history
    # --------------------------------------------------------

    history = load_history()

    if history:
        history_df = pd.DataFrame(history)

        st.markdown(
            '<div class="section-title">Recent Runs</div>',
            unsafe_allow_html=True,
        )

        recent = history_df.tail(
            8
        ).copy()

        if "timestamp" in recent.columns:
            recent["timestamp"] = pd.to_datetime(
                recent["timestamp"],
                errors="coerce",
            )

        columns = [
            "timestamp",
            "model",
            "runtime",
            "precision",
            "throughput",
            "tokens_per_second",
            "average_latency_ms",
        ]

        columns = [
            col
            for col in columns
            if col in recent.columns
        ]

        if columns:
            recent_display = recent[
                columns
            ].copy()

            rename = {
                "timestamp": "Timestamp",
                "model": "Model",
                "runtime": "Runtime",
                "precision": "Precision",
                "throughput": "FPS",
                "tokens_per_second": "Tokens/sec",
                "average_latency_ms": "Latency",
            }

            recent_display = recent_display.rename(
                columns=rename
            )

            if "FPS" in recent_display:
                recent_display["FPS"] = (
                    recent_display["FPS"].map(
                        fmt_fps
                    )
                )

            if "Tokens/sec" in recent_display:
                recent_display["Tokens/sec"] = (
                    recent_display["Tokens/sec"].map(
                        fmt_tok_s
                    )
                )

            if "Latency" in recent_display:
                recent_display["Latency"] = (
                    recent_display["Latency"].map(
                        fmt_ms
                    )
                )

            st.dataframe(
                recent_display,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# VISION
# ============================================================

with tab_vision:

    vision_df = filtered_df[
        filtered_df["model_type"] == "vision"
    ].copy()

    if vision_df.empty:
        st.info(
            "No vision results match the current filters."
        )
    else:

        st.markdown(
            '<div class="section-title">Vision Optimization</div>',
            unsafe_allow_html=True,
        )

        measurable = vision_df.dropna(
            subset=["throughput"]
        )

        best = (
            measurable.loc[
                measurable["throughput"].idxmax()
            ]
            if not measurable.empty
            else None
        )

        baseline_candidates = vision_df[
            vision_df["precision"] == "FP32"
        ].dropna(
            subset=["throughput"]
        )

        baseline = (
            baseline_candidates.loc[
                baseline_candidates["throughput"].idxmax()
            ]
            if not baseline_candidates.empty
            else None
        )

        best_latency = (
            vision_df.dropna(
                subset=["average_latency_ms"]
            )
        )

        best_latency_row = (
            best_latency.loc[
                best_latency["average_latency_ms"].idxmin()
            ]
            if not best_latency.empty
            else None
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            value = (
                fmt_fps(best["throughput"])
                if best is not None
                else "—"
            )

            caption = (
                str(best["precision"])
                if best is not None
                else "No data"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Best Throughput</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-caption">{caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            value = (
                fmt_ms(
                    best_latency_row[
                        "average_latency_ms"
                    ]
                )
                if best_latency_row is not None
                else "—"
            )

            caption = (
                str(
                    best_latency_row[
                        "precision"
                    ]
                )
                if best_latency_row is not None
                else "No data"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Best Latency</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-caption">{caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        speedup = None

        if (
            best is not None
            and baseline is not None
            and float(
                baseline["throughput"]
            ) != 0
        ):
            speedup = (
                float(best["throughput"])
                / float(baseline["throughput"])
            )

        with k3:
            value = (
                f"{speedup:.2f}×"
                if speedup is not None
                else "—"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Speedup vs FP32</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-caption">Best measured configuration</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        latency_reduction = None

        if (
            best is not None
            and baseline is not None
            and not is_missing(
                best["average_latency_ms"]
            )
            and not is_missing(
                baseline["average_latency_ms"]
            )
            and float(
                baseline["average_latency_ms"]
            ) != 0
        ):
            latency_reduction = (
                1
                - (
                    float(
                        best["average_latency_ms"]
                    )
                    / float(
                        baseline["average_latency_ms"]
                    )
                )
            ) * 100

        with k4:
            value = (
                f"{latency_reduction:.1f}%"
                if latency_reduction is not None
                else "—"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Latency Reduction</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-caption">vs FP32 baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # Main comparison table
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Configuration Comparison</div>',
            unsafe_allow_html=True,
        )

        table = vision_df[
            [
                "model",
                "runtime",
                "precision",
                "average_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "throughput",
                "accuracy_metric",
                "accuracy_value",
            ]
        ].copy()

        table.columns = [
            "Model",
            "Runtime",
            "Precision",
            "Avg Latency",
            "P50",
            "P95",
            "P99",
            "Throughput",
            "Quality Metric",
            "Quality",
        ]

        table["Avg Latency"] = table[
            "Avg Latency"
        ].map(fmt_ms)

        table["P50"] = table[
            "P50"
        ].map(fmt_ms)

        table["P95"] = table[
            "P95"
        ].map(fmt_ms)

        table["P99"] = table[
            "P99"
        ].map(fmt_ms)

        table["Throughput"] = table[
            "Throughput"
        ].map(fmt_fps)

        table["Quality"] = table[
            "Quality"
        ].map(fmt_percent)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Throughput
        # ----------------------------------------------------

        chart_df = vision_df[
            [
                "precision",
                "throughput",
            ]
        ].dropna(
            subset=["throughput"]
        )

        if not chart_df.empty:

            chart_df = (
                chart_df
                .groupby(
                    "precision",
                    as_index=True,
                )["throughput"]
                .max()
                .sort_values(
                    ascending=True
                )
            )

            st.markdown(
                '<div class="section-title">'
                "Throughput"
                "</div>",
                unsafe_allow_html=True,
            )

            st.bar_chart(
                chart_df
            )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latency_df = vision_df[
            [
                "precision",
                "average_latency_ms",
            ]
        ].dropna(
            subset=["average_latency_ms"]
        )

        if not latency_df.empty:

            latency_df = (
                latency_df
                .groupby(
                    "precision",
                    as_index=True,
                )["average_latency_ms"]
                .min()
                .sort_values(
                    ascending=True
                )
            )

            st.markdown(
                '<div class="section-title">'
                "Latency"
                "</div>",
                unsafe_allow_html=True,
            )

            st.bar_chart(
                latency_df
            )

        # ----------------------------------------------------
        # Optimization takeaway
        # ----------------------------------------------------

        if (
            best is not None
            and baseline is not None
            and speedup is not None
        ):
            best_precision = str(
                best["precision"]
            )

            st.markdown(
                '<div class="section-title">'
                "Optimization Takeaway"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="diagnostic-card">
                    <div class="diagnostic-title">
                        {best_precision} is currently the fastest vision configuration.
                    </div>
                    <div class="diagnostic-muted" style="margin-top:0.5rem;">
                        It reaches {fmt_fps(best["throughput"])}
                        compared with {fmt_fps(baseline["throughput"])}
                        for FP32 — a {speedup:.2f}× throughput improvement.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# LLM
# ============================================================

with tab_llm:

    llm_df = filtered_df[
        filtered_df["model_type"] == "llm"
    ].copy()

    if llm_df.empty:
        st.info(
            "No LLM results match the current filters."
        )
    else:

        st.markdown(
            '<div class="section-title">LLM Inference</div>',
            unsafe_allow_html=True,
        )

        measurable = llm_df.dropna(
            subset=["tokens_per_second"]
        )

        best = (
            measurable.loc[
                measurable["tokens_per_second"].idxmax()
            ]
            if not measurable.empty
            else None
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Generation Throughput</div>
                    <div class="metric-value">
                        {
                            fmt_tok_s(
                                best["tokens_per_second"]
                            )
                            if best is not None
                            else "—"
                        }
                    </div>
                    <div class="metric-caption">Autoregressive generation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">TTFT</div>
                    <div class="metric-value">
                        {
                            fmt_ms(best["ttft_ms"])
                            if best is not None
                            else "—"
                        }
                    </div>
                    <div class="metric-caption">Time to first token</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Latency</div>
                    <div class="metric-value">
                        {
                            fmt_ms(
                                best[
                                    "average_latency_ms"
                                ]
                            )
                            if best is not None
                            else "—"
                        }
                    </div>
                    <div class="metric-caption">End-to-end generation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">
                        {
                            str(best["precision"])
                            if best is not None
                            else "—"
                        }
                    </div>
                    <div class="metric-caption">Active model format</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # LLM table
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Configuration Comparison</div>',
            unsafe_allow_html=True,
        )

        table = llm_df[
            [
                "model",
                "runtime",
                "precision",
                "ttft_ms",
                "tokens_per_second",
                "average_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
            ]
        ].copy()

        table.columns = [
            "Model",
            "Runtime",
            "Precision",
            "TTFT",
            "Tokens/sec",
            "Total Latency",
            "P50",
            "P95",
            "P99",
        ]

        table["TTFT"] = table[
            "TTFT"
        ].map(fmt_ms)

        table["Tokens/sec"] = table[
            "Tokens/sec"
        ].map(fmt_tok_s)

        table["Total Latency"] = table[
            "Total Latency"
        ].map(fmt_ms)

        table["P50"] = table[
            "P50"
        ].map(fmt_ms)

        table["P95"] = table[
            "P95"
        ].map(fmt_ms)

        table["P99"] = table[
            "P99"
        ].map(fmt_ms)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Throughput chart
        # ----------------------------------------------------

        throughput_df = llm_df[
            [
                "precision",
                "tokens_per_second",
            ]
        ].dropna(
            subset=["tokens_per_second"]
        )

        if not throughput_df.empty:

            throughput_df = (
                throughput_df
                .groupby(
                    "precision",
                    as_index=True,
                )["tokens_per_second"]
                .max()
                .sort_values(
                    ascending=True
                )
            )

            st.markdown(
                '<div class="section-title">'
                "Generation Throughput"
                "</div>",
                unsafe_allow_html=True,
            )

            st.bar_chart(
                throughput_df
            )

        # ----------------------------------------------------
        # TTFT chart
        # ----------------------------------------------------

        ttft_df = llm_df[
            [
                "precision",
                "ttft_ms",
            ]
        ].dropna(
            subset=["ttft_ms"]
        )

        if not ttft_df.empty:

            ttft_df = (
                ttft_df
                .groupby(
                    "precision",
                    as_index=True,
                )["ttft_ms"]
                .min()
                .sort_values(
                    ascending=True
                )
            )

            st.markdown(
                '<div class="section-title">'
                "Time to First Token"
                "</div>",
                unsafe_allow_html=True,
            )

            st.bar_chart(
                ttft_df
            )


# ============================================================
# DIAGNOSTICS
# ============================================================

with tab_diagnostics:

    st.markdown(
        '<div class="section-title">Runtime Diagnostics</div>',
        unsafe_allow_html=True,
    )

    profile_summary = load_latest_summary()

    if profile_summary is None:

        st.info(
            "No profiling summary found. "
            "Run `python -m inferbench profile-onnx ...` "
            "to generate diagnostics."
        )

    else:

        providers = profile_summary.get(
            "active_providers",
            [],
        )

        cuda_active = (
            "CUDAExecutionProvider"
            in providers
        )

        memcpy_events = int(
            profile_summary.get(
                "memcpy_events",
                0,
            )
        )

        memcpy_time_us = float(
            profile_summary.get(
                "memcpy_total_us",
                0,
            )
        )

        cpu_events = int(
            profile_summary.get(
                "cpu_events",
                0,
            )
        )

        cuda_events = int(
            profile_summary.get(
                "cuda_events",
                0,
            )
        )

        inference_latency = profile_summary.get(
            "inference_latency_ms"
        )

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            status = (
                "Active"
                if cuda_active
                else "Unavailable"
            )

            status_class = (
                "status-good"
                if cuda_active
                else "status-bad"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">CUDA Provider</div>
                    <div class="metric-value">
                        <span class="{status_class}">
                            {status}
                        </span>
                    </div>
                    <div class="metric-caption">
                        {", ".join(providers) if providers else "No provider data"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Memcpy Events</div>
                    <div class="metric-value">
                        {memcpy_events:,}
                    </div>
                    <div class="metric-caption">
                        Host/device copy activity
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Memcpy Time</div>
                    <div class="metric-value">
                        {memcpy_time_us / 1000:.2f} ms
                    </div>
                    <div class="metric-caption">
                        Cumulative profile time
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">CPU Events</div>
                    <div class="metric-value">
                        {cpu_events:,}
                    </div>
                    <div class="metric-caption">
                        CPU-related activity
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # Status row
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Execution Health</div>',
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3)

        with s1:
            if cuda_active:
                st.success(
                    "CUDA execution provider is active."
                )
            else:
                st.error(
                    "CUDA execution provider is not active."
                )

        with s2:
            if memcpy_events > 0:
                st.warning(
                    f"{memcpy_events:,} memcpy events detected."
                )
            else:
                st.success(
                    "No memcpy events detected."
                )

        with s3:
            if cpu_events > 0:
                st.warning(
                    f"{cpu_events:,} CPU-related events detected."
                )
            else:
                st.success(
                    "No CPU execution activity detected."
                )

        # ----------------------------------------------------
        # Diagnosis
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Diagnosis</div>',
            unsafe_allow_html=True,
        )

        findings = diagnose(
            profile_summary
        )

        if findings:
            for finding in findings:
                st.warning(
                    finding
                )
        else:
            st.success(
                "No obvious runtime bottleneck detected."
            )

        # ----------------------------------------------------
        # Key metrics
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Profile Metrics</div>',
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "CUDA Events",
                f"{cuda_events:,}",
            )

        with p2:
            st.metric(
                "Profiled Latency",
                fmt_ms(inference_latency),
            )

        with p3:
            st.metric(
                "Total Profile Events",
                f"{int(profile_summary.get('total_events', 0)):,}",
            )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        if memcpy_events > 0:

            st.markdown(
                '<div class="section-title">'
                "Optimization Signal"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="diagnostic-card">
                    <div class="diagnostic-title">
                        Host/device transfer overhead is visible.
                    </div>
                    <div class="diagnostic-muted"
                         style="margin-top:0.5rem;">
                        InferBench detected memcpy activity during
                        the profiled CUDA execution path. This can
                        reduce the benefit of GPU acceleration,
                        especially for highly quantized graphs.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # Profile details
        # ----------------------------------------------------

        with st.expander(
            "Profiling details"
        ):
            st.write(
                "Model:",
                profile_summary.get(
                    "model",
                    "Unknown",
                ),
            )

            st.write(
                "Active providers:",
                providers,
            )

            st.write(
                "Input shape:",
                profile_summary.get(
                    "input_shape",
                    "Unknown",
                ),
            )

            st.write(
                "Profile file:",
                profile_summary.get(
                    "profile",
                    "Unknown",
                ),
            )

            st.json(
                profile_summary
            )


# ============================================================
# Developer details
# ============================================================

with st.expander(
    "Developer Details"
):

    st.markdown(
        "### Benchmark files"
    )

    for path in sorted(
        RESULTS_DIR.rglob("*.json")
    ):
        st.code(
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
        )

    st.markdown(
        "### Profiling summaries"
    )

    summaries = load_profile_summaries()

    if summaries:
        for summary in summaries:
            st.code(
                summary.get(
                    "_file",
                    "Unknown",
                )
            )
    else:
        st.write(
            "No profiling summaries found."
        )

    st.markdown(
        "### Raw benchmark JSON"
    )

    st.json(
        results
    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        InferBench · AI Inference Optimization & Benchmarking Engine
    </div>
    """,
    unsafe_allow_html=True,
)