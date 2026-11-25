"""Streamlit dashboard for telecom sentiment analysis - Clean Display Layer."""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Silence joblib physical core warning on Windows

import json
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Inject root project directory to system path so `utils` resolves from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force reload of utils to bypass Streamlit's module caching
if "utils" in sys.modules:
    del sys.modules["utils"]

from utils import clean_text, predict_with_rules

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Page config
st.set_page_config(
    page_title="Telecom Sentiment Intelligence",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark glassmorphic styling
st.markdown("""
<style>
/* Reset and background */
[data-testid="stAppViewContainer"] {
    background-color: #0b0e14 !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
.stApp {
    background-color: #0b0e14;
    color: #f8f9fa;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Main title styling */
.main-title {
    font-size: 28px;
    font-weight: 600;
    color: #f8f9fa;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}

.subtitle {
    font-size: 14px;
    color: #8b949e;
    margin-bottom: 30px;
    line-height: 1.5;
}

/* Model Card container */
.model-card {
    background-color: #11141e;
    border: 1px solid #1e2433;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 20px;
}

.model-title {
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-bottom: 4px;
}
.model-subtitle {
    font-size: 10px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 20px;
    letter-spacing: 0.5px;
}

/* Metrics rows */
.metric-row {
    margin-bottom: 12px;
}
.metric-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #cbd5e1;
    margin-bottom: 6px;
    align-items: center;
}
.metric-name {
    font-weight: 500;
}
.metric-score {
    font-weight: 700;
    font-family: ui-monospace, monospace;
}

/* Progress bar container */
.progress-bg {
    height: 4px;
    background-color: #1e2433;
    border-radius: 2px;
    width: 100%;
}
.progress-fill {
    height: 100%;
    border-radius: 2px;
}
.progress-fill.blue { background-color: #5b4ef6; }
.progress-fill.yellow { background-color: #fbbd23; }
.progress-fill.red { background-color: #ef4444; }
.progress-fill.green { background-color: #10b981; }

.f1-box {
    background-color: #1e2433;
    padding: 2px 6px;
    border-radius: 4px;
    color: white;
}

/* Header for Results */
.results-header-container {
    display: flex;
    align-items: center;
    margin-top: 30px;
    margin-bottom: 20px;
    position: relative;
}
.results-header {
    font-size: 20px;
    font-weight: 600;
    color: #f8f9fa;
}

/* Buttons */
div.stButton > button {
    background-color: #5b4ef6 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #4f42e2 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dashboard Main Title
# ---------------------------------------------------------
st.markdown('<div class="main-title">📶 Telecom Sentiment Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clean visual interface showcasing model performance metrics and confusion matrices calculated and verified strictly during the ML training pipeline.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Real-Time Interactive Inference Area
# ---------------------------------------------------------
st.markdown('<div class="results-header-container"><div class="results-header">⚡ Live Sentiment Telemetry (Inference)</div></div>', unsafe_allow_html=True)

with st.form("inference_form", clear_on_submit=False):
    default_text = "The network connection drops frequently during peak hours. Extremely frustrating when trying to work from home."
    review_text = st.text_area("review", value=default_text, label_visibility="collapsed")
    submit = st.form_submit_button("⚡ Analyze Customer Review")

if review_text.strip():
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    if not vectorizer_path.exists():
        st.warning("⚠️ No trained models found. Please run the training pipeline first: `python run_pipeline.py`")
    else:
        try:
            vec = joblib.load(vectorizer_path)
            cleaned = clean_text(review_text)
            x_vec = vec.transform([cleaned])
            
            lr_model = joblib.load(MODELS_DIR / "logistic.pkl")
            nb_model = joblib.load(MODELS_DIR / "naive_bayes.pkl")
            rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
            
            lr_pred = predict_with_rules(review_text, vec, lr_model)
            nb_pred = predict_with_rules(review_text, vec, nb_model)
            rf_pred = predict_with_rules(review_text, vec, rf_model)
            
            col_inf1, col_inf2, col_inf3 = st.columns(3)
            with col_inf1:
                st.markdown(f'<div class="model-card"><div class="model-title">Logistic Regression</div><div style="font-size: 16px; font-weight: 700; color: #818cf8; margin-top: 10px;">{lr_pred.upper()}</div></div>', unsafe_allow_html=True)
            with col_inf2:
                st.markdown(f'<div class="model-card"><div class="model-title">Naive Bayes</div><div style="font-size: 16px; font-weight: 700; color: #fbbd23; margin-top: 10px;">{nb_pred.upper()}</div></div>', unsafe_allow_html=True)
            with col_inf3:
                st.markdown(f'<div class="model-card"><div class="model-title">Random Forest</div><div style="font-size: 16px; font-weight: 700; color: #10b981; margin-top: 10px;">{rf_pred.upper()}</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Inference prediction failed: {e}")

st.markdown("---")

# ---------------------------------------------------------
# Show Model Evaluation results (Benchmarks & Confusion Matrices)
# ---------------------------------------------------------
st.markdown('<div class="results-header-container"><div class="results-header">📊 Model Evaluation Benchmarks</div></div>', unsafe_allow_html=True)

# File names
metric_files = {
    "logistic": RESULTS_DIR / "logistic_metrics.pkl",
    "naive_bayes": RESULTS_DIR / "nb_metrics.pkl",
    "random_forest": RESULTS_DIR / "rf_metrics.pkl"
}

# Verify files exist
missing_files = [k for k, path in metric_files.items() if not path.exists()]

if missing_files:
    st.warning("⚠️ Some or all metrics files are missing. Please execute the training pipeline to generate them: `python run_pipeline.py`")
else:
    # Load metrics
    metrics = {
        "logistic": joblib.load(metric_files["logistic"]),
        "naive_bayes": joblib.load(metric_files["naive_bayes"]),
        "random_forest": joblib.load(metric_files["random_forest"])
    }

    # Display Metrics Cards Side by Side
    col1, col2, col3 = st.columns(3)

    def render_metric(name: str, score: float, bar_color: str, is_f1: bool = False):
        if is_f1:
            score_html = f'<span class="metric-score f1-box">{score:.4f}</span>'
            bar_html = ""
        else:
            score_html = f'<span class="metric-score">{score:.4f}</span>'
            bar_html = f'<div class="progress-bg"><div class="progress-fill {bar_color}" style="width: {score * 100}%;"></div></div>'
        return f'<div class="metric-row"><div class="metric-header"><span class="metric-name">{name}</span>{score_html}</div>{bar_html}</div>'

    def render_model_card(title: str, subtitle: str, model_metrics: dict, acc_color: str, prec_color: str, rec_color: str):
        html = '<div class="model-card">'
        html += f'<div class="model-title">{title}</div>'
        html += f'<div class="model-subtitle">{subtitle}</div>'
        html += '<div style="height: 1px; background-color: #1e2433; margin-bottom: 16px;"></div>'
        html += render_metric("Accuracy", model_metrics["accuracy"], acc_color)
        html += render_metric("Precision (Weighted)", model_metrics["precision"], prec_color)
        html += render_metric("Recall (Weighted)", model_metrics["recall"], rec_color)
        html += render_metric("F1-Score (Weighted)", model_metrics["f1_score"], "", is_f1=True)
        html += '</div>'
        return html

    with col1:
        st.markdown(render_model_card(
            "Logistic Regression", "BASELINE LINEAR MODEL", metrics["logistic"], "blue", "yellow", "blue"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(render_model_card(
            "Multinomial Naive Bayes", "PROBABILISTIC CLASSIFIER", metrics["naive_bayes"], "yellow", "red", "blue"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(render_model_card(
            "Random Forest Classifier", "ENSEMBLE DECISION TREE", metrics["random_forest"], "blue", "blue", "green"
        ), unsafe_allow_html=True)

    # 📊 Confusion Matrices Section
    st.markdown('<div class="results-header-container"><div class="results-header">Confusion Matrices</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Logistic Regression Matrix", 
        "Naive Bayes Matrix", 
        "Random Forest Matrix"
    ])

    labels = ["Positive", "Neutral", "Negative"]
    models_keys = [("logistic", tab1), ("naive_bayes", tab2), ("random_forest", tab3)]

    for model_key, tab in models_keys:
        with tab:
            st.markdown(f'<div class="model-card" style="padding: 24px; margin-bottom: 0px;">', unsafe_allow_html=True)
            
            t_col1, t_col2 = st.columns([6.5, 3.5])
            with t_col1:
                st.markdown(f'<div style="font-size: 14px; font-weight: 600; color: #818cf8; margin-top: 6px;">Evaluation Partition Metrics (Labels: Positive, Neutral, Negative)</div>', unsafe_allow_html=True)
            with t_col2:
                view_type = st.radio("Format Options:", ["Heatmap View", "Table View"], horizontal=True, key=f"view_{model_key}", label_visibility="collapsed")
            
            cm_arr = metrics[model_key]["confusion_matrix"]
            
            df_cm = pd.DataFrame(
                cm_arr,
                index=[f"Actual {c}" for c in labels],
                columns=[f"Predicted {c}" for c in labels]
            )
            
            st.write("")
            
            if "Table View" in view_type:
                st.dataframe(df_cm, use_container_width=True)
            else:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_facecolor('#11141e')
                    ax.set_facecolor('#11141e')
                    
                    sns.heatmap(
                        df_cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax,
                        annot_kws={"size": 10, "weight": "bold", "color": "#f8f9fa"}
                    )
                    
                    ax.tick_params(colors='#8b949e', labelsize=8)
                    ax.set_xticklabels(ax.get_xticklabels(), color='#8b949e')
                    ax.set_yticklabels(ax.get_yticklabels(), color='#8b949e')
                    
                    for spine in ax.spines.values():
                        spine.set_color('#1e2433')
                        
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.dataframe(df_cm.style.background_gradient(cmap='Purples', axis=None), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
