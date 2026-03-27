import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
st.set_page_config(page_title="Big Five Personality Predictor", layout="wide", page_icon="🧠")

import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from app.model_utils import load_model, load_feature_means, load_scaler, predict_traits, scores_to_softmax_pct
from data.questions import questions, trait_names
from data.trait_descriptions import trait_descriptions

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f0f2f5 !important;
    color: #111 !important;
}

/* Force dark text on light bg — scoped to Streamlit markdown only */
.stMarkdown p, .stMarkdown li { color: #444 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #111 !important; }
.stMarkdown strong { color: #111 !important; }
.stAlert p { color: #111 !important; }
.stTabs [data-baseweb="tab-panel"] p,
.stTabs [data-baseweb="tab-panel"] li { color: #444 !important; }
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { display: none; }
#MainMenu { display: none; }
footer { display: none; }

/* Main container */
.block-container {
    padding: 0.5rem 2rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Topbar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 2rem;
}
.topbar-logo {
    font-size: 1.2rem; font-weight: 800; letter-spacing: -0.5px;
}
.topbar-logo span { color: #e8400c; }

/* ── Hero ── */
.hero-section {
    background: #fff;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1rem;
    border: 1px solid #e8e8e8;
}
.hero-tag {
    display: inline-block;
    background: #fff0ec; color: #e8400c;
    font-size: 0.75rem; font-weight: 700;
    padding: 0.3rem 0.8rem; border-radius: 50px;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem; font-weight: 800; color: #111;
    line-height: 1.15; letter-spacing: -1px; margin-bottom: 1rem;
}
.hero-title span { color: #e8400c; }
.hero-sub {
    font-size: 1.05rem; color: #555; line-height: 1.7;
    max-width: 520px; margin-bottom: 2rem;
}

/* ── Cards ── */
.card {
    background: #fff;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #e8e8e8;
    margin-bottom: 0.8rem;
    height: 100%;
}
.card-orange {
    background: #e8400c;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: none;
    margin-bottom: 0.8rem;
    color: white;
}
.card h3 { font-size: 0.9rem; font-weight: 700; margin: 0 0 0.7rem; color: #111; }
.card-orange h3 { color: white; margin: 0 0 0.7rem; font-size: 0.9rem; font-weight: 700; }

/* ── Stat boxes ── */
.stat-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.stat-box {
    background: #fff; border: 1px solid #e8e8e8; border-radius: 14px;
    padding: 0.8rem 1rem; flex: 1; text-align: center;
}
.stat-num { font-size: 2rem; font-weight: 800; color: #e8400c; }
.stat-label { font-size: 0.8rem; color: #777; margin-top: 0.2rem; }

/* ── Trait pill ── */
.trait-row {
    display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.45rem;
}
.trait-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.trait-name { font-weight: 600; font-size: 0.9rem; color: #111; }
.trait-desc { font-size: 0.82rem; color: #777; }

/* ── Question card ── */
.q-wrapper {
    background: #fff; border-radius: 20px;
    padding: 2.5rem 3rem; border: 1px solid #e8e8e8;
    margin-bottom: 1.5rem; min-height: 160px;
}
.q-tag {
    display: inline-block; background: #fff0ec; color: #e8400c;
    font-size: 0.72rem; font-weight: 700; padding: 0.25rem 0.7rem;
    border-radius: 50px; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.q-text {
    font-size: 1.4rem; font-weight: 700; color: #111;
    line-height: 1.4; margin-bottom: 0.3rem;
}
.q-hint { font-size: 0.88rem; color: #888; margin-bottom: 1.5rem; }

/* ── Radio override ── */
.stRadio > label { display: none !important; }
.stRadio { width: 100% !important; }
.stRadio > div {
    display: flex !important; gap: 0.5rem !important;
    flex-wrap: nowrap !important; width: 100% !important;
}
.stRadio > div > label {
    background: #f5f5f5 !important; border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important; padding: 0.75rem 0.4rem !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #333 !important; cursor: pointer !important;
    transition: all 0.15s ease !important;
    flex: 1 1 auto !important;
    text-align: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
/* Hide the default radio circle */
.stRadio > div > label > div:first-child { display: none !important; }
.stRadio > div > label * { color: #333 !important; }
.stRadio > div > label:hover {
    border-color: #e8400c !important; background: #fff0ec !important;
}
.stRadio > div > label:hover * { color: #e8400c !important; }
.stRadio > div > label[data-checked="true"],
.stRadio > div > label:has(input:checked) {
    background: #e8400c !important; border-color: #e8400c !important;
}
.stRadio > div > label[data-checked="true"] *,
.stRadio > div > label:has(input:checked) * { color: white !important; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: #e8400c !important; border-radius: 4px !important;
}
.stProgress > div > div { background: #e8e8e8 !important; border-radius: 4px !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; padding: 0.6rem 1.5rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: #e8400c !important; border: none !important; color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: #c73509 !important;
}
.stButton > button:not([kind="primary"]) {
    background: white !important; border: 2px solid #e0e0e0 !important; color: #333 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #e8400c !important; color: #e8400c !important;
}

/* ── Result dominant banner ── */
.dominant-banner {
    background: #e8400c; border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 1.5rem; color: white;
}
.dominant-tag {
    display: inline-block; background: rgba(255,255,255,0.2);
    font-size: 0.72rem; font-weight: 700; padding: 0.25rem 0.7rem;
    border-radius: 50px; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.dominant-name { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0.5rem; }
.dominant-sub { font-size: 1rem; opacity: 0.85; max-width: 480px; line-height: 1.6; }

/* ── Trait bar ── */
.tbar-wrap { margin-bottom: 1rem; }
.tbar-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
.tbar-name { font-size: 0.9rem; font-weight: 600; color: #111; }
.tbar-pct { font-size: 0.88rem; color: #777; }
.tbar-bg { background: #f0f0f0; border-radius: 6px; height: 8px; overflow: hidden; }
.tbar-fill { height: 100%; border-radius: 6px; }

/* ── SHAP tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent !important; }
.stTabs [data-baseweb="tab"] {
    background: #fff !important; border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.82rem !important; color: #555 !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #e8400c !important; border-color: #e8400c !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #fff; border-radius: 16px; padding: 1.5rem;
    border: 1px solid #e8e8e8; margin-top: 0.5rem;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: white !important; color: #111 !important;
    border: 2px solid #e0e0e0 !important; border-radius: 10px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    border-color: #e8400c !important; color: #e8400c !important;
}

/* ── Success/warning ── */
.stAlert { border-radius: 12px !important; }

/* Divider */
hr { border-color: #e8e8e8 !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Trait config ─────────────────────────────────────────────────────────────
TRAIT_COLORS = {
    "Extraversion":      "#e8400c",
    "Neuroticism":       "#f97316",
    "Agreeableness":     "#0ea5e9",
    "Conscientiousness": "#8b5cf6",
    "Openness":          "#10b981",
}
TRAIT_DESCS = {
    "Extraversion":      "Outgoing, energetic, socially driven",
    "Neuroticism":       "Emotional reactivity and stress sensitivity",
    "Agreeableness":     "Compassionate, cooperative, considerate",
    "Conscientiousness": "Organised, reliable, goal-directed",
    "Openness":          "Curious, imaginative, open to new ideas",
}

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_model()

@st.cache_resource
def get_scaler():
    return load_scaler()

@st.cache_data
def get_feature_means():
    return load_feature_means()

@st.cache_data
def get_question_ranking():
    path = os.path.join(os.path.dirname(__file__), "cat_question_ranking.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "page": "welcome",
        "q_index": 0,
        "responses": np.full(50, np.nan),
        "answers_log": [],
        "final_prediction": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo" style="color:#111;">Big<span style="color:#e8400c;">Five</span> Predictor</div>
    <div style="font-size:0.82rem; color:#888;">Powered by CatBoost + SHAP</div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def trait_bar(name, pct, color):
    st.markdown(f"""
    <div class="tbar-wrap">
        <div class="tbar-header">
            <span class="tbar-name">{name}</span>
            <span class="tbar-pct">{pct:.1f}%</span>
        </div>
        <div class="tbar-bg">
            <div class="tbar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# WELCOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title" style="color:#111;">Understand your<br>personality with <span style="color:#e8400c;">AI</span></div>
        <div class="hero-sub" style="color:#555;">
            Answer 25 carefully selected questions and get a detailed breakdown of your
            Big Five personality traits — powered by machine learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">25</div><div class="stat-label">Questions</div></div>
        <div class="stat-box"><div class="stat-num">5</div><div class="stat-label">Personality Traits</div></div>
        <div class="stat-box"><div class="stat-num">~5</div><div class="stat-label">Minutes</div></div>
        <div class="stat-box"><div class="stat-num">100%</div><div class="stat-label">Private</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        traits_inner = ""
        for name in trait_names:
            color = TRAIT_COLORS[name]
            desc = TRAIT_DESCS[name]
            traits_inner += (
                '<div style="display:flex;align-items:center;gap:1rem;padding:0.55rem 0;border-bottom:1px solid #f0f0f0;">'
                '<div style="width:34px;height:34px;border-radius:10px;background:rgba(0,0,0,0.05);'
                'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>'
                '</div>'
                '<div>'
                f'<div style="font-size:0.88rem;font-weight:700;color:#111;">{name}</div>'
                f'<div style="font-size:0.78rem;color:#888;margin-top:1px;">{desc}</div>'
                '</div></div>'
            )
        traits_html = (
            '<div class="card">'
            '<div style="font-size:0.7rem;font-weight:700;color:#e8400c;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.6rem;">The Big Five Traits</div>'
            + traits_inner +
            '</div>'
        )
        st.markdown(traits_html, unsafe_allow_html=True)

    with col_right:
        steps = [
            ("25 questions", "SHAP-ranked for maximum insight"),
            ("CatBoost models", "One trained model per trait"),
            ("SHAP explanations", "See exactly what drove your score"),
            ("Full report", "Download your results as CSV"),
        ]
        steps_inner = ""
        for i, (title, sub) in enumerate(steps):
            steps_inner += (
                '<div style="display:flex;gap:0.8rem;align-items:flex-start;margin-bottom:0.75rem;">'
                '<div style="width:22px;height:22px;border-radius:6px;background:rgba(255,255,255,0.2);'
                'font-size:0.72rem;font-weight:700;color:white;display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;">{i+1}</div>'
                '<div>'
                f'<div style="font-size:0.85rem;font-weight:700;color:white;">{title}</div>'
                f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.75);margin-top:1px;">{sub}</div>'
                '</div></div>'
            )

        how_it_works_html = (
            '<div class="card-orange" style="margin-bottom:0.8rem;">'
            '<div style="font-size:0.7rem;font-weight:700;color:rgba(255,255,255,0.7);'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">How it works</div>'
            + steps_inner +
            '</div>'
            '<div class="card" style="border-left:3px solid #e8400c;padding:0.9rem 1.1rem;">'
            '<div style="font-size:0.78rem;font-weight:700;color:#e8400c;margin-bottom:0.3rem;">Privacy first</div>'
            '<div style="font-size:0.8rem;color:#777;line-height:1.55;">'
            'No data stored or transmitted. Personal insight only — not a clinical instrument.'
            '</div></div>'
        )
        st.markdown(how_it_works_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([2, 4])
    with col_btn:
        if st.button("Start Assessment →", type="primary", use_container_width=True):
            st.session_state.page = "quiz"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# QUIZ PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "quiz":
    ranking = get_question_ranking()
    TOTAL = 25
    q_idx = st.session_state.q_index
    q_feature_idx = int(ranking[q_idx])
    q_text = questions[q_feature_idx]
    radio_key = f"q_{q_feature_idx}"

    # Progress
    progress_pct = (q_idx + 1) / TOTAL
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
        <span style="font-size:0.82rem; font-weight:600; color:#e8400c; text-transform:uppercase; letter-spacing:0.06em;">
            Question {q_idx + 1} of {TOTAL}
        </span>
        <span style="font-size:0.82rem; color:#999;">{int(progress_pct*100)}% complete</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(progress_pct)
    st.markdown("<br>", unsafe_allow_html=True)

    # Question card — no pill tag, just the question text
    st.markdown(f"""
    <div class="q-wrapper">
        <div class="q-text">{q_text}</div>
        <div class="q-hint">Select how much you agree with this statement</div>
    </div>
    """, unsafe_allow_html=True)

    saved_val = st.session_state.responses[q_feature_idx]
    default_idx = int(saved_val) - 1 if not np.isnan(saved_val) else None

    st.radio(
        "response",
        options=[1, 2, 3, 4, 5],
        index=default_idx,
        key=radio_key,
        format_func=lambda x: {
            1: "Strongly Disagree",
            2: "Disagree",
            3: "Neutral",
            4: "Agree",
            5: "Strongly Agree",
        }[x],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_spacer, col_next = st.columns([1, 3, 1])

    with col_back:
        if st.button("← Back", use_container_width=True):
            if q_idx > 0:
                st.session_state.q_index -= 1
            else:
                st.session_state.page = "welcome"
            st.rerun()

    with col_next:
        is_last = q_idx == TOTAL - 1
        if st.button("See Results →" if is_last else "Next →", type="primary", use_container_width=True):
            current_answer = st.session_state.get(radio_key)
            if current_answer is None:
                st.warning("Please select an answer before continuing.")
            else:
                st.session_state.responses[q_feature_idx] = current_answer
                st.session_state.answers_log.append({
                    "Question": q_text,
                    "Answer": current_answer,
                    "Label": {1:"Strongly Disagree",2:"Disagree",3:"Neutral",4:"Agree",5:"Strongly Agree"}[current_answer],
                })
                if is_last:
                    models = get_models()
                    scaler = get_scaler()
                    feature_means = get_feature_means()
                    resp = np.copy(st.session_state.responses)
                    nan_mask = np.isnan(resp)
                    resp[nan_mask] = feature_means[nan_mask]
                    raw_preds = predict_traits(models, resp, scaler)
                    st.session_state.final_prediction = raw_preds
                    st.session_state.page = "results"
                else:
                    st.session_state.q_index += 1
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":
    raw_preds = st.session_state.final_prediction
    percentages = scores_to_softmax_pct(raw_preds)

    model_keys = list(get_models().keys())
    trait_pct = {}
    for i, key in enumerate(model_keys):
        display = next((tn for tn in trait_names if tn.lower() == key.lower()), key.capitalize())
        trait_pct[display] = percentages[i]

    dominant_trait = max(trait_pct, key=trait_pct.get)
    dominant_pct = trait_pct[dominant_trait]

    # Dominant banner
    desc_lines = [l.strip() for l in trait_descriptions[dominant_trait].split('\n') if l.strip() and not l.startswith('**') and not l.startswith('-')]
    first_sentence = desc_lines[0][:160] if desc_lines else ""

    st.markdown(f"""
    <div class="dominant-banner">
        <div class="dominant-name">{dominant_trait}</div>
        <div class="dominant-sub">{first_sentence}</div>
    </div>
    """, unsafe_allow_html=True)

    col_scores, col_desc = st.columns([1, 1], gap="large")

    with col_scores:
        st.markdown('<div class="card"><h3>All Trait Scores</h3>', unsafe_allow_html=True)
        for name, pct in sorted(trait_pct.items(), key=lambda x: x[1], reverse=True):
            trait_bar(name, pct, TRAIT_COLORS.get(name, "#e8400c"))
        st.markdown('</div>', unsafe_allow_html=True)

        # Mini stat boxes
        top2 = sorted(trait_pct.items(), key=lambda x: x[1], reverse=True)[:2]
        st.markdown(f"""
        <div style="display:flex; gap:0.8rem; margin-top:0.5rem;">
            <div class="stat-box" style="flex:1;">
                <div class="stat-num" style="font-size:1.4rem;">{top2[0][1]:.1f}%</div>
                <div class="stat-label">{top2[0][0]}</div>
            </div>
            <div class="stat-box" style="flex:1;">
                <div class="stat-num" style="font-size:1.4rem; color:#555;">{top2[1][1]:.1f}%</div>
                <div class="stat-label">{top2[1][0]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_desc:
        st.markdown(f'<div class="card" style="border-top: 4px solid {TRAIT_COLORS.get(dominant_trait, "#e8400c")};">', unsafe_allow_html=True)
        st.markdown(f'<h3 style="color:{TRAIT_COLORS.get(dominant_trait,"#e8400c")};">{dominant_trait} — {dominant_pct:.1f}%</h3>', unsafe_allow_html=True)
        st.markdown(trait_descriptions[dominant_trait])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # SHAP section
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-size:0.75rem; font-weight:700; color:#e8400c; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;">+ Insights</div>
        <div style="font-size:1.4rem; font-weight:800; color:#111; margin-bottom:0.3rem;">What drove your results?</div>
        <div style="font-size:0.9rem; color:#777;">SHAP values show which answers had the most influence. Red = pushed score up, Blue = pushed score down.</div>
    </div>
    """, unsafe_allow_html=True)

    models = get_models()
    scaler = get_scaler()
    feature_means = get_feature_means()
    resp = np.copy(st.session_state.responses)
    nan_mask = np.isnan(resp)
    resp[nan_mask] = feature_means[nan_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled = scaler.transform(resp.reshape(1, -1))
    input_df_renamed = pd.DataFrame(scaled, columns=[str(i) for i in range(50)])
    cat_features = [f"Q{i+1}" for i in range(50)]

    shap_tabs = st.tabs(list(trait_names))
    for tab, trait_display in zip(shap_tabs, trait_names):
        with tab:
            model_key = next((k for k in models if k.lower() == trait_display.lower()), None)
            if model_key is None:
                st.info(f"No model found for {trait_display}")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = shap.Explainer(models[model_key])
                    shap_values = explainer(input_df_renamed)

                abs_vals = np.abs(shap_values.values[0])
                top_idx = np.argsort(abs_vals)[::-1][:10]

                bv = shap_values.base_values
                bv_scalar = float(bv[0]) if hasattr(bv, '__len__') else float(bv)
                data_slice = np.array(shap_values.data[0], dtype=np.float64)[top_idx]

                sv_top = shap.Explanation(
                    values=np.array(shap_values.values[0][top_idx], dtype=np.float64),
                    base_values=bv_scalar,
                    data=data_slice,
                    feature_names=[cat_features[i] for i in top_idx],
                )

                plt.close("all")
                shap.plots.waterfall(sv_top, max_display=10, show=False)
                fig = plt.gcf()

                # Save to buffer avoiding the NumPy 2.x Agg color bug
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight",
                            facecolor="white", edgecolor="none", dpi=150)
                buf.seek(0)
                plt.close("all")
                st.image(buf, width=860)

            except Exception as e:
                import traceback
                st.warning(f"Could not generate SHAP plot: {e}")
                st.code(traceback.format_exc())

    st.markdown("<hr>", unsafe_allow_html=True)

    col_dl, col_restart = st.columns([2, 1])
    with col_dl:
        answers_df = pd.DataFrame(st.session_state.answers_log)
        scores_df = pd.DataFrame([{"Trait": k, "Score (%)": f"{v:.1f}"} for k, v in sorted(trait_pct.items(), key=lambda x: x[1], reverse=True)])
        csv_data = pd.concat([answers_df, pd.DataFrame([{}]), scores_df], ignore_index=True)
        st.download_button(
            "Download Full Results (CSV)",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name="personality_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_restart:
        if st.button("Retake Assessment", use_container_width=True):
            for key in ["q_index", "responses", "answers_log", "final_prediction"]:
                del st.session_state[key]
            st.session_state.page = "welcome"
            st.rerun()
