import sys
import os

# Fix paths for HuggingFace Spaces — app runs from /app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import streamlit as st
from src.model import SpecialistMapper

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="MediMap — Symptom to Specialist",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f1e;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(99,102,241,0.08) 0%, transparent 60%),
        #0a0f1e;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; margin: 0 auto; }

/* ── Hero ── */
.hero { text-align: center; padding: 3.5rem 0 2.5rem; }
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 0.35rem 1rem; border-radius: 99px; margin-bottom: 1.4rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    line-height: 1.1; color: #f0f4ff; margin-bottom: 1rem;
}
.hero h1 em { font-style: italic; color: #38bdf8; }
.hero p {
    font-size: 1.05rem; color: #8892a4;
    max-width: 520px; margin: 0 auto;
    line-height: 1.7; font-weight: 300;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent);
    margin: 2rem 0;
}

/* ── Input ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    padding: 1.1rem 1.3rem !important;
    resize: none !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
}
.stTextArea textarea::placeholder { color: #4a5568 !important; }
.stTextArea label { color: #8892a4 !important; font-size: 0.85rem !important; }

/* ── Main button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Examples label ── */
.examples-label {
    font-size: 0.78rem; color: #4a5568; text-align: center;
    margin-bottom: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600;
}

/* ── Stats ── */
.stats-row {
    display: flex; gap: 1rem; justify-content: center;
    margin-bottom: 2rem; flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 99px; padding: 0.4rem 1rem;
    font-size: 0.78rem; color: #8892a4;
}
.stat-pill b { color: #e8eaf0; }

/* ── Result cards ── */
.results-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem; color: #f0f4ff;
    margin: 2.5rem 0 1.5rem; text-align: center;
}
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 1.6rem 1.8rem;
    margin-bottom: 1rem; position: relative;
    overflow: hidden; transition: border-color 0.2s, transform 0.2s;
}
.card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    border-radius: 18px 18px 0 0;
}
.card-rank1::before { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
.card-rank2::before { background: linear-gradient(90deg, #6366f1, #818cf8); }
.card-rank3::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.card:hover { border-color: rgba(56,189,248,0.2); transform: translateY(-2px); }
.card-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 1rem;
}
.card-rank {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a5568; margin-bottom: 0.2rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem; color: #f0f4ff;
}
.card-icon { font-size: 2rem; opacity: 0.8; }
.confidence-label {
    font-size: 0.78rem; color: #8892a4; margin-bottom: 0.4rem;
    font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
}
.confidence-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 99px; height: 6px;
    margin-bottom: 1.1rem; overflow: hidden;
}
.bar-1 { height:100%; border-radius:99px; background: linear-gradient(90deg,#0ea5e9,#38bdf8); }
.bar-2 { height:100%; border-radius:99px; background: linear-gradient(90deg,#6366f1,#818cf8); }
.bar-3 { height:100%; border-radius:99px; background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.card-match {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 8px; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; color: #8892a4;
    margin-bottom: 0.8rem; font-style: italic;
}
.card-match span { color: #38bdf8; font-weight: 500; font-style: normal; }
.card-desc { font-size: 0.88rem; color: #6b7280; line-height: 1.65; }
.conf-num-1 { font-size:1.6rem; font-weight:700; font-family:'DM Serif Display',serif; color:#38bdf8; }
.conf-num-2 { font-size:1.6rem; font-weight:700; font-family:'DM Serif Display',serif; color:#818cf8; }
.conf-num-3 { font-size:1.6rem; font-weight:700; font-family:'DM Serif Display',serif; color:#a78bfa; }

/* ── Loading state ── */
.loading-box {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 14px; padding: 2rem;
    text-align: center; color: #38bdf8;
    font-size: 0.95rem; margin: 2rem 0;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.15);
    border-radius: 12px; padding: 1rem 1.3rem;
    font-size: 0.83rem; color: #92836a;
    text-align: center; margin-top: 2rem; line-height: 1.6;
}
.disclaimer strong { color: #fbbf24; }

/* ── Error ── */
.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 12px; padding: 1rem 1.3rem;
    color: #fca5a5; font-size: 0.9rem; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────
SPECIALIST_ICONS = {
    "Neurologist":        "🧠",
    "Cardiologist":       "❤️",
    "Dermatologist":      "🩹",
    "Gastroenterologist": "🫁",
    "Pulmonologist":      "🌬️",
    "Orthopedist":        "🦴",
    "Endocrinologist":    "⚗️",
    "Psychiatrist":       "🧘",
    "Urologist":          "💧",
    "Ophthalmologist":    "👁️",
    "ENT Specialist":     "👂",
    "Oncologist":         "🔬",
    "Gynecologist":       "🌸",
    "Pediatrician":       "👶",
    "General Physician":  "🩺",
}

EXAMPLES = [
    "I have severe headaches with nausea and sensitivity to light",
    "Chest pain radiating to my left arm with shortness of breath",
    "Itchy red rashes spreading across my body for several weeks",
    "Burning sensation when urinating with lower back pain",
    "Feeling hopeless, sad and unable to sleep for months",
]

# ── Session state init ──────────────────────────────────────────
if "symptom_text" not in st.session_state:
    st.session_state.symptom_text   = ""
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

# ── Load model (cached, lazy) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mapper():
    try:
        mapper = SpecialistMapper()
        return mapper, None
    except Exception as e:
        return None, str(e)

# ── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🏥 AI-Powered Triage</div>
    <h1>Describe your symptoms.<br>Find your <em>specialist</em>.</h1>
    <p>Powered by semantic AI — not keyword search. Describe what
    you feel in plain language and get matched to the right doctor.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Stats ───────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-pill">📊 <b>3,831</b> symptom patterns</div>
    <div class="stat-pill">🏥 <b>15</b> specialties</div>
    <div class="stat-pill">⚡ <b>&lt;100ms</b> inference</div>
    <div class="stat-pill">🎯 <b>84.5%</b> top-3 accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── Example buttons ─────────────────────────────────────────────
st.markdown('<div class="examples-label">Try an example</div>',
            unsafe_allow_html=True)

ex_cols = st.columns(len(EXAMPLES))
for i, (col, example) in enumerate(zip(ex_cols, EXAMPLES)):
    with col:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.symptom_text   = example
            st.session_state.run_prediction = True
            st.rerun()

# ── Text area ───────────────────────────────────────────────────
symptom_input = st.text_area(
    "Describe your symptoms",
    value=st.session_state.symptom_text,
    placeholder="e.g. I have been experiencing sharp chest pain when I breathe deeply, "
                "along with dizziness and fatigue for the past three days...",
    height=130,
    label_visibility="collapsed",
    key="text_area_input",
)

# Sync manual typing back to session state
if symptom_input != st.session_state.symptom_text:
    st.session_state.symptom_text   = symptom_input
    st.session_state.run_prediction = False

st.markdown("<br>", unsafe_allow_html=True)

# ── Find button ─────────────────────────────────────────────────
if st.button("🔍  Find My Specialist", use_container_width=True):
    st.session_state.run_prediction = True

# ── Prediction ──────────────────────────────────────────────────
if st.session_state.run_prediction:

    if not st.session_state.symptom_text.strip():
        st.markdown(
            '<div class="error-box">⚠️ Please describe your symptoms before searching.</div>',
            unsafe_allow_html=True,
        )
        st.session_state.run_prediction = False

    else:
        with st.spinner("🔍 Analyzing your symptoms with semantic AI..."):
            mapper, err = load_mapper()

        if err:
            st.markdown(
                f'<div class="error-box">❌ Model failed to load: {err}<br><br>'
                f'Make sure model files exist in <code>models/embeddings/</code></div>',
                unsafe_allow_html=True,
            )
            st.session_state.run_prediction = False
            st.stop()

        with st.spinner("⚡ Finding best specialists..."):
            try:
                prediction = mapper.predict(st.session_state.symptom_text)
                results    = prediction["results"]
                inf_ms     = prediction["inference_ms"]
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Prediction error: {e}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.run_prediction = False
                st.stop()

        # ── Results ─────────────────────────────────────────────
        st.markdown(
            f'<div class="results-header">Top Recommendations '
            f'<span style="font-size:0.9rem;color:#4a5568;'
            f'font-family:\'DM Sans\',sans-serif;">· {inf_ms}ms</span></div>',
            unsafe_allow_html=True,
        )

        card_cls = ["card-rank1", "card-rank2", "card-rank3"]
        bar_cls  = ["bar-1",      "bar-2",      "bar-3"     ]
        conf_cls = ["conf-num-1", "conf-num-2", "conf-num-3"]
        rank_lbl = ["Best Match", "2nd Match",  "3rd Match" ]

        for i, result in enumerate(results):
            specialist = result["specialist"]
            confidence = result["confidence"]
            matched    = result["matched_symptoms"][:90]
            desc       = result["description"]
            icon       = SPECIALIST_ICONS.get(specialist, "🏥")
            bar_w      = min(int(confidence), 100)

            st.markdown(f"""
<div class="card {card_cls[i]}">
    <div class="card-header">
        <div>
            <div class="card-rank">{rank_lbl[i]}</div>
            <div class="card-title">{specialist}</div>
        </div>
        <div style="display:flex;align-items:center;gap:1rem;">
            <div class="{conf_cls[i]}">{confidence}%</div>
            <div class="card-icon">{icon}</div>
        </div>
    </div>
    <div class="confidence-label">Confidence Score</div>
    <div class="confidence-bar-bg">
        <div class="{bar_cls[i]}" style="width:{bar_w}%"></div>
    </div>
    <div class="card-match">
        <span>Matched pattern:</span> {matched}…
    </div>
    <div class="card-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Medical Disclaimer</strong><br>
    This tool is for informational purposes only and does <strong>not</strong>
    constitute a medical diagnosis. Always consult a qualified, licensed physician
    for proper medical evaluation and treatment.
</div>
""", unsafe_allow_html=True)

        st.session_state.run_prediction = False