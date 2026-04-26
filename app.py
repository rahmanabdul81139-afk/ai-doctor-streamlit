import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Doctor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – clean medical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #f5f7fa;
    --card:      #ffffff;
    --accent:    #1d6fa4;
    --accent2:   #0e9e82;
    --warn:      #e05c2e;
    --text:      #1a1e2e;
    --muted:     #6b7280;
    --border:    #dde3ec;
    --shadow:    0 2px 16px rgba(0,0,0,0.07);
    --radius:    14px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem; max-width: 900px; margin: auto; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 540px;
    margin: 0 auto;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.8rem 0 0.6rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

/* ── Disease result card ── */
.disease-card {
    background: linear-gradient(135deg, #e8f4fd 0%, #f0fbf7 100%);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
.disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: var(--accent);
}
.confidence-bar-bg {
    background: #dde9f5;
    border-radius: 99px;
    height: 8px;
    margin-top: 6px;
}
.confidence-bar {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 99px;
    height: 8px;
    transition: width 0.8s ease;
}

/* ── Test pills ── */
.tests-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.test-pill {
    background: #e8f4fd;
    color: var(--accent);
    border: 1px solid #b8d9f0;
    border-radius: 99px;
    padding: 0.3rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Alt diagnoses ── */
.alt-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.alt-name { font-weight: 500; font-size: 0.95rem; }
.alt-score {
    font-size: 0.8rem;
    color: var(--muted);
    background: #f0f4fa;
    padding: 2px 8px;
    border-radius: 99px;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #fff8f0;
    border: 1px solid #f5d5b0;
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    font-size: 0.82rem;
    color: #8a5320;
    margin-top: 2rem;
    line-height: 1.6;
}

/* ── History item ── */
.hist-item {
    font-size: 0.85rem;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: 0.4rem 0;
}

/* ── Symptom chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 0.8rem; }
.chip {
    background: #f0f4fa;
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.25rem 0.7rem;
    font-size: 0.8rem;
    color: var(--accent);
    cursor: default;
}

/* ── Stagger animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.45s ease both; }
.fade-up-2 { animation: fadeUp 0.45s 0.1s ease both; }
.fade-up-3 { animation: fadeUp 0.45s 0.2s ease both; }

/* ── Streamlit input overrides ── */
textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2.2rem !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #155d8c !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading medical knowledge base…")
def load_model():
    df = pd.read_excel("symptoms based medical test recommendations (2).xlsx")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "questions": "symptoms",
        "recommending medical tests": "test",
        "disease ": "disease",
    })
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["symptoms", "disease", "test"])
    df["symptoms"] = df["symptoms"].str.lower()
    df["disease"]  = df["disease"].str.strip()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    X = vectorizer.fit_transform(df["symptoms"])
    return df, vectorizer, X


df, vectorizer, X = load_model()

# Pre-compute unique diseases list
ALL_DISEASES = sorted(df["disease"].unique())


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s,]", " ", text)
    return text


def diagnose(user_text: str, top_n: int = 5, min_score: float = 0.05):
    """Return list of (disease, score, tests) sorted by score desc."""
    cleaned = preprocess(user_text)
    q_vec   = vectorizer.transform([cleaned])
    sims    = cosine_similarity(q_vec, X).flatten()

    # Aggregate by disease (weighted vote: take max sim per disease)
    disease_scores: dict[str, float] = {}
    disease_tests:  dict[str, list]  = {}

    for idx in np.argsort(sims)[::-1][:50]:   # consider top-50 rows
        score   = float(sims[idx])
        disease = df["disease"].iloc[idx]
        tests   = [t.strip() for t in str(df["test"].iloc[idx]).split(",")]

        if disease not in disease_scores or score > disease_scores[disease]:
            disease_scores[disease] = score
            disease_tests[disease]  = tests

    results = [
        (d, s, disease_tests[d])
        for d, s in sorted(disease_scores.items(), key=lambda x: -x[1])
        if s >= min_score
    ]
    return results[:top_n]


def extract_keywords(text: str) -> list[str]:
    """Pull likely symptom words to show as chips."""
    stopwords = {"i", "have", "a", "the", "am", "and", "my", "some", "is",
                 "are", "with", "in", "of", "to", "do", "what", "should",
                 "me", "it", "for", "can", "get", "any", "also", "been",
                 "feeling", "experiencing", "seems", "like", "little", "bit",
                 "very", "really", "also", "or", "but", "not", "no"}
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in dict.fromkeys(words) if w not in stopwords and len(w) > 2]


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of (query, top_disease)


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero fade-up">
  <h1>🩺 AI Doctor</h1>
  <p>Describe your symptoms in plain language — get an instant disease assessment &amp; recommended tests.</p>
</div>
""", unsafe_allow_html=True)

# ── Input area ──────────────────────────────
st.markdown('<p class="section-title">Describe your symptoms</p>', unsafe_allow_html=True)

EXAMPLES = [
    "fever, cough, body aches, headache and fatigue",
    "chest pain, shortness of breath, dizziness",
    "frequent urination, excessive thirst, blurred vision",
    "persistent cough with blood, night sweats, weight loss",
]

col_ex, _ = st.columns([3, 1])
with col_ex:
    example_choice = st.selectbox(
        "💡 Try an example",
        ["— pick an example or type below —"] + EXAMPLES,
        label_visibility="collapsed",
    )

default_text = "" if example_choice.startswith("—") else example_choice
user_input = st.text_area(
    "Symptoms",
    value=default_text,
    placeholder="e.g. I've been having fever, chills, severe headache and joint pain for 3 days…",
    height=110,
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([2, 1])
with col_btn:
    analyse_clicked = st.button("🔍  Analyse Symptoms")
with col_clear:
    if st.button("↺  Clear"):
        st.rerun()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if analyse_clicked:
    if not user_input.strip():
        st.warning("⚠️  Please enter your symptoms before analysing.")
    else:
        results = diagnose(user_input)

        if not results:
            st.error("No matching conditions found. Try describing symptoms in more detail.")
        else:
            top_disease, top_score, top_tests = results[0]
            confidence_pct = min(int(top_score * 180), 97)   # scale to ~0-97%

            # Save to history
            st.session_state.history.insert(
                0, {"query": user_input[:60], "disease": top_disease}
            )
            if len(st.session_state.history) > 8:
                st.session_state.history.pop()

            # ── Primary result ───────────────────────
            st.markdown('<p class="section-title">Primary Assessment</p>', unsafe_allow_html=True)
            keywords = extract_keywords(user_input)
            chips_html = "".join(f'<span class="chip">{k}</span>' for k in keywords[:12])

            st.markdown(f"""
            <div class="disease-card fade-up">
              <div style="margin-bottom:0.3rem;font-size:0.75rem;color:#6b7280;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">Most Likely Condition</div>
              <div class="disease-name">{top_disease}</div>
              <div style="margin-top:0.7rem;font-size:0.8rem;color:#6b7280;">Confidence</div>
              <div class="confidence-bar-bg"><div class="confidence-bar" style="width:{confidence_pct}%"></div></div>
              <div style="font-size:0.78rem;color:#6b7280;margin-top:3px;">{confidence_pct}% match based on symptom analysis</div>
              <div style="margin-top:1rem;font-size:0.8rem;color:#6b7280;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;">Detected Symptoms</div>
              <div class="chip-row">{chips_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Recommended tests ────────────────────
            st.markdown('<p class="section-title fade-up-2">Recommended Medical Tests</p>', unsafe_allow_html=True)
            pills_html = "".join(f'<span class="test-pill">🧪 {t}</span>' for t in top_tests if t)
            st.markdown(f"""
            <div class="card fade-up-2">
              <div class="tests-grid">{pills_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Alternate diagnoses ──────────────────
            if len(results) > 1:
                st.markdown('<p class="section-title fade-up-3">Other Possible Conditions</p>', unsafe_allow_html=True)
                for disease, score, tests in results[1:]:
                    alt_pct = min(int(score * 180), 97)
                    alt_tests_str = ", ".join(tests[:3])
                    st.markdown(f"""
                    <div class="alt-card fade-up-3">
                      <div>
                        <div class="alt-name">{disease}</div>
                        <div style="font-size:0.75rem;color:#9ca3af;margin-top:2px;">Tests: {alt_tests_str}{"…" if len(tests)>3 else ""}</div>
                      </div>
                      <div class="alt-score">{alt_pct}% match</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Disclaimer ───────────────────────────
            st.markdown("""
            <div class="disclaimer fade-up-3">
              ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational and educational purposes only. 
              It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified 
              healthcare professional for medical concerns. Do not ignore or delay professional medical 
              advice based on information from this app.
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR – history & disease explorer
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Session History")
    if st.session_state.history:
        for item in st.session_state.history:
            st.markdown(f"""
            <div class="hist-item">
              <strong>{item['disease']}</strong><br>
              <span style="font-size:0.75rem">"{item['query']}{"…" if len(item['query'])==60 else ""}"</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No analyses yet in this session.")

    st.markdown("---")
    st.markdown("### 🗂 Disease Coverage")
    st.caption(f"Knowledge base: **{len(df):,} symptom entries**, **{len(ALL_DISEASES)} diseases**")
    with st.expander("View all diseases"):
        for d in ALL_DISEASES:
            st.markdown(f"• {d}")
