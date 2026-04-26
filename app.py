import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Doctor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

:root {
    --bg:         #f4f6fb;
    --card:       #ffffff;
    --accent:     #1d6fa4;
    --accent2:    #0e9e82;
    --border:     #dde3ec;
    --text:       #1a1e2e;
    --muted:      #6b7280;
    --shadow:     0 2px 18px rgba(0,0,0,0.07);
    --radius:     14px;

    /* triage colours */
    --emergency:  #c0392b;
    --urgent:     #e67e22;
    --moderate:   #f1c40f;
    --routine:    #27ae60;

    --emergency-bg: #fdf0ef;
    --urgent-bg:    #fef6ec;
    --moderate-bg:  #fefde6;
    --routine-bg:   #edfbf4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 960px; margin: auto; }

/* ── Hero ── */
.hero { text-align:center; padding:2rem 1rem 1.2rem; }
.hero h1 { font-family:'DM Serif Display',serif; font-size:2.6rem; color:var(--accent); margin-bottom:.3rem; letter-spacing:-.5px; }
.hero p  { color:var(--muted); font-size:1rem; max-width:540px; margin:0 auto; }

/* ── Section label ── */
.sec { font-size:.72rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:1.6rem 0 .55rem; }

/* ── Generic card ── */
.card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:1.3rem 1.5rem; box-shadow:var(--shadow); margin-bottom:.9rem; }

/* ── Triage badge ── */
.triage-badge {
    display:inline-flex; align-items:center; gap:.45rem;
    font-weight:700; font-size:.82rem; letter-spacing:.06em; text-transform:uppercase;
    padding:.35rem .9rem; border-radius:99px; margin-bottom:.6rem;
}
.triage-EMERGENCY { background:var(--emergency-bg); color:var(--emergency); border:1.5px solid var(--emergency); }
.triage-URGENT    { background:var(--urgent-bg);    color:var(--urgent);    border:1.5px solid var(--urgent);    }
.triage-MODERATE  { background:var(--moderate-bg);  color:#b7860b;          border:1.5px solid var(--moderate);  }
.triage-ROUTINE   { background:var(--routine-bg);   color:var(--routine);   border:1.5px solid var(--routine);   }

.triage-banner {
    border-radius:var(--radius); padding:1.1rem 1.4rem; margin-bottom:.9rem;
    border-left:5px solid;
}
.triage-banner-EMERGENCY { background:var(--emergency-bg); border-color:var(--emergency); }
.triage-banner-URGENT    { background:var(--urgent-bg);    border-color:var(--urgent);    }
.triage-banner-MODERATE  { background:var(--moderate-bg);  border-color:var(--moderate);  }
.triage-banner-ROUTINE   { background:var(--routine-bg);   border-color:var(--routine);   }

/* ── Disease card ── */
.disease-card {
    background:linear-gradient(135deg,#e8f4fd,#f0fbf7);
    border-left:4px solid var(--accent); border-radius:var(--radius);
    padding:1.4rem 1.7rem; margin-bottom:.9rem;
}
.disease-name { font-family:'DM Serif Display',serif; font-size:1.65rem; color:var(--accent); }
.conf-bg { background:#dde9f5; border-radius:99px; height:8px; margin-top:5px; }
.conf-bar { background:linear-gradient(90deg,var(--accent),var(--accent2)); border-radius:99px; height:8px; }

/* ── Test pills ── */
.tests-wrap { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.5rem; }
.test-pill { background:#e8f4fd; color:var(--accent); border:1px solid #b8d9f0; border-radius:99px; padding:.28rem .8rem; font-size:.8rem; font-weight:500; }

/* ── XAI word highlights ── */
.xai-word { display:inline-block; padding:.15rem .45rem; border-radius:5px; font-weight:600; font-size:.88rem; margin:.1rem; }
.xai-high  { background:#d4edda; color:#1a6b30; }
.xai-med   { background:#fff3cd; color:#856404; }
.xai-low   { background:#f8d7da; color:#842029; }

/* ── Score bar row ── */
.score-row { display:flex; align-items:center; gap:.7rem; margin:.35rem 0; font-size:.82rem; }
.score-label { width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:500; }
.score-track { flex:1; background:#eef0f5; border-radius:99px; height:8px; }
.score-fill  { border-radius:99px; height:8px; background:var(--accent); }
.score-val   { width:38px; text-align:right; color:var(--muted); font-size:.75rem; }

/* ── Alt diagnosis ── */
.alt-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:.85rem 1.1rem; margin-bottom:.45rem; display:flex; justify-content:space-between; align-items:center; }
.alt-name { font-weight:500; font-size:.9rem; }
.alt-why  { font-size:.73rem; color:var(--muted); margin-top:2px; }
.alt-score { font-size:.78rem; color:var(--muted); background:#f0f4fa; padding:2px 8px; border-radius:99px; white-space:nowrap; }

/* ── Chip ── */
.chip-row { display:flex; flex-wrap:wrap; gap:.35rem; margin:.3rem 0 .7rem; }
.chip { background:#f0f4fa; border:1px solid var(--border); border-radius:99px; padding:.22rem .65rem; font-size:.78rem; color:var(--accent); }

/* ── HITL feedback ── */
.hitl-box { background:#f8faff; border:1px solid #c8d8f0; border-radius:var(--radius); padding:1.1rem 1.4rem; margin-top:.8rem; }
.hitl-title { font-weight:600; font-size:.9rem; margin-bottom:.5rem; }

/* ── Disclaimer ── */
.disclaimer { background:#fff8f0; border:1px solid #f5d5b0; border-radius:var(--radius); padding:1rem 1.4rem; font-size:.8rem; color:#8a5320; margin-top:1.6rem; line-height:1.6; }

/* ── History ── */
.hist-item { font-size:.82rem; color:var(--muted); border-bottom:1px solid var(--border); padding:.4rem 0; }

/* ── Doctor Mode ── */
.doctor-banner {
    background:linear-gradient(135deg,#0a2540,#0e4d7a);
    border-radius:var(--radius); padding:1.1rem 1.5rem;
    color:#fff; margin-bottom:1rem;
    display:flex; align-items:center; gap:1rem;
}
.doctor-banner-icon { font-size:2rem; }
.doctor-banner-title { font-family:'DM Serif Display',serif; font-size:1.2rem; margin-bottom:.1rem; }
.doctor-banner-sub   { font-size:.78rem; opacity:.75; }
.verified-badge {
    display:inline-flex; align-items:center; gap:.3rem;
    background:#0a2540; color:#4fc3f7;
    border:1.5px solid #4fc3f7; border-radius:99px;
    font-size:.72rem; font-weight:700; letter-spacing:.06em;
    padding:.2rem .65rem; text-transform:uppercase; margin-left:.5rem;
}
.doctor-stat-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:.9rem 1.1rem; text-align:center;
    box-shadow:var(--shadow);
}
.doctor-stat-val  { font-family:'DM Serif Display',serif; font-size:1.6rem; color:var(--accent); }
.doctor-stat-label{ font-size:.72rem; color:var(--muted); margin-top:.1rem; }
.doc-log-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:.5rem 0; border-bottom:1px solid var(--border); font-size:.8rem;
}
.doc-log-query   { color:var(--muted); font-size:.73rem; margin-top:2px; }
.doc-correct     { color:var(--accent2); font-weight:600; }
.doc-incorrect   { color:var(--emergency); font-weight:600; }

/* ── Animations ── */
@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.fu  { animation:fadeUp .4s ease both; }
.fu2 { animation:fadeUp .4s .1s ease both; }
.fu3 { animation:fadeUp .4s .2s ease both; }

/* ── Buttons ── */
.stButton>button {
    background:var(--accent) !important; color:#fff !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    font-size:.95rem !important; border:none !important;
    border-radius:10px !important; padding:.6rem 1.8rem !important;
    transition:background .2s,transform .1s !important; width:100%;
}
.stButton>button:hover { background:#155d8c !important; transform:translateY(-1px) !important; }
textarea { font-family:'DM Sans',sans-serif !important; font-size:.98rem !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEEDBACK_FILE  = "feedback_log.csv"
DOCTOR_PASSWORD = "medic2024"   # ← change this before deploying

# ── Triage rule map (disease → base level) ──────────────────────────────────
DISEASE_TRIAGE = {
    "Heart Attack (Myocardial Infarction)": "EMERGENCY",
    "Stroke":                               "EMERGENCY",
    "Ebola Virus Disease":                  "EMERGENCY",
    "Cholera":                              "EMERGENCY",
    "Brain Cancer":                         "EMERGENCY",
    "Leukemia":                             "EMERGENCY",
    "HIV/AIDS":                             "URGENT",
    "Tuberculosis (TB)":                    "URGENT",
    "Malaria":                              "URGENT",
    "COVID-19.":                            "URGENT",
    "Dengue Fever":                         "URGENT",
    "Lung Cancer":                          "URGENT",
    "Breast Cancer":                        "URGENT",
    "Coronary Artery Disease (CAD)":        "URGENT",
    "Arrhythmias (Irregular Heartbeat)":    "URGENT",
    "Hepatitis":                            "URGENT",
    "Prostate Cancer":                      "MODERATE",
    "Colorectal Cancer":                    "MODERATE",
    "Skin Cancer (Melanoma)":               "MODERATE",
    "Atherosclerosis":                      "MODERATE",
    "Interstitial Lung Disease":            "MODERATE",
    "Hypertension (High Blood Pressure)":   "MODERATE",
    "Asthma":                               "MODERATE",
    "Chronic Obstructive":                  "MODERATE",
    "Pulmonary Disease (COPD)":             "MODERATE",
    "Influenza (Flu)":                      "ROUTINE",
    "Diabetes":                             "ROUTINE",
    "Hypothyroidism (Underactive Thyroid)": "ROUTINE",
    "Hyperthyroidism (Overactive Thyroid)": "ROUTINE",
    "Addison's Disease":                    "ROUTINE",
}

# ── Emergency symptom keywords (bump triage up) ─────────────────────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
    "heart attack", "stroke", "unconscious", "fainting", "coughing blood",
    "vomiting blood", "severe bleeding", "paralysis", "sudden vision loss",
    "sudden numbness", "severe chest", "difficulty breathing", "crushing pain",
]

TRIAGE_META = {
    "EMERGENCY": {"icon": "🚨", "label": "Emergency",  "action": "Seek emergency care immediately — call 108 / go to ER now."},
    "URGENT":    {"icon": "⚠️",  "label": "Urgent",     "action": "See a doctor within 24 hours. Do not delay."},
    "MODERATE":  {"icon": "🔔", "label": "Moderate",   "action": "Schedule a doctor's appointment within a few days."},
    "ROUTINE":   {"icon": "✅",  "label": "Routine",    "action": "Monitor symptoms; consult a GP at your convenience."},
}

TRIAGE_ORDER = ["EMERGENCY", "URGENT", "MODERATE", "ROUTINE"]


# ─────────────────────────────────────────────
# DATA & MODEL  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading medical knowledge base…")
def load_model():
    df = pd.read_excel("symptoms based medical test recommendations (2).xlsx")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "questions":                   "symptoms",
        "recommending medical tests":  "test",
        "disease ":                    "disease",
    })
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["symptoms", "disease", "test"])
    df["symptoms"] = df["symptoms"].str.lower()
    df["disease"]  = df["disease"].str.strip()

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    X   = vec.fit_transform(df["symptoms"])

    # feature name list for XAI
    feature_names = vec.get_feature_names_out()
    return df, vec, X, feature_names


df, vec, X, feature_names = load_model()
ALL_DISEASES = sorted(df["disease"].unique())


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    return re.sub(r"[^a-z0-9\s,]", " ", text.lower().strip())


def get_triage(disease: str, confidence: float, user_text: str) -> str:
    """
    Combine three signals:
      1. Rule-based disease severity map
      2. Confidence score (high conf → stay / escalate; very low → demote)
      3. Emergency keyword detection in user text
    """
    base = DISEASE_TRIAGE.get(disease, "ROUTINE")
    level_idx = TRIAGE_ORDER.index(base)

    # Signal 3 – keyword detection (always escalate to EMERGENCY if hit)
    lower_text = user_text.lower()
    if any(kw in lower_text for kw in EMERGENCY_KEYWORDS):
        return "EMERGENCY"

    # Signal 2 – confidence modulation
    if confidence < 0.12 and level_idx < 2:   # low confidence: demote 1 step
        level_idx = min(level_idx + 1, 3)
    elif confidence >= 0.55 and level_idx > 0: # very high confidence: escalate 1 step
        level_idx = max(level_idx - 1, 0)

    return TRIAGE_ORDER[level_idx]


def diagnose(user_text: str, top_n: int = 5):
    cleaned = preprocess(user_text)
    q_vec   = vec.transform([cleaned])
    sims    = cosine_similarity(q_vec, X).flatten()

    disease_scores: dict[str, float] = {}
    disease_tests:  dict[str, list]  = {}
    disease_row:    dict[str, int]   = {}

    for idx in np.argsort(sims)[::-1][:80]:
        score   = float(sims[idx])
        disease = df["disease"].iloc[idx]
        tests   = [t.strip() for t in str(df["test"].iloc[idx]).split(",")]
        if disease not in disease_scores or score > disease_scores[disease]:
            disease_scores[disease] = score
            disease_tests[disease]  = tests
            disease_row[disease]    = idx

    results = sorted(disease_scores.items(), key=lambda x: -x[1])
    return [
        {
            "disease":    d,
            "score":      s,
            "tests":      disease_tests[d],
            "row_idx":    disease_row[d],
        }
        for d, s in results if s >= 0.04
    ][:top_n]


def xai_explain(user_text: str, top_result: dict) -> dict:
    """
    Returns:
      - driving_words: list of (word, weight_bin) — words in user query with high TF-IDF weight
      - score_breakdown: list of (disease, score) for chart
      - ruled_lower: why alt diseases scored less
      - plain_summary: plain English
    """
    cleaned = preprocess(user_text)
    q_vec   = vec.transform([cleaned])

    # ── Driving words ──────────────────────────────────────────────────
    q_arr   = q_vec.toarray()[0]
    nonzero = [(feature_names[i], q_arr[i]) for i in np.where(q_arr > 0)[0]]
    nonzero.sort(key=lambda x: -x[1])
    max_w   = nonzero[0][1] if nonzero else 1.0

    def weight_bin(w):
        r = w / max_w
        if r >= 0.65:  return "high"
        if r >= 0.30:  return "med"
        return "low"

    driving_words = [(word, weight_bin(w)) for word, w in nonzero[:14]]

    # ── Ruled-lower reasons ────────────────────────────────────────────
    results    = diagnose(user_text, top_n=5)
    top_score  = results[0]["score"] if results else 1.0
    ruled_lower = []
    for r in results[1:]:
        gap  = top_score - r["score"]
        pct  = int(gap / top_score * 100) if top_score > 0 else 0
        ruled_lower.append({
            "disease": r["disease"],
            "score":   r["score"],
            "reason":  f"Scored {pct}% lower — fewer overlapping symptom terms matched",
        })

    # ── Plain summary ──────────────────────────────────────────────────
    top_words = [w for w, _ in driving_words[:4]]
    plain = (
        f"The model identified **{top_result['disease']}** as the most likely condition "
        f"because your description strongly matched symptom patterns for this disease. "
        f"The key terms driving this decision were: **{', '.join(top_words)}**. "
        f"The match confidence was **{min(int(top_result['score']*180), 97)}%**, "
        f"based on how closely your words aligned with known symptom descriptions in the database."
    )

    return {
        "driving_words": driving_words,
        "score_breakdown": [(r["disease"], r["score"]) for r in results],
        "ruled_lower": ruled_lower,
        "plain_summary": plain,
    }


def save_feedback(query: str, predicted: str, confirmed: bool,
                  correct_disease: str = "", doctor_verified: bool = False,
                  doctor_name: str = ""):
    exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "query", "predicted", "confirmed",
                             "correct_disease", "doctor_verified", "doctor_name"])
        writer.writerow([
            datetime.now().isoformat(),
            query[:200],
            predicted,
            confirmed,
            correct_disease,
            doctor_verified,
            doctor_name,
        ])


def load_feedback_stats():
    if not os.path.exists(FEEDBACK_FILE):
        return None
    try:
        fb = pd.read_csv(FEEDBACK_FILE)
        # back-fill columns added later
        for col in ["doctor_verified", "doctor_name", "correct_disease"]:
            if col not in fb.columns:
                fb[col] = "" if col == "doctor_name" else False

        total   = len(fb)
        correct = int(fb["confirmed"].sum())
        acc     = round(correct / total * 100, 1) if total else 0

        doc_fb  = fb[fb["doctor_verified"] == True]
        doc_total   = len(doc_fb)
        doc_correct = int(doc_fb["confirmed"].sum()) if doc_total else 0
        doc_acc     = round(doc_correct / doc_total * 100, 1) if doc_total else 0

        return {
            "total": total, "correct": correct, "accuracy": acc,
            "doc_total": doc_total, "doc_correct": doc_correct, "doc_acc": doc_acc,
            "df": fb,
        }
    except Exception:
        return None


def extract_chips(text: str) -> list[str]:
    stop = {"i","have","a","the","am","and","my","some","is","are","with","in",
            "of","to","do","what","should","me","it","for","can","get","any",
            "also","been","feeling","experiencing","seems","like","little","bit",
            "very","really","or","but","not","no","since","past","days","day",
            "been","having","getting","feel","im","ive"}
    return [w for w in dict.fromkeys(re.findall(r"[a-zA-Z]+", text.lower()))
            if w not in stop and len(w) > 2][:12]


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("history",       []),
    ("results",       None),
    ("user_input",    ""),
    ("xai_data",      None),
    ("feedback_done", False),
    ("doctor_mode",   False),
    ("doctor_name",   ""),
    ("active_tab",    "Patient"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero fu">
  <h1>🩺 AI Doctor</h1>
  <p>Describe your symptoms — get triage priority, AI diagnosis with explanations, and recommended tests.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS  (Patient | 🔒 Doctor — hidden in plain sight)
# ─────────────────────────────────────────────
tab_patient, tab_doctor = st.tabs(["🧑‍💼  Patient View", "🔒  Doctor Portal"])


# ══════════════════════════════════════════════
# DOCTOR PORTAL TAB
# ══════════════════════════════════════════════
with tab_doctor:

    if not st.session_state.doctor_mode:
        # ── Login form ────────────────────────
        st.markdown('<p class="sec">Doctor Access — Enter Credentials</p>', unsafe_allow_html=True)
        with st.form("doctor_login"):
            d_name = st.text_input("Your Name / ID", placeholder="Dr. Ananya Sharma")
            d_pass = st.text_input("Access Code", type="password", placeholder="••••••••")
            login  = st.form_submit_button("🔓  Unlock Doctor Mode")
        if login:
            if d_pass == DOCTOR_PASSWORD and d_name.strip():
                st.session_state.doctor_mode = True
                st.session_state.doctor_name = d_name.strip()
                st.rerun()
            elif not d_name.strip():
                st.error("Please enter your name.")
            else:
                st.error("❌ Incorrect access code.")

    else:
        # ── Doctor is authenticated ───────────
        doc = st.session_state.doctor_name
        st.markdown(f"""
        <div class="doctor-banner fu">
          <div class="doctor-banner-icon">👨‍⚕️</div>
          <div>
            <div class="doctor-banner-title">Welcome, {doc}</div>
            <div class="doctor-banner-sub">Doctor Verified Mode is active — your feedback is weighted separately and marked with a ✦ verified badge.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Doctor feedback on current diagnosis ──
        results_now = st.session_state.results
        q_now       = st.session_state.user_input

        if results_now:
            top_now = results_now[0]
            st.markdown('<p class="sec">✦ Verify Current Diagnosis</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="hitl-box">
              <div class="hitl-title">
                Predicted: <em>{top_now['disease']}</em>
                <span class="verified-badge">✦ Doctor Review</span>
              </div>
              <div style="font-size:.78rem;color:var(--muted);">
                Patient query: "{q_now[:120]}{"…" if len(q_now)>120 else ""}"
              </div>
            </div>
            """, unsafe_allow_html=True)

            doc_fb_key = "doc_feedback_done"
            if doc_fb_key not in st.session_state:
                st.session_state[doc_fb_key] = False

            if not st.session_state[doc_fb_key]:
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("✅  Clinically Correct", key="doc_yes"):
                        save_feedback(q_now, top_now["disease"], True,
                                      doctor_verified=True, doctor_name=doc)
                        st.session_state[doc_fb_key] = True
                        st.rerun()
                with dc2:
                    if st.button("❌  Clinically Incorrect", key="doc_no"):
                        st.session_state[doc_fb_key] = "deny"
                        st.rerun()

                if st.session_state[doc_fb_key] == "deny":
                    doc_correct = st.selectbox(
                        "Correct diagnosis", ["— select —"] + ALL_DISEASES, key="doc_correct_sel"
                    )
                    doc_notes = st.text_area(
                        "Clinical notes (optional)",
                        placeholder="e.g. Patient presentation more consistent with dengue given thrombocytopenia…",
                        height=80, key="doc_notes"
                    )
                    if st.button("Submit Doctor Correction", key="doc_submit") and doc_correct != "— select —":
                        save_feedback(q_now, top_now["disease"], False, doc_correct,
                                      doctor_verified=True, doctor_name=doc)
                        st.session_state[doc_fb_key] = True
                        st.rerun()
            else:
                st.success("✦ Doctor-verified feedback recorded. Thank you.")
        else:
            st.info("Run a patient symptom analysis first (Patient View tab), then return here to verify.")

        st.markdown("---")

        # ── Doctor-only stats dashboard ───────
        st.markdown('<p class="sec">✦ Doctor-Verified Accuracy Dashboard</p>', unsafe_allow_html=True)
        stats = load_feedback_stats()
        if stats and stats["doc_total"] > 0:
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.markdown(f'<div class="doctor-stat-card"><div class="doctor-stat-val">{stats["doc_total"]}</div><div class="doctor-stat-label">Doctor Reviews</div></div>', unsafe_allow_html=True)
            sc2.markdown(f'<div class="doctor-stat-card"><div class="doctor-stat-val">{stats["doc_acc"]}%</div><div class="doctor-stat-label">Clinical Accuracy</div></div>', unsafe_allow_html=True)
            sc3.markdown(f'<div class="doctor-stat-card"><div class="doctor-stat-val">{stats["doc_correct"]}</div><div class="doctor-stat-label">Confirmed ✓</div></div>', unsafe_allow_html=True)
            sc4.markdown(f'<div class="doctor-stat-card"><div class="doctor-stat-val">{stats["doc_total"]-stats["doc_correct"]}</div><div class="doctor-stat-label">Corrected ✗</div></div>', unsafe_allow_html=True)

            st.markdown("**vs. General User Accuracy**")
            gen_only_total   = stats["total"] - stats["doc_total"]
            gen_only_correct = stats["correct"] - stats["doc_correct"]
            gen_acc = round(gen_only_correct / gen_only_total * 100, 1) if gen_only_total else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="doctor-stat-card">
                  <div class="doctor-stat-val" style="color:var(--accent2);">{stats['doc_acc']}%</div>
                  <div class="doctor-stat-label">✦ Doctor-Verified Accuracy</div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="doctor-stat-card">
                  <div class="doctor-stat-val" style="color:var(--muted);">{gen_acc}%</div>
                  <div class="doctor-stat-label">👤 General User Accuracy</div>
                </div>""", unsafe_allow_html=True)

            # Recent doctor log
            st.markdown("**Recent Doctor Feedback Log**")
            doc_df = stats["df"][stats["df"]["doctor_verified"] == True].tail(10).iloc[::-1]
            for _, row in doc_df.iterrows():
                badge = '<span class="doc-correct">✓ Correct</span>' if row["confirmed"] else f'<span class="doc-incorrect">✗ → {row.get("correct_disease","?")}</span>'
                st.markdown(f"""
                <div class="doc-log-row">
                  <div>
                    <div><strong>{row['predicted']}</strong> {badge}
                      <span class="verified-badge">✦ {row.get('doctor_name','Dr.')}</span>
                    </div>
                    <div class="doc-log-query">"{str(row['query'])[:90]}…"</div>
                  </div>
                  <div style="font-size:.72rem;color:var(--muted);white-space:nowrap;margin-left:.5rem;">
                    {str(row['timestamp'])[:16]}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No doctor-verified feedback yet. Use the form above after running an analysis.")

        st.markdown("---")
        if st.button("🔒  Log Out of Doctor Mode"):
            st.session_state.doctor_mode = False
            st.session_state.doctor_name = ""
            if "doc_feedback_done" in st.session_state:
                del st.session_state["doc_feedback_done"]
            st.rerun()


# ══════════════════════════════════════════════
# PATIENT VIEW TAB
# ══════════════════════════════════════════════
with tab_patient:

    st.markdown('<p class="sec">Describe your symptoms</p>', unsafe_allow_html=True)

    EXAMPLES = [
        "fever, cough, body aches, headache and fatigue",
        "chest pain, shortness of breath and dizziness",
        "frequent urination, excessive thirst and blurred vision",
        "persistent cough with blood, night sweats, weight loss",
        "sudden severe headache, facial drooping, arm weakness",
    ]

    example = st.selectbox("💡 Try an example", ["— or type your own below —"] + EXAMPLES,
                           label_visibility="collapsed")
    default_txt = "" if example.startswith("—") else example

    user_input = st.text_area(
        "Symptoms",
        value=default_txt,
        placeholder="e.g. I've had a fever, chills, severe headache and joint pain for 3 days…",
        height=115, label_visibility="collapsed",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        analyse = st.button("🔍  Analyse Symptoms")
    with c2:
        if st.button("↺  Clear"):
            st.session_state.results       = None
            st.session_state.xai_data      = None
            st.session_state.feedback_done = False
            st.rerun()

    # ── Run Analysis ──────────────────────────
    if analyse:
        if not user_input.strip():
            st.warning("⚠️  Please enter your symptoms before analysing.")
        else:
            results  = diagnose(user_input)
            xai_data = xai_explain(user_input, results[0]) if results else None
            st.session_state.results       = results
            st.session_state.xai_data      = xai_data
            st.session_state.user_input    = user_input
            st.session_state.feedback_done = False
            if "doc_feedback_done" in st.session_state:
                del st.session_state["doc_feedback_done"]
            if results:
                st.session_state.history.insert(0, {
                    "query":   user_input[:60],
                    "disease": results[0]["disease"],
                    "triage":  get_triage(results[0]["disease"],
                                          results[0]["score"], user_input),
                })
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop()

    # ── Display Results ───────────────────────
    results  = st.session_state.results
    xai_data = st.session_state.xai_data
    q_text   = st.session_state.user_input

    if results:
        top   = results[0]
        score = top["score"]
        conf  = min(int(score * 180), 97)
        triage_level = get_triage(top["disease"], score, q_text)
        tmeta        = TRIAGE_META[triage_level]

        # ── Doctor-verified banner (if logged in) ──
        if st.session_state.doctor_mode:
            st.markdown(f"""
            <div style="background:#e8f5e9;border:1.5px solid #4fc3f7;border-radius:10px;
                        padding:.6rem 1rem;margin-bottom:.7rem;font-size:.82rem;color:#0a2540;">
              ✦ <strong>Doctor Mode Active</strong> — Switch to the
              <em>Doctor Portal</em> tab to submit verified clinical feedback for this case.
              Logged in as: <strong>{st.session_state.doctor_name}</strong>
            </div>
            """, unsafe_allow_html=True)

        # ══════════════════════════════════════
        # 1. TRIAGE BANNER
        # ══════════════════════════════════════
        st.markdown('<p class="sec fu">① Triage Priority</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="triage-banner triage-banner-{triage_level} fu">
          <div class="triage-badge triage-{triage_level}">{tmeta['icon']} {tmeta['label']}</div>
          <div style="font-size:.9rem;font-weight:500;margin-bottom:.2rem;">{tmeta['action']}</div>
          <div style="font-size:.75rem;color:var(--muted);">
            Based on: disease severity classification · symptom keyword detection · AI confidence score ({conf}%)
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════
        # 2. PRIMARY DIAGNOSIS
        # ══════════════════════════════════════
        st.markdown('<p class="sec fu">② Primary Diagnosis</p>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="chip">{w}</span>' for w in extract_chips(q_text))
        st.markdown(f"""
        <div class="disease-card fu">
          <div style="font-size:.7rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.2rem;">Most Likely Condition</div>
          <div class="disease-name">{top['disease']}</div>
          <div style="margin-top:.7rem;font-size:.78rem;color:var(--muted);">Confidence</div>
          <div class="conf-bg"><div class="conf-bar" style="width:{conf}%"></div></div>
          <div style="font-size:.75rem;color:var(--muted);margin-top:3px;">{conf}% match</div>
          <div style="margin-top:.9rem;font-size:.75rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em;">Detected Symptom Keywords</div>
          <div class="chip-row">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)

        pills = "".join(f'<span class="test-pill">🧪 {t}</span>' for t in top["tests"] if t)
        st.markdown(f"""
        <div class="card fu2">
          <div class="sec" style="margin:0 0 .5rem;">Recommended Medical Tests</div>
          <div class="tests-wrap">{pills}</div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════
        # 3. EXPLAINABLE AI
        # ══════════════════════════════════════
        st.markdown('<p class="sec fu2">③ Why this diagnosis? (Explainable AI)</p>', unsafe_allow_html=True)

        with st.expander("🧠  View AI Explanation", expanded=True):
            st.markdown("**📝 Plain-English Reasoning**")
            st.info(xai_data["plain_summary"])

            st.markdown("**🔑 Symptom Words That Drove the Diagnosis**")
            bin_colors = {"high": "xai-high", "med": "xai-med", "low": "xai-low"}
            bin_labels = {"high": "strongly matched", "med": "partially matched", "low": "weakly matched"}
            words_html = " ".join(
                f'<span class="xai-word {bin_colors[b]}" title="{bin_labels[b]}">{w}</span>'
                for w, b in xai_data["driving_words"]
            )
            st.markdown(f"""
            <div class="card" style="margin-bottom:.6rem;">
              <div style="margin-bottom:.4rem;font-size:.75rem;color:var(--muted);">
                <span style="background:#d4edda;padding:1px 6px;border-radius:4px;color:#1a6b30;font-size:.72rem;">■ Strong</span>&nbsp;
                <span style="background:#fff3cd;padding:1px 6px;border-radius:4px;color:#856404;font-size:.72rem;">■ Partial</span>&nbsp;
                <span style="background:#f8d7da;padding:1px 6px;border-radius:4px;color:#842029;font-size:.72rem;">■ Weak</span>
              </div>
              {words_html}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**📊 Similarity Score Breakdown — Top Conditions**")
            max_s = xai_data["score_breakdown"][0][1] if xai_data["score_breakdown"] else 1.0
            bars  = ""
            for disease, s in xai_data["score_breakdown"]:
                pct    = int(s / max_s * 100) if max_s else 0
                disp_s = min(int(s * 180), 97)
                bars  += f"""
                <div class="score-row">
                  <div class="score-label">{disease}</div>
                  <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
                  <div class="score-val">{disp_s}%</div>
                </div>"""
            st.markdown(f'<div class="card" style="margin-bottom:.6rem;">{bars}</div>', unsafe_allow_html=True)

            if xai_data["ruled_lower"]:
                st.markdown("**🔍 Why Other Conditions Ranked Lower**")
                for item in xai_data["ruled_lower"]:
                    alt_conf = min(int(item["score"] * 180), 97)
                    st.markdown(f"""
                    <div class="alt-card">
                      <div>
                        <div class="alt-name">{item['disease']}</div>
                        <div class="alt-why">💡 {item['reason']}</div>
                      </div>
                      <div class="alt-score">{alt_conf}% match</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ══════════════════════════════════════
        # 4. OTHER POSSIBLE CONDITIONS
        # ══════════════════════════════════════
        if len(results) > 1:
            st.markdown('<p class="sec fu3">Other Possible Conditions</p>', unsafe_allow_html=True)
            for r in results[1:]:
                alt_conf   = min(int(r["score"] * 180), 97)
                alt_triage = get_triage(r["disease"], r["score"], q_text)
                alt_tmeta  = TRIAGE_META[alt_triage]
                alt_tests  = ", ".join(r["tests"][:3]) + ("…" if len(r["tests"]) > 3 else "")
                st.markdown(f"""
                <div class="alt-card fu3">
                  <div>
                    <div class="alt-name">{r['disease']}
                      <span class="triage-badge triage-{alt_triage}" style="font-size:.66rem;padding:.18rem .55rem;margin-left:.4rem;">{alt_tmeta['icon']} {alt_tmeta['label']}</span>
                    </div>
                    <div class="alt-why">Tests: {alt_tests}</div>
                  </div>
                  <div class="alt-score">{alt_conf}% match</div>
                </div>
                """, unsafe_allow_html=True)

        # ══════════════════════════════════════
        # 5. PATIENT FEEDBACK (HITL)
        # ══════════════════════════════════════
        st.markdown('<p class="sec fu3">④ Human-in-the-Loop — Your Feedback</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="hitl-box fu3">
          <div class="hitl-title">🩺 Was <em>{top['disease']}</em> the correct diagnosis?</div>
          <div style="font-size:.8rem;color:var(--muted);margin-bottom:.7rem;">
            Your feedback helps improve future predictions and is saved to <code>feedback_log.csv</code>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.feedback_done:
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                if st.button("👍  Yes, this is correct"):
                    save_feedback(q_text, top["disease"], True,
                                  doctor_verified=False, doctor_name="")
                    st.session_state.feedback_done = True
                    st.rerun()
            with fb_col2:
                if st.button("👎  No, this is incorrect"):
                    st.session_state.feedback_done = "deny"
                    st.rerun()

            if st.session_state.feedback_done == "deny":
                correct = st.selectbox("What is the correct disease?", ["— select —"] + ALL_DISEASES)
                if st.button("Submit Correction") and correct != "— select —":
                    save_feedback(q_text, top["disease"], False, correct,
                                  doctor_verified=False, doctor_name="")
                    st.session_state.feedback_done = True
                    st.rerun()
        else:
            st.success("✅ Thank you for your feedback! It has been recorded.")

        # ── Disclaimer ──
        st.markdown("""
        <div class="disclaimer fu3">
          ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational and educational purposes only.
          It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare
          professional. In an emergency, call <strong>108</strong> immediately.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 AI Doctor")
    st.caption(f"Knowledge base: **{len(df):,} entries · {len(ALL_DISEASES)} diseases**")
    if st.session_state.doctor_mode:
        st.markdown(f'<span class="verified-badge">✦ {st.session_state.doctor_name}</span>',
                    unsafe_allow_html=True)
    st.markdown("---")

    # ── Feedback stats ───────────────────────
    st.markdown("### 📊 Feedback Statistics")
    stats = load_feedback_stats()
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Total", stats["total"])
        c2.metric("Overall Acc.", f"{stats['accuracy']}%")
        if stats["doc_total"] > 0:
            c1.metric("✦ Doctor Reviews", stats["doc_total"])
            c2.metric("✦ Clinical Acc.", f"{stats['doc_acc']}%")
        if st.checkbox("Show feedback log"):
            st.dataframe(stats["df"].tail(10), use_container_width=True)
    else:
        st.caption("No feedback submitted yet.")

    st.markdown("---")

    # ── Triage legend ────────────────────────
    st.markdown("### 🚦 Triage Legend")
    for level, meta in TRIAGE_META.items():
        st.markdown(f"{meta['icon']} **{meta['label']}** — {meta['action']}")

    st.markdown("---")

    # ── Session history ──────────────────────
    st.markdown("### 🕑 Session History")
    if st.session_state.history:
        for item in st.session_state.history:
            tmeta = TRIAGE_META[item["triage"]]
            st.markdown(f"""
            <div class="hist-item">
              {tmeta['icon']} <strong>{item['disease']}</strong><br>
              <span style="font-size:.73rem">"{item['query']}{"…" if len(item['query'])==60 else ""}"</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No analyses yet.")

    st.markdown("---")

    # ── Disease explorer ─────────────────────
    st.markdown("### 🗂 Disease Coverage")
    with st.expander("View all 30 diseases"):
        for d in ALL_DISEASES:
            t = DISEASE_TRIAGE.get(d, "ROUTINE")
            st.markdown(f"{TRIAGE_META[t]['icon']} {d}")
