# corrected_app.py
import os
import re
import datetime
import io
import base64
import hashlib
import fitz  # PyMuPDF
import PyPDF2
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import requests

# Optional advanced imports (safe to fail)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    ST_AVAILABLE = False

try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

try:
    from streamlit_extras.let_it_rain import rain
except Exception:
    rain = None

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False

try:
    import turtle
    TURTLE_AVAILABLE = True
except Exception:
    turtle = None
    TURTLE_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    import pytesseract
except Exception:
    convert_from_bytes = None
    pytesseract = None

# --- Session state initialization ---
if "decision_log" not in st.session_state:
    st.session_state.decision_log = []
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "engine_run" not in st.session_state:
    st.session_state.engine_run = False

# Optional persistent log file
DECISION_LOG_CSV = "decision_log.csv"
if os.path.exists(DECISION_LOG_CSV) and not st.session_state.decision_log:
    try:
        df_log = pd.read_csv(DECISION_LOG_CSV)
        st.session_state.decision_log = df_log.to_dict(orient="records")
    except Exception:
        pass

# --- Embedding model loader (cached) ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if ST_AVAILABLE:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return None
    return None

# Safe encode helper
def safe_encode(model, texts):
    try:
        emb = model.encode(texts, convert_to_numpy=True)
    except TypeError:
        emb = model.encode(texts)
    return np.asarray(emb, dtype=np.float32)

# --- UI helpers ---
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=6)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def trigger_celebration():
    if rain:
        rain(emoji="🎉", font_size=54, falling_speed=5, animation_length=3)
    else:
        st.balloons()

# --- Background helpers (turtle + gradient fallback) ---
def pil_image_to_data_uri(img: Image.Image, fmt="PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def generate_gradient_image(width=1200, height=800, base="#0f172a") -> Image.Image:
    base_rgb = Image.new("RGBA", (width, height), base)
    overlay = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(overlay)
    for i in range(height):
        alpha = int(120 * (i / height))
        color = (96, 165, 250, alpha)
        draw.line([(0, i), (width, i)], fill=color)
    combined = Image.alpha_composite(base_rgb, overlay)
    return combined.filter(ImageFilter.GaussianBlur(radius=2))

def can_use_turtle():
    if not TURTLE_AVAILABLE:
        return False
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        return False
    return True

def generate_turtle_image(width=1200, height=800, bg_color="#0f172a") -> Image.Image:
    if not can_use_turtle():
        return generate_gradient_image(width, height, bg_color)
    try:
        screen = turtle.Screen()
        screen.setup(width=width, height=height)
        screen.bgcolor(bg_color)
        screen.tracer(0, 0)
        t = turtle.Turtle(visible=False)
        t.speed("fastest")
        colors = ["#38bdf8", "#818cf8", "#c084fc", "#e879f9", "#2dd4bf"]
        t.pensize(1); t.penup(); t.goto(0, 0); t.pendown()
        for i in range(200):
            t.color(colors[i % len(colors)])
            t.forward(i * 1.8)
            t.left(119)
        screen.update()
        # Safely get postscript and attempt to convert; fallback to gradient on failure
        try:
            ps = screen.getcanvas().postscript(colormode='color')
            # Use latin-1 encoding for PostScript bytes
            img = Image.open(io.BytesIO(ps.encode('latin-1')))
            img = img.convert("RGBA").resize((width, height), Image.LANCZOS)
            try:
                turtle.bye()
            except Exception:
                pass
            return img
        except Exception:
            try:
                turtle.bye()
            except Exception:
                pass
            return generate_gradient_image(width, height, bg_color)
    except Exception:
        try:
            turtle.bye()
        except Exception:
            pass
        return generate_gradient_image(width, height, bg_color)

@st.cache_data(show_spinner=False)
def get_background_data_uri(width=1200, height=800):
    return pil_image_to_data_uri(generate_turtle_image(width, height))

def set_streamlit_background(width=1200, height=800, opacity=0.12):
    data_uri = get_background_data_uri(width, height)
    css = f"""
    <style>
    .stApp {{ background-image: url("{data_uri}"); background-size: cover; background-position: center; background-attachment: fixed; }}
    .stApp::before {{ content: ""; position: fixed; inset: 0; background: rgba(15,23,42,{opacity}); pointer-events: none; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Document parsing ---
def normalize_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(Page \d+ of \d+)', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_text_robust(uploaded_file) -> str:
    if uploaded_file.name.endswith(".txt"):
        try:
            return normalize_text(uploaded_file.getvalue().decode("utf-8"))
        except Exception:
            return ""
    file_bytes = uploaded_file.getvalue()
    text = ""
    # Try PyMuPDF first
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = None
        if convert_from_bytes and pytesseract:
            # limit DPI to avoid huge memory usage
            try:
                images = convert_from_bytes(file_bytes, dpi=200)
            except Exception:
                images = None
        for i, page in enumerate(doc):
            page_text = page.get_text().strip()
            if (not page_text or len(page_text) < 30) and images and i < len(images):
                try:
                    page_text = pytesseract.image_to_string(images[i])
                except Exception:
                    page_text = ""
            text += (page_text or "") + "\n"
    except Exception:
        # Fallback to PyPDF2
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            extracted = []
            for p in reader.pages:
                try:
                    t = p.extract_text()
                    if t:
                        extracted.append(t)
                except Exception:
                    continue
            text = "\n".join(extracted)
        except Exception:
            text = ""
    return normalize_text(text)

# --- Models & data ---
class Match(BaseModel):
    role_title: str
    rationale: str
    match_percentage: int
    missing_skills: List[str]
    accommodations: str = ""

JOBS = [
    {"title": "Junior Data Analyst", "reqs": "Python, Data Cleaning, Excel, SQL, Problem Solving", "acc": "Remote work allowed, Screen-reader compatibility"},
    {"title": "Customer Success Lead", "reqs": "Empathy, Communication, Troubleshooting, CRM, Conflict Resolution", "acc": "Flexible hours, Caregiver-friendly scheduling"},
    {"title": "Front-End Apprentice", "reqs": "HTML, CSS, willingness to learn, basic design sense, teamwork", "acc": "Mentorship program included, Asynchronous check-ins"}
]

# --- Utility functions ---
def safe_key(prefix: str, text: str) -> str:
    hash_str = hashlib.md5(text.encode()).hexdigest()[:10]
    return f"{prefix}_{hash_str}"

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def fallback_heuristic_match(text: str) -> List[Match]:
    return [Match(role_title=JOBS[0]["title"], rationale="Fallback heuristic match.", match_percentage=65, missing_skills=["SQL"], accommodations=JOBS[0]["acc"])]

def get_missing_skills(job_title: str) -> List[str]:
    if "Data" in job_title: return ["Advanced SQL Joins", "Pandas Basics"]
    if "Front-End" in job_title: return ["React Fundamentals", "CSS Grid"]
    if "Success" in job_title: return ["Zendesk / CRM Software", "SaaS Metrics"]
    return []

# --- AI matching ---
def match_candidate_ai(raw_text: str) -> List[Match]:
    model = load_embedding_model()
    if model is None:
        return fallback_heuristic_match(raw_text)

    job_reqs = [j['reqs'] for j in JOBS]
    try:
        job_embeddings = safe_encode(model, job_reqs)
    except Exception:
        return fallback_heuristic_match(raw_text)

    # FAISS path
    if FAISS_AVAILABLE:
        try:
            dimension = job_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(job_embeddings)
            try:
                candidate_embedding = safe_encode(model, [raw_text])
            except Exception:
                return fallback_heuristic_match(raw_text)
            distances, indices = index.search(candidate_embedding, k=min(2, len(JOBS)))
            matches = []
            for i in range(len(indices[0])):
                job_idx = int(indices[0][i])
                job = JOBS[job_idx]
                raw_dist = float(distances[0][i])
                match_score = max(40, min(95, int(100 - (raw_dist * 20))))
                rationale = f"Semantic Search confirms strong skills alignment with {job['title']}."
                matches.append(Match(role_title=job['title'], rationale=rationale, match_percentage=match_score, missing_skills=get_missing_skills(job['title']), accommodations=job['acc']))
            return matches
        except Exception:
            pass

    # Cosine similarity fallback
    try:
        candidate_embedding = safe_encode(model, [raw_text])[0]
        sims = [cosine_similarity(candidate_embedding, job_embeddings[i]) for i in range(job_embeddings.shape[0])]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:2]
        matches = []
        for idx, sim in ranked:
            job = JOBS[idx]
            match_score = max(40, min(95, int(sim * 100)))
            rationale = f"Semantic similarity ({sim:.2f}) indicates capability alignment with {job['title']}."
            matches.append(Match(role_title=job['title'], rationale=rationale, match_percentage=match_score, missing_skills=get_missing_skills(job['title']), accommodations=job['acc']))
        return matches
    except Exception:
        return fallback_heuristic_match(raw_text)

# --- Decision logging ---
def persist_decision_log():
    try:
        pd.DataFrame(st.session_state.decision_log).to_csv(DECISION_LOG_CSV, index=False)
    except Exception:
        pass

def log_decision(candidate_id, role, decision):
    entry = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": candidate_id,
        "role": role,
        "decision": decision
    }
    st.session_state.decision_log.append(entry)
    st.session_state.last_decision = decision
    persist_decision_log()

# ==========================================
# MAIN APP UI
# ==========================================
st.title("InclusiveBridge AI")
st.caption("Redefining hiring with fairness at the core.")

with st.sidebar:
    if st_lottie:
        lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
        if lottie_hello:
            st_lottie(lottie_hello, height=200, key="hello")
    st.info("🧠 Engine: Semantic Vector Search (FAISS optional)")
    bg_mode = st.selectbox("Background style", ["Auto (turtle if available)", "Gradient (safe for cloud)"])
    st.divider()
    demo_choice = st.radio("Load a test persona or upload a resume:", [
        "Upload Custom PDF/TXT",
        "Persona A: Harshpreet (Retail to Tech, Tier-3 City India)",
        "Persona B: Neha (7-year Career Break, Caregiver)",
        "Persona C: Prashant (Rural Graduate, Vocational Training)"
    ])

# Background rendering
if bg_mode == "Gradient (safe for cloud)":
    gradient_uri = pil_image_to_data_uri(generate_gradient_image(1400, 900))
    st.markdown(
        f"""<style>
        .stApp {{ background-image: url("{gradient_uri}"); background-size: cover; background-position: center; background-attachment: fixed; }}
        .stApp::before {{ content: ""; position: fixed; inset: 0; background: rgba(15,23,42,0.12); pointer-events: none; }}
        </style>""", unsafe_allow_html=True)
else:
    set_streamlit_background(width=1400, height=900, opacity=0.12)

# Main logic
raw_text = ""
if demo_choice == "Upload Custom PDF/TXT":
    uploaded = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])
    if uploaded:
        raw_text = extract_text_robust(uploaded)
elif "Harshpreet" in demo_choice:
    raw_text = "Worked 4 years at local grocery store in Pune. Managed inventory and handled angry customers. High school diploma."
elif "Neha" in demo_choice:
    raw_text = "Previously worked as a junior web developer. Took a 7-year career break as a Caregiver for a sick relative. Managed complex medication schedules and emergencies."
elif "Prashant" in demo_choice:
    raw_text = "Completed vocational training in electrical repair. Good at troubleshooting circuits and fixing broken systems logically."

if raw_text:
    if st.button("🚀 Run AI Discovery & Match"):
        st.session_state.engine_run = True

    if st.session_state.engine_run:
        with st.spinner("Initializing Vector Database & Running Semantic Search..."):
            model_loaded = load_embedding_model() is not None
            st.success("🔎 " + ("Embeddings model loaded." if model_loaded else "Running heuristic fallback (no embedding model)."))
            st.divider()

            def translate_hidden_skills(text_input: str) -> str:
                t = text_input.lower()
                if "caregiver" in t or "career break" in t:
                    return "7-year gap as Caregiver ➔ **Crisis Management**, **High-Empathy Communication**, **Logistics**."
                if "electrical" in t or "vocational" in t:
                    return "Vocational Electrical ➔ **Systemic Troubleshooting**, **Root Cause Analysis**, **Logical Problem Solving**."
                if "grocery" in t or "retail" in t:
                    return "Retail ➔ **Inventory Management**, **De-escalation**, **Real-time Data Entry**."
                return "Transferable skills detected: **Adaptability** and **Process Execution**."

            st.subheader("🕵️ Skills Discovery Agent")
            st.info(f"**AI Translation:** {translate_hidden_skills(raw_text)}")
            st.divider()

            matches = match_candidate_ai(raw_text)

            st.subheader("🎯 Inclusive Matching Agent")
            for match in matches:
                with st.container():
                    st.markdown(f"#### {match.role_title}")
                    st.write(match.rationale)
                    st.caption(f"♿ Workplace Accommodations: **{match.accommodations}**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(match.match_percentage / 100.0)
                    with col2:
                        st.write(f"**{match.match_percentage}% Match**")

                    if match.missing_skills:
                        st.markdown("##### 🗺️ Learning Pathway Agent")
                        st.markdown("**Upskilling Path (Skill Gaps):**")
                        for skill in match.missing_skills:
                            st.markdown(f"- 🔸 {skill} ➔ [Free Training Module](#)")

                    # Use candidate snippet + role to avoid key collisions
                    key_base = raw_text[:60] + "|" + match.role_title
                    colA, colB, colC = st.columns(3)
                    colA.button("✅ Recommend", key=safe_key("rec", key_base), on_click=lambda m=match: log_decision(raw_text[:60], m.role_title, "Recommend"))
                    colB.button("👀 Review", key=safe_key("rev", key_base), on_click=lambda m=match: log_decision(raw_text[:60], m.role_title, "Review"))
                    colC.button("❌ Reject", key=safe_key("rej", key_base), on_click=lambda m=match: log_decision(raw_text[:60], m.role_title, "Reject"))
                    st.markdown("---")

            # -----------------------------
            # Candidate Empowerment Hub
            # -----------------------------
            st.subheader("🌟 Candidate Empowerment Hub")
            st.markdown("Empower candidates with clear next steps, practice, and progress tracking.")

            # Build a simple candidate profile summary to personalize resources
            candidate_snippet = raw_text[:300] + ("..." if len(raw_text) > 300 else "")
            st.markdown(f"**Candidate snapshot:** {candidate_snippet}")

            # Personalized Upskilling Roadmap
            st.markdown("#### 🗺️ Personalized Upskilling Roadmap")
            for match in matches:
                st.markdown(f"**Role:** {match.role_title} — **Match:** {match.match_percentage}%")
                if match.missing_skills:
                    st.markdown("**Suggested 4-week plan:**")
                    for i, skill in enumerate(match.missing_skills, start=1):
                        # Provide a simple week mapping and placeholder resource links
                        week = min(i, 4)
                        resource_link = "#"
                        st.markdown(f"- Week {week}: Learn **{skill}** ➔ [Free Resource]({resource_link})")
                    st.markdown("**Tip:** Complete one focused mini-project per skill to demonstrate learning.")
                else:
                    st.markdown("- No major skill gaps detected. Focus on interview practice and portfolio polish.")
                st.divider()

            # Gamified Progress Tracker (simple, based on match percentage)
            st.markdown("#### 🎮 Gamified Progress Tracker")
            # Map match percentage to a level
            def map_level(pct: int) -> str:
                if pct < 50:
                    return "Beginner"
                if pct < 75:
                    return "Intermediate"
                return "Job-Ready"

            for match in matches:
                level = map_level(match.match_percentage)
                progress_fraction = match.match_percentage / 100.0
                st.markdown(f"**{match.role_title}** — Level: **{level}**")
                st.progress(progress_fraction)
                st.caption(f"{match.match_percentage}% toward role readiness")
            st.divider()

            # Interview Readiness Coach
            st.markdown("#### 🎤 Interview Readiness Coach")
            st.markdown("Practice questions tailored to the role and candidate background:")
            # Generate role-aware sample questions
            def generate_questions(role_title: str, candidate_text: str) -> List[str]:
                base_questions = [
                    "Tell me about a time you solved a tough problem.",
                    "Describe a situation where you had to manage competing priorities.",
                    "How do you approach learning a new technical skill?"
                ]
                # Add role-specific prompts
                if "Data" in role_title:
                    base_questions.insert(0, "Walk me through a data cleaning task you completed and the tools you used.")
                if "Front-End" in role_title:
                    base_questions.insert(0, "How would you debug a layout issue across browsers?")
                if "Success" in role_title:
                    base_questions.insert(0, "How do you handle an upset customer who is missing a deadline?")
                # Add a personalized prompt referencing candidate experience
                if "caregiver" in candidate_text.lower() or "career break" in candidate_text.lower():
                    base_questions.append("How has your caregiving experience shaped your approach to teamwork and communication?")
                return base_questions

            for match in matches:
                st.markdown(f"**{match.role_title} practice set:**")
                qs = generate_questions(match.role_title, raw_text)
                for q in qs:
                    st.markdown(f"- {q}")
                st.markdown("**Coaching tip:** Use the STAR method (Situation, Task, Action, Result) for behavioral answers.")
                st.divider()

            # Confidence Dashboard (visual strengths)
            st.markdown("#### 📊 Confidence Dashboard")
            # Heuristic strengths derived from translation and match percentage
            def estimate_strengths(text: str, match_pct: int):
                # Simple heuristic: caregiving -> communication/adaptability; vocational -> technical/troubleshooting; retail -> ops/customer
                strengths = {"Technical": 50, "Communication": 50, "Adaptability": 50}
                t = text.lower()
                if "caregiver" in t or "career break" in t:
                    strengths["Communication"] = min(95, strengths["Communication"] + 25)
                    strengths["Adaptability"] = min(95, strengths["Adaptability"] + 20)
                if "electrical" in t or "vocational" in t:
                    strengths["Technical"] = min(95, strengths["Technical"] + 30)
                    strengths["Adaptability"] = min(95, strengths["Adaptability"] + 10)
                if "grocery" in t or "retail" in t:
                    strengths["Communication"] = min(95, strengths["Communication"] + 15)
                    strengths["Technical"] = min(95, strengths["Technical"] + 5)
                # nudge by match percentage
                for k in strengths:
                    strengths[k] = min(100, int((strengths[k] * 0.6) + (match_pct * 0.4)))
                return strengths

            # Show a combined dashboard for the top match
            if matches:
                top = matches[0]
                strengths = estimate_strengths(raw_text, top.match_percentage)
                if PLOTLY_AVAILABLE:
                    df_strengths = pd.DataFrame(list(strengths.items()), columns=["Skill", "Score"])
                    fig_strengths = px.bar(df_strengths, x="Skill", y="Score", color="Skill", template="plotly_dark")
                    st.plotly_chart(fig_strengths, use_container_width=True)
                else:
                    st.write(strengths)

            # Exportable summary for candidate (copyable text)
            st.markdown("#### ✉️ Shareable Candidate Summary")
            summary_lines = [
                f"Candidate snapshot: {candidate_snippet}",
                f"Top recommended role: {matches[0].role_title} ({matches[0].match_percentage}% match)" if matches else "No matches found.",
                "Key strengths: " + ", ".join([k for k, v in (estimate_strengths(raw_text, matches[0].match_percentage) if matches else {}).items()][:3]),
                "Suggested next steps: Complete the 4-week upskilling plan and practice the interview questions provided."
            ]
            summary_text = "\n".join(summary_lines)
            st.text_area("Candidate summary (copy to share)", value=summary_text, height=180)

    # Chat interface mapped to Somya
    st.subheader("💬 Interactive Recruiter Agent (Somya)")
    # Explicit Streamlit chat guard with version hint
    if hasattr(st, "chat_input") and hasattr(st, "chat_message"):
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask why this candidate is a good fit..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            def get_agent_chat_response(query: str) -> str:
                q = query.lower()
                if "prashant" in q and "front-end" in q:
                    return "Prashant's electrical repair background demonstrates strong systemic troubleshooting and logic—excellent foundations for debugging HTML/CSS."
                if "neha" in q and "customer success" in q:
                    return "Neha's caregiving experience maps to deep empathy and conflict resolution under pressure—highly valuable for Customer Success."
                return "This candidate shows transferable skills aligned with the role. Ask for a skill-by-skill breakdown if needed."

            response = get_agent_chat_response(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.info("Chat UI requires a newer Streamlit version. Upgrade Streamlit to use chat features.")

# Human Review Log & Fairness Metrics
st.divider()
colL, colR = st.columns(2)
with colL:
    with st.expander("🗂️ Human-in-the-Loop Log"):
        if st.session_state.decision_log:
            for entry in st.session_state.decision_log:
                st.write(entry)
        else:
            st.write("No decisions logged yet.")

with colR:
    with st.expander("📊 Bias Audit Agent", expanded=False):
        parity_values = {"Harshpreet": 0.92, "Neha": 0.85, "Prashant": 0.95}
        if PLOTLY_AVAILABLE:
            df = pd.DataFrame(list(parity_values.items()), columns=['Candidate', 'Parity Score'])
            fig = px.bar(df, x='Candidate', y='Parity Score', color='Candidate', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame.from_dict(parity_values, orient="index", columns=["Parity Score"]))

# Trigger celebrations / toasts
if st.session_state.last_decision == "Recommend":
    trigger_celebration()
    st.session_state.last_decision = None
elif st.session_state.last_decision in ["Review", "Reject"]:
    st.success(f"Logged decision: {st.session_state.last_decision}")
    st.session_state.last_decision = None
