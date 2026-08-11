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
from typing import List
from PIL import Image, ImageDraw, ImageFilter
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

# --- Gradient background helpers ---
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
    st.divider()
    demo_choice = st.radio("Load a test persona or upload a resume:", [
        "Upload Custom PDF/TXT",
        "Persona A: Harshpreet (Retail to Tech, Tier-3 City India)",
        "Persona B: Neha (7-year Career Break, Caregiver)",
        "Persona C: Prashant (Rural Graduate, Vocational Training)"
    ])

# --- Permanent gradient background ---
gradient_uri = pil_image_to_data_uri(generate_gradient_image(1400, 900))
st.markdown(
    f"""<style>
    .stApp {{
        background-image: url("{gradient_uri}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(15,23,42,0.12);
        pointer-events: none;
    }}
    </style>""",
    unsafe_allow_html=True
)
