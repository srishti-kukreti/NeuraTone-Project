# ══════════════════════════════════════════════════════════════════════
# utils.py  ·  NeuraTone v7 — "Kore" Luxe-Clinical Edition
#
# Changes from v6:
#   • GLOBAL_CSS: Londrina Solid headings (BIG, uppercase), glassmorphism
#     cards (rgba(255,255,255,0.40) + 16px blur)
#   • Unicode sanitizer for PDF safety (em-dashes, ellipses, etc.)
#   • Vault helpers: init_vault / vault_set / vault_get / vault_clear_one
#   • Severity/status badges preserved; informal section emojis removed
# ══════════════════════════════════════════════════════════════════════

import io
import os
import tempfile
import numpy as np


# ── Severity colour palette ───────────────────────────────────────────
SEVERITY_COLORS = {
    "Minimal"           : {"bg": "#d1fae5", "text": "#065f46", "border": "#34d399", "badge": "#10b981"},
    "Mild"              : {"bg": "#fef9c3", "text": "#713f12", "border": "#fbbf24", "badge": "#f59e0b"},
    "Moderate"          : {"bg": "#ffedd5", "text": "#7c2d12", "border": "#fb923c", "badge": "#f97316"},
    "Moderately Severe" : {"bg": "#fee2e2", "text": "#7f1d1d", "border": "#f87171", "badge": "#ef4444"},
    "Severe"            : {"bg": "#fce7f3", "text": "#500724", "border": "#f472b6", "badge": "#ec4899"},
}

# Professional severity/status icons — KEPT (clinical identity badges)
SEVERITY_EMOJI = {
    "Minimal"           : "🟢",
    "Mild"              : "🟡",
    "Moderate"          : "🟠",
    "Moderately Severe" : "🔴",
    "Severe"            : "🚨",
}

STATUS_OPTIONS = [
    "Pending Review",
    "Review Complete",
    "Follow-up Required",
    "Referred to Specialist",
    "Discharged",
]

CLINICAL_QUESTIONS = [
    ("mood",     "Over the past week, how would you describe your overall mood? Please speak freely."),
    ("sleep",    "How has your sleep been recently — falling asleep, staying asleep, or waking early?"),
    ("energy",   "Can you tell me about your energy levels and how motivated you feel day to day?"),
    ("appetite", "Have you noticed any changes in your appetite or interest in food lately?"),
    ("interest", "What activities used to bring you joy, and are you still engaging with them?"),
]

PHQ9_QUESTIONS = [
    ("phq_pleasure",    "Lately, have you found it hard to get excited about things you usually enjoy?"),
    ("phq_depressed",   "How has your mood been? Have you felt particularly heavy or hopeless recently?"),
    ("phq_sleep",       "How’s your sleep? Are you finding it tough to drift off, or maybe feeling like you can't get out of bed?"),
    ("phq_tired",       "Do you feel like your battery is constantly running low, even after resting?"),
    ("phq_appetite",    "Have you noticed any changes in how you’re eating? Maybe skipping meals or reaching for food more than usual?"),
    ("phq_worthless",   "Do you ever find yourself being really hard on yourself, or feeling like you’ve let people down?"),
    ("phq_concentrate", "Have you found it difficult to focus lately, like when you’re reading or just trying to watch a show?"),
    ("phq_moving",      "Have others noticed you’re moving a bit slower, or perhaps feeling really restless and fidgety?"),
    ("phq_thoughts",    "This is a heavy one, but have you had any thoughts about hurting yourself or felt like it would be easier if you weren't here?"),
]


# ════════════════════════════════════════════════════════════════════
# VAULT HELPERS — Bulletproof Redo Logic (Req 3)
#
# st.session_state.vault = { question_id: (np.ndarray, sr) }
# Clicking "Redo Q1" calls vault_clear_one(state, 1)
# Q2–Q5 data are completely untouched.
# ════════════════════════════════════════════════════════════════════

def init_vault(state) -> None:
    """Ensure st.session_state.vault dict exists."""
    if "vault" not in state:
        state.vault = {}


def vault_set(state, question_id, audio_data) -> None:
    """Store (arr, sr) for a specific question id without touching others."""
    init_vault(state)
    state.vault[question_id] = audio_data


def vault_get(state, question_id):
    """Retrieve (arr, sr) for a question id, or None if not recorded."""
    init_vault(state)
    return state.vault.get(question_id)


def vault_clear_one(state, question_id) -> None:
    """
    Clear ONLY the specified question's audio.
    All other question data remains completely preserved.
    """
    init_vault(state)
    state.vault.pop(question_id, None)


def vault_all_segments(state) -> list:
    """Return [(arr, sr), ...] in question-id order for stitching."""
    init_vault(state)
    result = []
    for k in sorted(state.vault.keys()):
        entry = state.vault[k]
        if entry is not None:
            result.append(entry)
    return result


# ════════════════════════════════════════════════════════════════════
# UNICODE / PDF SANITIZER (Req 4)
#
# Call sanitize_for_pdf(text) on EVERY string before passing it
# to fpdf2's Latin-1 renderer to prevent UnicodeEncodeError crashes.
# ════════════════════════════════════════════════════════════════════

_UNICODE_MAP = {
    "\u2014": "--",     # em dash       —
    "\u2013": "-",      # en dash       –
    "\u2026": "...",    # ellipsis      …
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote / apostrophe
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u00b7": "*",      # middle dot
    "\u2022": "*",      # bullet
    "\u00a0": " ",      # non-breaking space
    "\u2028": "\n",     # line separator
    "\u2029": "\n\n",   # paragraph separator
    "\u00e9": "e",      # é
    "\u00e8": "e",      # è
    "\u00ea": "e",      # ê
    "\u00e0": "a",      # à
    "\u00e2": "a",      # â
    "\u00f4": "o",      # ô
    "\u00fb": "u",      # û
    "\u00e7": "c",      # ç
    "\u00fc": "u",      # ü
    "\u00f6": "o",      # ö
    "\u00e4": "a",      # ä
    "\u00df": "ss",     # ß
    "\u2122": "(TM)",   # ™
    "\u00ae": "(R)",    # ®
    "\u00a9": "(C)",    # ©
}

def sanitize_for_pdf(text: str) -> str:
    """
    Replace fancy Unicode characters with ASCII equivalents before
    any string is passed to fpdf2's Latin-1 renderer.
    Unknown non-Latin-1 bytes are replaced with '?'.
    """
    if not isinstance(text, str):
        text = str(text)
    for uni_char, replacement in _UNICODE_MAP.items():
        text = text.replace(uni_char, replacement)
    # Final safety pass: encode to Latin-1, replacing unknowns
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


# ── Audio preprocessing ───────────────────────────────────────────────

def decode_audio_bytes(audio_bytes: bytes, source_sr: int = None) -> tuple:
    TARGET_SR = 16000
    errors    = []

    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = _resample(audio, sr, TARGET_SR)
        return audio, TARGET_SR
    except Exception as e:
        errors.append(f"soundfile: {e}")

    try:
        from pydub import AudioSegment
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tf:
            tf.write(audio_bytes)
            tmp_path = tf.name
        try:
            seg = AudioSegment.from_file(tmp_path)
            seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0
            return samples, TARGET_SR
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"pydub: {e}")

    raise RuntimeError(
        "Could not decode audio bytes.\n" + "\n".join(errors) +
        "\nEnsure ffmpeg is installed: sudo apt-get install ffmpeg"
    )


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def stitch_audio_segments(segments: list, sr: int = 16000,
                           silence_ms: int = 500) -> np.ndarray:
    silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.float32)
    parts   = []
    for i, (arr, seg_sr) in enumerate(segments):
        if seg_sr != sr:
            arr = _resample(arr, seg_sr, sr)
        parts.append(arr.astype(np.float32))
        if i < len(segments) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def save_temp_wav(audio: np.ndarray, sr: int = 16000) -> str:
    import soundfile as sf
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tf.name, audio, sr)
    return tf.name


def save_session_wav(audio: np.ndarray, sr: int = 16000,
                     patient_name: str = "anonymous",
                     data_dir: str = None) -> str:
    """
    Save a high-quality WAV to engine/data/ for the v4 inference engine.
    Returns the absolute path.
    """
    import soundfile as sf
    import datetime
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in patient_name.lower())[:20]
    filename  = f"session_{safe_name}_{ts}.wav"
    path      = os.path.join(data_dir, filename)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


# ════════════════════════════════════════════════════════════════════
# v7 "KORE" GLOBAL CSS — Luxe-Clinical Aesthetics
#
# Design system:
#   Background : #D6F3CE  (Mint)
#   Forest     : #47682C
#   Headings   : Londrina Solid — BIG, UPPERCASE, impactful
#   Body       : Inter — clinical readability
#   Cards      : rgba(255,255,255,0.40) + backdrop-filter:blur(16px)
# ════════════════════════════════════════════════════════════════════


GLOBAL_CSS = """
<style>
/* ── 1. FONT & ICON IMPORTS ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Helper class for Google Icons */
    .material-symbols-outlined {
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    vertical-align: middle;
    color: inherit;
    }


/* ── 1. FONT IMPORT (Inter for everything, JetBrains Mono for data) ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  /* ── 2. COLOR DESIGN TOKENS ── */
  --brand-1: #2e8b57;
  --brand-2: #246E45;
  --light-1: #F4FAF7; /* Menu Background */
  --light-2: #FFFFFF; /* Pure White Main Background */
  --light-3: #DEEDE5; /* Sage Green Box Color */
  --dark-1:  #173624; /* Primary Dark 1 (Header Text) */
  --dark-2:  #2D533D;
  --dark-3:  #476B57;

  /* ── 3. MASSIVE TYPOGRAPHY ── */
  --font-size-heading-1: clamp(2.5rem, 1.2rem + 4vw, 4.5rem); 
  --font-size-heading-2: clamp(1.8125rem, 1.4868rem + 1.4474vw, 2.5rem);
  --font-size-heading-5: clamp(1.25rem, 1.1612rem + 0.3947vw, 1.4375rem);
}

/* ── 4. GLOBAL APP SHELL (PURE WHITE CONTENT) ── */
html, body, [class*="css"], .stApp, .main, .stMain, [data-testid="stAppViewContainer"], [data-testid="stMainViewContainer"], .block-container {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--light-2) !important;
    background: var(--light-2) !important;
    color: var(--dark-1);
}

/* ── 5. NAVIGATION MENU (TINTED #F4FAF7) ── */
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stHeader"], [data-testid="stSidebarHeader"] {
    background-color: var(--light-1) !important;
    border-right: 1px solid var(--light-3) !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── 6. TYPOGRAPHY APPLICATION ── */
h1 { 
    font-size: var(--font-size-heading-1) !important; 
    color: var(--dark-1) !important; 
    font-weight: 800 !important; 
    letter-spacing: -0.04em !important; 
    margin-bottom: 1.5rem !important;
    line-height: 1.1 !important;
}

h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: var(--dark-1) !important;
    font-weight: 700 !important;
}

h2 { font-size: var(--font-size-heading-2) !important; letter-spacing: -0.01em !important; }

/* Bold H5 for Agent Banner */
h5 { 
    font-size: var(--font-size-heading-5) !important; 
    color: var(--dark-1) !important; 
    font-weight: 800 !important; 
    text-transform: uppercase !important; 
    letter-spacing: 0.02em !important; 
    margin: 0 0 8px 0 !important;
}

p, div { font-size: 1rem; line-height: 1.6; }

/* ── 7. CONTAINERS & CARDS ── */
/* General Card Style (Sage Green Background) */
.live-call-banner, .cv-card, .rq-row {
    background-color: var(--light-3) !important;
    border: 1px solid var(--light-3) !important;
    border-radius: 12px !important;
    padding: 28px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 2px 10px rgba(23, 54, 36, 0.05) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* Question Cards with Thick Left Border */
.q-card {
    padding: 18px 24px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid var(--light-3) !important;
    border-left: 6px solid var(--brand-1) !important;
    margin-bottom: 16px;
}
.q-answered {
    border-left-color: #10b981 !important;
    background: rgba(255, 255, 255, 0.6) !important;
}

/* ── 8. BUTTONS (ROUNDED PILLS) ── */
.stButton > button {
    border-radius: 50px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.24s ease !important;
}

.stButton > button[kind="primary"] {
    background: var(--brand-1) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(71, 104, 44, 0.2) !important;
}

/* Redo Button Fix */
[key*="redo"] button {
    background: white !important;
    border: 1px solid var(--light-3) !important;
    color: var(--dark-2) !important;
}

/* ── 9. COMPONENTS ── */
/* Session mode toggle container */
.session-toggle {
    display: flex;
    background: var(--light-1);
    border-radius: 50px;
    border: 1px solid var(--light-3);
    padding: 6px;
    width: fit-content;
    margin-bottom: 24px;
}

/* Pulse animation for Dr. Maya */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}
.live-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 8px;
    animation: pulse-dot 1.5s infinite;
}

/* ── 10. METRICS & DATA ── */
div[data-testid="stMetricValue"] {
    color: var(--brand-1) !important;
    font-weight: 800 !important;
}
.bm-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
</style>
"""

'''GLOBAL_CSS = """
<style>
    
    @import url('https://fonts.googleapis.com/css2?family=Londrina+Solid:wght@300;400;900&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Core backgrounds (The Screenshot Gradient) ─────────────── */
    [data-testid="stAppViewContainer"] {
        /* This matches the smooth, bright mint-to-sage look in your image */
        background: linear-gradient(145deg, #f0f7ee 0%, #d9e8d4 100%) !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(71,104,44,0.12) !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* ── Typography ─────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #2d3a25;
    }
    h1 {
        font-size: var(--font-size-heading-1) !important; 
        color: #173624 !important; /* Your Primary Dark 1 color */
        font-weight: 800 !important; 
        letter-spacing: -0.04em !important; /* Tighter tracking for large text */
        margin-bottom: 1rem !important;
        line-height: 1.1 !important;
    }
    h2, h3 {
        font-family: 'Londrina Solid', cursive !important;
        text-transform: uppercase !important;
        color: #47682C !important;
    }
    h2 { font-size: 2.1rem !important; letter-spacing: 0.04em !important; font-weight: 400 !important; }
    h3 { font-size: 1.55rem !important; letter-spacing: 0.03em !important; font-weight: 400 !important; }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* ── Glassmorphism card (Used for results & dashboard) ──────── */
    .cv-card {
        background: rgba(255, 255, 255, 0.70) !important; /* Brighter white for contrast */
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0 8px 30px rgba(71, 104, 44, 0.06), inset 0 1px 0 rgba(255,255,255,0.8) !important;
        padding: 22px 26px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .cv-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(71, 104, 44, 0.1);
    }

    /* ── Metric tiles ── */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 20px rgba(71, 104, 44, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
    }

    /* ── Buttons ────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 50px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.24s ease !important;
    }
    
    /* Primary Action (Save & Next / Analyze) */
    .stButton > button[kind="primary"] {
        background: #47682C !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(71, 104, 44, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3b5724 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(71, 104, 44, 0.4) !important;
    }
    
    /* Secondary Action (Skip / Start Over) */
    .stButton > button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(71, 104, 44, 0.2) !important;
        color: #47682C !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: #ffffff !important;
        border-color: #47682C !important;
    }

    /* The "🗑️ Redo" button fix */
    [key*="redo"] button {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(71, 104, 44, 0.2) !important;
        color: #2d3a25 !important; 
    }

    /* ── Inputs & Expanders ─────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 14px !important;
        border: 1.5px solid rgba(71, 104, 44, 0.15) !important;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px) !important;
    }
    [data-testid="stExpander"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.6) !important;
    }

    /* ── Session mode toggle (Rounded Pill) ─────────────────────── */
    .session-toggle {
        display: flex;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(16px);
        border-radius: 50px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        padding: 6px;
        width: fit-content;
        margin: 0 auto 24px auto;
        box-shadow: 0 4px 20px rgba(71, 104, 44, 0.05);
    }

    /* ── Kore Live Agent banner ─────────────────────────────────── */
    .live-call-banner {
        background: linear-gradient(135deg, #3b5724 0%, #47682C 50%, #6b8e4e 100%);
        border-radius: 24px;
        padding: 30px 40px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(71, 104, 44, 0.25);
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* ── Live dot pulse ── */
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #10b981; /* Brighter green for active state */
        margin-right: 8px;
        animation: pulse-dot 1.5s infinite;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
    }

    /* ── Question cards (As seen in the screenshot) ─────────────── */
    .q-card {
        padding: 18px 24px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-left: 4px solid #47682C;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(71, 104, 44, 0.04);
        transition: transform 0.2s ease;
    }
    .q-answered {
        border-left: 4px solid #10b981 !important;
        background: rgba(255, 255, 255, 0.6) !important; /* Slightly more faded when done */
    }

    /* ── Step dots ── */
    .step-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: rgba(71, 104, 44, 0.15);
        margin: 0 4px;
    }
    .step-dot.active {
        background: #47682C;
        box-shadow: 0 0 0 4px rgba(71, 104, 44, 0.15);
    }
    .step-dot.done { background: #10b981; }

    /* ── Audio Player Customization ── */
    audio {
        width: 100%;
        height: 45px;
        border-radius: 12px;
        margin-top: 12px;
        opacity: 0.9;
    }

    hr { border-color: rgba(71, 104, 44, 0.15) !important; }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #3b5724 !important;
        font-family: 'Inter', sans-serif !important;
    }
</style>
"""
'''

# ════════════════════════════════════════════════════════════════════
# BACKEND CLEANUP UTILITY (Req 5)
#
# purge_session_data() gives a clean demo start:
#   • Deletes all *.wav files in engine/data/
#   • Resets reports.json to []
#
# Safe to call from Streamlit (e.g. "Clear All Data" admin button)
# or from a shell script before a demo.
# ════════════════════════════════════════════════════════════════════

def purge_session_data(
    data_dir:     str = None,
    reports_path: str = None,
) -> dict:
    """
    Purge all session WAV files and reset the reports database.

    Parameters
    ----------
    data_dir     : path to engine/data/ (auto-detected if None)
    reports_path : path to reports.json (auto-detected if None)

    Returns
    -------
    dict with keys:
        "wavs_deleted"   : int  — number of WAV files removed
        "reports_reset"  : bool — True if reports.json was reset
        "errors"         : list[str] — any non-fatal errors encountered
    """
    import glob

    _engine_dir = os.path.dirname(os.path.abspath(__file__))

    if data_dir is None:
        data_dir = os.path.join(_engine_dir, "data")
    if reports_path is None:
        # reports.json lives one level above engine/
        reports_path = os.path.join(os.path.dirname(_engine_dir), "reports.json")

    errors      = []
    wavs_deleted = 0

    # ── 1. Delete WAV files ───────────────────────────────────────────
    if os.path.isdir(data_dir):
        pattern = os.path.join(data_dir, "*.wav")
        for wav_path in glob.glob(pattern):
            try:
                os.remove(wav_path)
                wavs_deleted += 1
            except OSError as e:
                errors.append(f"Could not delete {wav_path}: {e}")
    else:
        # data_dir doesn't exist yet — nothing to delete
        pass

    # ── 2. Reset reports.json to empty array ─────────────────────────
    reports_reset = False
    try:
        with open(reports_path, "w") as f:
            f.write("[]")
        reports_reset = True
    except OSError as e:
        errors.append(f"Could not reset reports.json: {e}")

    return {
        "wavs_deleted" : wavs_deleted,
        "reports_reset": reports_reset,
        "errors"       : errors,
    }
