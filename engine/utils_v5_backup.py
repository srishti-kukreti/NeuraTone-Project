# ══════════════════════════════════════════════════════════════════════
# utils.py  ·  Shared helpers for Streamlit pages
# ══════════════════════════════════════════════════════════════════════

import io
import os
import tempfile
import numpy as np

# ── Severity colour palette (mirrors v4 PHQ8_BANDS) ─────────────────
SEVERITY_COLORS = {
    "Minimal"           : {"bg": "#d1fae5", "text": "#065f46", "border": "#34d399", "badge": "#10b981"},
    "Mild"              : {"bg": "#fef9c3", "text": "#713f12", "border": "#fbbf24", "badge": "#f59e0b"},
    "Moderate"          : {"bg": "#ffedd5", "text": "#7c2d12", "border": "#fb923c", "badge": "#f97316"},
    "Moderately Severe" : {"bg": "#fee2e2", "text": "#7f1d1d", "border": "#f87171", "badge": "#ef4444"},
    "Severe"            : {"bg": "#fce7f3", "text": "#500724", "border": "#f472b6", "badge": "#ec4899"},
}

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


# ── Audio preprocessing ───────────────────────────────────────────────

def decode_audio_bytes(audio_bytes: bytes, source_sr: int = None) -> tuple:
    """
    Convert raw audio bytes (WebM/Opus from browser, or WAV/MP3) to
    a float32 mono numpy array at 16 kHz.

    Returns (audio_array: np.ndarray, sample_rate: int)
    Raises RuntimeError if all decoders fail.
    """
    TARGET_SR = 16000
    errors    = []

    # ── Try soundfile first (handles WAV, FLAC, raw PCM) ──────────────
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

    # ── Fallback: pydub (handles MP3, WebM/Opus, OGG) ─────────────────
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
    """Simple linear resample via librosa or scipy."""
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
    """
    Concatenate a list of (audio_array, sr) tuples with a short silence
    between each, resampling to `sr` as needed.

    Returns a single float32 mono array.
    """
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
    """Write audio array to a temporary WAV file. Returns the path."""
    import soundfile as sf
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tf.name, audio, sr)
    return tf.name


# ── Streamlit CSS injection ──────────────────────────────────────────

GLOBAL_CSS = """
<style>
    /* Hide default Streamlit hamburger & footer */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}

    /* Card component */
    .card {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .card-sm {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* Severity badge */
    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Biomarker bar */
    .bm-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .bm-label { flex: 0 0 200px; color: rgba(128,128,128,0.9); }
    .bm-bar-wrap { flex: 1; background: rgba(128,128,128,0.12); border-radius: 4px; height: 8px; }
    .bm-bar { height: 8px; border-radius: 4px; }
    .bm-val { flex: 0 0 52px; text-align: right; font-variant-numeric: tabular-nums; }

    /* Question card in interview */
    .q-card {
        border-left: 3px solid #6366f1;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        background: rgba(99,102,241,0.06);
        margin-bottom: 12px;
    }
    .q-answered {
        border-left: 3px solid #10b981;
        background: rgba(16,185,129,0.06);
    }

    /* Report queue row */
    .rq-row {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(128,128,128,0.15);
        margin-bottom: 8px;
        cursor: pointer;
        transition: background 0.15s;
    }
    .rq-row:hover { background: rgba(128,128,128,0.06); }

    /* Step indicator */
    .step-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: rgba(128,128,128,0.3);
        margin: 0 3px;
    }
    .step-dot.active { background: #6366f1; }
    .step-dot.done   { background: #10b981; }
</style>
"""
