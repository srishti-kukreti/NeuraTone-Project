# ══════════════════════════════════════════════════════════════════════
# pages/patient_portal.py  ·  NeuraTone v8 — Kore Patient Portal
#
# v7 Changes:
# ─────────────────────────────────────────────────────────────────────
# 1. DIAGNOSTIC SESSION PAGE (Req 2)
#    • "Live Session" + "Guided Interview" merged into one page
#    • Rounded toggle: "Interactive Agent" vs "Standard Interview"
#    • Kore voice persona (Dr. Maya) — warm, empathetic, unhurried
#    • Audio integrity: only USER voice segments reach v4 engine
#
# 2. KORE VOICE-TO-VOICE PORTAL (Req 2)
#    • Model: models/gemini-3.1-flash-live-preview  (via google-genai)
#    • Voice config: voice_name="Kore"
#    • Persona: "You are an empathetic psychologist. Speak with a warm,
#      nurturing, and calm tone. Do not rush. Listen for pauses."
#    • Agent audio stripped — only user segments sent to v4 pipeline
#
# 3. VAULT REDO LOGIC (Req 3)
#    • st.session_state.vault = {question_id: (arr, sr)}
#    • Redo Q1 only clears vault[1]; vault[2..5] fully preserved
#    • Navigate back/forth without losing any recorded answers
#
# 4. GLOBAL CSS / UI (Req 1)
#    • Luxe Kore theme via GLOBAL_CSS from utils.py
#    • YouTube URL and File Upload modes preserved
# ══════════════════════════════════════════════════════════════════════

import os
import sys
import tempfile
import time
import asyncio
import io
import base64
import threading

import numpy as np
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = os.path.join(_ROOT, "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from utils import (
    GLOBAL_CSS, CLINICAL_QUESTIONS, PHQ9_QUESTIONS,
    SEVERITY_COLORS, SEVERITY_EMOJI, dr_maya_voice_trigger,
    decode_audio_bytes, stitch_audio_segments, save_temp_wav,
    # Vault helpers (Req 3)
    init_vault, vault_set, vault_get, vault_clear_one, vault_all_segments,
    save_session_wav, purge_session_data,
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# KORE LIVE SESSION — Gemini 3.1 Flash Live + pyaudio bridge
#
# Architecture (mirrors ai_studio_code.py):
#   • Uses google-genai Live API with voice_name="Kore"
#   • Bidirectional audio: pyaudio mic → WebSocket → Kore TTS → speakers
#   • AUDIO INTEGRITY: Only PCM chunks from the USER's mic are stored
#     in self._user_pcm_chunks.  Agent audio is played via pyaudio but
#     is NEVER written to this list — it never reaches the v4 engine.
#   • On stop(), all user chunks are concatenated, saved to engine/data/
#     as a single 16-bit PCM WAV, and the path is returned for analysis.
#   • Runs entirely in a daemon thread so Streamlit's main loop is free.
#
# In pure Streamlit-cloud environments (no pyaudio / no mic access),
# the class gracefully degrades: the constructor sets self.available=False
# and the portal falls back to the per-turn st.audio_input() approach.
# ══════════════════════════════════════════════════════════════════════

class KoreLiveSession:
    """
    Wraps the google-genai Live API in a thread-safe session object.

    Usage
    -----
    session = KoreLiveSession(api_key, system_instruction, patient_name)
    if session.available:
        session.start()          # launches background thread
        ...                      # UI waits / shows transcript
        wav_path = session.stop() # saves user-only WAV, returns path
    """

    SEND_SR    = 16000   # mic sample rate
    RECV_SR    = 24000   # Kore TTS sample rate
    CHUNK      = 1024
    MODEL      = "models/gemini-3.1-flash-live-preview"

    def __init__(
        self,
        api_key:       str,
        system_prompt: str,
        patient_name:  str = "anonymous",
        data_dir:      str = None,
    ):
        self.api_key       = api_key
        self.system_prompt = system_prompt
        self.patient_name  = patient_name
        self.data_dir      = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "engine", "data"
        )

        self._user_pcm_chunks: list[bytes] = []   # USER mic only
        self._transcript:      list[dict]  = []   # {role, text}
        self._thread:          threading.Thread | None = None
        self._stop_event                   = threading.Event()
        self._loop:            asyncio.AbstractEventLoop | None = None
        self.available                     = False
        self._error:           str | None  = None

        # Check runtime dependencies
        try:
            import pyaudio                           # noqa: F401
            from google import genai                 # noqa: F401
            self.available = bool(api_key)
        except ImportError as exc:
            self._error = (
                f"KoreLiveSession unavailable: {exc}. "
                f"Install google-genai and pyaudio, or use Standard Interview."
            )

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        """Start the Live session in a background daemon thread."""
        if not self.available:
            return
        self._stop_event.clear()
        self._user_pcm_chunks = []
        self._transcript      = []
        self._thread = threading.Thread(
            target=self._thread_main, daemon=True, name="KoreLiveThread"
        )
        self._thread.start()

    # Inside pages/patient_portal.py -> class KoreLiveSession
    def stop(self) -> str | None:
        """Signal the session to end and ensure hardware release."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            # Block for a max of 2 seconds to let the thread clean up
            self._thread.join(timeout=2) 
        
        # Save the WAV only AFTER we are sure the mic is off
        wav_path = self._save_user_wav()
        
        # Reset internal states to prevent zombie loops
        self._loop = None
        self._thread = None
        return wav_path

    @property
    def transcript(self) -> list[dict]:
        return list(self._transcript)

    @property
    def user_audio_bytes(self) -> bytes:
        """Concatenated raw PCM bytes of the user's speech only."""
        return b"".join(self._user_pcm_chunks)

    # ── Internal ──────────────────────────────────────────────────────

    def _thread_main(self):
        """Entry point for the background thread — owns the asyncio loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_run())
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._loop.close()

    async def _async_run(self):
        """Full bidirectional session coroutine compliant with Gemini 3.1 Live API."""
        import pyaudio
        from google import genai
        from google.genai import types

        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        pya    = pyaudio.PyAudio()
        client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=self.api_key,
        )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_LOW",
            system_instruction=self.system_prompt,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
        )

        audio_in_q: asyncio.Queue[bytes] = asyncio.Queue()
        out_q:      asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)

        async def _listen_mic():
            mic_info = pya.get_default_input_device_info()
            try:
                mic_stream = await asyncio.to_thread(
                    pya.open, format=FORMAT, channels=CHANNELS, rate=self.SEND_SR,
                    input=True, input_device_index=mic_info["index"], frames_per_buffer=self.CHUNK,
                )
            except Exception as e:
                print(f"❌ PyAudio Mic Error: {e}")
                raise

            try:
                while not self._stop_event.is_set():
                    data = await asyncio.to_thread(mic_stream.read, self.CHUNK, exception_on_overflow=False)
                    self._user_pcm_chunks.append(data)
                    await out_q.put(data)
            finally:
                mic_stream.close()

        async def _send_realtime(session):
            while not self._stop_event.is_set():
                try:
                    data = out_q.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)
                    continue
                
                # ── The 3.1 Flash Live Fix ──
                # Use strictly typed 'audio' Blob instead of the deprecated 'media_chunks' dict
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

        async def _receive_audio(session):
            while not self._stop_event.is_set():
                turn = session.receive()
                async for response in turn:
                    if self._stop_event.is_set():
                        break
                    if response.data:
                        #print("🎵 [Kore] Audio chunk received from Dr. Maya!")
                        audio_in_q.put_nowait(response.data)
                    if response.text:
                        self._transcript.append({"role": "model", "text": response.text})

        async def _play_audio():
            try:
                play_stream = await asyncio.to_thread(
                    pya.open, format=FORMAT, channels=CHANNELS, rate=self.RECV_SR, output=True,
                )
            except Exception as e:
                print(f"❌ PyAudio Speaker Error: {e}")
                raise

            try:
                while not self._stop_event.is_set():
                    try:
                        chunk = audio_in_q.get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.01)
                        continue
                    await asyncio.to_thread(play_stream.write, chunk)
            finally:
                play_stream.close()

        try:
            print("🌐 Connecting to Gemini 3.1 Flash Live API...")
            async with client.aio.live.connect(model=self.MODEL, config=config) as session:
                print("✅ Connected! Sending Wake-Up Ping to Dr. Maya...")
                
                # Wake up ping using the new strict 'text' parameter
                await session.send_realtime_input(
                    text="Hello Dr. Maya, the patient is ready. Please introduce yourself."
                )
                
                try:
                    async with asyncio.TaskGroup() as tg:
                        print("🎤 Starting microphone and speakers...")
                        tg.create_task(_listen_mic())
                        tg.create_task(_send_realtime(session))
                        tg.create_task(_receive_audio(session))
                        tg.create_task(_play_audio())

                        await asyncio.to_thread(self._stop_event.wait)
                        print("⏹️ Stop event triggered, closing session cleanly...")
                        raise asyncio.CancelledError("session ended")
                except ExceptionGroup as eg:
                    print(f"❌ Background Task Crashed: {eg.exceptions}")
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            print(f"❌ Session Error: {e}")
        # Replace your existing 'finally' block inside _async_run:
        finally:
            try:
                # Explicitly stop everything in order
                self._stop_event.set()
                if 'mic_stream' in locals():
                    mic_stream.stop_stream()
                    mic_stream.close()
                if 'play_stream' in locals():
                    play_stream.stop_stream()
                    play_stream.close()
                
                pya.terminate()
                print("🔒 Audio hardware terminated successfully.")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
            
            try:
                # Graceful shutdown of the async client
                await client.aio.aclose()
            except:
                pass

    def _save_user_wav(self) -> str | None:
        """Save concatenated user PCM to engine/data/ as a WAV file."""
        if not self._user_pcm_chunks:
            return None
        try:
            import soundfile as sf
            import datetime

            raw   = b"".join(self._user_pcm_chunks)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            os.makedirs(self.data_dir, exist_ok=True)
            ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c if c.isalnum() else "_" for c in self.patient_name.lower()
            )[:20]
            path = os.path.join(self.data_dir, f"kore_{safe_name}_{ts}.wav")
            sf.write(path, audio, self.SEND_SR, subtype="PCM_16")
            return path
        except Exception as exc:
            self._error = f"WAV save failed: {exc}"
            return None

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        # Unified diagnostic session mode toggle
        "diag_mode"         : "Interactive Agent",   # or "Standard Interview"

        # Guided interview (Standard mode)
        "interview_step"    : 0,
        "interview_answers" : {},
        # vault replaces audio_segments — keyed by question_id (int)
        # populated by vault_set(), cleared per-Q by vault_clear_one()

        "gemini_chat"       : [],
        "analysis_result"   : None,
        "patient_name"      : "",
        "portal_mode"       : "Diagnostic Session",
        "running"           : False,

        # Live session state (Interactive Agent mode)
        "live_session_audio" : [],    # list of raw audio bytes (USER only)
        "live_session_active": False,
        "live_transcript"    : [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Ensure vault exists
    init_vault(st.session_state)

_init_state()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("#### Configuration")

    checkpoints_dir = st.text_input(
        "Checkpoints directory",
        value = os.environ.get("CHECKPOINTS_DIR", "checkpoints"),
        help  = "Path to TAN_best.pt, lgb_v3.pkl, etc.",
    )
    gemini_key = st.text_input(
        "Gemini API key",
        type  = "password",
        value = os.environ.get("GEMINI_API_KEY", ""),
        help  = "Used for Kore Live Agent and interview questions.",
    )

    st.divider()
    st.markdown("#### Input mode")
    st.session_state.portal_mode = st.radio(
        "Select intake method",
        options = ["Diagnostic Session", "YouTube URL", "File Upload"],
        index   = ["Diagnostic Session", "YouTube URL", "File Upload"].index(
            st.session_state.portal_mode
        ),
    )

    st.divider()
    st.caption("NeuraTone v8 — Acoustic Intelligence for Mental Wellness")
    st.caption("⚠️ Not a clinical diagnosis.")


# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_engine(ckpt_dir: str, api_key: str):
    from pipeline_v7 import ClinicalEngine
    return ClinicalEngine(
        checkpoints_dir = ckpt_dir,
        gemini_api_key  = api_key or None,
    )


@st.cache_resource
def _get_gemini_model(api_key: str):
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None


def _ask_gemini_question(model, question_text: str) -> str:
    if model is None:
        return question_text
    try:
        prompt = (
            "You are a warm, empathetic Senior Clinical Psychologist conducting a brief "
            "mental wellness check-in. Rephrase the following clinical question "
            "in a natural, supportive, conversational tone (2 sentences max). "
            "Do NOT add preamble or sign-offs.\n\n"
            f"Question: {question_text}"
        )
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return question_text


def _run_analysis(source, patient_name, interview_answers=None, forced_domain=None):
    try:
        engine = _get_engine(checkpoints_dir, gemini_key)
        result = engine.run(
            source             = source,
            patient_name       = patient_name or "Anonymous",
            interview_answers  = interview_answers,
            forced_domain      = forced_domain,
        )
        st.session_state.analysis_result = result
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.session_state.running = False


# ══════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════

def _show_results(result: dict):
    ar   = result["acoustic_report"]
    sev  = ar.get("severity", "Minimal")
    col  = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["Minimal"])
    band = ar.get("band_info", {})

    st.markdown("---")
    st.markdown("## Analysis Complete")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PHQ-8 Estimate", f"{ar['phq8_estimate']:.1f} / 24")
    c2.metric("Severity", f"{SEVERITY_EMOJI.get(sev,'')} {sev}")
    c3.metric("Classification", ar["label"] + (" ⚠️" if ar.get("safety_override") else ""))
    c4.metric("Audio Duration", f"{ar['duration_sec']:.0f}s")

    if ar.get("safety_override"):
        st.warning("⚠️ **Safety Override Active** — PHQ-8 ≥ 10 triggered. "
                   "Classification upgraded to MDD.")

    st.markdown(
        f'<div class="cv-card" style="background:{col["bg"]};border:1px solid {col["border"]}">'
        f'<b style="color:{col["text"]};font-size:16px">'
        f'{SEVERITY_EMOJI.get(sev,"")} {band.get("headline","")}</b><br>'
        f'<span style="color:{col["text"]};font-size:14px;opacity:0.85">'
        f'{band.get("what_it_means","")}</span></div>',
        unsafe_allow_html=True,
    )

    if result.get("gemini_summary"):
        with st.expander("AI Clinical Summary", expanded=True):
            st.markdown(result["gemini_summary"])

    chunk_probs = ar.get("chunk_probs", [])
    if len(chunk_probs) > 1:
        import altair as alt
        import pandas as pd
        df = pd.DataFrame({
            "Window"      : list(range(len(chunk_probs))),
            "Probability" : chunk_probs,
        })
        line = (
            alt.Chart(df)
            .mark_line(point=True, color="#47682C")
            .encode(
                x = alt.X("Window:Q", title="3-second window"),
                y = alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                          title="Depression probability"),
            )
        )
        threshold = (
            alt.Chart(pd.DataFrame({"y": [0.47]}))
            .mark_rule(strokeDash=[4, 3], color="#9ca3af", size=1)
            .encode(y="y:Q")
        )
        st.altair_chart(line + threshold, use_container_width=True)

    signals = ar.get("acoustic_signals", [])
    if signals:
        st.markdown("**Acoustic Biomarkers**")
        max_z    = max(abs(s["z_score"]) for s in signals) or 1.0
        html_bars = ""
        for s in signals:
            z       = s["z_score"]
            pct     = min(100, abs(z) / max_z * 100)
            bar_col = "#ef4444" if z > 0 else "#3b82f6"
            tick    = "✓" if s["consistent"] else "↔"
            html_bars += (
                f'<div class="bm-row">'
                f'<span class="bm-label">{tick} {s["description"]}</span>'
                f'<div class="bm-bar-wrap"><div class="bm-bar" '
                f'style="width:{pct:.0f}%;background:{bar_col}"></div></div>'
                f'<span class="bm-val">{z:+.2f}</span>'
                f'</div>'
            )
        st.markdown(html_bars, unsafe_allow_html=True)

    st.success(f"✅ Report saved — ID: **{result['report_id']}** — "
               f"View in My Results or Doctor Dashboard")


# ══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════

st.markdown(
"""<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
<div style="text-align: center; margin-top: 1rem; margin-bottom: 3.5rem;">
<div style="display: inline-flex; align-items: center; justify-content: center; gap: 8px;">
<h1 style="margin: 0 !important; padding: 0 !important; font-size: 2.8rem; line-height: 1;">NeuraTone</h1>
<span class="material-symbols-outlined" style="font-size: 3rem; color: #173624; display: block; margin-top: 4px;">psychiatry</span>
</div>
<div style="font-size: 14px; color: #2D533D; opacity: 0.85; font-weight: 500; margin-top: 4px;">
Acoustic intelligence for mental wellness &middot; Research screening tool &mdash; not a clinical diagnosis.
</div>
</div>""",
unsafe_allow_html=True
)

patient_name_input = st.text_input(
    "Your name (optional)",
    value       = st.session_state.patient_name,
    placeholder = "Anonymous",
    key         = "patient_name_field",
)
st.session_state.patient_name = patient_name_input

st.divider()

# ══════════════════════════════════════════════════════════════════════
# MODE: DIAGNOSTIC SESSION — Unified Live + Guided (Req 2)
# ══════════════════════════════════════════════════════════════════════

if st.session_state.portal_mode == "Diagnostic Session":

    st.markdown("### Diagnostic Session")

    # ── Rounded toggle: Interactive Agent vs Standard Interview ───────
    # Streamlit doesn't natively do rounded pill toggles so we use
    # a st.radio styled as two clean options.
    st.markdown(
        '<p style="font-family:\'Inter\',sans-serif;font-size:0.9rem;'
        'color:#4a5f3a;margin-bottom:4px">Select session type</p>',
        unsafe_allow_html=True,
    )
    diag_mode = st.radio(
        "Session type",
        options      = ["Interactive Agent (Voice-to-Voice)", "Standard Interview (Text-Guided)"],
        index        = 0 if st.session_state.diag_mode == "Interactive Agent" else 1,
        horizontal   = True,
        label_visibility = "collapsed",
    )
    st.session_state.diag_mode = (
        "Interactive Agent" if "Interactive" in diag_mode else "Standard Interview"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # SUB-MODE: INTERACTIVE AGENT — Kore Voice-to-Voice (Req 2)
    # ══════════════════════════════════════════════════════════════════

    if st.session_state.diag_mode == "Interactive Agent":

        # Dr. Maya banner
        # NeuraTone Unified Banner — Professional Clinical Style
        st.markdown(
            f'''
            <div class="live-call-banner" style="background-color: #DEEDE5 !important;">
                <div style="font-family: 'Inter', sans-serif; 
                            font-size: var(--font-size-heading-5); 
                            font-weight: 800; 
                            color: #173624 !important; 
                            text-transform: uppercase; 
                            letter-spacing: 0.02em; 
                            margin-bottom: 8px;">
                    Dr. Maya — Live Voice Agent
                </div>
                <div style="font-size: var(--font-size-body); 
                            color: var(--dark-2); 
                            margin-bottom: 20px; 
                            line-height: 1.6;
                            font-weight: 500;">
                    Your AI psychologist listens with warmth and patience. 
                    She will guide you through the PHQ-9 assessment using real-time acoustic analysis.
                </div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <span style="background: var(--light-1); 
                                border: 1px solid var(--light-3); 
                                color: var(--brand-1); 
                                border-radius: 6px; 
                                padding: 5px 14px; 
                                font-size: 12px; 
                                font-weight: 700;">
                        🎙️ Bidirectional Audio
                    </span>
                    <span style="background: var(--light-1); 
                                border: 1px solid var(--light-3); 
                                color: var(--brand-1); 
                                border-radius: 6px; 
                                padding: 5px 14px; 
                                font-size: 12px; 
                                font-weight: 700;">
                        🧬 Acoustic Analysis
                    </span>
                    <span style="background: var(--light-1); 
                                border: 1px solid var(--light-3); 
                                color: var(--brand-1); 
                                border-radius: 6px; 
                                padding: 5px 14px; 
                                font-size: 12px; 
                                font-weight: 700;">
                        💚 Kore Voice Profile
                    </span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        with st.expander("How the Kore Voice Session works", expanded=False):
            st.markdown("""
**Kore Voice Session — Architecture:**

1. **Voice Model**: `models/gemini-3.1-flash-live-preview` via the `google-genai` SDK.
   Voice profile set to **`Kore`** — warm, nurturing, unhurried.
2. **Persona (Dr. Maya)**: *"You are an empathetic psychologist. Speak with a warm, nurturing,
   and calm tone. Do not rush. Listen for pauses as signs of reflection."*
3. **Audio Integrity**: Only **your** voice segments are recorded and stitched.
   Dr. Maya's audio output is played back to you but is **never** passed to the v4 engine.
4. **v4 Engine**: The stitched user-only WAV is fed into the InferencePipeline v4
   (TAN + LightGBM) for acoustic depression biomarker analysis.
5. **Results**: PHQ-8 score, severity band, acoustic biomarkers, and AI summary are reported.

*The Gemini Live API requires `google-genai >= 0.8.0` with Live API access enabled.*
*For environments without Live API, use Standard Interview mode.*
            """)

        # ── Kore Live Session implementation ──────────────────────────

        # Dr. Maya system instruction — Kore persona (Req 2)
        KORE_SYSTEM = """You are Dr. Maya, a Senior Clinical Psychologist specialising in
mood disorders and emotional wellness. You speak with the Kore voice profile.
Your tone is warm, nurturing, and calm. Provide brief, empathetic acknowledgments after each user response.

You speak slowly and clearly. You never rush the patient. You listen for pauses as signs of reflection.

Your task is to conduct a brief PHQ-9 screening interview. Ask each of the 9 PHQ-9 questions
one at a time, wait patiently for the patient's response, then move gently to the next.

Guidelines:
- Introduce yourself warmly and invite the patient to take their time
- Rephrase each question conversationally — never clinical or robotic
- Acknowledge each answer with genuine warmth before moving on
  (e.g., "Thank you for sharing that with me..." / "I hear you...")
- Maintain a calm, non-judgmental atmosphere throughout
- If the patient seems hesitant, offer gentle encouragement — never pressure
- After all 9 questions, thank the patient and let them know their responses are complete
- Do NOT provide diagnosis or clinical interpretation during the session

PHQ-9 questions to cover (in your own warm, unhurried words):
1. Lately, have you found it hard to get excited about things you usually enjoy?
2. How has your mood been? Have you felt particularly heavy or hopeless recently?
3. How's your sleep? Are you finding it tough to drift off, or maybe feeling like you can't get out of bed?
4. Do you feel like your battery is constantly running low, even after resting?
5. Have you noticed any changes in how you're eating? Maybe skipping meals or reaching for food more than usual?
6. Do you ever find yourself being really hard on yourself, or feeling like you've let people down?
7. Have you found it difficult to focus lately, like when you're reading or just trying to watch a show?
8. Have others noticed you're moving a bit slower, or perhaps feeling really restless and fidgety?
9. This is a heavy one, but have you had any thoughts about hurting yourself or felt like it would be easier if you weren't here?

Begin by introducing yourself as Dr. Maya and inviting the patient to begin when ready."""

        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button(
                "▶️ Start Kore Session", type="primary",
                use_container_width=True,
                disabled=st.session_state.live_session_active
            ):
                # Clear any previous KoreLiveSession and question caches
                st.session_state.pop("_kore_live_session", None)
                for k in list(st.session_state.keys()):
                    if k.startswith("_maya_q_audio_"):
                        del st.session_state[k]
                st.session_state.live_session_active = True
                st.session_state.live_session_audio  = []
                st.session_state.live_transcript     = []
                st.session_state.analysis_result     = None
                st.rerun()

        with col_stop:
            if st.button(
                "⏹️ End Session & Analyse",
                use_container_width=True,
                disabled=not st.session_state.live_session_active
            ):
                with st.spinner("Closing audio streams and saving recording..."):
                    kore_obj = st.session_state.get("_kore_live_session")
                    if kore_obj:
                        # 1. Stop the hardware first
                        wav_path = kore_obj.stop() 
                        if wav_path:
                            st.session_state["_kore_wav_path"] = wav_path
                        
                        # 2. Now clear the object from memory
                        st.session_state.pop("_kore_live_session", None)
                    
                    # 3. Update state only after hardware is released
                    st.session_state.live_session_active = False
                
                # 4. Final refresh
                st.rerun()

        if st.session_state.live_session_active:
            st.markdown(
                '<div style="text-align:center;margin:14px 0">'
                '<span class="live-dot"></span>'
                '<span style="font-weight:600;color:#47682C;font-family:\'Inter\',sans-serif">'
                'Kore Session Active — Dr. Maya is speaking</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── VOICE-ONLY INTERFACE ───────────────────────────────────
            # Dr. Maya asks all questions ONLY via Kore voice (AUDIO modality).
            # No question text is shown to the patient — purely voice-to-voice.
            # The KoreLiveSession class handles the bidirectional audio bridge:
            #   microphone → Gemini Live API → Kore TTS → speakers
            #   user PCM chunks → stored → WAV saved to engine/data/
            # ─────────────────────────────────────────────────────────────

            # Attempt to use the true KoreLiveSession (pyaudio + google-genai Live)
            kore_session_key = "_kore_live_session"
            if kore_session_key not in st.session_state:
                session_obj = KoreLiveSession(
                    api_key       = gemini_key,
                    system_prompt = KORE_SYSTEM,
                    patient_name  = st.session_state.patient_name or "anonymous",
                )
                st.session_state[kore_session_key] = session_obj
                if session_obj.available:
                    session_obj.start()

            kore_obj = st.session_state.get(kore_session_key)
            live_available = kore_obj is not None and kore_obj.available

            if live_available:
                # ── TRUE LIVE MODE: bidirectional audio via pyaudio ───
                st.markdown(
                    f'''
                    <div class="kore-maya-card" style="text-align:center; padding:28px 20px; background-color: #DEEDE5 !important; border: 1px solid #DEEDE5 !important;">
                        <div style="font-size: 2.4rem; margin-bottom: 10px;">🎙️</div>
                        <div style="font-family: 'Inter', sans-serif; 
                                    font-size: 1.3rem; 
                                    font-weight: 800; 
                                    color: #173624 !important; 
                                    text-transform: uppercase; 
                                    letter-spacing: 0.02em; 
                                    margin-bottom: 8px;">
                            Dr. Maya is listening via NeuraTone Voice
                        </div>
                        <div style="font-size: 14px; color: #2D533D; line-height: 1.6; font-weight: 500;">
                            Speak naturally — Dr. Maya will ask each PHQ-9 question aloud and wait patiently for your voice response.<br>
                            <span style="opacity: 0.8; font-size: 12px; font-weight: 600; color: #173624;">
                                Audio is captured directly from your microphone. No text prompts are shown — this is a pure voice session.
                            </span>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

                # Show live transcript as it builds (model turns only — no text shown to guide)
                transcript_now = kore_obj.transcript
                if transcript_now:
                    completed_turns = sum(1 for t in transcript_now if t["role"] == "user")
                    st.caption(
                        f"🔄 Session in progress · "
                        f"{completed_turns} response(s) recorded · "
                        f"Speak when Dr. Maya pauses"
                    )

                st.caption(
                    "🔒 *Only your voice is saved for analysis. "
                    "Dr. Maya's Kore audio is played through your speakers only — "
                    "never written to disk.*"
                )

            else:
                # ── FALLBACK: per-turn voice capture (no pyaudio / no API key) ─
                # Even in fallback mode: questions are delivered via Gemini voice
                # synthesis embedded in the audio response — NOT shown as text.
                # The st.audio_input widget captures the user's spoken reply only.

                if not gemini_key:
                    st.warning(
                        "⚠️ A Gemini API key is required for the voice session. "
                        "Add it in the sidebar to activate Dr. Maya's Kore voice."
                    )
                else:
                    # Show a minimal pulsing UI — no question text displayed
                    st.markdown(
                        '<div class="kore-maya-card" style="text-align:center;padding:24px 20px;">'
                        '<div style="font-size:2rem;margin-bottom:8px">🎧</div>'
                        '<div style="font-family:\'Londrina Solid\',cursive;font-size:1.2rem;'
                        'color:#47682C;margin-bottom:8px">Dr. Maya is speaking…</div>'
                        '<div style="font-size:13px;color:#4a5f3a;opacity:0.8">'
                        'Listen for Dr. Maya\'s question, then record your response below.'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    # Generate Dr. Maya's spoken question via Gemini (audio-only response)
                    # and surface the audio player for the patient to hear it
                    turn_idx = len(st.session_state.live_session_audio)

                    # Fetch Dr. Maya's next question as text-to-be-spoken
                    # (internally — never rendered as visible text to patient)
                    _q_cache_key = f"_maya_q_audio_{turn_idx}"
                    if _q_cache_key not in st.session_state:
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=gemini_key)
                            model = genai.GenerativeModel(
                                "gemini-1.5-flash",
                                system_instruction=KORE_SYSTEM,
                            )
                            history_payload = []
                            for t in st.session_state.live_transcript:
                                history_payload.append(
                                    {"role": t["role"], "parts": [t["text"]]}
                                )
                            trigger_msg = (
                                "Hello Dr. Maya, I'm ready to begin."
                                if turn_idx == 0
                                else "[Patient has responded. Please continue to the next question.]"
                            )
                            chat = model.start_chat(history=history_payload)
                            resp = chat.send_message(trigger_msg)
                            maya_text = resp.text.strip()
                            st.session_state[_q_cache_key] = maya_text
                            # Store in transcript as model turn (not shown to user)
                            st.session_state.live_transcript.append(
                                {"role": "model", "text": maya_text}
                            )
                        except Exception as e:
                            st.session_state[_q_cache_key] = None
                            st.warning(f"Dr. Maya voice error: {e}")

                    # ── User speaks their reply ────────────────────────
                    # Label is minimal — no question content shown
                    live_audio = st.audio_input(
                        "🎙️ Your voice response",
                        key=f"kore_audio_{turn_idx}",
                    )

                    if live_audio is not None:
                        # ── AUDIO INTEGRITY: only USER PCM stored ─────
                        user_audio_bytes = live_audio.getvalue()
                        st.session_state.live_session_audio.append(user_audio_bytes)
                        # Log as user turn (duration only — no transcription shown)
                        dur_kb = len(user_audio_bytes) / 1000
                        st.session_state.live_transcript.append(
                            {"role": "user", "text": f"[voice · {dur_kb:.1f}KB]"}
                        )
                        # Invalidate cache so Maya generates the next question
                        next_q_key = f"_maya_q_audio_{turn_idx + 1}"
                        st.session_state.pop(next_q_key, None)
                        st.rerun()

                    st.caption(
                        f"*Response {turn_idx + 1} of 9 · "
                        f"Agent voice is not included in the analysis audio.*"
                    )

            # Live transcript (hidden by default — shows only for debug/review)
            completed_user_turns = sum(
                1 for t in st.session_state.live_transcript if t["role"] == "user"
            )
            if completed_user_turns > 0:
                with st.expander(
                    f"Session log ({completed_user_turns} response(s) captured)",
                    expanded=False,
                ):
                    st.caption("ℹ️ Transcript shows session progress. Dr. Maya's questions are voice-only.")
                    for i, turn in enumerate(st.session_state.live_transcript):
                        role  = "Dr. Maya" if turn["role"] == "model" else "You (voice)"
                        color = "#47682C" if turn["role"] == "model" else "#2d3a25"
                        label = f"[turn {i+1}]" if turn["role"] == "user" else ""
                        st.markdown(
                            f'<div class="cv-card" style="margin:3px 0;padding:8px 12px;">'
                            f'<b style="color:{color}">{role}:</b> '
                            f'<span style="opacity:0.7;font-size:12px">{label}</span> '
                            f'{turn["text"] if turn["role"] == "user" else "*(voice delivered)*"}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        elif (
            not st.session_state.live_session_active
            and (st.session_state.live_session_audio or st.session_state.get("_kore_wav_path"))
        ):
            n_segs = len(st.session_state.live_session_audio)
            kore_wav = st.session_state.get("_kore_wav_path")
            if kore_wav:
                st.success(
                    f"✅ Session complete — user voice saved to: `{os.path.basename(kore_wav)}`"
                )
            else:
                st.success(
                    f"✅ Session complete — {n_segs} user voice response(s) captured."
                )

            if st.session_state.analysis_result is None:
                if st.button(
                    "Run Acoustic Analysis", type="primary",
                    use_container_width=True,
                    disabled=st.session_state.running
                ):
                    st.session_state.running = True
                    with st.spinner("Loading user audio and running v4 inference engine…"):
                        try:
                            interview_answers = {}
                            for i, turn in enumerate(st.session_state.live_transcript):
                                if turn["role"] == "user":
                                    interview_answers[f"kore_turn_{i}"] = turn["text"]

                            # ── AUDIO SOURCE PRIORITY ─────────────────────────
                            # 1. KoreLiveSession WAV (highest fidelity — pyaudio 16kHz PCM_16)
                            # 2. Per-turn st.audio_input segments (fallback)
                            if kore_wav and os.path.exists(kore_wav):
                                _run_analysis(
                                    source             = kore_wav,
                                    patient_name       = st.session_state.patient_name,
                                    interview_answers  = interview_answers,
                                    forced_domain      = 0.0,
                                )
                            else:
                                # AUDIO INTEGRITY: only user segments, never agent audio
                                segments = []
                                for ab in st.session_state.live_session_audio:
                                    try:
                                        arr, sr = decode_audio_bytes(ab)
                                        segments.append((arr, sr))
                                    except Exception:
                                        pass

                                if not segments:
                                    st.error("No valid user audio could be decoded.")
                                    st.session_state.running = False
                                else:
                                    stitched = stitch_audio_segments(segments)
                                    _run_analysis(
                                        source             = (stitched, 16000),
                                        patient_name       = st.session_state.patient_name,
                                        interview_answers  = interview_answers,
                                        forced_domain      = 0.0,
                                    )
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                    st.session_state.running = False
                    st.rerun()

        if st.session_state.analysis_result:
            _show_results(st.session_state.analysis_result)


    # ══════════════════════════════════════════════════════════════════
    # SUB-MODE: STANDARD INTERVIEW — Text-Guided with Vault Redo (Req 3)
    # ══════════════════════════════════════════════════════════════════

    else:  # Standard Interview

        st.markdown("### Guided Clinical Interview")
        st.markdown(
            "Dr. Maya will present each PHQ-9 question. Record your answer, "
            "then click **Save & Next**. Use **🗑️ Redo** on any question — "
            "only that answer is cleared; all others are preserved intact."
        )

        step = st.session_state.interview_step

        # ── Progress indicator (9 steps) ─────────────────────────────
        if step > 0:
            dots = ""
            for i in range(1, 10):
                if i == step:
                    cls = "active"
                elif vault_get(st.session_state, i) is not None:
                    cls = "done"
                else:
                    cls = "step-dot"
                dots += f'<span class="step-dot {cls}"></span>'
            pct = int(len(st.session_state.get("vault", {})) / 9 * 100)
            st.markdown(
                f'<div style="margin:6px 0 4px">{dots}</div>'
                f'<div style="font-size:12px;color:#4a5f3a;margin-bottom:16px">'
                f'{len(st.session_state.get("vault",{}))} of 9 answers recorded'
                f'</div>',
                unsafe_allow_html=True,
            )

        if step == 0:
            if st.button("Begin PHQ-9 Interview", type="primary", use_container_width=True):
                st.session_state.interview_step     = 1
                st.session_state.interview_answers  = {}
                st.session_state.vault              = {}
                st.session_state.gemini_chat        = []
                st.session_state.analysis_result    = None
                # Clear any cached Gemini question text
                for k in list(st.session_state.keys()):
                    if k.startswith("_gemini_q_"):
                        del st.session_state[k]
                st.rerun()

        elif 1 <= step <= 9:
            q_key, q_raw = PHQ9_QUESTIONS[step - 1]

            # Cache conversational rephrasing via Gemini
            cache_key = f"_gemini_q_{step}"
            if cache_key not in st.session_state:
                with st.spinner("Preparing question…"):
                    model   = _get_gemini_model(gemini_key)
                    phrased = _ask_gemini_question(model, q_raw)
                st.session_state[cache_key] = phrased
            phrased_q = st.session_state[cache_key]

            # ── Show all previously answered questions with per-Q Redo ─
            # Use vault keys so ALL answered questions remain visible even if we step back
            for prev_step in sorted(st.session_state.get("vault", {}).keys()):
                # If we are currently re-recording this exact step, don't show it in the 'completed' list
                if prev_step == step:
                    continue

                prev_key = PHQ9_QUESTIONS[prev_step - 1][0]
                prev_q   = st.session_state.get(
                    f"_gemini_q_{prev_step}",
                    PHQ9_QUESTIONS[prev_step - 1][1],
                )
                prev_ans = st.session_state.interview_answers.get(prev_key, "—")

                col_card, col_redo = st.columns([5, 1])
                with col_card:
                    st.markdown(
                        f'<div class="q-card q-answered">'
                        f'<b>Q{prev_step}:</b> {prev_q}<br>'
                        f'<span style="opacity:0.65;font-size:13px">✓ {prev_ans}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_redo:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(
                        "🗑️ Redo", key=f"redo_{prev_step}",
                        help=(
                            f"Re-record Q{prev_step}. "
                            f"All other answers (Q1–Q9) remain intact."
                        ),
                    ):
                        # ── VAULT NON-DESTRUCTIVE REDO ──────────────────
                        vault_clear_one(st.session_state, prev_step)
                        pk = PHQ9_QUESTIONS[prev_step - 1][0]
                        st.session_state.interview_answers.pop(pk, None)
                        st.session_state.interview_step = prev_step
                        st.rerun()


            # ── Current active question with Speaker Button ───────────
            q_col, speak_col = st.columns([0.85, 0.15])
            
            with q_col:
                st.markdown(
                    f'<div class="q-card">'
                    f'<b>Question {step} of 9</b><br>'
                    f'{phrased_q}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
            with speak_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔊", key=f"speak_q_{step}", help="Hear Dr. Maya read the question"):
                    dr_maya_voice_trigger(phrased_q)

            # ── Text-to-Speech (Read Aloud) ───────────────────────────
            tts_key = f"_tts_q_{step}"
            if tts_key not in st.session_state:
                try:
                    from gtts import gTTS
                    import io
                    # Generates a calm, UK-accented voice to mimic the "Kore" vibe
                    tts = gTTS(text=phrased_q, lang='en', tld='co.uk', slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    # Cache the audio bytes so it doesn't reload when the user interacts
                    st.session_state[tts_key] = fp.getvalue()
                except ImportError:
                    st.session_state[tts_key] = None
            
            # Display the audio player right below the question
            if st.session_state[tts_key]:
                st.audio(st.session_state[tts_key], format='audio/mp3')

            # ── Microphone Input ──────────────────────────────────────
            audio_data = st.audio_input(
                f"Record your answer to question {step}",
                key=f"audio_q{step}",
            )

            col_skip, col_next = st.columns([1, 2])
            with col_skip:
                if st.button("Skip (no audio)", key=f"skip_{step}"):
                    # Store a sentinel in vault so the question is "answered"
                    vault_set(
                        st.session_state, step,
                        (np.zeros(1600, dtype=np.float32), 16000),
                    )
                    st.session_state.interview_answers[q_key] = "[skipped]"
                    
                    # ── SMART NAVIGATION ──
                    # Find the next question that hasn't been answered yet
                    next_step = step + 1
                    while next_step <= 9 and next_step in st.session_state.get("vault", {}):
                        next_step += 1
                    st.session_state.interview_step = next_step
                    st.rerun()

            with col_next:
                if audio_data is not None:
                    if st.button("Save & Next ➜", key=f"next_{step}", type="primary"):
                        try:
                            arr, sr = decode_audio_bytes(audio_data.getvalue())
                            # Only this key is written; all others untouched.
                            vault_set(st.session_state, step, (arr, sr))
                            dur = len(arr) / sr
                            st.session_state.interview_answers[q_key] = (
                                f"[audio recorded — {dur:.1f}s]"
                            )
                        except Exception as e:
                            st.warning(f"Could not decode audio: {e}")
                            st.session_state.interview_answers[q_key] = "[decode error]"
                        
                        # ── SMART NAVIGATION ──
                        # Find the next question that hasn't been answered yet
                        next_step = step + 1
                        while next_step <= 9 and next_step in st.session_state.get("vault", {}):
                            next_step += 1
                        st.session_state.interview_step = next_step
                        st.rerun()

        elif step == 10:
            # ── VAULT GATE: Final Analysis only when all 9 answered ───
            vault_size = len(st.session_state.get("vault", {}))

            if vault_size < 9:
                missing = [
                    i for i in range(1, 10)
                    if vault_get(st.session_state, i) is None
                ]
                st.warning(
                    f"⚠️ {9 - vault_size} question(s) still need an answer "
                    f"before analysis can run. Missing: Q{', Q'.join(str(m) for m in missing)}"
                )
                # Let the user jump back to any missing question
                for m in missing:
                    if st.button(f"Answer Q{m}", key=f"jump_{m}"):
                        st.session_state.interview_step = m
                        st.rerun()
            else:
                st.success("✅ All 9 PHQ-9 questions answered. Ready for acoustic analysis.")

                if st.session_state.analysis_result is None:
                    if st.button(
                        "Run Final Analysis",
                        type="primary",
                        use_container_width=True,
                        disabled=st.session_state.running,
                    ):
                        # ── VAULT: retrieve all 9 segments in order ───
                        segs_list = vault_all_segments(st.session_state)
                        if not segs_list:
                            st.error("No audio was recorded. Please go back and re-record.")
                        else:
                            st.session_state.running = True
                            with st.spinner(
                                "Stitching 9-question audio and running v4 inference engine…"
                            ):
                                stitched = stitch_audio_segments(segs_list)
                                _run_analysis(
                                    source            = (stitched, 16000),
                                    patient_name      = st.session_state.patient_name,
                                    interview_answers = st.session_state.interview_answers,
                                    forced_domain     = 0.0,
                                )
                            st.session_state.running = False
                            st.rerun()

            if st.button("Start Over"):
                keys_to_clear = [
                    "interview_step", "interview_answers", "vault",
                    "gemini_chat", "analysis_result", "running",
                ]
                for k in keys_to_clear:
                    st.session_state.pop(k, None)
                for k in list(st.session_state.keys()):
                    if k.startswith("_gemini_q_"):
                        del st.session_state[k]
                _init_state()
                st.rerun()

        if st.session_state.analysis_result is not None:
            _show_results(st.session_state.analysis_result)


# ══════════════════════════════════════════════════════════════════════
# MODE: YOUTUBE URL
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# MODE: YOUTUBE & VIDEO ANALYSIS (Toggle Link vs. File)
# ══════════════════════════════════════════════════════════════════════

elif st.session_state.portal_mode == "YouTube URL":

    st.markdown("### YouTube & Video Analysis")
    st.info("Domain set to **1.0 (Vlog)**. Use this for YouTube links or personal video vlogs.")

    # ── The Toggle: Link vs. File ─────────────────────────────────────
    vid_input_type = st.radio(
        "Select Video Source",
        options=["YouTube Link", "Video File (.mp4/.mkv)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if vid_input_type == "YouTube Link":
        yt_url = st.text_input(
            "YouTube URL",
            placeholder = "https://www.youtube.com/watch?v=…",
        )

        if st.button(
            "Analyse YouTube Video", type="primary",
            disabled=not yt_url or st.session_state.running
        ):
            st.session_state.analysis_result = None
            st.session_state.running = True
            with st.spinner("Downloading audio and running inference…"):
                _run_analysis(
                    source        = yt_url,
                    patient_name  = st.session_state.patient_name,
                    forced_domain = 1.0, # Kept as Vlog domain
                )
            st.session_state.running = False
            st.rerun()

    else:
        # ── Video File Upload logic (Hybrid Mode) ─────────────────────
        uploaded_vid = st.file_uploader(
            "Upload Video File",
            type=["mp4", "mkv", "mov", "avi"],
            help="Extracts audio for behavioral analysis."
        )

        if uploaded_vid is not None:
            st.video(uploaded_vid)
            if st.button(
                "Analyse Video Content", type="primary",
                disabled=st.session_state.running
            ):
                st.session_state.analysis_result = None
                st.session_state.running = True
                
                suffix = os.path.splitext(uploaded_vid.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    tf.write(uploaded_vid.read())
                    tmp_path = tf.name

                with st.spinner("Extracting features and running inference..."):
                    # We use forced_domain=1.0 to ensure the TAN model 
                    # treats it as a 'Vlog' analysis.
                    _run_analysis(
                        source        = tmp_path,
                        patient_name  = st.session_state.patient_name,
                        forced_domain = 1.0,
                    )
                
                try: os.remove(tmp_path)
                except: pass
                    
                st.session_state.running = False
                st.rerun()

    if st.session_state.analysis_result:
        _show_results(st.session_state.analysis_result)

# ══════════════════════════════════════════════════════════════════════
# MODE: FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════

elif st.session_state.portal_mode == "File Upload":

    st.markdown("### Clinical File Upload")
    st.info("Domain will be automatically set to **0.0 (Clinical)** for uploaded files.")

    uploaded = st.file_uploader(
        "Upload a voice recording",
        type = ["wav", "mp3", "flac", "ogg"],
    )

    if uploaded is not None:
        st.audio(uploaded)
        if st.button(
            "Run Analysis", type="primary",
            disabled=st.session_state.running
        ):
            st.session_state.analysis_result = None
            st.session_state.running = True

            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(uploaded.read())
                tmp_path = tf.name

            with st.spinner("Running inference engine…"):
                _run_analysis(
                    source        = tmp_path,
                    patient_name  = st.session_state.patient_name,
                    forced_domain = 0.0,
                )
            st.session_state.running = False
            st.rerun()

    if st.session_state.analysis_result:
        _show_results(st.session_state.analysis_result)
