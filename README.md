# NeuraTone v7 — "Kore" Luxe-Clinical Edition

A high-end, voice-first clinical depression screening platform powered by
the Kore voice profile, Gemini Live API, and the TAN + LightGBM Inference
Engine v4.

---

## What's New in v7

### 1. "Dramatic but Playful" Luxe UI
- **Background**: `#D6F3CE` (Mint) with radial depth gradients
- **Headings**: `Londrina Solid` — BIG, uppercase, impactful (900 weight)
- **Body**: `Inter` for clinical readability
- **Glassmorphism**: All cards use `rgba(255,255,255,0.40)` + `backdrop-filter: blur(16px)`
- **Buttons**: Rounded pill shape with Forest green gradient

### 2. Kore Voice-to-Voice Portal (Dr. Maya)
- **Model**: `models/gemini-3.1-flash-live-preview` via `google-genai` SDK
- **Voice**: `voice_name="Kore"` — warm, nurturing, unhurried
- **Persona**: Dr. Maya — *"You are an empathetic psychologist. Speak with a warm,
  nurturing, and calm tone. Do not rush. Listen for pauses as signs of reflection."*
- **Unified UI**: "Live Session" + "Guided Interview" merged into one
  **Diagnostic Session** page with a rounded pill toggle:
  - `Interactive Agent (Voice-to-Voice)` — Kore Live API
  - `Standard Interview (Text-Guided)` — per-question audio capture

### 3. Audio Integrity — Agent Voice Stripping
The Kore agent's voice output is **never** stored or passed to the v4 engine.
Only the **user's** audio segments are:
1. Captured per-turn into `st.session_state.live_session_audio`
2. Stitched together with 500ms silence gaps
3. Fed into the v4 InferencePipeline (TAN + LightGBM)

### 4. Bulletproof "Redo" Logic — The Vault
```python
st.session_state.vault = {
    1: (arr_q1, sr),   # Question 1 audio
    2: (arr_q2, sr),   # Question 2 audio
    ...
}
```
- `vault_set(state, q_id, data)` — store audio for one question
- `vault_clear_one(state, q_id)` — clear ONLY that question's audio
- `vault_get(state, q_id)` — retrieve a specific question's audio
- `vault_all_segments(state)` — ordered list for stitching

Clicking "Redo" on Question 1 calls `vault_clear_one(state, 1)`.
**All data for Q2–Q5 is completely untouched.**

### 5. Unicode-Ready PDF Engine
The `sanitize_for_pdf()` function in `utils.py` is called on **every** string
before it reaches fpdf2's Latin-1 renderer:

```python
sanitize_for_pdf("Patient noted feeling hopeless — struggling daily…")
# → "Patient noted feeling hopeless -- struggling daily..."
```

Handles: `—` `–` `…` `'` `'` `"` `"` `•` `·` `™` `®` `©` `é` `ü` `ö` and more.

---

## Setup

### 1. Install dependencies
```bash
pip install streamlit google-generativeai fpdf2 soundfile pydub librosa \
            scipy numpy altair pandas
```

### 2. DejaVu Font (optional but recommended)
For full Unicode PDF support with fpdf2, download `DejaVuSans.ttf` and place it in:
```
engine/font/DejaVuSans.ttf
```
Download from: https://dejavu-fonts.github.io/Download.html

If the font file is not present, the PDF engine falls back to Helvetica with
the `sanitize_for_pdf()` safety layer (ASCII-safe).

### 3. Checkpoints
Place your v4 model checkpoints in a directory (default: `checkpoints/`):
```
checkpoints/
  TAN_best.pt
  lgb_v3.pkl
  scaler_v3.pkl
  feature_names_v3.json
```

### 4. Run
```bash
streamlit run app.py
```

---

## Architecture

```
app.py
  └─ pages/
       ├─ patient_portal.py   — Diagnostic Session (Kore + Standard Interview)
       ├─ my_results.py       — Patient result lookup + PDF download
       └─ doctor_dashboard.py — Clinical review queue + feedback

engine/
  ├─ utils.py                 — Kore CSS, Vault helpers, PDF sanitizer
  ├─ pipeline_v7.py           — Orchestration (Unicode-safe PDF, ClinicalEngine)
  ├─ inference_engine_v4.py   — TAN + LightGBM acoustic engine (never bypassed)
  ├─ real_time_feature_extractor.py
  ├─ data/                    — Session WAVs saved here
  └─ font/
       └─ DejaVuSans.ttf      — (place here for Unicode PDF support)

reports.json                  — Persistent report store
```

---

## Kore Live API (Production Configuration)

In production with full `google-genai` Live API access, replace the
`gemini-1.5-flash` text fallback with:

```python
from google import genai as gai

client = gai.Client(api_key=gemini_key)
config = gai.types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=gai.types.SpeechConfig(
        voice_config=gai.types.VoiceConfig(
            prebuilt_voice_config=gai.types.PrebuiltVoiceConfig(
                voice_name="Kore"
            )
        )
    ),
    system_instruction=KORE_SYSTEM,
)
async with client.aio.live.connect(
    model="models/gemini-3.1-flash-live-preview",
    config=config
) as session:
    # Stream user audio in, receive Kore voice back
    # Store only user audio chunks in live_session_audio
    ...
```

Requires: `google-genai >= 0.8.0` with Live API access enabled on your project.

---

## Disclaimer

This tool is for research and clinical support purposes only.
It does not constitute a medical diagnosis.
All findings must be reviewed by a qualified clinician.
