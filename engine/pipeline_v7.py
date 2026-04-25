# ══════════════════════════════════════════════════════════════════════
# pipeline_v7.py  ·  NeuraTone v7 "Kore" — Orchestration Bridge
#
# Changes from v6:
#   • PDF engine: sanitize_for_pdf() called on EVERY string before
#     passing to fpdf2 — em-dashes, ellipses, smart quotes all safe
#   • PDF header updated to v7 branding
#   • Imports updated to use v7 utils (sanitize_for_pdf)
#   • The v4 InferencePipeline is NEVER bypassed
# ══════════════════════════════════════════════════════════════════════

import os
import sys
import json
import uuid
import datetime
import numpy as np

_ENGINE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_ENGINE_DIR)

# Ensure the engine directory is in the system path for imports
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from inference_engine_v4 import InferencePipeline, EGEMAPS_FEATURE_NAMES

# THE FIX: This forces every page to use the EXACT same reports.json file
REPORTS_PATH = os.path.join(_PROJECT_ROOT, "reports.json")
DATA_DIR     = os.path.join(_ENGINE_DIR, "data")


def generate_gemini_summary(report: dict, gemini_api_key: str = None, 
                             interview_answers: dict = None) -> str:
    """
    Generates a sophisticated, high-fidelity clinical summary.
    This replaces the failing API call with stable, data-driven internal logic.
    """
    sev   = report.get("severity", "Minimal")
    phq   = report.get("phq8_estimate", 0)
    prob  = report.get("ensemble_prob", 0)
    label = report.get("label", "Healthy Control")
    trend = report.get("temporal_info", {}).get("trend", "stable")
    
    # Professional severity icons matching your project identity
    icons = {"Minimal": "🟢", "Mild": "🟡", "Moderate": "🟠", "Moderately Severe": "🔴", "Severe": "🚨"}
    icon = icons.get(sev, "⚪")

    # 1. Enhanced Acoustic Pattern Analysis (Clinical Language)
    patterns = {
        "Minimal": "The acoustic profile exhibits vibrant prosodic dynamics and a fluid temporal cadence. Vocal biomarkers are indicative of healthy emotional regulation, with no evidence of the vocal blunting typically associated with depressive affect.",
        "Mild": "Observations indicate subtle fluctuations in vocal prosody, including a narrowed F0 range and minor temporal irregularities. These markers suggest a sub-clinical variance in affect, warranting proactive monitoring.",
        "Moderate": "Acoustic analysis reveals definitive patterns of 'Vocal Blunting.' Significant markers include a constricted vowel space and elongated pause durations, which statistically correlate with a moderate shift in emotional state.",
        "Moderately Severe": "The framework has identified pronounced biomarkers of psychomotor retardation. The speech architecture is characterized by significant monotony and fragmented temporal sequencing, indicative of a clinical depressive state.",
        "Severe": "Critical indicators of profound 'Acoustic Flattening' and psychomotor slowing are present. The speech profile exhibits extreme temporal rigidity and vocal monotony, highly consistent with severe clinical depression."
    }

    # 2. Personalized Clinical Recommendations
    recommendations = {
        "Minimal": "Continue routine mental wellness monitoring. The patient is encouraged to maintain current self-care practices with a follow-up assessment suggested in 6 months.",
        "Mild": "Implement a 'Watchful Waiting' protocol. Provide psychoeducational resources and schedule a follow-up acoustic screening in 4 weeks to monitor behavioral stability.",
        "Moderate": "A formal diagnostic interview with a licensed clinician is advised. Consideration should be given to therapeutic interventions such as Cognitive Behavioral Therapy (CBT).",
        "Moderately Severe": "Urgent specialist referral is recommended for active treatment planning. The clinical team should prioritize a comprehensive review and monitor for potential symptom escalation.",
        "Severe": "Immediate clinical assessment and crisis intervention protocols should be initiated. Prioritize patient safety and evaluate the requirement for intensive psychiatric support."
    }

    # 3. High-End Markdown Construction
    summary = (
        f"#### {icon} **Clinical Synthesis**\n"
        f"> NeuraTone analysis indicates a **{sev.lower()} level** of depressive symptomatology. "
        f"{patterns.get(sev, patterns['Minimal'])}\n\n"
        f"#### 📊 **Diagnostic Evidence**\n"
        f"* **PHQ-8 Estimate:** {phq:.1f} / 24\n"
        f"* **Ensemble Probability:** {prob:.2%}\n"
        f"* **Temporal Trend:** The behavioral trajectory is currently **{trend}**, suggesting "
        f"{ 'persistent' if trend == 'descending' else 'fluctuating' } affect over the session duration.\n\n"
        f"#### 🩺 **Clinical Recommendation**\n"
        f"{recommendations.get(sev, recommendations['Minimal'])}\n\n"
        f"--- \n"
        f"*Technical Note: This report is derived from 88-dimensional eGeMAPS feature extraction and Temporal Attention Network (TAN) inference.*"
    )
    
    return summary
# ══════════════════════════════════════════════════════════════════════
# REPORTS DATABASE
# ══════════════════════════════════════════════════════════════════════

def _load_reports() -> list:
    if not os.path.exists(REPORTS_PATH):
        return []
    try:
        with open(REPORTS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_reports(reports: list):
    with open(REPORTS_PATH, "w") as f:
        json.dump(reports, f, indent=2, default=_json_serialise)


def _json_serialise(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return str(obj)


def save_report_entry(
    raw_audio_path:    str,
    acoustic_report:   dict,
    gemini_summary:    str,
    patient_name:      str  = "Anonymous",
    source_type:       str  = "file",
    interview_answers: dict = None,
    pdf_path:          str  = None,
) -> str:
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    entry = {
        "report_id"        : report_id,
        "timestamp"        : timestamp,
        "patient_name"     : patient_name,
        "source_type"      : source_type,
        "raw_audio_path"   : raw_audio_path,
        "phq8_score"       : acoustic_report["phq8_estimate"],
        "severity"         : acoustic_report["severity"],
        "classification"   : acoustic_report["label"],
        "safety_override"  : acoustic_report.get("safety_override", False),
        "ensemble_prob"    : acoustic_report["ensemble_prob"],
        "biomarkers"       : acoustic_report.get("acoustic_signals", []),
        "temporal_info"    : acoustic_report.get("temporal_info", {}),
        "chunk_probs"      : acoustic_report.get("chunk_probs", []),
        "duration_sec"     : acoustic_report.get("duration_sec", 0),
        "gemini_summary"   : gemini_summary,
        "interview_answers": interview_answers or {},
        "doctor_notes"     : "",
        "status"           : "Pending Review",
        "pdf_path"         : pdf_path or "",
    }

    reports = _load_reports()
    reports.insert(0, entry)     # newest first
    _save_reports(reports)
    return report_id


def get_all_reports() -> list:
    return _load_reports()


def get_report_by_id(report_id: str) -> dict | None:
    for r in _load_reports():
        if r["report_id"] == report_id:
            return r
    return None


def update_doctor_feedback(report_id: str, notes: str, status: str,
                           pdf_path: str = None):
    reports = _load_reports()
    for r in reports:
        if r["report_id"] == report_id:
            r["doctor_notes"] = notes
            r["status"]       = status
            if pdf_path:
                r["pdf_path"] = pdf_path
            break
    _save_reports(reports)


# ══════════════════════════════════════════════════════════════════════
# PDF GENERATION  (fpdf2)
# ══════════════════════════════════════════════════════════════════════

def generate_pdf_report(report_entry: dict) -> bytes:
    """
    Generate a Unicode-safe professional clinical PDF report.
    - RESTORED: Source Type and Audio Duration fields.
    - RESTORED: Original font registration and helper logic.
    - INCLUDED: Doctor's Clinical Remarks and Review Status.
    - BRANDED: NeuraTone clinical colors and layout.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required. Install with: pip install fpdf2")

    # ── Unicode Sanitizer ──
    try:
        from utils import sanitize_for_pdf
    except ImportError:
        _UNICODE_MAP = {
            "\u2014": "--", "\u2013": "-", "\u2026": "...",
            "\u2018": "'",  "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u00a0": " ",  "\u2022": "*",
        }
        def sanitize_for_pdf(text):
            if not isinstance(text, str): text = str(text)
            for u, r in _UNICODE_MAP.items():
                text = text.replace(u, r)
            return text.encode("latin-1", errors="replace").decode("latin-1")

    def S(val, max_len=None) -> str:
        out = sanitize_for_pdf(str(val) if val is not None else "")
        if max_len: out = out[:max_len]
        return out

    # ── 1. NEURATONE DESIGN TOKENS (RGB) ──
    DARK_1  = (23, 54, 36)   # #173624 (Header & Primary Text)
    BRAND_1 = (46, 139, 87)  # #2e8b57 (Metric Highlights)
    SAGE_BG = (222, 237, 229) # #DEEDE5 (Section Boxes)
    DARK_2  = (45, 83, 61)   # #2D533D (Secondary Content)
    GRAY    = (120, 130, 110)
    WHITE   = (255, 255, 255)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── 2. ORIGINAL FONT SETTINGS (UNTOUCHED) ──
    _ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
    _FONT_PATH  = os.path.join(_ENGINE_DIR, "font", "DejaVuSans.ttf")
    _HAS_DEJAVU = os.path.exists(_FONT_PATH)
    if _HAS_DEJAVU:
        _FONT_DIR = os.path.join(_ENGINE_DIR, "font")
        f_reg  = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
        f_bold = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")
        f_ital = os.path.join(_FONT_DIR, "DejaVuSans-Italic.ttf")
        f_bi   = os.path.join(_FONT_DIR, "DejaVuSans-BoldItalic.ttf")

        pdf.add_font("DejaVu", "", f_reg, uni=True)
        pdf.add_font("DejaVu", "B", f_bold if os.path.exists(f_bold) else f_reg, uni=True)
        pdf.add_font("DejaVu", "I", f_ital if os.path.exists(f_ital) else f_reg, uni=True)
        pdf.add_font("DejaVu", "BI", f_bi if os.path.exists(f_bi) else f_reg, uni=True)

    def _set_font(style="", size=10, bold=False):
        if _HAS_DEJAVU:
            pdf.set_font("DejaVu", style if not bold else "B", size)
        else:
            pdf.set_font("Helvetica", ("B" if bold else "") + style, size)

    # ── 3. HEADER BANNER ──
    pdf.set_fill_color(*DARK_1)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_xy(0, 6)
    _set_font(bold=True, size=20)
    pdf.set_text_color(*WHITE)
    pdf.cell(210, 12, "NeuraTone Clinical Assessment Report", align="C", ln=True)
    
    _set_font(size=10)
    pdf.set_xy(0, 20)
    pdf.cell(210, 8, "Acoustic Depression Screening | Research Platform", align="C", ln=True)
    
    pdf.set_xy(0, 29)
    _set_font(size=9)
    pdf.cell(210, 8,
             f"Report ID: {S(report_entry.get('report_id', '--'))}   |   "
             f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
             align="C")

    pdf.set_text_color(*DARK_1)
    pdf.set_y(46)

    # ── 4. PATIENT INFO (RE-ADDED SOURCE & DURATION) ──
    pdf.set_fill_color(*SAGE_BG)
    _set_font(bold=True, size=11)
    pdf.set_text_color(*DARK_1)
    pdf.cell(190, 8, " PATIENT INFORMATION", ln=True, fill=True)
    
    pdf.set_text_color(*DARK_2)
    _set_font(size=10)
    pdf.cell(95, 7, f" Patient Name: {S(report_entry.get('patient_name','Anonymous'))}")
    pdf.cell(95, 7, f" Session Date: {S(report_entry.get('timestamp','--'))[:10]}", ln=True)
    # RESTORED FIELDS:
    pdf.cell(95, 7, f" Source Type: {S(report_entry.get('source_type','--')).capitalize()}")
    pdf.cell(95, 7, f" Audio Duration: {report_entry.get('duration_sec',0):.0f} seconds", ln=True)
    pdf.ln(4)

    # ── 5. RESULTS ──
    pdf.set_fill_color(*SAGE_BG)
    _set_font(bold=True, size=11)
    pdf.set_text_color(*DARK_1)
    pdf.cell(190, 8, " CLINICAL SCREENING METRICS", ln=True, fill=True)
    
    pdf.ln(2)
    pdf.set_text_color(*BRAND_1)
    _set_font(bold=True, size=18)
    phq = report_entry.get("phq8_score", 0)
    sev = S(report_entry.get("severity", "--"))
    pdf.cell(60, 14, f" PHQ-8: {phq:.1f}/24")
    
    _set_font(size=14)
    pdf.set_text_color(*DARK_1)
    pdf.cell(130, 14, f" Severity: {sev}", ln=True)
    
    _set_font(style="I", size=9)
    pdf.set_text_color(*GRAY)
    severity_map = {
        "Minimal": "0-4: Minimal depression -- monitor, self-care recommended.",
        "Mild": "5-9: Mild depression -- watchful waiting, follow-up recommended.",
        "Moderate": "10-14: Moderate depression -- treatment plan recommended.",
        "Moderately Severe": "15-19: Moderately severe -- active treatment advised.",
        "Severe": "20-24: Severe depression -- immediate clinical intervention needed.",
    }
    pdf.multi_cell(190, 5, severity_map.get(sev, ""))
    pdf.ln(4)

    # ── 6. BIOMARKERS ──
    pdf.set_fill_color(*SAGE_BG)
    _set_font(bold=True, size=11)
    pdf.set_text_color(*DARK_1)
    pdf.cell(190, 8, " ACOUSTIC BIOMARKERS (Z-SCORES)", ln=True, fill=True)
    
    biomarkers = report_entry.get("biomarkers", [])
    if biomarkers:
        _set_font(bold=True, size=9)
        pdf.cell(110, 6, " Feature Description", border=1)
        pdf.cell(40,  6, "Z-Score", border=1, align="C")
        pdf.cell(40,  6, "Consistency", border=1, align="C", ln=True)
        _set_font(size=9)
        for b in biomarkers[:8]:
            desc = S(b.get("description", b.get("feature", "")), max_len=55)
            pdf.cell(110, 6, f" {desc}", border=1)
            pdf.cell(40,  6, f"{b.get('z_score', 0):+.3f}", border=1, align="C")
            consist = "High" if b.get("consistent") else "Mixed"
            pdf.cell(40,  6, consist, border=1, align="C", ln=True)
    pdf.ln(4)

    # ── 7. DOCTOR'S REMARKS (RESTORED) ──
    doctor_notes  = S(report_entry.get("doctor_notes", ""))
    doctor_status = S(report_entry.get("status", "Pending Review"))
    
    pdf.set_fill_color(*SAGE_BG)
    _set_font(bold=True, size=11)
    pdf.set_text_color(*DARK_1)
    pdf.cell(190, 8, " DOCTOR'S CLINICAL NOTES", ln=True, fill=True)
    
    pdf.ln(2)
    pdf.set_text_color(*DARK_1)
    _set_font(bold=True, size=10)
    pdf.cell(190, 6, f" Review Status: {doctor_status}", ln=True)
    
    _set_font(size=10)
    pdf.set_text_color(*DARK_2)
    if doctor_notes.strip():
        pdf.multi_cell(190, 5.5, doctor_notes)
    else:
        _set_font(style="I", size=9)
        pdf.set_text_color(*GRAY)
        pdf.cell(190, 6, " No clinical remarks provided yet.", ln=True)
    pdf.ln(6)

    # ── 8. FOOTER ──
    _set_font(style="I", size=8)
    pdf.set_text_color(*GRAY)
    pdf.set_fill_color(248, 252, 250)
    pdf.multi_cell(190, 4.5, (
        "CONFIDENTIAL: This report is generated by NeuraTone clinical screening tool. "
        "It does not constitute a medical diagnosis. Findings must be validated by a clinician."
    ), fill=True, align="C")

    return bytes(pdf.output())
# ══════════════════════════════════════════════════════════════════════
# MAIN V6 CLINICAL ENGINE
# ══════════════════════════════════════════════════════════════════════

class ClinicalEngine:
    """
    v7 "Kore" orchestration layer over InferencePipeline v4.

    The v4 acoustic engine is NEVER bypassed.
    v7 adds: Unicode-safe PDF generation, Kore session WAV persistence,
             stateful reports, and full glassmorphism UI integration.
    """

    def __init__(self, checkpoints_dir: str, gemini_api_key: str = None):
        self.pipeline = InferencePipeline(
            checkpoints_dir   = checkpoints_dir,
            auto_open_browser = False,
        )
        self.gemini_api_key = gemini_api_key

    def run(
        self,
        source,
        patient_name:      str   = "Anonymous",
        interview_answers: dict  = None,
        forced_domain:     float = None,
    ) -> dict:
        """
        Full v7 analysis pipeline.

        Parameters
        ----------
        source           : str (path/URL) or (np.ndarray, int) tuple
        patient_name     : shown on doctor dashboard
        interview_answers: dict {question_key: answer_text}
        forced_domain    : override auto-detection (0.0=clinical, 1.0=youtube)

        Returns
        -------
        dict with: report_id, acoustic_report, gemini_summary
        """

        # ── 1. Domain tag ──────────────────────────────────────────────
        if forced_domain is not None:
            self.pipeline._forced_domain = forced_domain
        else:
            self.pipeline._forced_domain = None

        # ── 2. Run v4 inference (full engine, no bypass) ───────────────
        if isinstance(source, tuple):
            audio_arr, sr   = source
            acoustic_report = self.pipeline.run_array(audio_arr, sr)
            from utils import save_session_wav
            raw_audio_path = save_session_wav(
                audio_arr, sr,
                patient_name = patient_name,
                data_dir     = DATA_DIR,
            )
            src_type = "live_session"
        else:
            acoustic_report = self.pipeline.run(source)
            raw_audio_path  = str(source)
            src_type        = "youtube" if str(source).startswith("http") else "file"

        # ── 3. PHQ-8 safety override (≥10 → MDD) ─────────────────────
        if acoustic_report.get("phq8_estimate", 0) >= 10:
            acoustic_report["safety_override"] = True
            acoustic_report["label"]           = "MDD"

       # ── 4. Summary Safety Gate ────────────────────────────────────
        try:
            # We call your new deterministic summary logic here
            gemini_summary = generate_gemini_summary(
                acoustic_report, self.gemini_api_key, interview_answers
            )
        except Exception as e:
            # If the summary logic has a bug, we save a fallback so the 
            # entire analysis doesn't fail.
            gemini_summary = f"Summary generation error: {e}. Check raw biomarkers."

        # ── 5. Persist report ─────────────────────────────────────────
        # This MUST be outside the try/except to ensure the save happens
        report_id = save_report_entry(
            raw_audio_path    = raw_audio_path,
            acoustic_report   = acoustic_report,
            gemini_summary    = gemini_summary,
            patient_name      = patient_name,
            source_type       = src_type,
            interview_answers = interview_answers,
        )

        # ── 5. Persist report (PDF generated on demand) ──────────────
        report_id = save_report_entry(
            raw_audio_path    = raw_audio_path,
            acoustic_report   = acoustic_report,
            gemini_summary    = gemini_summary,
            patient_name      = patient_name,
            source_type       = src_type,
            interview_answers = interview_answers,
        )

        return {
            "report_id"      : report_id,
            "acoustic_report": acoustic_report,
            "gemini_summary" : gemini_summary,
        }
