# ══════════════════════════════════════════════════════════════════════
# pages/my_results.py  ·  NeuraTone v7 — Patient "My Results" Portal
# ══════════════════════════════════════════════════════════════════════

import os
import sys
import datetime
import streamlit as st

# ── 1. PATH SETUP ───────────────────────────────────────────────────
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = os.path.join(_ROOT, "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from utils import GLOBAL_CSS, SEVERITY_COLORS, SEVERITY_EMOJI
from pipeline_v7 import get_report_by_id, generate_pdf_report

# Apply NeuraTone Global Styles (Loads Google Icons and Fonts)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── 2. SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About My Results")
    st.markdown(
        "Enter your **Report ID** to view your clinical findings.\n\n"
        "Status will update automatically once a clinician has reviewed your session."
    )
    st.divider()
    st.caption("NeuraTone v7 · Clinical Research")
    st.caption("⚠️ Not a medical diagnosis.")

# ── 3. CENTERED HERO LAYOUT ─────────────────────────────────────────
# Fixes 'off' composition by centering the lookup experience
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Page Header
    st.markdown(
        '''
        <div style="text-align: center; margin-bottom: 40px; margin-top: 10px;">
            <h1 style="margin-bottom: 8px;">My Results</h1>
            <p style="color: #476B57; font-weight: 500; opacity: 0.8; font-size: 15px;">
                Securely retrieve your acoustic screening report and clinician notes.
            </p>
        </div>
        ''', 
        unsafe_allow_html=True
    )

    # Lookup Card (Sage #DEEDE5 / Google Icons)
    st.markdown(
        f'''
        <div class="results-lookup" style="
            text-align: center; 
            padding: 50px 30px; 
            background-color: #DEEDE5 !important; 
            border: 1px solid #DEEDE5;
            border-radius: 24px;
            box-shadow: 0 8px 30px rgba(23, 54, 36, 0.05);
        ">
            <span class="material-symbols-outlined" style="font-size: 48px; color: #2e8b57; margin-bottom: 16px;">
                shield_person
            </span>
            <h5 style="margin-bottom: 12px; color: #173624 !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.02em;">
                Clinical Report Access
            </h5>
            <div style="font-size: 14px; color: #2D533D; font-weight: 500; opacity: 0.9;">
                Your unique Report ID was provided at the end of your session.
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Controls
    report_id_input = st.text_input(
        "Report ID",
        placeholder = "e.g. 79B490DD",
        max_chars   = 8,
        label_visibility = "collapsed",
    )

    lookup_triggered = st.button(
        "Retrieve Records", 
        type="primary", 
        use_container_width=True,
        disabled=not report_id_input.strip()
    )

# ── 4. DYNAMIC RESULTS DISPLAY ──────────────────────────────────────
if lookup_triggered:
    rid    = report_id_input.strip().upper()
    report = get_report_by_id(rid)

    with col2:
        if report is None:
            st.error(f"No records found for ID: {rid}. Please check your entry.")
        else:
            st.success(f"Report identified for {report.get('patient_name', 'Anonymous')}")
            st.divider()

            # ── Case Status Banner ──
            status = report.get("status", "Pending Review")
            is_complete = status in ("Review Complete", "Discharged", "Referred to Specialist")

            status_meta = {
                "Pending Review"         : ("#F4FAF7", "#173624", "#DEEDE5", "pending_actions"),
                "Review Complete"        : ("#DEEDE5", "#173624", "#2e8b57", "verified"),
                "Follow-up Required"     : ("#FFF9E6", "#713f12", "#fbbf24", "event_note"),
                "Referred to Specialist" : ("#F0F4FF", "#173624", "#DEEDE5", "move_down"),
                "Discharged"             : ("#DEEDE5", "#173624", "#2e8b57", "check_circle"),
            }.get(status, ("#FFFFFF", "#173624", "#DEEDE5", "help_center"))

            bg, txt, brd, icon = status_meta
            st.markdown(
                f'''
                <div style="background:{bg}; border:2px solid {brd}; border-radius:16px; 
                            padding:24px; text-align:center; margin-bottom:30px;">
                    <span class="material-symbols-outlined" style="font-size: 36px; color: {brd}; margin-bottom: 8px;">
                        {icon}
                    </span>
                    <div style="font-weight:800; font-size:14px; color:{txt}; text-transform:uppercase; letter-spacing:0.05em;">
                        Case Status: {status}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            # ── Metrics Grid ──
            phq = report.get("phq8_score", 0)
            sev = report.get("severity", "Minimal")
            
            m1, m2 = st.columns(2)
            m1.metric("PHQ-8 Score", f"{phq:.1f} / 24")
            m2.metric("Screening Severity", f"{sev}")

            # ── Doctor's Review Section (Unicode-Safe Quotes) ──
            st.markdown(
                '''#### <span class="material-symbols-outlined" style="vertical-align:middle; font-size:20px; margin-right:8px;">clinical_notes</span>Clinician Review''', 
                unsafe_allow_html=True
            )

            if is_complete:
                notes = report.get("doctor_notes", "").strip()
                st.markdown(
                    f'''
                    <div class="cv-card" style="background-color: #F4FAF7 !important; border-left: 5px solid #2e8b57 !important;">
                        <h5 style="font-size: 0.9rem; color: #173624; margin-bottom: 12px; font-weight: 800; text-transform: uppercase;">
                            Doctor's Clinical Notes
                        </h5>
                        <p style="font-size: 15px; color: #2D533D; line-height: 1.6;">
                            {notes if notes else "Clinical review finalized. No additional comments recorded."}
                        </p>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '''
                    <div class="cv-card" style="text-align:center; background-color: #F4FAF7 !important; border: 1px dashed #DEEDE5 !important;">
                        <span class="material-symbols-outlined" style="font-size: 32px; color: #476B57; margin-bottom: 10px;">
                            hourglass_empty
                        </span>
                        <div style="font-weight: 800; color: #173624; text-transform: uppercase; font-size: 13px; letter-spacing: 0.02em;">
                            Awaiting Clinical Validation
                        </div>
                        <div style="font-size: 13px; color: #476B57; opacity: 0.8; margin-top: 4px;">
                            Your session is currently in the medical review queue.
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            # ── PDF Download (Unicode-Safe Quotes) ──
            st.divider()
            st.markdown(
                '''#### <span class="material-symbols-outlined" style="vertical-align:middle; font-size:20px; margin-right:8px;">picture_as_pdf</span>Official Report''', 
                unsafe_allow_html=True
            )
            
            try:
                pdf_bytes = generate_pdf_report(report)
                st.download_button(
                    label = "Download Official PDF Report",
                    data = pdf_bytes,
                    file_name = f"NeuraTone_Report_{rid}.pdf",
                    mime = "application/pdf",
                    use_container_width = True,
                )
            except Exception:
                st.info("The secure PDF report is currently being finalized...")