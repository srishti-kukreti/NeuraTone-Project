# ══════════════════════════════════════════════════════════════════════
# pages/doctor_dashboard.py  ·  NeuraTone v6 Doctor Dashboard
#
# New in v6:
#   • Verification Players — YouTube: st.video | File/Session: st.audio
#   • Closed-Loop Feedback — notes/status saved to reports.json,
#                            immediately visible in My Results portal
#   • PDF generation & download button in the dashboard
# ══════════════════════════════════════════════════════════════════════

import os
import sys
import datetime

import streamlit as st

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = os.path.join(_ROOT, "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from utils import GLOBAL_CSS, SEVERITY_COLORS, SEVERITY_EMOJI, STATUS_OPTIONS
from pipeline_v7 import (
    get_all_reports, update_doctor_feedback,
    generate_pdf_report, DATA_DIR,
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _fmt_ts(iso_str: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        return iso_str


def _severity_badge(severity: str) -> str:
    col = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["Minimal"])
    em  = SEVERITY_EMOJI.get(severity, "")
    return (
        f'<span class="severity-badge" '
        f'style="background:{col["bg"]};color:{col["text"]};'
        f'border:1px solid {col["border"]}">'
        f'{em} {severity}</span>'
    )


def _phq_color(score: float) -> str:
    if score < 5:  return "#10b981"
    if score < 10: return "#f59e0b"
    if score < 15: return "#f97316"
    if score < 20: return "#ef4444"
    return "#ec4899"


def _status_dot(status: str) -> str:
    return {
        "Pending Review"         : "#9ca3af",
        "Review Complete"        : "#10b981",
        "Follow-up Required"     : "#f59e0b",
        "Referred to Specialist" : "#6366f1",
        "Discharged"             : "#3b82f6",
    }.get(status, "#9ca3af")


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔍 Filters")
    filter_status = st.multiselect(
        "Status",
        options = STATUS_OPTIONS,
        default = [],
        help    = "Leave empty to show all.",
    )
    filter_sev = st.multiselect(
        "Severity",
        options = ["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"],
        default = [],
    )
    sort_by = st.selectbox(
        "Sort by",
        options = ["Newest first", "Oldest first",
                   "PHQ-8 high → low", "PHQ-8 low → high"],
    )

    st.divider()
    if st.button("🔄 Refresh reports"):
        st.cache_data.clear()
        st.rerun()

    st.caption("NeuraTone v6 · Doctor Dashboard")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

st.title("Doctor Dashboard")

all_reports = get_all_reports()

if not all_reports:
    st.info("No reports yet. Ask a patient to complete a session in the **Patient Portal**.")
    st.stop()

# ── Filter & sort ─────────────────────────────────────────────────────
shown = all_reports
if filter_status:
    shown = [r for r in shown if r.get("status", "") in filter_status]
if filter_sev:
    shown = [r for r in shown if r.get("severity", "") in filter_sev]

if sort_by == "Oldest first":
    shown = list(reversed(shown))
elif sort_by == "PHQ-8 high → low":
    shown = sorted(shown, key=lambda r: r.get("phq8_score", 0), reverse=True)
elif sort_by == "PHQ-8 low → high":
    shown = sorted(shown, key=lambda r: r.get("phq8_score", 0))

if "open_report_id" not in st.session_state:
    st.session_state.open_report_id = None

left, right = st.columns([1, 2], gap="large")

# ══════════════════════════════════════════════════════════════════════
# LEFT: REPORT QUEUE
# ══════════════════════════════════════════════════════════════════════

with left:
    st.markdown(f"#### Queue ({len(shown)} reports)")

    for r in shown:
        rid     = r["report_id"]
        sev     = r.get("severity", "—")
        phq     = r.get("phq8_score", 0)
        col     = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["Minimal"])
        em      = SEVERITY_EMOJI.get(sev, "")
        status  = r.get("status", "—")
        is_open = (st.session_state.open_report_id == rid)

        border = (
            f"border:2px solid {col['border']};"
            if is_open else
            "border:1px solid rgba(71,104,44,0.15);"
        )
        bg = "rgba(71,104,44,0.06)" if is_open else "rgba(255,255,255,0.55)"

        st.markdown(
            f'<div class="rq-row" style="{border}background:{bg}">'
            f'<div style="flex:1">'
            f'<div style="font-weight:600;font-size:14px">{r.get("patient_name","—")}</div>'
            f'<div style="font-size:12px;opacity:0.6">{_fmt_ts(r.get("timestamp",""))}</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:13px;font-weight:700;color:{_phq_color(phq)}">'
            f'PHQ {phq:.1f}</div>'
            f'<div style="font-size:11px;opacity:0.7">{em} {sev}</div>'
            f'</div>'
            f'<div style="width:8px;height:8px;border-radius:50%;'
            f'background:{_status_dot(status)};flex-shrink:0"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button(f"Open  {rid}", key=f"open_{rid}", use_container_width=True):
            st.session_state.open_report_id = rid
            st.rerun()

    st.caption(f"Total in database: {len(all_reports)}")


# ══════════════════════════════════════════════════════════════════════
# RIGHT: DIAGNOSTIC VIEW
# ══════════════════════════════════════════════════════════════════════

with right:
    open_id = st.session_state.open_report_id

    if open_id is None:
        st.markdown(
            '<div style="opacity:0.45;text-align:center;padding:60px 0">'
            '<div style="font-size:56px">🩺</div>'
            '<div style="font-size:16px;margin-top:10px">Select a report from the queue</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    rpt = next((r for r in all_reports if r["report_id"] == open_id), None)
    if rpt is None:
        st.warning("Report not found.")
        st.stop()

    sev      = rpt.get("severity", "—")
    phq      = rpt.get("phq8_score", 0)
    col      = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["Minimal"])
    em       = SEVERITY_EMOJI.get(sev, "")
    label    = rpt.get("classification", "—")
    override = rpt.get("safety_override", False)

    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:22px;'
        f'font-weight:700;color:#47682C">{rpt.get("patient_name","—")}</span>'
        f'<span style="opacity:0.45;font-size:13px">{_fmt_ts(rpt.get("timestamp",""))}</span>'
        f'<span style="opacity:0.40;font-size:12px">· ID {open_id}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PHQ-8",     f"{phq:.1f} / 24")
    m2.metric("Severity",  f"{em} {sev}")
    m3.metric("Class",     label + (" ⚠️" if override else ""))
    m4.metric("Duration",  f"{rpt.get('duration_sec',0):.0f}s")

    if override:
        st.warning("⚠️ Safety Override triggered — PHQ-8 ≥ 10 forced MDD label.")

    st.markdown(
        f'<div style="background:{col["bg"]};border-left:4px solid {col["border"]};'
        f'border-radius:0 12px 12px 0;padding:12px 18px;margin:10px 0">'
        f'<b style="color:{col["text"]}">{em} {sev} Depression Band</b><br>'
        f'<span style="font-size:12px;color:{col["text"]};opacity:0.8">'
        f'PHQ-8 range — '
        f'{"0–4 (Minimal)" if sev=="Minimal" else "5–9 (Mild)" if sev=="Mild" else "10–14 (Moderate)" if sev=="Moderate" else "15–19 (Moderately Severe)" if sev=="Moderately Severe" else "20–24 (Severe)"}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── Verification Audio/Video Player ──────────────────────────────
    src_type  = rpt.get("source_type", "file")
    audio_path = rpt.get("raw_audio_path", "")

    st.markdown("**🎧 Verification Player**")
    if src_type == "youtube" and audio_path.startswith("http"):
        st.video(audio_path)
    elif audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
    elif src_type == "live_session":
        st.info(
            f"Session WAV saved to `engine/data/`. "
            f"Path: `{audio_path}`"
        )
        # Try to find and play from data dir
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
    else:
        st.caption(f"Audio source: `{audio_path}`")

    # ── Temporal Sparkline ────────────────────────────────────────────
    chunk_probs = rpt.get("chunk_probs", [])
    if len(chunk_probs) > 1:
        st.markdown("**⏱️ Temporal Probability Sparkline**")
        import altair as alt
        import pandas as pd

        df = pd.DataFrame({
            "Window"      : list(range(len(chunk_probs))),
            "Probability" : chunk_probs,
        })
        line = (
            alt.Chart(df)
            .mark_area(
                line  = {"color": "#47682C", "strokeWidth": 2},
                color = alt.Gradient(
                    gradient = "linear",
                    stops    = [
                        alt.GradientStop(color="#47682C33", offset=0),
                        alt.GradientStop(color="#47682C08", offset=1),
                    ],
                    x1=0, x2=0, y1=1, y2=0,
                ),
            )
            .encode(
                x = alt.X("Window:Q", title="3-second window"),
                y = alt.Y("Probability:Q", scale=alt.Scale(domain=[0,1]),
                          title="P(Depression)"),
            )
        )
        thresh = (
            alt.Chart(pd.DataFrame({"y": [0.47]}))
            .mark_rule(strokeDash=[4,3], color="#9ca3af", size=1)
            .encode(y="y:Q")
        )
        st.altair_chart(line + thresh, use_container_width=True)

        ti = rpt.get("temporal_info", {})
        if ti:
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Peak window", ti.get("peak_window_start", "—"))
            tc2.metric("Peak prob",   f"{ti.get('peak_prob',0):.3f}")
            tc3.metric("Trend",       ti.get("trend", "—").capitalize())

    # ── Acoustic Biomarkers ───────────────────────────────────────────
    biomarkers = rpt.get("biomarkers", [])
    if biomarkers:
        st.markdown("**🎙️ Top Acoustic Biomarkers (Z-scores)**")
        max_z   = max(abs(b["z_score"]) for b in biomarkers) or 1.0
        bm_html = ""
        for b in biomarkers:
            z       = b["z_score"]
            pct     = min(100, abs(z) / max_z * 100)
            bar_col = "#ef4444" if z > 0 else "#3b82f6"
            tick    = "✓" if b.get("consistent") else "↔"
            bm_html += (
                f'<div class="bm-row">'
                f'<span class="bm-label">{tick} {b.get("description", b.get("feature",""))}</span>'
                f'<div class="bm-bar-wrap"><div class="bm-bar" '
                f'style="width:{pct:.0f}%;background:{bar_col}"></div></div>'
                f'<span class="bm-val">{z:+.2f}</span>'
                f'</div>'
            )
        st.markdown(bm_html, unsafe_allow_html=True)

    # ── Gemini Summary ────────────────────────────────────────────────
    if rpt.get("gemini_summary"):
        with st.expander("🤖 AI Clinical Summary", expanded=True):
            st.markdown(rpt["gemini_summary"])

    # ── Interview Answers ─────────────────────────────────────────────
    answers = rpt.get("interview_answers", {})
    if answers:
        with st.expander("💬 Patient Responses"):
            q_map = {
                "mood": "Mood", "sleep": "Sleep", "energy": "Energy",
                "appetite": "Appetite", "interest": "Interest",
            }
            for key, val in answers.items():
                label_txt = q_map.get(key, key.replace("_", " ").capitalize())
                st.markdown(f"**{label_txt}:** {val}")

    # ── Source Info ───────────────────────────────────────────────────
    with st.expander("📁 Source Info"):
        st.code(rpt.get("raw_audio_path", "—"), language=None)
        st.caption(f"Source type: {rpt.get('source_type','—')}")

    # ════════════════════════════════════════════════════════════════
    # CLINICAL FEEDBACK — CLOSED-LOOP
    # ════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("#### 📝 Clinical Feedback")
    st.caption("Changes are saved to reports.json and immediately visible to the patient in **My Results**.")

    current_notes  = rpt.get("doctor_notes", "")
    current_status = rpt.get("status", "Pending Review")

    status_idx = STATUS_OPTIONS.index(current_status) if current_status in STATUS_OPTIONS else 0
    new_status = st.selectbox(
        "Case status",
        options = STATUS_OPTIONS,
        index   = status_idx,
        key     = f"status_{open_id}",
    )

    new_notes = st.text_area(
        "Doctor's notes",
        value       = current_notes,
        placeholder = "Clinical observations, differential diagnosis, next steps…",
        height      = 130,
        key         = f"notes_{open_id}",
    )

    col_save, col_pdf, col_del = st.columns([5, 5, 3]) # Added a 3rd column for Delete

    with col_save:
        if st.button("💾 Save Feedback", type="primary", key=f"save_{open_id}",
                     use_container_width=True):
            update_doctor_feedback(open_id, new_notes, new_status)
            st.success("✅ Feedback saved — now visible to patient in My Results.")
            st.rerun()

    with col_pdf:
        # Generate PDF on demand
        try:
            # Merge updated notes into report for PDF
            rpt_for_pdf = dict(rpt)
            rpt_for_pdf["doctor_notes"] = new_notes
            rpt_for_pdf["status"]       = new_status
            pdf_bytes = generate_pdf_report(rpt_for_pdf)

            st.download_button(
                label     = "📄 Download PDF Report",
                data      = pdf_bytes,
                file_name = f"NeuraTone_Report_{open_id}.pdf",
                mime      = "application/pdf",
                use_container_width = True,
                key       = f"pdf_{open_id}",
            )
        except Exception as e:
            st.button(
                f"📄 PDF unavailable",
                disabled    = True,
                use_container_width = True,
                key         = f"pdf_disabled_{open_id}",
            )

    with col_del:
        if st.button("🗑️ Delete", type="secondary", key=f"del_{open_id}", use_container_width=True):
            import json
            import os
            
            # 1. Hunt down the file (Check the 3 most likely locations)
            possible_paths = [
                os.path.join(DATA_DIR, "reports.json"),           # The engine/data folder
                os.path.join(_ROOT, "reports.json"),              # The main project folder
                os.path.join(os.getcwd(), "reports.json")         # The terminal working directory
            ]
            
            actual_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    actual_path = p
                    break
            
            # 2. Execute the delete if we found it
            if not actual_path:
                st.error("Path Error: Could not find reports.json in your project folders.")
            else:
                try:
                    with open(actual_path, "r") as f:
                        db = json.load(f)
                    
                    # Force string comparison to kill the ghost bug
                    db = [r for r in db if str(r.get("report_id", "")) != str(open_id)]
                    
                    # Save the updated database
                    with open(actual_path, "w") as f:
                        json.dump(db, f, indent=4)
                        
                    # Clear Streamlit's memory completely
                    st.cache_data.clear()
                    if hasattr(st, "cache_resource"):
                        st.cache_resource.clear()
                    
                    # Close the view and refresh
                    st.session_state.open_report_id = None 
                    st.rerun()
                except Exception as e:
                    st.error(f"Crash during delete: {e}")