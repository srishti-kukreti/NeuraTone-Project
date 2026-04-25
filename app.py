'''# ══════════════════════════════════════════════════════════════════════
# app.py  ·  NeuraTone v7 — "Kore" Luxe-Clinical Edition
#
# Entry point — three-page Streamlit app using st.navigation()
#
# Run with:
#   streamlit run app.py
# ══════════════════════════════════════════════════════════════════════

import streamlit as st

st.set_page_config(
    page_title            = "NeuraTone v7 — Kore",
    page_icon             = "🧠",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Pages ────────────────────────────────────────────────────────────
patient_page = st.Page(
    "pages/patient_portal.py",
    title = "Patient Portal",
    icon  = "🎙️",
)
results_page = st.Page(
    "pages/my_results.py",
    title = "My Results",
    icon  = "📋",
)
doctor_page = st.Page(
    "pages/doctor_dashboard.py",
    title = "Doctor Dashboard",
    icon  = "🩺",
)

pg = st.navigation([patient_page, results_page, doctor_page])
pg.run()'''




import streamlit as st

st.set_page_config(
    page_title="NeuraTone — Acoustic Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Styling ──
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #F0F7EE 0%, #D9E8D4 100%); }
    </style>
    """, unsafe_allow_html=True)

# ── Navigation with Material Icons ──
patient_page = st.Page(
    "pages/patient_portal.py",
    title="Patient Portal",
    icon=":material/person:",
)
results_page = st.Page(
    "pages/my_results.py",
    title="My Results",
    icon=":material/history:",
)
doctor_page = st.Page(
    "pages/doctor_dashboard.py",
    title="Doctor Dashboard",
    icon=":material/monitoring:",
)

pg = st.navigation({
    "Services": [patient_page, results_page],
    "Administration": [doctor_page]
})

pg.run()
