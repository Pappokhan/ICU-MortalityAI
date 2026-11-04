"""
ICU-MortalityAI — Streamlit entry point.

Single-page, two-column layout:
    left   header + patient input form + Predict button
    right  result panel (risk band, cards, partitions, drivers) + report download

MVC layout
    models/       M  loading artefacts, preprocessing, prediction, risk bands, report data
    views/        V  HTML templates (views/templates), CSS (assets/style.css), widgets, PDF report
    controllers/  C  maps form input -> models -> Prediction / CaseReport

Run:  streamlit run app.py
"""
import streamlit as st

from controllers.prediction_controller import PredictionController
from models.artifact_store import ArtifactStore
from views.form_view import render_form
from views.layout import (inject_css, render_error, render_footer, render_header,
                          render_placeholder, section)
from views.report_pdf import build_pdf
from views.result_view import render_results

st.set_page_config(page_title="ICU-MortalityAI", page_icon="🏥", layout="wide",
                   initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading models…")
def get_controller() -> PredictionController:
    return PredictionController(ArtifactStore.load())


inject_css()

try:
    controller = get_controller()
except FileNotFoundError as e:
    render_header()
    render_error("Missing artefact file",
                 f"Could not find '{e.filename}'. Copy the 9 notebook artefacts into the "
                 f"artifacts/ folder (see README.md).")
    st.stop()

left, right = st.columns([1.05, 1], gap="large")

# ---------------- LEFT: header + form + predict ----------------
with left:
    render_header()
    form_values = render_form(controller.features)
    predict_clicked = st.button("Predict mortality risk", type="primary")

if predict_clicked:
    try:
        pred = controller.predict(form_values)
        st.session_state["case"] = controller.build_report(form_values, pred)
        st.session_state.pop("pdf", None)          # invalidate cached PDF
        st.session_state.pop("error", None)
    except ValueError as e:
        st.session_state.pop("case", None)
        st.session_state["error"] = str(e)

# ---------------- RIGHT: results + report ----------------
with right:
    section("Result", first=True)
    case = st.session_state.get("case")
    if st.session_state.get("error"):
        render_error("Invalid input", st.session_state["error"])
    elif case is None:
        render_placeholder()
    else:
        render_results(case.prediction)

        if "pdf" not in st.session_state:
            st.session_state["pdf"] = build_pdf(case)
        st.download_button(
            "📄 Download report (PDF)",
            data=st.session_state["pdf"],
            file_name=f"ICU-MortalityAI_report_{case.created:%Y%m%d_%H%M%S}.pdf",
            mime="application/pdf",
            key="download_report",
            use_container_width=True,
        )
    render_footer()
