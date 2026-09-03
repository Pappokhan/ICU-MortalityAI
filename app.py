import streamlit as st

from config import APP_ICON, APP_NAME
from controller.prediction_controller import predict_mortality_risk
from model.ml_model import ModelLoadError, get_model
from view import ui_components as ui
from view.report import generate_pdf_report

st.set_page_config(page_title=f"{APP_NAME} — ICU Mortality Risk", page_icon=APP_ICON, layout="wide")

ui.init_session_state()
ui.inject_css()
ui.render_header()

try:
    ensemble = get_model()
except ModelLoadError as exc:
    ui.render_error(str(exc))
    st.stop()

assess_tab, about_tab = st.tabs(["Assess a patient", "How this works"])

with assess_tab:
    ui.render_example_bar()
    st.write("")

    left, right = st.columns([2.75, 1], gap="large")

    with left:
        form_input = ui.render_intake_form()

    with right:
        ui.render_result_column_anchor()
        if form_input is None:
            ui.render_empty_result_panel()
        else:
            with st.spinner("Running federated ensemble inference..."):
                result = predict_mortality_risk(form_input)
            ui.render_result_panel(result, n_models=ensemble.n_models)
            report_pdf = generate_pdf_report(form_input, result, ensemble.n_models)
            ui.render_download_report_button(report_pdf)

    ui.render_footnote()

with about_tab:
    ui.render_how_it_works()
