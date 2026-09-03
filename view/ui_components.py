from __future__ import annotations

import math
import os

import streamlit as st

from config import (
    AGE_RISK_LABELS,
    APP_NAME,
    APP_TAGLINE,
    EXAMPLE_PATIENTS,
    FEATURE_META,
    FORM_SECTIONS,
    RISK_THRESHOLDS,
    age_to_risk_category,
)
from controller.prediction_controller import FormInput

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

RISK_COLORS = {
    "low": "#2F9E63",
    "moderate": "#D9A441",
    "high": "#DB6B3D",
    "critical": "#C1443C",
}
RISK_COPY = {
    "low": "Low predicted risk",
    "moderate": "Moderate predicted risk",
    "high": "High predicted risk",
    "critical": "Critical predicted risk",
}

AGE_DEFAULT = 64
ALL_INPUT_KEYS = [f"inp_{feat}" for feat in FEATURE_META] + ["inp_age"]


def init_session_state() -> None:
    for feat, meta in FEATURE_META.items():
        st.session_state.setdefault(f"inp_{feat}", meta["default"])
    st.session_state.setdefault("inp_age", AGE_DEFAULT)


def _apply_example(name: str) -> None:
    example = EXAMPLE_PATIENTS[name]["values"]
    for feat, value in example.items():
        st.session_state[f"inp_{feat}"] = value


def _reset_defaults() -> None:
    for feat, meta in FEATURE_META.items():
        st.session_state[f"inp_{feat}"] = meta["default"]
    st.session_state["inp_age"] = AGE_DEFAULT


def inject_css() -> None:
    css_path = os.path.join(STATIC_DIR, "css", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    # Built as concatenated single-line segments rather than one big
    # indented triple-quoted block. The earlier "Per-client model votes"
    # bug came from a whitespace-only line accidentally appearing inside
    # a raw-HTML block, which made Streamlit's markdown parser bail out
    # of HTML mode partway through — sometimes that shows as literal
    # code text, but depending on where the split lands it can just as
    # easily swallow the content before the break (an empty-looking
    # region) instead. The brand row here (logo + "ICU-MortalityAI")
    # was the first real content inside the header block, so it's
    # exactly the part that would go missing. Flattening removes any
    # chance of a stray blank line breaking the block.
    brand_name_html = APP_NAME.replace("-", '<span class="accent">-</span>')
    logo_svg = (
        '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">'
        '<path d="M1 10H5L7 4L11 16L13 10H19" stroke="#4FC3CE" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    header_html = (
        '<div class="clinical-header">'
        '<div class="header-copy">'
        '<div class="brand-row">'
        f'<div class="brand-mark">{logo_svg}</div>'
        f'<span class="brand-name">{brand_name_html}</span>'
        '</div>'
        '<h1>ICU patient mortality risk estimator</h1>'
        f"<p>{APP_TAGLINE}. Enter a patient's neurological, vital-sign and "
        'severity-score readings from their first 24 hours to get a '
        'risk estimate from a 5-site federated XGBoost ensemble.</p>'
        '</div>'
        '<div class="badge-col">'
        '<span class="badge"><span class="dot"></span>'
        '<span>5-MODEL FEDERATED ENSEMBLE</span></span>'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_error(message: str) -> None:
    st.markdown(
        f"""
        <div class="error-card">
          <span class="e-title">Couldn't load the model</span>
          {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_bar() -> None:
    st.markdown('<p class="example-label">Try an example patient</p>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_PATIENTS) + 1)
    for col, (key, patient) in zip(cols, EXAMPLE_PATIENTS.items()):
        with col:
            st.button(
                patient["label"],
                help=patient["description"],
                on_click=_apply_example,
                args=(key,),
                use_container_width=True,
                type="secondary",
            )
    with cols[-1]:
        st.button(
            "Reset to defaults",
            on_click=_reset_defaults,
            use_container_width=True,
            type="primary",
        )


def _slider_for(feat_key: str, col=None):
    meta = FEATURE_META[feat_key]
    target = col if col is not None else st
    unit = f" ({meta['unit']})" if meta["unit"] else ""
    return target.slider(
        meta["label"] + unit,
        min_value=meta["min"],
        max_value=meta["max"],
        step=meta["step"],
        help=meta["help"],
        key=f"inp_{feat_key}",
    )


def render_age_picker() -> int:
    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        age = st.slider(
            "Patient age (years)",
            min_value=16, max_value=100, step=1,
            help="Used to derive the model's age-risk category feature.",
            key="inp_age",
        )
    with c2:
        risk_cat = age_to_risk_category(age)
        # A label of matching height/spacing sits above the readout box so
        # its top edge lines up with the slider track next to it, instead
        # of floating a few pixels higher the way a single unlabeled div did.
        st.markdown(
            f"""
            <div class="age-readout-label">Age-risk category</div>
            <div class="age-readout"><strong>{AGE_RISK_LABELS[risk_cat]}</strong></div>
            """,
            unsafe_allow_html=True,
        )
    return age


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _balanced_columns(sections):
    """Split FORM_SECTIONS (in order) into two side-by-side groups whose
    total field counts are as close to even as possible — mirrors the
    two-column card layout in the reference design, where the left
    sub-column holds the shorter early sections (Neurological status,
    Oxygenation) and the right sub-column carries the taller remainder
    (Blood pressure, Temperature, APACHE)."""
    total = sum(len(s["fields"]) for s in sections)
    half = total / 2
    running = 0
    best_i, best_diff = 0, abs(0 - half)
    for i, section in enumerate(sections, start=1):
        running += len(section["fields"])
        diff = abs(running - half)
        if diff < best_diff:
            best_diff = diff
            best_i = i
    return sections[:best_i], sections[best_i:]


def _render_section_group(sections, start_index: int) -> None:
    for offset, section in enumerate(sections):
        _section_start(str(start_index + offset), section["title"], section["subtitle"])
        for row_fields in _chunk(section["fields"], 3):
            row_cols = st.columns(len(row_fields))
            for feat_key, col in zip(row_fields, row_cols):
                _slider_for(feat_key, col)
        _section_end()


def render_intake_form() -> FormInput | None:
    group_a, group_b = _balanced_columns(FORM_SECTIONS)

    with st.form("intake_form", clear_on_submit=False):
        col_a, col_b = st.columns([1.15, 1], gap="large")
        with col_a:
            age = render_age_picker()
            _render_section_group(group_a, start_index=1)
        with col_b:
            _render_section_group(group_b, start_index=len(group_a) + 1)

        submitted = st.form_submit_button("Calculate mortality risk")

    if not submitted:
        return None

    values = {feat: st.session_state[f"inp_{feat}"] for feat in FEATURE_META}
    return FormInput(age=age, **values)


def _section_start(index: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
          <h3><span class="section-index">{index}</span>{title}</h3>
          <p class="section-sub">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def _section_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_column_anchor() -> None:
    """Marks the current column so CSS can pin it (:has-based sticky hook)."""
    st.markdown('<span class="sticky-anchor"></span>', unsafe_allow_html=True)


def render_empty_result_panel() -> None:
    st.markdown(
        """
        <div class="result-card">
          <div class="empty-icon">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M1 10H5L7 4L11 16L13 10H19" stroke="#4FC3CE" stroke-width="1.6"
                    stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <p class="r-eyebrow">Result</p>
          <h2>No estimate yet</h2>
          <p class="empty-result">
            Fill in the form (or load an example patient above) and select
            <strong>Estimate mortality risk</strong> to see the federated
            ensemble's prediction, a per-model breakdown, and the overall
            risk tier.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _angle_for(v: float) -> float:
    return 180 * (1 - v)


def _point_at(angle_deg: float, cx: float = 110, cy: float = 110, r: float = 90):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


def _zone_arc(v0: float, v1: float, color: str) -> str:
    x1, y1 = _point_at(_angle_for(v0))
    x2, y2 = _point_at(_angle_for(v1))
    return (
        f'<path d="M {x1:.2f} {y1:.2f} A 90 90 0 0 1 {x2:.2f} {y2:.2f}" '
        f'fill="none" stroke="{color}" stroke-width="14" stroke-linecap="butt" />'
    )


def _gauge_svg(prob: float) -> str:
    """Server-rendered semicircle risk gauge — same geometry the old
    canvas/JS version drew, but as plain markup so it sits in normal
    document flow instead of a fixed-height iframe."""
    t_low = RISK_THRESHOLDS["low"]
    t_moderate = RISK_THRESHOLDS["moderate"]
    t_high = RISK_THRESHOLDS["high"]

    zones = (
        _zone_arc(0, t_low, RISK_COLORS["low"])
        + _zone_arc(t_low, t_moderate, RISK_COLORS["moderate"])
        + _zone_arc(t_moderate, t_high, RISK_COLORS["high"])
        + _zone_arc(t_high, 1, RISK_COLORS["critical"])
    )
    nx, ny = _point_at(_angle_for(prob), r=74)

    return f"""
    <svg class="gauge-svg" width="220" height="128" viewBox="0 0 220 128" overflow="visible">
      {zones}
      <line x1="110" y1="110" x2="{nx:.2f}" y2="{ny:.2f}"
            stroke="#ffffff" stroke-width="4" stroke-linecap="round" />
      <circle cx="110" cy="110" r="7" fill="#ffffff" />
      <circle cx="110" cy="110" r="3" fill="#10233B" />
      <text x="18" y="122" fill="#B9C7D6" font-size="10" font-family="IBM Plex Mono, monospace">0%</text>
      <text x="188" y="122" fill="#B9C7D6" font-size="10" font-family="IBM Plex Mono, monospace">100%</text>
    </svg>
    """


def render_result_panel(result, n_models: int) -> None:
    color = RISK_COLORS[result.risk_level]
    pct = round(result.probability * 100, 1)
    agreement_pct = round(result.agreement * 100, 1)
    verdict = "Predicted outcome: did not survive stay" if result.predicted_class == 1 else "Predicted outcome: survived stay"
    plain_language = (
        f"Out of 100 similar patients, this model expects about "
        f"<strong>{round(pct)}</strong> would not survive their hospital stay, "
        f"based on the values entered."
    )

    st.markdown(
        f"""
        <div class="result-card">
          <p class="r-eyebrow">Result — federated ensemble ({n_models} models)</p>
          <h2>{verdict}</h2>
          <div class="gauge-wrap">{_gauge_svg(result.probability)}</div>
          <div class="gauge-readout">
            <span class="pct" style="color:{color}">{pct}%</span>
            <span class="lbl">predicted probability of hospital death</span>
          </div>
          <div class="pill-row">
            <span class="risk-pill {result.risk_level}">{RISK_COPY[result.risk_level]}</span>
            <span class="agreement-pill">Model agreement: {agreement_pct}%</span>
          </div>
          <p class="plain-language">{plain_language}</p>
          <div class="model-votes">
            <div class="mv-title">Per-client model votes</div>
            {_mv_rows_html(result.per_model_probabilities)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_report_button(report_pdf: bytes) -> None:
    st.download_button(
        label="⬇ Download clinical report (.pdf)",
        data=report_pdf,
        file_name="icu_mortality_ai_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _mv_row(index: int, prob: float) -> str:
    # Kept on one line deliberately: each row used to be its own indented,
    # multi-line f-string. Concatenating those in _mv_rows_html left a
    # whitespace-only line between every pair of rows, which is exactly
    # what a blank line inside a markdown HTML block means to Streamlit's
    # renderer — "the raw-HTML block just ended". Everything after that
    # point (the rest of the rows) then got re-parsed as an indented
    # markdown code block and shown as literal text instead of being
    # rendered, which is the bug the user saw. A single line per row has
    # no internal newline, so no blank line can ever appear between rows.
    pct = round(prob * 100, 1)
    return (
        f'<div class="mv-row">'
        f'<span class="mv-name">Client {index + 1}</span>'
        f'<div class="mv-bar-track">'
        f'<div class="mv-bar-fill" style="--w:{pct}%; animation-delay:{index * 70}ms;"></div>'
        f'</div>'
        f'<span class="mv-val">{pct}%</span>'
        f'</div>'
    )


def _mv_rows_html(probabilities) -> str:
    return "".join(_mv_row(i, p) for i, p in enumerate(probabilities))


def render_how_it_works() -> None:
    st.markdown(
        """
        <div class="section-card">
          <h3><span class="section-index">i</span>What is a federated ensemble?</h3>
          <p class="section-sub" style="margin-bottom:14px;">
            Instead of pooling patient data from every hospital into one place,
            <strong>federated learning</strong> trains a separate model at each
            hospital (or "client") on its own local data, keeping that data
            private. This app's model is the resulting <strong>global
            ensemble</strong>: 5 client models whose predictions are averaged
            together for every patient, similar to asking 5 specialists for a
            second opinion and taking the average of their answers.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
          <h3><span class="section-index">i</span>What goes into the prediction?</h3>
          <p class="section-sub" style="margin-bottom:10px;">
            The model looks at 15 clinical signals, selected because they were
            most strongly associated with patient outcomes in the training
            data:
          </p>
          <ul style="font-size:13.5px;color:#3E5470;line-height:1.9;margin-top:0;padding-left:33px;">
            <li>Two APACHE IVa severity scores (the hospital's own risk model)</li>
            <li>Glasgow Coma Scale — verbal, eye, and motor response</li>
            <li>Oxygen saturation (SpO₂) in the first hour and first day</li>
            <li>Systolic, diastolic and mean blood pressure minimums</li>
            <li>Minimum core body temperature</li>
            <li>An age-based risk category (young / middle-aged / elderly / very elderly)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
          <h3><span class="section-index">i</span>How to read the result</h3>
          <table style="width:100%;font-size:13.5px;color:#3E5470;border-collapse:collapse;margin:0 0 16px 33px;width:calc(100% - 33px);">
            <tr style="border-bottom:1px solid #E5EBEE;">
              <th style="text-align:left;padding:8px 6px;color:#10233B;">Risk tier</th>
              <th style="text-align:left;padding:8px 6px;color:#10233B;">Predicted probability</th>
            </tr>
            <tr style="border-bottom:1px solid #E5EBEE;"><td style="padding:8px 6px;">Low</td><td style="padding:8px 6px;">Below 20%</td></tr>
            <tr style="border-bottom:1px solid #E5EBEE;"><td style="padding:8px 6px;">Moderate</td><td style="padding:8px 6px;">20% – 45%</td></tr>
            <tr style="border-bottom:1px solid #E5EBEE;"><td style="padding:8px 6px;">High</td><td style="padding:8px 6px;">45% – 70%</td></tr>
            <tr><td style="padding:8px 6px;">Critical</td><td style="padding:8px 6px;">70% and above</td></tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout">
          <strong>A note on accuracy.</strong> The original training pipeline
          normalised vitals using a scaler fit on the full training set, but
          only the 5 trained models were saved — not that scaler. This app
          rebuilds an equivalent normalisation from clinically plausible
          ranges instead, so results are a very close approximation rather
          than a bit-for-bit match to the original research notebook. See
          the project README for how to plug in the exact scaler if you
          export it later.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footnote() -> None:
    st.markdown(
        f"""
        <div class="footnote">
          {APP_NAME} is a research demo built on the WiDS Datathon ICU
          mortality dataset and a federated XGBoost ensemble; it is
          <strong>not</strong> a certified clinical decision-support device
          and must not be used for real patient care.
        </div>
        """,
        unsafe_allow_html=True,
    )
