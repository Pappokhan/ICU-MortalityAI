from __future__ import annotations

import html
import io
from datetime import datetime

from xhtml2pdf import pisa

from config import (
    AGE_RISK_LABELS,
    APP_NAME,
    FEATURE_META,
    FORM_SECTIONS,
    RISK_THRESHOLDS,
    age_to_risk_category,
)
from controller.prediction_controller import FormInput
from model.ml_model import PredictionResult

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


def _esc(value) -> str:
    return html.escape(str(value))


def _input_rows(form: FormInput) -> str:
    rows = []
    age_cat = age_to_risk_category(form.age)
    rows.append(
        f"""<tr>
            <td>Patient age</td>
            <td>{form.age} years</td>
            <td>{_esc(AGE_RISK_LABELS[age_cat])}</td>
        </tr>"""
    )
    for section in FORM_SECTIONS:
        for feat in section["fields"]:
            meta = FEATURE_META[feat]
            value = getattr(form, feat)
            unit = f" {meta['unit']}" if meta["unit"] else ""
            rows.append(
                f"""<tr>
                    <td>{_esc(meta['label'])}</td>
                    <td>{value}{_esc(unit)}</td>
                    <td>{_esc(section['title'])}</td>
                </tr>"""
            )
    return "\n".join(rows)


def _model_rows(result: PredictionResult) -> str:
    rows = []
    for i, prob in enumerate(result.per_model_probabilities):
        pct = round(prob * 100, 1)
        rows.append(
            f"""<tr>
                <td>Federated client {i + 1}</td>
                <td>{pct}%</td>
            </tr>"""
        )
    return "\n".join(rows)


def generate_html_report(form: FormInput, result: PredictionResult, n_models: int) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    color = RISK_COLORS[result.risk_level]
    pct = round(result.probability * 100, 1)
    verdict = "Predicted outcome: did not survive stay" if result.predicted_class == 1 else "Predicted outcome: survived stay"
    agreement_pct = round(result.agreement * 100, 1)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_esc(APP_NAME)} — Mortality Risk Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
  *{{box-sizing:border-box;}}
  body{{margin:0;padding:32px;background:#F5F8F9;color:#10233B;font-family:'IBM Plex Sans',sans-serif;}}
  .sheet{{max-width:820px;margin:0 auto;background:#fff;border:1px solid #E1E8EB;border-radius:8px;padding:32px 36px;}}
  h1{{font-family:'Source Serif 4',serif;font-size:24px;margin:0 0 4px 0;}}
  .meta{{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#3E5470;margin-bottom:24px;}}
  .result-banner{{background:#10233B;color:#fff;border-radius:8px;padding:22px 24px;margin-bottom:24px;}}
  .result-banner .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#9FD9DE;margin:0 0 4px 0;}}
  .result-banner h2{{font-family:'Source Serif 4',serif;margin:0 0 12px 0;font-size:20px;}}
  .pct{{font-size:36px;font-weight:600;color:{color};font-family:'IBM Plex Mono',monospace;}}
  .pill{{display:inline-block;margin-left:12px;padding:4px 12px;border-radius:999px;font-size:13px;font-weight:600;background:rgba(255,255,255,0.14);}}
  .agreement{{font-size:12.5px;color:#B9C7D6;margin-top:10px;}}
  h3{{font-family:'Source Serif 4',serif;font-size:16px;border-bottom:1px solid #E1E8EB;padding-bottom:8px;margin:28px 0 10px 0;}}
  table{{width:100%;border-collapse:collapse;font-size:13.5px;}}
  th{{text-align:left;padding:8px 6px;color:#10233B;border-bottom:1px solid #E1E8EB;}}
  td{{padding:7px 6px;border-bottom:1px solid #F0F3F4;color:#3E5470;}}
  .disclaimer{{margin-top:28px;padding:14px 16px;background:#FDF1EF;border-left:4px solid #C1443C;border-radius:4px;font-size:12.5px;line-height:1.6;color:#10233B;}}
  .footer{{margin-top:24px;font-size:11.5px;color:#3E5470;text-align:center;}}
  @media print{{ body{{background:#fff;padding:0;}} .sheet{{border:none;padding:0;}} }}
</style>
</head>
<body>
  <div class="sheet">
    <h1>{_esc(APP_NAME)} — Mortality Risk Report</h1>
    <div class="meta">Generated {_esc(generated_at)} · Federated ensemble, {n_models} client models</div>

    <div class="result-banner">
      <p class="eyebrow">RESULT</p>
      <h2>{_esc(verdict)}</h2>
      <span class="pct">{pct}%</span>
      <span class="pill">{_esc(RISK_COPY[result.risk_level])}</span>
      <div class="agreement">Model agreement across the {n_models} federated clients: {agreement_pct}%</div>
    </div>

    <h3>Patient inputs</h3>
    <table>
      <tr><th>Field</th><th>Value</th><th>Category</th></tr>
      {_input_rows(form)}
    </table>

    <h3>Per-model votes</h3>
    <table>
      <tr><th>Model</th><th>Predicted probability of death</th></tr>
      {_model_rows(result)}
    </table>

    <h3>Risk tiers</h3>
    <table>
      <tr><th>Tier</th><th>Predicted probability</th></tr>
      <tr><td>Low</td><td>Below {int(RISK_THRESHOLDS['low'] * 100)}%</td></tr>
      <tr><td>Moderate</td><td>{int(RISK_THRESHOLDS['low'] * 100)}% – {int(RISK_THRESHOLDS['moderate'] * 100)}%</td></tr>
      <tr><td>High</td><td>{int(RISK_THRESHOLDS['moderate'] * 100)}% – {int(RISK_THRESHOLDS['high'] * 100)}%</td></tr>
      <tr><td>Critical</td><td>{int(RISK_THRESHOLDS['high'] * 100)}% and above</td></tr>
    </table>

    <div class="disclaimer">
      This report is generated by a research demo built on the WiDS Datathon ICU
      mortality dataset and a federated XGBoost ensemble. It is <strong>not</strong>
      a certified clinical decision-support device and must not be used for real
      patient care or as a substitute for clinical judgment.
    </div>

    <div class="footer">{_esc(APP_NAME)} · Federated XGBoost mortality risk estimation for ICU patients</div>
  </div>
</body>
</html>"""


def _pdf_html(form: FormInput, result: PredictionResult, n_models: int) -> str:
    """A PDF-safe variant of the report markup.

    xhtml2pdf renders with reportlab and only understands a subset of
    CSS 2.1 — no @import'd web fonts, no rgba()/box-shadow, limited
    flexbox/grid. This trims the styling to what it can actually draw
    while keeping the same content and layout.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    color = RISK_COLORS[result.risk_level]
    pct = round(result.probability * 100, 1)
    verdict = "Predicted outcome: did not survive stay" if result.predicted_class == 1 else "Predicted outcome: survived stay"
    agreement_pct = round(result.agreement * 100, 1)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: Helvetica, sans-serif; color: #10233B; font-size: 10.5pt; }}
  h1 {{ font-size: 18pt; margin: 0 0 4px 0; }}
  .meta {{ font-family: Courier, monospace; font-size: 8.5pt; color: #3E5470; margin-bottom: 16px; }}
  .banner-table {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; }}
  .banner-table td {{ background-color: #10233B; color: #ffffff; padding: 14px 16px; border-bottom: none; }}
  .eyebrow {{ font-family: Courier, monospace; font-size: 8.5pt; color: #9FD9DE; margin: 0 0 6px 0; }}
  .verdict {{ font-size: 13pt; font-weight: bold; color: #ffffff; margin: 0 0 10px 0; }}
  .pct {{ font-size: 22pt; font-weight: bold; color: {color}; font-family: Courier, monospace; }}
  .pill {{ font-size: 9.5pt; font-weight: bold; color: #9FD9DE; }}
  .agreement {{ font-size: 8.5pt; color: #B9C7D6; margin-top: 8px; }}
  h3 {{ font-size: 12pt; border-bottom: 1px solid #E1E8EB; padding-bottom: 6px; margin: 18px 0 8px 0; }}
  table.data {{ width: 100%; border-collapse: collapse; font-size: 9.5pt; }}
  table.data th {{ text-align: left; padding: 6px; color: #10233B; border-bottom: 1px solid #E1E8EB; background-color: #F5F8F9; }}
  table.data td {{ padding: 5px 6px; border-bottom: 1px solid #F0F3F4; color: #3E5470; }}
  .disclaimer {{ margin-top: 20px; padding: 10px 12px; background-color: #FDF1EF; border-left: 3px solid #C1443C; font-size: 8.5pt; line-height: 1.5; color: #10233B; }}
  .footer {{ margin-top: 18px; font-size: 8pt; color: #3E5470; text-align: center; }}
</style>
</head>
<body>
  <h1>{_esc(APP_NAME)} &mdash; Mortality Risk Report</h1>
  <div class="meta">Generated {_esc(generated_at)} &middot; Federated ensemble, {n_models} client models</div>

  <table class="banner-table">
    <tr>
      <td>
        <div class="eyebrow">RESULT</div>
        <div class="verdict">{_esc(verdict)}</div>
        <span class="pct">{pct}%</span>&nbsp;&nbsp;<span class="pill">{_esc(RISK_COPY[result.risk_level])}</span>
        <div class="agreement">Model agreement across the {n_models} federated clients: {agreement_pct}%</div>
      </td>
    </tr>
  </table>

  <h3>Patient inputs</h3>
  <table class="data">
    <tr><th>Field</th><th>Value</th><th>Category</th></tr>
    {_input_rows(form)}
  </table>

  <h3>Per-model votes</h3>
  <table class="data">
    <tr><th>Model</th><th>Predicted probability of death</th></tr>
    {_model_rows(result)}
  </table>

  <h3>Risk tiers</h3>
  <table class="data">
    <tr><th>Tier</th><th>Predicted probability</th></tr>
    <tr><td>Low</td><td>Below {int(RISK_THRESHOLDS['low'] * 100)}%</td></tr>
    <tr><td>Moderate</td><td>{int(RISK_THRESHOLDS['low'] * 100)}% &ndash; {int(RISK_THRESHOLDS['moderate'] * 100)}%</td></tr>
    <tr><td>High</td><td>{int(RISK_THRESHOLDS['moderate'] * 100)}% &ndash; {int(RISK_THRESHOLDS['high'] * 100)}%</td></tr>
    <tr><td>Critical</td><td>{int(RISK_THRESHOLDS['high'] * 100)}% and above</td></tr>
  </table>

  <div class="disclaimer">
    This report is generated by a research demo built on the WiDS Datathon ICU
    mortality dataset and a federated XGBoost ensemble. It is <strong>not</strong>
    a certified clinical decision-support device and must not be used for real
    patient care or as a substitute for clinical judgment.
  </div>

  <div class="footer">{_esc(APP_NAME)} &middot; Federated XGBoost mortality risk estimation for ICU patients</div>
</body>
</html>"""


def generate_pdf_report(form: FormInput, result: PredictionResult, n_models: int) -> bytes:
    """Render the clinical report as PDF bytes, ready for a download button."""
    html_source = _pdf_html(form, result, n_models)
    buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=html_source, dest=buffer)
    if pisa_status.err:
        raise RuntimeError(f"Failed to render PDF report ({pisa_status.err} error(s)).")
    return buffer.getvalue()
