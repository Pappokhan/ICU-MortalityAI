"""VIEW layer — renders a CaseReport as a one-page landscape PDF (reportlab)."""
from io import BytesIO

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from models.report import CaseReport
from models.schema import label_for

ACCENT = colors.HexColor("#0f766e")
BAND_COLOR = {"low": colors.HexColor("#15803d"), "moderate": colors.HexColor("#d97706"),
              "high": colors.HexColor("#dc2626")}
GREY = colors.HexColor("#6b7280")
LIGHT = colors.HexColor("#f3f4f6")
LINE = colors.HexColor("#d1d5db")

PAGE_W, PAGE_H = landscape(A4)
MARGIN = 11 * mm
CONTENT_W = PAGE_W - 2 * MARGIN          # ~275 mm
LEFT_W = 122 * mm
GAP = 6 * mm
RIGHT_W = CONTENT_W - LEFT_W - GAP


def _t(text: str) -> str:
    """Base-14 PDF fonts have no subscript/≥ glyphs; map them to plain text."""
    for a, b in {"₂": "2", "≥": ">=", "≤": "<=", "–": "-", "·": "|", "—": "-"}.items():
        text = text.replace(a, b)
    return text


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _hex(c) -> str:
    return "#" + c.hexval()[2:]


def _bars(rows, width, row_h=14) -> Drawing:
    """rows: [(label, fraction 0..1, value_text, color)]; width in points."""
    label_w, val_w = 170, 42
    bar_w = width - label_w - val_w - 6
    d = Drawing(width, row_h * len(rows))
    for i, (label, frac, val, col) in enumerate(rows):
        y = row_h * (len(rows) - i - 1)
        d.add(String(0, y + 4, _t(label)[:46], fontName="Helvetica", fontSize=7.5, fillColor=colors.black))
        d.add(Rect(label_w, y + 3, bar_w, 8, fillColor=LIGHT, strokeColor=LINE, strokeWidth=.4))
        d.add(Rect(label_w, y + 3, max(0.0, min(1.0, frac)) * bar_w, 8, fillColor=col, strokeColor=None))
        d.add(String(width, y + 4, val, fontName="Helvetica-Bold", fontSize=7.5,
                     textAnchor="end", fillColor=colors.black))
    return d


def build_pdf(report: CaseReport) -> bytes:
    pred, band = report.prediction, report.band
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], alignment=0, fontSize=18, leading=20,
                        textColor=ACCENT, spaceAfter=0)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=8, textColor=GREY)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, leading=12, textColor=ACCENT,
                        spaceBefore=0, spaceAfter=3)
    note = ParagraphStyle("note", parent=ss["Normal"], fontSize=7, leading=9, textColor=GREY)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=8, leading=9.5)
    big = ParagraphStyle("big", parent=body, fontSize=16, leading=19)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=10)
    tiny = ParagraphStyle("tiny", parent=body, fontSize=6.5, leading=8)
    bandlbl = ParagraphStyle("bandlbl", parent=body, fontSize=15, leading=18)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=10 * mm, bottomMargin=9 * mm,
                            title="ICU-MortalityAI report", author="ICU-MortalityAI")
    s = []

    # ---- title ----
    s.append(Paragraph("ICU-MortalityAI - Risk Report", h1))
    s.append(Spacer(1, 2))
    s.append(Paragraph(f"Generated {report.created:%Y-%m-%d %H:%M} | Research prototype - not for clinical use", sub))
    s.append(Spacer(1, 5))

    # ---- row 1: risk banner + 3 metric cards ----
    bc = BAND_COLOR[band.key]
    cons = "all agree" if pred.agree == pred.n_clients else "partial agreement"
    banner = Table([[Paragraph(f'<font color="{_hex(bc)}"><b>{_t(band.label)}</b></font>', bandlbl)],
                    [Paragraph(_t(f"Federated ensemble - {band.text}"), body)]], colWidths=[80 * mm])
    banner.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 4, bc), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                                ("BOX", (0, 0), (-1, -1), .5, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))

    def card(title, value, tag, tag_color):
        c = Table([[Paragraph(f'<font color="#6b7280">{title}</font>', tiny)],
                   [Paragraph(f'<b>{value}</b>', big)],
                   [Paragraph(f'<font color="{_hex(tag_color)}"><b>{_t(tag)}</b></font>', small)]],
                  colWidths=[(CONTENT_W - 80 * mm - 3 * 4 * mm) / 3])
        c.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .5, LINE), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        return c

    row1 = Table([[banner,
                   card("CENTRAL XGBOOST", _pct(pred.p_central), report.central_band.label,
                        BAND_COLOR[report.central_band.key]),
                   card("FEDERATED ENSEMBLE", _pct(pred.p_fl), band.label, bc),
                   card("PARTITION CONSENSUS", f"{pred.agree} / {pred.n_clients}", cons, GREY)]],
                 colWidths=[80 * mm + 4 * mm] + [(CONTENT_W - 84 * mm) / 3] * 3)
    row1.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                              ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                              ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s.append(row1)
    s.append(Spacer(1, 8))

    # ---- row 2 left: inputs ----
    data = [["Parameter", "Value", "Note"]] + [
        [Paragraph(_t(r.label), body), r.value, Paragraph(_t(r.note), body)] for r in report.inputs]
    inputs = Table(data, colWidths=[LEFT_W - 52 * mm, 16 * mm, 36 * mm], repeatRows=1)
    inputs.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                                ("GRID", (0, 0), (-1, -1), .3, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
    left = [Paragraph("Patient inputs", h2), inputs]

    # ---- row 2 right: partitions + drivers ----
    bar_w = RIGHT_W - 2
    right = [Paragraph("Per-partition predictions", h2),
             _bars([(f"Partition {i + 1}", p, _pct(p), ACCENT) for i, p in enumerate(pred.client_probs)], bar_w),
             Paragraph("The five partitions are stratified splits of one dataset, not independent hospitals. "
                       "The ensemble averages predicted probabilities; it is not parameter-averaging FedAvg.", note),
             Spacer(1, 8)]
    if pred.drivers:
        big = max(abs(c) for _, c in pred.drivers) or 1.0
        right += [Paragraph("Top drivers (central model, log-odds)", h2),
                  _bars([(label_for(f), abs(c) / big, f"{c:+.3f}",
                          BAND_COLOR["high"] if c > 0 else BAND_COLOR["low"]) for f, c in pred.drivers], bar_w),
                  Paragraph("Red pushes the prediction toward higher mortality risk, green toward lower.", note)]

    row2 = Table([[left, "", right]], colWidths=[LEFT_W, GAP, RIGHT_W])
    row2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    s.append(row2)
    s.append(Spacer(1, 8))

    # ---- notes ----
    s.append(Paragraph(_t(
        f"<b>Interpretation.</b> Risk bands: Low &lt; {pred.t_youden:.3f} (Youden's J, high-sensitivity cut-off) | "
        f"Moderate {pred.t_youden:.3f}-{pred.t_f1max:.3f} | High &gt;= {pred.t_f1max:.3f} (F1-maximising cut-off). "
        "Models were trained on a SMOTE-balanced set, so probabilities are not calibrated to real-world "
        "mortality - interpret the bands, not the raw percentages."), note))
    s.append(Spacer(1, 3))
    s.append(Paragraph("<b>Disclaimer.</b> ICU-MortalityAI is a thesis prototype. Predictions are for research "
                       "demonstration only and must not guide patient care.", note))
    doc.build(s)
    return buf.getvalue()
