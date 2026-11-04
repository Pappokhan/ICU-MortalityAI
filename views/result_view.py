"""VIEW layer — renders a Prediction as HTML cards, bars and tables."""
from models.predictor import Prediction
from models.risk import classify
from models.schema import label_for
from views.layout import esc, render_template, show_html


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _bar(name: str, fraction: float, value: str, cls: str) -> str:
    width = max(0.0, min(100.0, fraction * 100))
    return render_template("bar_row", name=esc(name), width=f"{width:.1f}",
                           value=esc(value), cls=cls)


def render_results(pred: Prediction) -> None:
    band = classify(pred.p_fl, pred.t_youden, pred.t_f1max)
    central = classify(pred.p_central, pred.t_youden, pred.t_f1max)

    consensus = "all agree" if pred.agree == pred.n_clients else "partial agreement"
    show_html(render_template(
        "summary",
        band_key=band.key, band_label=esc(band.label), band_text=esc(band.text),
        central_key=central.key, central_label=esc(central.label),
        p_central=_pct(pred.p_central), p_fl=_pct(pred.p_fl),
        agree=pred.agree, n_clients=pred.n_clients, consensus_text=consensus,
    ))

    partition_rows = "".join(
        _bar(f"Partition {i + 1}", p, _pct(p), "accent")
        for i, p in enumerate(pred.client_probs)
    )

    drivers_panel = ""
    if pred.drivers:
        biggest = max(abs(c) for _, c in pred.drivers) or 1.0
        rows = "".join(
            _bar(label_for(f), abs(c) / biggest, f"{c:+.3f}", "up" if c > 0 else "down")
            for f, c in pred.drivers
        )
        drivers_panel = render_template("drivers_panel", driver_rows=rows)

    show_html(render_template(
        "results",
        partition_rows=partition_rows, drivers_panel=drivers_panel,
        t_youden=f"{pred.t_youden:.3f}", t_f1max=f"{pred.t_f1max:.3f}",
    ))
