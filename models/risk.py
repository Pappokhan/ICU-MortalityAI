"""MODEL layer — risk band logic."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskBand:
    key: str        # css class suffix: low | moderate | high
    label: str
    text: str


def classify(p: float, t_youden: float, t_f1max: float) -> RiskBand:
    if p < t_youden:
        return RiskBand("low", "Low risk",
                        f"Below the high-sensitivity cut-off ({t_youden:.3f}).")
    if p < t_f1max:
        return RiskBand("moderate", "Moderate risk",
                        f"Between the high-sensitivity ({t_youden:.3f}) and F1-optimal ({t_f1max:.3f}) cut-offs.")
    return RiskBand("high", "High risk",
                    f"At or above the F1-optimal cut-off ({t_f1max:.3f}).")
