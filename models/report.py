"""MODEL layer — plain data describing one prediction case (used by the report)."""
from dataclasses import dataclass
from datetime import datetime
from typing import List

from models.predictor import Prediction
from models.risk import RiskBand


@dataclass
class InputRow:
    label: str
    value: str
    note: str = ""


@dataclass
class CaseReport:
    created: datetime
    band: RiskBand            # federated ensemble band
    central_band: RiskBand    # central XGBoost band
    prediction: Prediction
    inputs: List[InputRow]
