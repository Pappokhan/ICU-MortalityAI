"""CONTROLLER layer — turns form input into a prediction using the models."""
import math
from datetime import datetime
from typing import Dict, List, Optional

from models.artifact_store import ArtifactStore
from models.predictor import Prediction, Predictor
from models.report import CaseReport, InputRow
from models.risk import classify
from models.preprocessor import Preprocessor, age_risk_category
from models.schema import FIELDS, label_for


class PredictionController:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.preprocessor = Preprocessor(store.scaler, store.imputer, store.features)
        self.predictor = Predictor(store.central, store.clients,
                                   store.t_youden, store.t_f1max)

    @property
    def features(self) -> List[str]:
        return self.store.features

    # -- form values (as entered by the user) -> raw numeric feature dict ----
    def build_raw(self, form: Dict[str, object]) -> Dict[str, Optional[float]]:
        raw: Dict[str, Optional[float]] = {}
        for key in self.features:
            v = form.get(key)
            kind = FIELDS[key].kind if key in FIELDS else "float"
            if v is None:
                raw[key] = None
            elif kind == "age":
                raw[key] = float(age_risk_category(float(v)))
            elif kind == "choice":                       # ventilated_apache
                raw[key] = {"Unknown": None, "No": 0.0, "Yes": 1.0}[str(v)]
            else:
                x = float(v)
                if not math.isfinite(x):
                    raise ValueError(f"Invalid value for {key}")
                raw[key] = x
        return raw

    def predict(self, form: Dict[str, object]) -> Prediction:
        raw = self.build_raw(form)
        X = self.preprocessor.transform(raw)
        return self.predictor.predict(X)

    # -- snapshot of one prediction (inputs + results) used for the downloadable report ----
    def build_report(self, form: Dict[str, object], pred: Prediction) -> CaseReport:
        rows: List[InputRow] = []
        for key in self.features:
            v = form.get(key)
            f = FIELDS.get(key)
            label = label_for(key)
            if v is None:
                med = self.preprocessor.median_for(key)
                rows.append(InputRow(label, "N/A", f"median {med:.3g} used"))
            elif f is not None and f.kind == "choice":
                rows.append(InputRow(label, str(v), ""))
            elif f is not None and f.kind == "age":
                cat = age_risk_category(float(v))
                rows.append(InputRow(label, f"{int(v)}", f"age risk category {cat}"))
            else:
                rows.append(InputRow(label, f"{float(v):g}", ""))
        return CaseReport(
            created=datetime.now(),
            band=classify(pred.p_fl, pred.t_youden, pred.t_f1max),
            central_band=classify(pred.p_central, pred.t_youden, pred.t_f1max),
            prediction=pred,
            inputs=rows,
        )
