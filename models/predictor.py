"""MODEL layer — central XGBoost + 5-partition probability-averaging ensemble."""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class Prediction:
    p_central: float
    p_fl: float
    client_probs: List[float]
    agree: int                      # partitions agreeing with the ensemble decision
    n_clients: int
    t_youden: float
    t_f1max: float
    drivers: List[Tuple[str, float]] = field(default_factory=list)  # (feature, log-odds)


class Predictor:
    def __init__(self, central, clients, t_youden: float, t_f1max: float):
        self.central = central
        self.clients = clients
        self.t_youden = t_youden
        self.t_f1max = t_f1max

    def predict(self, X: pd.DataFrame, top_k: int = 8) -> Prediction:
        p_central = float(self.central.predict_proba(X)[:, 1][0])
        client_p = [float(m.predict_proba(X)[:, 1][0]) for m in self.clients]
        p_fl = float(np.mean(client_p))        # probability averaging (not FedAvg)

        ensemble_flag = p_fl >= self.t_youden
        agree = sum((p >= self.t_youden) == ensemble_flag for p in client_p)

        return Prediction(p_central, p_fl, client_p, agree, len(client_p),
                          self.t_youden, self.t_f1max, self._drivers(X, top_k))

    def _drivers(self, X: pd.DataFrame, top_k: int):
        try:
            c = self.central.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
            pairs = sorted(zip(X.columns, map(float, c)), key=lambda t: -abs(t[1]))
            return pairs[:top_k]
        except Exception:
            return []
