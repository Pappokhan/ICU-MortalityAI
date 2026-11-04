"""MODEL layer — reproduces Notebook 1 preprocessing (median impute -> MinMax scale)."""
import math
from typing import Dict, List, Optional

import pandas as pd


def age_risk_category(age: float) -> int:
    """Same rule as Notebook 1."""
    if age < 45:
        return 0
    if age <= 65:
        return 1
    if age <= 80:
        return 2
    return 3


class Preprocessor:
    def __init__(self, scaler, imputer, features: List[str]):
        self.features = features
        # The imputer/scaler were fitted on many columns; keep stats for ours.
        self._median = dict(zip(imputer.feature_names_in_, imputer.statistics_))
        cols = list(scaler.feature_names_in_)
        self._scale = dict(zip(cols, scaler.scale_))
        self._shift = dict(zip(cols, scaler.min_))

    def median_for(self, key: str):
        """Training-set median used when a value is marked N/A."""
        return self._median.get(key)

    def transform(self, raw: Dict[str, Optional[float]]) -> pd.DataFrame:
        row = []
        for f in self.features:
            v = raw.get(f)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = self._median[f]                     # training median
            row.append(v * self._scale[f] + self._shift[f])   # MinMaxScaler.transform
        return pd.DataFrame([row], columns=self.features)
