from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import DERIVED_FEATURES, FEATURE_META, FEATURE_ORDER, age_to_risk_category
from model.ml_model import PredictionResult, get_model


@dataclass
class FormInput:
    age: float
    apache_4a_hospital_death_prob: float
    apache_4a_icu_death_prob: float
    gcs_verbal_apache: int
    gcs_eyes_apache: int
    gcs_motor_apache: int
    d1_spo2_min: float
    d1_sysbp_min: float
    d1_sysbp_noninvasive_min: float
    h1_spo2_max: float
    d1_mbp_min: float
    d1_mbp_noninvasive_min: float
    d1_temp_min: float
    d1_diasbp_min: float
    d1_diasbp_noninvasive_min: float


def _min_max_scale(value: float, feat_name: str) -> float:
    meta = FEATURE_META[feat_name]
    lo, hi = meta["scale_min"], meta["scale_max"]
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def build_feature_vector(form: FormInput) -> pd.DataFrame:
    raw = asdict(form)
    raw["age_risk_category"] = age_to_risk_category(form.age)

    row = {}
    for feat in FEATURE_ORDER:
        row[feat] = float(raw[feat]) if feat in DERIVED_FEATURES else _min_max_scale(raw[feat], feat)
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_mortality_risk(form: FormInput) -> PredictionResult:
    ensemble = get_model()
    feature_vector = build_feature_vector(form)
    return ensemble.predict(feature_vector)
