from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Union

import numpy as np
import pandas as pd

from config import FEATURE_ORDER, MODEL_PATH, RISK_THRESHOLDS


class ModelLoadError(RuntimeError):
    pass


@dataclass
class PredictionResult:
    probability: float
    predicted_class: int
    per_model_probabilities: List[float]
    risk_level: str
    agreement: float


class FederatedXGBoostEnsemble:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.models = self._load_models(model_path)

    @staticmethod
    def _load_models(model_path: str):
        if not os.path.exists(model_path):
            raise ModelLoadError(
                f"Model file not found at '{model_path}'. Make sure "
                f"FL_XGBoost_Global_Ensemble.pkl is inside the model/ folder."
            )
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as exc:
            raise ModelLoadError(f"Could not unpickle the model file: {exc}") from exc

        models = obj if isinstance(obj, list) else [obj]

        for i, clf in enumerate(models):
            if not hasattr(clf, "predict_proba"):
                raise ModelLoadError(
                    f"Object #{i} inside the pickle has no predict_proba method "
                    f"— expected an XGBClassifier-like estimator."
                )
            n_in = getattr(clf, "n_features_in_", None)
            if n_in is not None and n_in != len(FEATURE_ORDER):
                raise ModelLoadError(
                    f"Model #{i} expects {n_in} features but this app builds "
                    f"{len(FEATURE_ORDER)} (see config.FEATURE_ORDER). Update "
                    f"FEATURE_ORDER to match the model's training columns."
                )
        return models

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_features(self) -> int:
        return len(FEATURE_ORDER)

    def predict(self, feature_vector: Union[np.ndarray, pd.DataFrame]) -> PredictionResult:
        if isinstance(feature_vector, pd.DataFrame):
            X = feature_vector
        else:
            X = pd.DataFrame(
                np.asarray(feature_vector, dtype=float).reshape(1, -1),
                columns=FEATURE_ORDER,
            )

        per_model_probs = [float(clf.predict_proba(X)[0][1]) for clf in self.models]
        avg_prob = float(np.mean(per_model_probs))
        agreement = float(1.0 - np.std(per_model_probs))

        return PredictionResult(
            probability=avg_prob,
            predicted_class=int(avg_prob >= 0.5),
            per_model_probabilities=per_model_probs,
            risk_level=self._risk_level(avg_prob),
            agreement=agreement,
        )

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < RISK_THRESHOLDS["low"]:
            return "low"
        elif prob < RISK_THRESHOLDS["moderate"]:
            return "moderate"
        elif prob < RISK_THRESHOLDS["high"]:
            return "high"
        return "critical"


@lru_cache(maxsize=1)
def get_model() -> FederatedXGBoostEnsemble:
    return FederatedXGBoostEnsemble()
