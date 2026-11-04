"""MODEL layer — loads every saved artefact from the notebooks."""
import json
import pickle
from dataclasses import dataclass
from typing import Any, List

import config


def _load_pickle(name: str):
    with open(config.find_artifact(name), "rb") as f:
        return pickle.load(f)


def _force_cpu(model):
    """Models were trained with device='cuda'; make them runnable on CPU."""
    try:
        model.set_params(device="cpu")
    except Exception:
        pass
    try:
        model.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass
    return model


@dataclass
class ArtifactStore:
    arts: dict
    scaler: Any
    imputer: Any
    central: Any
    clients: List[Any]

    @classmethod
    def load(cls) -> "ArtifactStore":
        with open(config.find_artifact("pipeline_artefacts.json")) as f:
            arts = json.load(f)
        scaler = _load_pickle("minmax_scaler.pkl")
        imputer = _load_pickle("num_imputer.pkl")
        central = _force_cpu(_load_pickle("tuned_xgb.pkl"))
        clients = [_force_cpu(_load_pickle(f"fl_client_{i}.pkl"))
                   for i in range(1, config.N_CLIENTS + 1)]
        return cls(arts, scaler, imputer, central, clients)

    @property
    def features(self) -> List[str]:
        return list(self.arts["top_features"])   # exact training order

    @property
    def t_youden(self) -> float:
        return float(self.arts.get("xgb_threshold_youden_tuned",
                     self.arts.get("xgb_threshold_youden", config.DEFAULT_YOUDEN)))

    @property
    def t_f1max(self) -> float:
        return float(self.arts.get("xgb_threshold_f1max", config.DEFAULT_F1MAX))
