"""Central configuration: paths and default constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Artefacts are looked up in these folders, in order.
# 1) $ICU_ARTIFACT_DIR (optional override)  2) ./artifacts  3) project root
_env = os.environ.get("ICU_ARTIFACT_DIR")
ARTIFACT_DIRS = ([Path(_env)] if _env else []) + [ROOT / "artifacts", ROOT]

N_CLIENTS = 5

# Fallbacks used only if pipeline_artefacts.json does not contain the values.
DEFAULT_YOUDEN = 0.0718
DEFAULT_F1MAX = 0.264

ASSETS_DIR = ROOT / "assets"
TEMPLATES_DIR = ROOT / "views" / "templates"


def find_artifact(name: str) -> Path:
    for d in ARTIFACT_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(2, "Artefact not found", name)
