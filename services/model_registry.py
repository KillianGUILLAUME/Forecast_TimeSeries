from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_ROOT = _PROJECT_ROOT / "models"


def get_models_root() -> Path:
    """Return the directory that stores persisted model artifacts.

    The directory is created on demand so that callers can immediately write
    checkpoints without having to guard the path creation on their side.
    """

    _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return _MODELS_ROOT


def suggest_model_dir() -> Path:
    base_model = get_models_root()
    base = base_model / "Forecast" / "artifacts" / "models"
    latest = base / "latest"
    if latest.exists():
        return latest
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"Aucun run trouvé dans {base}")
    runs.sort(key=lambda p: p.name)
    return runs[-1]


def register_model_dir(path: Path) -> None:
    """Record a newly saved model directory.

    Currently this simply ensures the results directory exists, but keeping
    the helper centralised allows future improvements (like writing a pointer to
    the latest run) without touching the training logic again.
    """

    resolved = path.resolve()
    if not resolved.exists():
        # The caller is expected to create the directory before registering it.
        raise FileNotFoundError(f"Model directory does not exist: {resolved}")
    get_models_root()