from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_ROOT = _PROJECT_ROOT / "results"


def get_models_root() -> Path:
    """Return the directory that stores persisted model artifacts.

    The directory is created on demand so that callers can immediately write
    checkpoints without having to guard the path creation on their side.
    """

    _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return _MODELS_ROOT


def suggest_model_dir(prefix: str = "run") -> Path:
    """Return a timestamped directory inside :func:'get_models_root'.

    The directory is not created automatically because Streamlit forms allow
    users to edit the suggested value before launching the training. The caller
    is expected to create the directory when the training actually starts.
    """

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return get_models_root() / f"{prefix}-{ts}"


def _iter_candidate_directories(root: Path) -> Iterable[Path]:
    for entry in root.iterdir():
        if entry.is_dir():
            yield entry


def resolve_latest_model_dir() -> Optional[Path]:
    """Return the most recent model directory available for inference.

    The helper looks for directories containing a meta.json file, which is
    produced by :meth:prediction_lstm_model.LSTMPredictorProba.save. When no
    such directory exists we fallback to returning the results directory if
    it contains a model.pt file so legacy single-file checkpoints remain
    usable.
    """

    root = get_models_root()
    candidates = []
    for entry in _iter_candidate_directories(root):
        if (entry / "meta.json").exists():
            candidates.append(entry)
    if candidates:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    legacy_checkpoint = root / "model.pt"
    if legacy_checkpoint.exists():
        return root

    return None


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