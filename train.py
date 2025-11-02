import os, sys
print("CWD:", os.getcwd(), "\nFILES:", os.listdir())
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


# --- Bootstrap Kaggle/local ---
from pathlib import Path
import os, sys, torch, platform, time

# Dossier de travail pour outputs (Kaggle ou local)
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
OUT_DIR = WORK_DIR / "Forecast" / "artifacts" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("WORK_DIR:", WORK_DIR)
print("OUT_DIR:", OUT_DIR)
# Diagnostiques utiles
try:
    import subprocess; subprocess.run(["nvidia-smi"])
except Exception: pass
print("CUDA:", torch.cuda.is_available())

# Si tes modules .py sont à la racine du projet que tu as poussé
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))



from main import run_lstm_training


def resolve_work_dir() -> Path:
    """Renvoie /kaggle/working si dispo, sinon le dossier courant."""
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

def resolve_save_dir(project_name="Forecast") -> Path:
    base = resolve_work_dir() / project_name / "artifacts" / "models"
    run = time.strftime("%Y%m%d-%H%M%S")
    save_dir = base / run
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir



DEFAULT_TRAIN_TICKERS: List[str] = [ "SPY"]#, "ISF.L", "CAC.PA", "EXS1.DE", "IAEX.AS", "1321.T", "XIC.TO", "2800.HK", "STW.AX", "510300.SS"]
FOREX_TICKERS: Dict[str, str] = {
    "EURUSD=X": "FX_EURUSD",
    "GBPUSD=X": "FX_GBPUSD",
    "USDJPY=X": "FX_USDJPY",
}

LSTM_FEATURE_COLUMNS: List[str] = [
    "volume_log",
    "ret",
    "ret_mean_5",
    "ret_mean_20",
    "ret_mean_50",
    "ret_std_20",
    "ret_std_50",
    "price_zscore_20",
    "RSI_14",
] + [f"{alias}_RET" for alias in FOREX_TICKERS.values()]

TARGET_COLUMN: str = "ret"

hp = {
    "window_size": 60, #100
    "hidden_size": 48, #64
    "num_layers": 1, #2
    "lr": 1e-3,
    "epochs": 700, #200
    "horizon": 10,
    "residual_boosting": False,
}

hp_small = {
    "window_size": 1, #100
    "hidden_size": 1, #64
    "num_layers": 1, #2
    "lr": 1e-3,
    "epochs": 1, #200
    "horizon": 10,
    "residual_boosting": False,
}


save_dir = resolve_save_dir("Forecast")
best_path = save_dir / "best.pt"   # meilleur modèle (val la + basse)
last_path = save_dir / "last.pt"   # dernier état (pour reprise)
print("Saving to:", save_dir)


run_lstm_training(
    hp=hp_small,
    save_dir=str(save_dir),
    tickers= DEFAULT_TRAIN_TICKERS,
    period = "max",
    interval = "1d",
    plot_training = False,
    plot_dir  = False
)