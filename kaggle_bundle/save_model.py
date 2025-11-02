from pathlib import Path
import time, json, torch, os

def resolve_save_dir(project_name="Forecast", file:str = 'models_saved'):
    base = Path("/Users/killianguillaume/Desktop/Forecast_MP") / project_name
    run = time.strftime("%Y%m%d-%H%M%S")
    save_dir = base / "artifacts" / "models" / run
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

