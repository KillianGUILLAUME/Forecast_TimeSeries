import argparse
import threading
import numpy as np
import pandas as pd

"""
Script principal pour l'analyse et la visualisation des ETF européens

Architecture:
1. etf_collector.py -> Collecte des données
2. etf_visualizer.py -> Visualisations
3. main.py -> Orchestration (ce fichier)
"""

from etf_collector import EuropeanETFCollector
from etf_visualizer import ETFVisualizer

import sys
import os, json
from typing import Dict, List, Optional




from data_preprocessing import (
    LSTM_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_prepared_ticker,
    build_lstm_features,
    prepare_lstm_training_datasets,
    resolve_training_universe,
)


def env_var_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', '0', 'no', 'n'}:
        return False
    return default

def get_config_from_env():
    """Récupère la configuration depuis les variables d'environnement (interface GUI)"""
    tickers_str = os.getenv('ETF_TICKERS', '').strip()            # NEW
    tickers = [t.strip() for t in tickers_str.split(',') if t.strip()] if tickers_str else []
    return {
        'period': os.getenv('ETF_PERIOD', '5y'),
        'interval': os.getenv('ETF_INTERVAL', '1d'),
        'light': env_var_to_bool(os.getenv('ETF_LIGHT'), False),
        'max': env_var_to_bool(os.getenv('ETF_MAX'), True),
        'action': os.getenv('ETF_ACTION', ''),
        'log_level': env_var_to_bool(os.getenv('ETF_LOG_PLOTS'), False),
        'tickers': tickers
    }

DEFAULT_LSTM_HP = {
    "window_size": 1, #100
    "hidden_size": 1, #64
    "num_layers": 1, #2
    "lr": 1e-3,
    "epochs": 1, #200
    "horizon": 10,
    "residual_boosting": False,
    "boosting_params": {
        "n_estimators": 130, #200
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 31, #31
    },    
}

HP_CASTERS = {
    "window_size": int,
    "hidden_size": int,
    "num_layers": int,
    "lr": float,
    "epochs": int,
    "horizon": int,
}




def get_lstm_from_env():
    action   = (os.getenv("LSTM_ACTION", "") or "").lower()
    raw_hp   = os.getenv("LSTM_HP", "{}")
    try:
        hp = json.loads(raw_hp) if raw_hp else {}
    except json.JSONDecodeError:
        print("Hyperparamètres LSTM invalides, utilisation des valeurs par défaut.")
        hp = {}
    ticker   = (os.getenv("LSTM_TICKER", "") or "").strip()
    load_dir = (os.getenv("LSTM_LOAD_DIR", "") or "").strip()
    save_dir = (os.getenv("LSTM_SAVE_DIR", "") or "").strip()
    return action, hp, ticker, load_dir, save_dir


def normalize_lstm_hp(raw_hp: Dict) -> Dict[str, float]:
    """Normalise les hyperparamètres reçus (convertit en bons types + défauts)."""
    hp = DEFAULT_LSTM_HP.copy()
    hp["boosting_params"] = dict(DEFAULT_LSTM_HP.get("boosting_params", {}))    
    if not isinstance(raw_hp, dict):
        return hp
    for key, caster in HP_CASTERS.items():
        if key not in raw_hp:
            continue
        try:
            hp[key] = caster(raw_hp[key])
        except (TypeError, ValueError):
            print(f"Hyperparamètre '{key}' invalide ({raw_hp[key]!r}), valeur par défaut {hp[key]} conservée.")

    if "residual_boosting" in raw_hp:
        hp["residual_boosting"] = env_var_to_bool(raw_hp["residual_boosting"], hp["residual_boosting"])

    def _coerce(value, caster, fallback):
        try:
            return caster(value)
        except (TypeError, ValueError):
            return fallback

    raw_boost = raw_hp.get("boosting_params") if isinstance(raw_hp, dict) else None
    if isinstance(raw_boost, str):
        try:
            raw_boost = json.loads(raw_boost)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_boost = None
    if isinstance(raw_boost, dict):
        for key, value in raw_boost.items():
            if value is None:
                continue
            if key in {"n_estimators", "max_depth", "num_leaves"}:
                hp["boosting_params"][key] = _coerce(value, int, hp["boosting_params"].get(key))
            elif key == "learning_rate":
                hp["boosting_params"][key] = _coerce(value, float, hp["boosting_params"].get(key))
            else:
                hp["boosting_params"][key] = value

    alias_map = {
        "boosting_n_estimators": ("n_estimators", int),
        "boosting_learning_rate": ("learning_rate", float),
        "boosting_max_depth": ("max_depth", int),
        "boosting_num_leaves": ("num_leaves", int),
    }
    for alias, (target_key, caster) in alias_map.items():
        if alias in raw_hp:
            hp["boosting_params"][target_key] = _coerce(raw_hp[alias], caster, hp["boosting_params"].get(target_key))

    return hp




def print_progress(message: str, step: int = None, total_steps: int = None):
    """Affiche les messages de progression pour l'interface"""
    if step and total_steps:
        progress = f"[{step}/{total_steps}] "
    else:
        progress = ""
    print(f"{progress}{message}")
    sys.stdout.flush()  


def build_pipeline(df0: pd.DataFrame) -> pd.DataFrame:
    """Construit le df des indicateurs utilisé par le LSTM.
        Utilisé pour les prédictions sur une série unique."""

    if not isinstance(df0, pd.DataFrame):
        raise TypeError("build_pipeline attend un DataFrame en entrée")

    feature = build_lstm_features(df0)
    if feature.empty:
        raise ValueError("Données insuffisantes pour construire les indicateurs.")
    
    missing = [col for col in LSTM_FEATURES if col not in feature.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après construction des indicateurs: {missing}")
    
    result = feature[LSTM_FEATURES].copy()
    if "adj_close" in feature.columns and "adj_close" not in result.columns:
        result["adj_close"] = feature["adj_close"]
    return result


def build_features_for_ticker(collector: EuropeanETFCollector, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_to_pred = collector.get_one_frame(ticker)
    if df_to_pred is None or df_to_pred.empty:
        raise ValueError(f"Aucune donnée disponible pour le ticker {ticker}.")
    dfp = build_pipeline(df_to_pred)
    if dfp.empty:
        raise ValueError(f"Données insuffisantes après préparation pour {ticker}.")
    return df_to_pred, dfp


LSTM_FEATURES = LSTM_FEATURE_COLUMNS
TARGET_FEATURE = 'ret'




def run_pipeline_for_graphics(
    tickers: Optional[List[str]] = None,
    period: str = "5y",
    interval: str = "1d",
    adjusted: bool = True,
    light: bool = False,
    max=True,
    log_level: bool = False,
    action: str = "single") -> None:
    """
    Orchestration complète : collecte -> résumé -> visualisations

    Args:
        tickers: Liste de tickers à analyser. Par défaut, tous les ETF connus.
        start: Date de début (YYYY-MM-DD)
        end: Date de fin (YYYY-MM-DD) ou None pour aujourd'hui
        interval: Intervalle Yahoo Finance (ex: "1d", "1wk", "1mo")
        adjusted: Utiliser les prix ajustés
        exchange_filter: Filtrer par place de cotation (ex: ".PA", ".L", ".DE")
        light: Si True, affiche des visuels essentiels seulement
    """
    print("=== Analyse complète des ETF Européens ===\n")

    # ========================================
    # ÉTAPE 1: COLLECTE DES DONNÉES
    # ========================================
    print("📊 ÉTAPE 1: Collecte des données")
    collector = EuropeanETFCollector() if not tickers else EuropeanETFCollector(tickers=tickers)

    if not tickers:
        tickers = collector.get_tickers()
    else:
        collector.tickers = tickers


    dfs: List[pd.DataFrame] = collector.get_etf_frames(
        period= "max" if max else period,
        interval=interval)

    if not dfs:
        print("Aucune donnée n'a pu être téléchargée. Arrêt.")
        sys.exit(1)

    tickers = collector.get_tickers()

    print(f"Tickers sélectionnés ({len(tickers)}): {', '.join(tickers)}")



    # ========================================
    # ÉTAPE 2: RÉSUMÉ & TABLEAU DE BORD
    # ========================================


    summary_df: pd.DataFrame = collector.build_summary()
    if summary_df.empty:
        print("Résumé vide — impossible de poursuivre les visualisations.")
        sys.exit(1)

    # Tri par rendement total décroissant pour affichage

    summary_df = summary_df.sort_values("total_return_%", ascending=False).reset_index(drop=True)

    # collector.print_results(summary_df, title="Résumé des ETF")

    prices = {
        row['ticker']: row['price_series'].to_frame(name='adj_close')
        for _, row in summary_df.iterrows()
    }


    # ========================================
    # ÉTAPE 3: VISUALISATIONS
    # ========================================
    
    viz = ETFVisualizer()

    # 3.1 Courbes individuelles 
    subset_for_compare = {t: prices[t] for t in tickers if t in prices}

    frames_by_ticker: Dict[str, pd.DataFrame] = {}
    for t, df in zip(tickers, dfs):
        if df is None:
            continue
        if 'ticker' not in df.columns:
            df = df.copy()
            df['ticker'] = t
        frames_by_ticker[t] = df


    if action == "single":
        t = tickers[0] if isinstance(tickers, list) else tickers
        df_viz = frames_by_ticker.get(t)
        viz.plot_single_etf(df_viz, ticker=t, title_suffix="(Top rendement)", log=log_level)

    # 3.2 Comparaison multi-ETF (normalisé base 100)
    else:
        viz.plot_multiple_etfs(subset_for_compare, normalize=True, title_suffix="— Top 4 par rendement", log=log_level)


        if not light:
            # 3.3 Distribution des rendements
            viz.plot_returns_distribution(prices)
            # 3.4 Matrice de corrélation (sur prix ajustés)
            viz.plot_correlation_matrix(prices)

            # 3.5 Nuage de points risque/rendement
            viz.plot_risk_return_scatter(prices)

            # 3.6 Dashboard récapitulatif
            viz.plot_summary_dashboard(summary_df)

    print("\n✅ Pipeline terminé.")



def get_collector(
    tickers: Optional[List[str]] = None,
    period: str = "5y",
    interval: str = "1d",
    adjusted: bool = True,
    light: bool = False,
    max=True
) -> EuropeanETFCollector:
    """
    Fonction pour obtenir un collecteur avec les données chargées

    Args:
        tickers: Liste de tickers à analyser. Par défaut, tous les ETF connus.
        start: Date de début (YYYY-MM-DD)
        end: Date de fin (YYYY-MM-DD) ou None pour aujourd'hui
        interval: Intervalle Yahoo Finance (ex: "1d", "1wk", "1mo")
        adjusted: Utiliser les prix ajustés
        exchange_filter: Filtrer par place de cotation (ex: ".PA", ".L", ".DE")
        light: Si True, affiche des visuels essentiels seulement

    Returns:
        Instance de EuropeanETFCollector avec les données téléchargées
    """
    print("📊 Initialisation du collecteur de données")
    collector = EuropeanETFCollector()

    if tickers is None or len(tickers) == 0:
        tickers = collector.get_tickers()

    print(f"Tickers sélectionnés ({len(tickers)}): {', '.join(tickers)}")

    dfs: List[pd.DataFrame] = collector.get_etf_frames(
        period= "max" if max else "5y",
        interval=interval)

    if not dfs:
        raise ValueError("Aucune donnée n'a pu être téléchargée.")

    return collector, dfs



def run_lstm_prediction(collector : EuropeanETFCollector, ticker: str, hp: Dict, load_dir: str):
    from prediction_lstm_model import LSTMPredictorProba
    from plot_prediction import plot_overlay

    if not isinstance(load_dir, str):
        raise TypeError(
            f"'load_dir' doit être une chaîne de caractères, reçu {type(load_dir).__name__}."
        )
    if not isinstance(hp, dict):
        raise TypeError(
            f"'hp' doit être un dictionnaire d'hyperparamètres, reçu {type(hp).__name__}."
        )

    if not load_dir:
        print("Pour l'action 'predict', le répertoire de chargement doit être spécifié via LSTM_LOAD_DIR.")
        sys.exit(1)
    if not ticker:
        print("Pour l'action 'predict', le ticker doit être spécifié via LSTM_TICKER.")
        sys.exit(1)


    predictor = LSTMPredictorProba.load(load_dir)
    print(f"modele chargé depuis {load_dir}")

    if getattr(predictor, "use_residual_boosting", False) and getattr(predictor, "residual_models_", None):
        print("Modèles LightGBM résiduels chargés pour l'ajustement des quantiles.")    

    df_to_pred = collector.get_one_frame(ticker)
    if df_to_pred is None or df_to_pred.empty:
        print(f"Aucune donnée disponible pour le ticker {ticker}.")
        sys.exit(1)

    dfp = build_pipeline(df_to_pred)
    if len(dfp) < predictor.window_size:
        print(
            f"Données insuffisantes pour effectuer une prédiction (fenêtre requise: {predictor.window_size}, observations: {len(dfp)})."
        )
        sys.exit(1)

    predictions = predictor.predict(dfp, asset=ticker)
    predicted_horizon = len(predictions)
    expected_horizon = hp.get('horizon')

    if expected_horizon is not None and predicted_horizon != expected_horizon:
        print(
            f"Avertissement : l'horizon prédit ({predicted_horizon}) ne correspond pas à l'horizon attendu ({expected_horizon})."
        )

    start = dfp.index[-1] + pd.offsets.BDay(1)
    idx_future = pd.bdate_range(start=start, periods=hp['horizon'])

    df_pred_fan = pd.DataFrame({
        'adj_close_P025': predictions[:, 0],
        'adj_close_P50':  predictions[:, 1],
        'adj_close_P975': predictions[:, 2],
    }, index=idx_future)

    plot_overlay(df_to_pred, df_pred_fan, feature='adj_close', ticker=ticker)


def run_lstm_training(
    hp: Dict[str, float],
    save_dir: str,
    *,
    tickers: Optional[List[str]] = None,
    period: str = "max",
    interval: str = "1d",
    plot_training: bool = False,
    plot_dir: Optional[str] = None,):

    from prediction_lstm_model import LSTMPredictorProba

    if not save_dir:
        print("Pour l'action 'train', le répertoire de sauvegarde doit être spécifié via LSTM_SAVE_DIR.")
        sys.exit(1)

    window_size = hp['window_size']

    datasets, metadata = prepare_lstm_training_datasets(
        tickers=tickers,
        period=period,
        interval=interval,
        window_size=window_size,
    )
    #dataset : List[Tuple[str, pd.DataFrame]]

    kept = metadata.get('kept', [])
    dropped = metadata.get('dropped', {})

    if kept:
        print("Séries retenues pour l'entraînement :")
        for name, length in kept:
            print(f"  • {name}: {length} observations après préparation")
    if dropped:
        print("Séries ignorées :")
        for name, reason in dropped.items():
            print(f"  • {name}: {reason}")

    if not datasets:
        print("Aucune série suffisante pour l'entraînement après préparation des données.")
        sys.exit(1)


    print(f'modele training with {len(datasets)} series, window_size={window_size}')

    residual_boosting = env_var_to_bool(hp.get('residual_boosting'), False)
    boosting_params = dict(hp.get('boosting_params', {}))
    print('on rentre danns le training')
    predictor = LSTMPredictorProba(
        feature=LSTM_FEATURES,
        target_feature=TARGET_FEATURE,
        window_size=hp['window_size'],
        hidden_size=hp['hidden_size'],
        num_layers=hp['num_layers'],
        lr=hp['lr'],
        epochs=hp['epochs'],
        horizon=hp['horizon'],
        plot_training=plot_training,
        plot_dir=plot_dir,
        residual_boosting=residual_boosting,
        boosting_params=boosting_params or None,
        )
    print('init du modele ok')

    if residual_boosting:
        print(
            "LightGBM residual boosting activé avec paramètres:",
            {k: boosting_params.get(k) for k in sorted(boosting_params)},
        )
    print("Démarrage de l'entraînement...")
    predictor.fit(datasets)
    print('on a fini')
    os.makedirs(save_dir, exist_ok=True)
    predictor.save(save_dir)
    print("✅ Entraînement terminé et modèle sauvegardé.")
    collector = EuropeanETFCollector()
    run_lstm_prediction(collector, datasets[0][0], hp, save_dir)


def run_graphics():
    config = get_config_from_env()
    sel_tickers = config.get('tickers') or []
    action = (config.get('action') or '').lower()

    if action not in {'single', 'compare'}:
        # déduction simple à partir du nombre de tickers
        action = 'single' if len(sel_tickers) == 1 else ('compare' if len(sel_tickers) >= 2 else '')
    print(f"[DEBUG] inferred_action={action} raw_action={config.get('action')}")


    log_level = bool(config.get('log_level', False))


    if action == 'single' and len(sel_tickers or []) > 1:
        sel_tickers = sel_tickers[:1]

    print(f"[DEBUG] action={action} tickers={sel_tickers}")

    run_pipeline_for_graphics(
        tickers=sel_tickers,
        period=config['period'],
        interval=config['interval'],
        light=config['light'],
        max=config['max'],
        log_level=log_level,
        action =action
    )



def run_prediction():

    action, raw_hp, ticker, load_dir, save_dir = get_lstm_from_env()
    hp = normalize_lstm_hp(raw_hp)

    env_config = get_config_from_env()
    period = "max" if env_config.get('max', True) else env_config.get('period', '5y')
    interval = env_config.get('interval', '1d')

    try:
        collector, dfs = get_collector(
            period="max",
            interval="1d"
        )
    except ValueError as e:
        print(f"Erreur lors de la collecte des données : {e}")
        sys.exit(1)

    

    if action not in {'train', 'predict'}:
        action = 'predict' if ticker else 'train'

    if action == 'predict' and not ticker:
        print("Pour l'action 'predict', le ticker doit être spécifié via LSTM_TICKER.")
        sys.exit(1)
    if action == 'train' and not save_dir:
        print("Pour l'action 'train', le répertoire de sauvegarde doit être spécifié via LSTM_SAVE_DIR.")
        sys.exit(1)


    if action =='predict':
        try:
            run_lstm_prediction(collector, ticker, hp, load_dir)
        except Exception as exc:
            print(f"Erreur lors de la prédiction LSTM: {exc}")
            sys.exit(1)
    else:
        env_candidates = [
            os.getenv('LSTM_TRAIN_TICKERS'),
            os.getenv('LSTM_TICKERS'),
            os.getenv('ETF_TICKERS'),
        ]
        tickers = resolve_training_universe(env_candidates)
        run_lstm_training(
            hp=hp,
            save_dir=save_dir,
            tickers=tickers,
            period=period,
            interval=interval,
        )



def launch_ui():
    """Lance la fenêtre Tkinter et connecte les boutons aux sous-processus main.py --mode ..."""
    import tkinter as tk
    import ttkbootstrap as ttk
    from client_UI import ETFAnalysisGUI  # on va passer le chemin de main.py à l'UI
    from pathlib import Path
    import sys

    root = ttk.Window(themename="superhero")
    root.title("ETF Tools")

    # On transmet à l'UI l'emplacement de ce script et l'interpréteur Python courant.
    ETFAnalysisGUI(
        root,
        script_path=Path(__file__).resolve(),
        python_exe=sys.executable
    )
    root.mainloop()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "graphics", "prediction"], default="ui",
                        help="Lancer l'interface (ui) ou exécuter directement un pipeline.")
    args = parser.parse_args()

    if args.mode == "graphics":
        run_graphics()
    elif args.mode == "prediction":
        run_prediction()
    else:
        launch_ui()

        
if __name__ == "__main__":
    main()