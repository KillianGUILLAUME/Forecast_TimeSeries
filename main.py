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

from typing import List, Dict
import sys
import os, json
from typing import List, Dict, Optional



from prediction_lstm_model import LSTMPredictor, LSTMPredictorProba



from plot_prediction import plot_overlay



def get_config_from_env():
    """Récupère la configuration depuis les variables d'environnement (interface GUI)"""
    tickers_str = os.getenv('ETF_TICKERS', '').strip()            # NEW
    tickers = [t.strip() for t in tickers_str.split(',') if t.strip()] if tickers_str else []
    return {
        'period': os.getenv('ETF_PERIOD', '5y'),
        'interval': os.getenv('ETF_INTERVAL', '1d'),
        'light': os.getenv('ETF_LIGHT', 'False').lower() == 'true',
        'max': os.getenv('ETF_MAX', 'True').lower() == 'true',
        'action': os.getenv('ETF_ACTION', ''),
        'log_level': os.getenv('ETF_LOG_PLOTS', 'False').lower() == 'true',
        'tickers': tickers
    }

DEFAULT_LSTM_HP = {
    "window_size": 100,
    "hidden_size": 64,
    "num_layers": 2,
    "lr": 1e-3,
    "epochs": 200,
    "horizon": 10,
}



def get_lstm_from_env():
    action   = (os.getenv("LSTM_ACTION", "") or "").lower()
    try:
        hp = json.loads(os.getenv("LSTM_HP", "{}"))  # dict
    except json.JSONDecodeError:
        hp = {}
    ticker   = (os.getenv("LSTM_TICKER", "") or "").strip()
    load_dir = (os.getenv("LSTM_LOAD_DIR", "") or "").strip()
    save_dir = (os.getenv("LSTM_SAVE_DIR", "") or "").strip()
    return action, hp, ticker, load_dir, save_dir


def normalize_hyperparameters(hp: Dict) -> Dict:
    """Complète les hyperparamètres manquants avec les valeurs par défaut"""
    normalized = DEFAULT_LSTM_HP.copy()
    if isinstance(hp, dict):
        normalized.update({k: hp.get(k,v) for k, v in DEFAULT_LSTM_HP.items() if v is not None})

    def _as_int(name):
        try:
            hp[name] = int(hp[name])
        except (ValueError, TypeError):
            hp[name] = int(DEFAULT_LSTM_HP[name])

    def _as_float(name):
        try:
            hp[name] = float(hp[name])
        except (ValueError, TypeError):
            hp[name] = float(DEFAULT_LSTM_HP[name])

    for key in ("window_size", "hidden_size", "num_layers", "epochs", "horizon"):
        _as_int(key)
    _as_float("lr")
    return hp



def print_progress(message: str, step: int = None, total_steps: int = None):
    """Affiche les messages de progression pour l'interface"""
    if step and total_steps:
        progress = f"[{step}/{total_steps}] "
    else:
        progress = ""
    print(f"{progress}{message}")
    sys.stdout.flush()  


def build_pipeline(df0,collector):
    df_base = df0[['adj_close', 'volume']].copy()
    df_base['ret'] = np.log(df_base['adj_close']).diff()
    ind0 = collector.get_indicator([df_base])[0]
    if 'ticker' in ind0.columns: ind0 = ind0.drop(columns=['ticker'])
    ind0 = ind0[['SMA_5', 'SMA_50', 'RSI_14']]
    dfp = df_base[['adj_close','volume','ret']].join(ind0, how='inner')
    dfp['volume'] = np.log1p(dfp['volume'].clip(lower=0))
    return dfp.dropna()

def build_features_for_ticker(collector: EuropeanETFCollector, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_to_pred = collector.get_one_frame(ticker)
    if df_to_pred is None or df_to_pred.empty:
        raise ValueError(f"Aucune donnée disponible pour le ticker {ticker}.")
    dfp = build_pipeline(df_to_pred, collector=collector)
    if dfp.empty:
        raise ValueError(f"Données insuffisantes après préparation pour {ticker}.")
    return df_to_pred, dfp



def run_pipeline(
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
    collector = EuropeanETFCollector(tickers=tickers)

    if not tickers:
        tickers = collector.get_tickers()
    else:
        collector.tickers = tickers


    print(f"Tickers sélectionnés ({len(tickers)}): {', '.join(tickers)}")

    dfs: List[pd.DataFrame] = collector.get_etf_frames(
        period= "max" if max else period,
        interval=interval)

    if not dfs:
        print("Aucune donnée n'a pu être téléchargée. Arrêt.")
        sys.exit(1)


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



def pipeline_predict(
    df: pd.DataFrame,
    feature: List[str],
    target_feature: str,
    window_size: int=10,
    hidden_size: int=50,
    num_layers: int=1,
    lr: float=0.001,
    epochs: int=100,
    horizon: int=5
) -> np.ndarray:
    """
    Pipeline de bout en bout pour la prédiction avec LSTM

    Args:
        df: DataFrame avec les données historiques
        feature: Liste des colonnes à utiliser comme caractéristiques
        window_size: Taille de la fenêtre temporelle
        hidden_size: Nombre de neurones cachés dans LSTM
        num_layers: Nombre de couches LSTM
        lr: Taux d'apprentissage
        epochs: Nombre d'époques d'entraînement
        n_steps: Nombre de pas de temps à prédire

    Returns:
        np.ndarray avec les prédictions pour les n_steps futurs
    """
    print("🤖 Pipeline de prédiction LSTM")
    predictor = LSTMPredictor(
        feature=feature,
        target_feature = target_feature,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon = horizon,
        lr=lr,
        epochs=epochs
    )

    print("🔧 Entraînement du modèle...")
    predictor.fit(df)

    print(f"🔮 Prédiction pour les {horizon} prochains pas...")
    predictions = predictor.predict(df)

    print("✅ Prédiction terminée.")
    return predictions


def pipeline_predict_proba_training(
    df: pd.DataFrame,
    feature: List[str],
    target_feature: str,
    window_size: int = 10,
    hidden_size: int = 50,
    num_layers: int = 1,
    lr: float = 0.001,
    epochs: int = 100,
    horizon: int = 5,
    evaluate: bool = True,
    alpha: float = 0.05,
    save_dir: str | None = "checkpoints/lstm_proba_latest"
):
    """
    Pipeline de bout en bout pour la prédiction probabiliste avec LSTM (quantiles).

    Args:
        df: DataFrame avec les données historiques (doit contenir 'adj_close')
        feature: Colonnes utilisées comme features (incluant celles nécessaires au modèle)
        target_feature: Colonne cible (p.ex. 'ret' = log-returns)
        window_size: Taille de fenêtre (T)
        hidden_size: Taille cachée LSTM
        num_layers: Nombre de couches LSTM
        lr: Learning rate
        epochs: Nombre d'époques d'entraînement
        horizon: Nombre de pas futurs H
        evaluate: Si True, calcule les métriques de coverage sur le set de test
        alpha: Niveau pour l’intervalle (0.05 => 95%)

    Returns:
        - Si evaluate=False: np.ndarray (H, 3) avec colonnes [P_low, P_med, P_high]
        - Si evaluate=True:  (preds, metrics) où
              preds  = np.ndarray (H, 3)
              metrics = dict de vecteurs (H,) {coverage, lower_tail, upper_tail, mpiwidth, interval_score}
    """


    print("🤖 Pipeline LSTM PROBA (quantiles)")
    predictor = LSTMPredictorProba(
        feature=feature,
        target_feature=target_feature,
        window_size=window_size,
        hidden_size=hidden_size,
        horizon=horizon,
        num_layers=num_layers,
        lr=lr,
        epochs=epochs,
    )

    print("🔧 Entraînement du modèle proba...")
    predictor.fit(df)

    print(f'exemple de prédiction pour le ticker : {df["ticker"].iloc[0]}' if 'ticker' in df.columns else '')
    print(f"🔮 Prédiction des quantiles pour les {horizon} prochains pas...")
    preds = predictor.predict(df[0]) 
    print("✅ Prédiction terminée.")

    if save_dir:
        predictor.save(save_dir)

    if evaluate and hasattr(predictor, "X_test_scaled_torch") and len(predictor.X_test_scaled_torch) > 0:
        print("📏 Évaluation du coverage sur l'échantillon test...")
        metrics = predictor.evaluate_coverage_on_test(alpha=alpha)
        return preds, metrics

    return preds


def run_lstm_prediction(collector : EuropeanETFCollector, ticker: str, hp: Dict, load_dir: str):

    predictor = LSTMPredictorProba.load(load_dir)
    print(f"modele chargé depuis {load_dir}")

    """Exécute la prédiction LSTM pour un ticker donné avec les hyperparamètres spécifiés"""
    df_to_pred, dfp = build_features_for_ticker(collector, ticker)

    predictions = predictor.predict(dfp)
    horizon = predictions.shape
    expected_horizon = hp.get('horizon')

    if expected_horizon and horizon != expected_horizon:
        print(f"[WARNING] Le modèle prédit {horizon} pas, différent du paramètre attendu {expected_horizon}.")

    start = dfp.index[-1] + pd.offsets.BDay(1)
    idx_future = pd.bdate_range(start=start, periods=hp['horizon'])

    df_pred_fan = pd.DataFrame({
        'adj_close_P025': predictions[:, 0],
        'adj_close_P50':  predictions[:, 1],
        'adj_close_P975': predictions[:, 2],
    }, index=idx_future)

    plot_overlay(df_to_pred, df_pred_fan, feature='adj_close', ticker=ticker)


def run_lstm_training(collector: EuropeanETFCollector, dfs: List[pd.DataFrame], hp : Dict[str, float], save_dir: str):

    dfs_pipeline = [build_pipeline(df0, collector=collector) for df0 in dfs]
    feature = ['adj_close','volume', 'ret', 'SMA_5', 'SMA_50', 'RSI_14']
    target_feature = 'ret'

    pipeline_predict_proba_training(
        df=dfs_pipeline,
        feature=feature,
        target_feature=target_feature,
        window_size=hp['window_size'],
        hidden_size=hp['hidden_size'],
        num_layers=hp['num_layers'],
        lr=hp['lr'],
        epochs=hp['epochs'],
        horizon=hp['horizon'],
        evaluate=False,
        save_dir=save_dir,
    )

"""     Point d'entrée principal    """
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
        sel_tickers = [sel_tickers[:1]]

    print(f"[DEBUG] action={action} tickers={sel_tickers}")

    run_pipeline(
        tickers=sel_tickers,
        period=config['period'],
        interval=config['interval'],
        light=config['light'],
        max=config['max'],
        log_level=log_level,
        action =action
    )



def run_prediction():

    action, hp_raw, ticker, load_dir, save_dir = get_lstm_from_env()
    hp = normalize_hyperparameters(hp_raw)
    collector_df = get_collector(
        period="max",
        interval="1d"
        )
    
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
            run_lstm_prediction(collector, ticker, load_dir, hp)
        except Exception as exc:
            print(f"Erreur lors de la prédiction LSTM: {exc}")
            sys.exit(1)
    else:
        try:
            run_lstm_training(collector, dfs, hp, save_dir)
        except Exception as exc:
            print(f"Erreur lors de l'entraînement LSTM: {exc}")
            sys.exit(1)



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