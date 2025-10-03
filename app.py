"""Streamlit dashboard for the Forecast_TimeSeries project."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st

from data_preprocessing import (
    build_lstm_features,
    prepare_lstm_training_datasets,
    resolve_training_universe,
)
from etf_collector import EuropeanETFCollector
from etf_visualizer import ETFVisualizer
from main import DEFAULT_LSTM_HP, run_lstm_training
from prediction_lstm_model import LSTMPredictorProba
from services.genai_service import fetch_economic_answer


st.set_page_config(
    page_title="QuantIA – ETF & LSTM Analytics",
    page_icon="📈",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logo_QuantIA.png"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _format_ticker_list(raw: Iterable[str]) -> List[str]:
    tickers: List[str] = []
    for value in raw:
        if not value:
            continue
        tickers.extend(t.strip() for t in value.split(",") if t.strip())
    return tickers


@st.cache_data(show_spinner=False)
def load_etf_data(
    tickers: Sequence[str],
    *,
    period: str,
    interval: str,
    use_max: bool,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Download ETF history and expose both the summary and tidy frames."""

    collector = EuropeanETFCollector(tickers=list(tickers) or None)
    frames = collector.get_etf_frames(period="max" if use_max else period, interval=interval)
    summary = collector.build_summary()

    price_frames: Dict[str, pd.DataFrame] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        ticker = frame.get("ticker")
        if ticker is None:
            continue
        ticker_name = str(ticker.iloc[0]) if hasattr(ticker, "iloc") else str(ticker)
        tidy = frame.reset_index().rename(columns={"index": "date", "Date": "date"})
        if "date" not in tidy.columns:
            tidy["date"] = pd.to_datetime(frame.index)
        tidy["date"] = pd.to_datetime(tidy["date"])
        tidy = tidy.sort_values("date")
        price_frames[ticker_name] = tidy

    return summary, price_frames


def ensure_price_column(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "adj_close" not in work.columns:
        if "close" in work.columns:
            work["adj_close"] = work["close"]
        else:
            candidates = [c for c in work.columns if c not in {"date", "volume", "ticker"}]
            if len(candidates) == 1:
                work["adj_close"] = work[candidates[0]]
    return work


def yahoo_suggest(prefix: str, count: int = 20) -> List[Dict[str, str]]:
    """Return quote suggestions from Yahoo Finance's search API."""

    if not prefix:
        return []

    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": prefix, "quotesCount": int(count), "newsCount": 0}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=6)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for quote in data.get("quotes", []):
        symbol = quote.get("symbol")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        name = quote.get("shortname") or quote.get("longname") or ""
        exchange = quote.get("exchDisp") or quote.get("exchange") or ""
        out.append({"symbol": symbol, "name": name, "exchange": exchange})

    return out


def make_single_candlestick(df: pd.DataFrame, ticker: str, *, log_scale: bool) -> go.Figure:
    work = ensure_price_column(df)
    if "date" not in work.columns:
        work = work.reset_index().rename(columns={"index": "date"})
    open_values = work.get("open", work["adj_close"])
    close_values = work.get("close", work["adj_close"])
    high_values = work.get("high", work["adj_close"])
    low_values = work.get("low", work["adj_close"])

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=work["date"],
                open=open_values,
                high=high_values,
                low=low_values,
                close=close_values,
                name=ticker,
            )
        ]
    )
    fig.update_layout(
        title=f"{ticker} — Évolution du prix ajusté",
        xaxis_title="Date",
        yaxis_title="Prix",
        template="plotly_dark",
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def make_multi_performance(
    prices: Dict[str, pd.DataFrame],
    *,
    normalize: bool,
    log_scale: bool,
) -> plt.Figure:
    viz = ETFVisualizer()
    fig, ax = plt.subplots(figsize=(12, 7))

    for ticker, frame in prices.items():
        work = ensure_price_column(frame)
        work = work.sort_values("date")
        series = work["adj_close"].astype(float)
        if normalize and not series.empty:
            series = (series / series.iloc[0]) * 100
            ylabel = "Performance indexée (base 100)"
        else:
            ylabel = "Prix"
        label = f"{ticker}"
        ax.plot(work["date"], series, label=label, linewidth=2.0, color=viz._get_color(ticker))

    ax.set_title("Comparaison des ETF sélectionnés", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def make_returns_distribution(prices: Dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    tickers = list(prices.keys())[:4]

    for idx, ticker in enumerate(tickers):
        work = ensure_price_column(prices[ticker])
        returns = work["adj_close"].pct_change() * 100
        axes[idx].hist(returns.dropna(), bins=40, color="#4ECDC4", edgecolor="black", alpha=0.7)
        axes[idx].axvline(returns.mean(), color="red", linestyle="--", label="Moyenne")
        axes[idx].set_title(f"{ticker} — Rendements quotidiens")
        axes[idx].set_xlabel("Rendement (%)")
        axes[idx].set_ylabel("Fréquence")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    for idx in range(len(tickers), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Distribution des rendements quotidiens", fontweight="bold")
    fig.tight_layout()
    return fig


def make_correlation_matrix(prices: Dict[str, pd.DataFrame]) -> Optional[plt.Figure]:
    returns = {}
    for ticker, frame in prices.items():
        work = ensure_price_column(frame)
        series = work["adj_close"].pct_change()
        if series.dropna().empty:
            continue
        returns[ticker] = series

    if len(returns) < 2:
        return None

    returns_df = pd.DataFrame(returns).dropna()
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, cbar_kws={"label": "Corrélation"}, ax=ax)
    ax.set_title("Matrice de corrélation des rendements", fontweight="bold")
    fig.tight_layout()
    return fig


def make_risk_return(prices: Dict[str, pd.DataFrame]) -> Optional[plt.Figure]:
    points: List[Tuple[str, float, float]] = []
    for ticker, frame in prices.items():
        work = ensure_price_column(frame)
        if len(work) < 2:
            continue
        series = work["adj_close"].astype(float)
        if series.dropna().empty:
            continue
        total_return = ((series.iloc[-1] / series.iloc[0]) ** (365.25 / max(len(series), 1)) - 1) * 100
        volatility = series.pct_change().std() * (252 ** 0.5) * 100
        points.append((ticker, total_return, volatility))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker, ret, vol in points:
        ax.scatter(vol, ret, s=100, alpha=0.7, label=ticker)
        ax.annotate(ticker, (vol, ret), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Volatilité annualisée (%)")
    ax.set_ylabel("Rendement annualisé (%)")
    ax.set_title("Profil risque/rendement", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_summary_dashboard(summary: pd.DataFrame) -> plt.Figure:
    df = summary.copy()
    if "exchange" not in df.columns:
        df["exchange"] = df["ticker"].str.extract(r"(\.[A-Z]+)$").fillna("UNKNOWN")
    if "country" not in df.columns:
        mapping = {
            ".DE": "Germany",
            ".AS": "Netherlands",
            ".PA": "France",
            ".L": "United Kingdom",
            ".MI": "Italy",
            ".SW": "Switzerland",
            "UNKNOWN": "Unknown",
        }
        df["country"] = df["exchange"].map(mapping).fillna("Unknown")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    country_returns = df.groupby("country")["total_return_%"].mean().sort_values(ascending=False)
    ax1.bar(country_returns.index, country_returns.values, color="#5C7CFA")
    ax1.set_title("Rendement moyen par pays", fontweight="bold")
    ax1.set_ylabel("Rendement (%)")
    ax1.tick_params(axis="x", rotation=45)

    if {"annual_volatility_%", "total_return_%"}.issubset(df.columns):
        ax2.scatter(df["annual_volatility_%"], df["total_return_%"], c="#4ECDC4", s=60, alpha=0.7)
        ax2.set_xlabel("Volatilité annuelle (%)")
        ax2.set_ylabel("Rendement total (%)")
        ax2.set_title("Volatilité vs rendement", fontweight="bold")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis("off")

    if "avg_volume" in df.columns:
        exchange_volume = df.groupby("exchange")["avg_volume"].mean().sort_values(ascending=False)
        ax3.barh(exchange_volume.index, exchange_volume.values, color="#FF6B6B")
        ax3.set_title("Volume moyen par bourse", fontweight="bold")
        ax3.set_xlabel("Volume moyen")
    else:
        ax3.axis("off")

    if "total_return_%" in df.columns:
        ax4.hist(df["total_return_%"], bins=15, color="#96CEB4", edgecolor="black", alpha=0.8)
        ax4.axvline(df["total_return_%"].mean(), color="red", linestyle="--", label="Moyenne")
        ax4.set_title("Distribution des rendements", fontweight="bold")
        ax4.set_xlabel("Rendement (%)")
        ax4.set_ylabel("Fréquence")
        ax4.legend()
    else:
        ax4.axis("off")

    fig.suptitle("Tableau de bord des ETF", fontweight="bold")
    fig.tight_layout()
    return fig


def make_prediction_plot(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    ticker: str,
    history_window: int = 126,
) -> plt.Figure:
    work_hist = ensure_price_column(df_hist)
    work_hist = work_hist.sort_values("date").tail(history_window)
    if "date" not in work_hist.columns:
        work_hist = work_hist.reset_index().rename(columns={"index": "date"})
    work_hist["date"] = pd.to_datetime(work_hist["date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(work_hist["date"], work_hist["adj_close"], label="Historique", linewidth=2)

    ax.fill_between(
        df_pred.index,
        df_pred["adj_close_P025"],
        df_pred["adj_close_P975"],
        color="#5C7CFA",
        alpha=0.2,
        label="Intervalle 95%",
    )
    ax.plot(
        df_pred.index,
        df_pred["adj_close_P50"],
        linestyle="--",
        color="#1E2749",
        linewidth=2,
        label="Médiane",
    )

    ax.axvline(work_hist["date"].iloc[-1], color="gray", linestyle=":", alpha=0.6)
    ax.set_title(f"{ticker} — Prédiction LSTM", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def compute_future_index(last_index: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_index + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=horizon)


# ---------------------------------------------------------------------------
# Streamlit layout helpers
# ---------------------------------------------------------------------------

def show_header() -> None:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=120)
    with col_title:
        st.title("QuantIA — Analyse ETF & Prédictions LSTM")
        st.caption(
            "Interface web interactive propulsée par Streamlit pour explorer les ETF européens,"
            " entraîner un modèle LSTM probabiliste et interroger l'assistant économique."
        )


def render_summary_metrics(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("Aucun résumé disponible pour les paramètres sélectionnés.")
        return

    top = summary.sort_values("total_return_%", ascending=False).iloc[0]
    cols = st.columns(3)
    cols[0].metric("ETF le plus performant", top["ticker"], f"{top['total_return_%']:.1f}%")
    cols[1].metric("Dernier prix", f"{top['last_price']:.2f}")
    cols[2].metric("Volatilité annuelle", f"{top['annual_volatility_%']:.1f}%")

    st.dataframe(
        summary.set_index("ticker")[["start", "end", "last_price", "total_return_%", "annual_volatility_%"]]
    )
    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le résumé (CSV)", data=csv, file_name="etf_summary.csv", mime="text/csv")


def render_visualisations(summary: pd.DataFrame, prices: Dict[str, pd.DataFrame], log_scale: bool) -> None:
    if not prices:
        st.warning("Sélectionnez au moins un ETF pour afficher les graphiques.")
        return

    viz_tab, compare_tab, dist_tab, corr_tab, risk_tab, dash_tab = st.tabs(
        [
            "Graphique individuel",
            "Comparaison",
            "Distribution",
            "Corrélation",
            "Risque vs rendement",
            "Dashboard",
        ]
    )

    first_ticker = next(iter(prices))
    with viz_tab:
        target = st.selectbox("Choisir l'ETF à visualiser", list(prices.keys()), key="single_ticker")
        fig = make_single_candlestick(prices[target], target, log_scale=log_scale)
        st.plotly_chart(fig, use_container_width=True)

    with compare_tab:
        fig = make_multi_performance(prices, normalize=True, log_scale=log_scale)
        st.pyplot(fig)
        plt.close(fig)

    with dist_tab:
        fig = make_returns_distribution(prices)
        st.pyplot(fig)
        plt.close(fig)

    with corr_tab:
        fig = make_correlation_matrix(prices)
        if fig is None:
            st.info("Pas assez d'ETF pour calculer une corrélation.")
        else:
            st.pyplot(fig)
            plt.close(fig)

    with risk_tab:
        fig = make_risk_return(prices)
        if fig is None:
            st.info("Pas assez de données pour tracer le profil risque/rendement.")
        else:
            st.pyplot(fig)
            plt.close(fig)

    with dash_tab:
        fig = make_summary_dashboard(summary)
        st.pyplot(fig)
        plt.close(fig)


def render_lstm_prediction() -> None:
    st.subheader("Prédiction LSTM probabiliste")
    defaults = DEFAULT_LSTM_HP.copy()

    with st.form("prediction_form"):
        ticker = st.text_input("Ticker à prédire", value="VWCE.DE")
        load_dir = st.text_input("Répertoire du modèle sauvegardé", value="checkpoints")
        period = st.selectbox("Période de téléchargement", ["1y", "5y", "10y", "max"], index=1)
        interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
        horizon = st.number_input("Horizon de prédiction (jours ouvrés)", min_value=1, max_value=90, value=int(defaults["horizon"]))
        submit = st.form_submit_button("Lancer la prédiction")

    if not submit:
        return

    ticker = ticker.strip().upper()
    if not ticker:
        st.error("Veuillez renseigner un ticker valide.")
        return

    try:
        predictor = LSTMPredictorProba.load(load_dir)
    except Exception as exc:  # pragma: no cover - runtime safety
        st.error(f"Impossible de charger le modèle: {exc}")
        return

    try:
        collector = EuropeanETFCollector(tickers=[ticker])
        frames = collector.get_etf_frames(period=period, interval=interval)
    except Exception as exc:  # pragma: no cover - runtime safety
        st.error(f"Erreur lors du téléchargement des données: {exc}")
        return

    if not frames:
        st.error("Aucune donnée téléchargée pour ce ticker.")
        return

    raw_frame = frames[0]
    if raw_frame is None or raw_frame.empty:
        st.error("Données historiques vides pour ce ticker.")
        return

    features = build_lstm_features(raw_frame)
    if features.empty:
        st.error("Impossible de construire les indicateurs requis pour le modèle.")
        return

    if len(features) < predictor.window_size:
        st.error(
            f"Données insuffisantes ({len(features)}) pour la fenêtre du modèle ({predictor.window_size})."
        )
        return

    try:
        preds = predictor.predict(features)
    except Exception as exc:  # pragma: no cover - runtime safety
        st.error(f"Erreur lors de la prédiction: {exc}")
        return

    future_index = compute_future_index(features.index[-1], horizon=horizon)
    preds_df = pd.DataFrame(
        preds[: len(future_index)],
        columns=["adj_close_P025", "adj_close_P50", "adj_close_P975"],
        index=future_index,
    )

    work_hist = raw_frame.reset_index().rename(columns={"index": "date"})
    work_hist["date"] = pd.to_datetime(work_hist["date"])

    fig = make_prediction_plot(work_hist, preds_df, ticker=ticker)
    st.pyplot(fig)
    plt.close(fig)

    st.write("Prévisions (premières lignes):")
    st.dataframe(preds_df.head())


def render_lstm_training() -> None:
    st.subheader("Entraînement du modèle LSTM")
    defaults = DEFAULT_LSTM_HP.copy()
    default_tickers = ", ".join(resolve_training_universe([])[:5])

    with st.form("training_form"):
        tickers_text = st.text_input(
            "Tickers pour l'entraînement (séparés par des virgules)",
            value=default_tickers,
        )
        save_dir = st.text_input("Répertoire de sauvegarde", value="checkpoints/experiment")
        period = st.selectbox("Période", ["1y", "5y", "10y", "max"], index=1)
        interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
        window_size = st.number_input("Fenêtre temporelle", min_value=10, max_value=400, value=int(defaults["window_size"]))
        hidden_size = st.number_input("Taille cachée", min_value=16, max_value=512, value=int(defaults["hidden_size"]))
        num_layers = st.number_input("Nombre de couches", min_value=1, max_value=6, value=int(defaults["num_layers"]))
        lr = st.number_input("Taux d'apprentissage", min_value=1e-5, max_value=1e-1, value=float(defaults["lr"]), format="%e")
        epochs = st.number_input("Nombre d'époques", min_value=10, max_value=2000, value=int(defaults["epochs"]))
        horizon = st.number_input("Horizon de sortie", min_value=1, max_value=90, value=int(defaults["horizon"]))
        submit = st.form_submit_button("Lancer l'entraînement")

    if not submit:
        return

    tickers = _format_ticker_list([tickers_text])
    if not tickers:
        st.error("Veuillez renseigner au moins un ticker d'entraînement.")
        return

    hp = {
        "window_size": int(window_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "lr": float(lr),
        "epochs": int(epochs),
        "horizon": int(horizon),
    }

    with st.spinner("Préparation des jeux de données..."):
        datasets, metadata = prepare_lstm_training_datasets(
            tickers=tickers,
            period=period,
            interval=interval,
            window_size=hp["window_size"],
        )

    kept = metadata.get("kept", [])
    dropped = metadata.get("dropped", {})

    if dropped:
        with st.expander("Séries ignorées"):
            for name, reason in dropped.items():
                st.write(f"- **{name}**: {reason}")

    st.success(f"{len(datasets)} séries prêtes pour l'entraînement.")

    if kept:
        kept_df = pd.DataFrame(kept, columns=["ticker", "observations"])
        st.write("Séries retenues pour l'entraînement :")
        st.dataframe(kept_df.set_index("ticker"))

    confirm = st.checkbox(
        "Confirmer le lancement de l'entraînement (peut prendre plusieurs minutes)",
        value=False,
    )
    if not confirm:
        return

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            run_lstm_training(
                hp=hp,
                save_dir=save_dir,
                tickers=tickers,
                period=period,
                interval=interval,
            )
    except Exception as exc:  # pragma: no cover - runtime safety
        st.error(f"Erreur lors de l'entraînement: {exc}")
    else:
        st.success("Entraînement terminé et modèle sauvegardé.")
    finally:
        logs = buffer.getvalue()
        if logs:
            st.text_area("Journal d'exécution", logs, height=200)


def render_economic_assistant() -> None:
    st.subheader("Assistant économique (Mistral AI)")
    with st.form("econ_form"):
        question = st.text_area("Posez votre question économique", height=150)
        submit = st.form_submit_button("Obtenir une réponse")

    if not submit:
        return

    question = (question or "").strip()
    if not question:
        st.error("Veuillez saisir une question avant de soumettre.")
        return

    with st.spinner("Interrogation du modèle Mistral AI..."):
        try:
            answer = fetch_economic_answer(question)
        except Exception as exc:  # pragma: no cover - runtime safety
            st.error(f"Erreur lors de l'appel au service: {exc}")
            return

    st.markdown(answer)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    def _init_state():
        defaults = {
            "extra_tickers": [],
            "ticker_suggestions": [],
            "manual_ticker_input": "",
            "suggestion_selection": [],
            "ticker_search_query": "",
            "ticker_search_count": 20,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    _init_state()
    show_header()

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choisissez une section",
        (
            "Analyse ETF",
            "Prédictions LSTM",
            "Assistant économique",
        ),
    )

    if section == "Analyse ETF":
        st.header("Analyse des ETF européens")

        base_tickers = EuropeanETFCollector().get_tickers()

        # -----------------------
        # 1) Ajouts AVANT le multiselect
        # -----------------------
        with st.expander("Rechercher des symboles sur Yahoo Finance"):
            with st.form("yahoo_search_form"):
                query = st.text_input("Recherche de symbole ou de nom", key="ticker_search_query")
                max_results = st.slider("Nombre maximum de résultats", 5, 50, 20, key="ticker_search_count")
                submit_search = st.form_submit_button("Lancer la recherche")

            if submit_search:
                suggestions = yahoo_suggest(query.strip(), count=max_results) or []
                st.session_state["ticker_suggestions"] = suggestions
                if not suggestions:
                    st.info("Aucun symbole correspondant trouvé.")

            suggestions = st.session_state.get("ticker_suggestions", [])
            if suggestions:
                suggestion_df = pd.DataFrame(suggestions)
                st.dataframe(suggestion_df, use_container_width=True)

                options_map = {}
                st.session_state["_options_map"] = options_map
                st.session_state["_base_tickers"] = base_tickers

                for item in suggestions:
                    symbol = item.get("symbol", "")
                    name = item.get("name", "")
                    exchange = item.get("exchange", "")
                    label_parts = [symbol]
                    if name:     label_parts.append(f"— {name}")
                    if exchange: label_parts.append(f"({exchange})")
                    options_map[" ".join(label_parts)] = symbol

                chosen_labels = st.multiselect(
                    "Sélectionnez les symboles à ajouter",
                    list(options_map.keys()),
                    key="suggestion_selection",
                )
                def _on_add_suggestions():
                    options_map = st.session_state.get("_options_map", {})
                    base_tickers = st.session_state.get("_base_tickers", [])
                    extras = st.session_state.setdefault("extra_tickers", [])
                    sel = st.session_state.get("selected_tickers") or []
                    chosen = st.session_state.get("suggestion_selection", [])

                    added = []
                    for label in chosen:
                        sym = options_map[label]
                        if sym not in extras and sym not in base_tickers:
                            extras.append(sym)
                        if sym not in sel:
                            sel.append(sym); added.append(sym)

                    st.session_state["selected_tickers"] = sel
                    st.session_state["suggestion_selection"] = []   # <- OK dans un callback
                    # Message à afficher après rerun
                    st.session_state["_add_feedback"] = (
                        "success",
                        "Ajouté : " + ", ".join(sorted(added)) if added else "Les symboles sélectionnés étaient déjà présents."
                    )
                    st.rerun()

                st.button("Ajouter à la sélection", key="add_suggestion_button", on_click=_on_add_suggestions)

                _fb = st.session_state.pop("_add_feedback", None)
                if _fb:
                    kind, msg = _fb
                    (st.success if kind == "success" else st.info)(msg)


        st.markdown("---")

        def _on_add_manual():
            txt = st.session_state.get("manual_ticker_input", "")
            manual_syms = _format_ticker_list([txt])
            if not manual_syms:
                st.session_state["_manual_feedback"] = ("warning", "Veuillez saisir au moins un symbole valide.")
            else:
                extras = st.session_state.setdefault("extra_tickers", [])
                sel = st.session_state.get("selected_tickers") or []
                added = []
                for sym in manual_syms:
                    if sym not in extras and sym not in base_tickers:
                        extras.append(sym)
                    if sym not in sel:
                        sel.append(sym); added.append(sym)
                st.session_state["selected_tickers"] = sel
                st.session_state["_manual_feedback"] = (
                    "success" if added else "info",
                    "Ajouté : " + ", ".join(sorted(added)) if added else "Les symboles saisis étaient déjà présents."
                )
            st.session_state["manual_ticker_input"] = ""  # OK dans le callback
            st.rerun()

        # widgets
        manual_col, button_col = st.columns([3, 1])
        with manual_col:
            st.text_input("Ajouter manuellement des symboles (séparés par des virgules)", key="manual_ticker_input")
        with button_col:
            st.button("Ajouter", key="manual_add_button", on_click=_on_add_manual)

        _fb2 = st.session_state.pop("_manual_feedback", None)
        if _fb2:
            kind, msg = _fb2
            getattr(st, kind)(msg)

            
        # -----------------------
        # 2) Multiselect UNIQUE (après ajouts)
        # -----------------------
        extras = st.session_state.get("extra_tickers", [])
        available_tickers = sorted(set(base_tickers + extras))

        default_selection = st.session_state.get("selected_tickers")
        if not default_selection:
            default_selection = [t for t in base_tickers[:2] if t in available_tickers]
        else:
            default_selection = [t for t in default_selection if t in available_tickers]

        selected = st.multiselect(
            "Sélectionnez les ETF ou actions",
            options=available_tickers,
            default=default_selection,
            key="selected_tickers",
        )

        # -----------------------
        # 3) Paramètres & rendu
        # -----------------------
        period = st.selectbox("Période", ["6mo", "1y", "5y", "10y", "max"], index=2)
        interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
        use_max = st.checkbox("Ignorer la période et télécharger l'historique complet", value=True)
        log_scale = st.checkbox("Afficher l'échelle logarithmique", value=False)

        if not selected:
            st.info("Sélectionnez au moins un ETF pour lancer la collecte.")
            st.stop()

        with st.spinner("Téléchargement des données..."):
            summary, prices = load_etf_data(tuple(selected), period=period, interval=interval, use_max=use_max)

        render_summary_metrics(summary)
        render_visualisations(summary, prices, log_scale)

    elif section == "Prédictions LSTM":
        render_lstm_prediction()
        st.markdown("---")
        render_lstm_training()
    else:
        render_economic_assistant()


if __name__ == "__main__":
    main()