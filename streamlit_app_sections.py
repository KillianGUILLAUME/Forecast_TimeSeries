"""Reusable building blocks for the Streamlit multi-page application."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from data_preprocessing import (
    download_price_history,
    build_lstm_features,
    prepare_lstm_training_datasets,
    resolve_training_universe,
)
from etf_collector import EuropeanETFCollector
from etf_visualizer import ETFVisualizer
from main import DEFAULT_LSTM_HP, run_lstm_training
from prediction_lstm_model import LSTMPredictorProba
from services.genai_service import fetch_economic_answer


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logo_QuantIA.png"


StateFactory = Callable[[], object]
StateDefault = Union[object, StateFactory]


STATE_DEFAULTS: Dict[str, StateDefault] = {
    "extra_tickers": list,
    "ticker_suggestions": list,
    "manual_ticker_input": str,
    "suggestion_selection": list,
    "ticker_search_query": str,
    "ticker_search_count": lambda: 20,
    "selected_tickers": list,
    "prediction_search_query": lambda: "VWCE",
    "prediction_last_symbol": lambda: "VWCE.DE",
    "prediction_selected_label": str,
}


def _resolve_default(default: StateDefault) -> object:
    return default() if callable(default) else default


def init_session_state() -> None:
    """Ensure all expected keys exist in ``st.session_state``."""

    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = _resolve_default(default)


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

    has_volume = "volume" in work.columns and not work["volume"].isna().all()

    fig = make_subplots(
        rows=2 if has_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3] if has_volume else [1.0],
    )
    fig.add_trace(
        go.Candlestick(
            x=work["date"],
            open=open_values,
            high=high_values,
            low=low_values,
            close=close_values,
            name=ticker,
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(title_text="Prix", row=1, col=1)

    if has_volume:
        fig.add_trace(
            go.Bar(
                x=work["date"],
                y=work["volume"],
                marker_color="#5C7CFA",
                name="Volume",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(
        title=f"{ticker} — Évolution du prix ajusté",
        xaxis_title="Date",
        template="plotly_dark",
        showlegend=True,
    )
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
    return fig


def make_multi_performance(
    prices: Dict[str, pd.DataFrame],
    *,
    normalize: bool,
    log_scale: bool,
) -> go.Figure:
    viz = ETFVisualizer()
    fig = go.Figure()

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
        fig.add_trace(
            go.Scatter(
                x=work["date"],
                y=series,
                mode="lines",
                name=label,
                line=dict(color=viz._get_color(ticker), width=2.0),
            )
        )

    fig.update_layout(
        title="Comparaison des ETF sélectionnés",
        xaxis_title="Date",
        yaxis_title=ylabel,
        template="plotly_dark",
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def make_returns_distribution(prices: Dict[str, pd.DataFrame]) -> go.Figure:
    tickers = list(prices.keys())[:4]
    titles = [f"{ticker} — Rendements quotidiens" for ticker in tickers]
    while len(titles) < 4:
        titles.append("")

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.1, vertical_spacing=0.15)

    for idx, ticker in enumerate(tickers):
        work = ensure_price_column(prices[ticker])
        returns = work["adj_close"].pct_change() * 100
        row = idx // 2 + 1
        col = idx % 2 + 1
        cleaned = returns.dropna()
        fig.add_trace(
            go.Histogram(
                x=cleaned,
                nbinsx=40,
                marker=dict(color="#4ECDC4"),
                name=ticker,
                opacity=0.75,
            ),
            row=row,
            col=col,
        )
        mean_value = cleaned.mean()
        if pd.notna(mean_value):
            fig.add_vline(
                x=mean_value,
                line=dict(color="red", dash="dash"),
                annotation_text="Moyenne",
                annotation_position="top right",
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Rendement (%)", row=row, col=col)
        fig.update_yaxes(title_text="Fréquence", row=row, col=col)

    fig.update_layout(
        title="Distribution des rendements quotidiens",
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def make_correlation_matrix(prices: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
    returns: Dict[str, pd.Series] = {}
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
    heatmap = go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Corrélation"),
        text=corr.round(2).values,
        texttemplate="%{text}",
    )
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="Matrice de corrélation des rendements",
        template="plotly_dark",
    )
    return fig


def make_risk_return(prices: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
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

    fig = go.Figure()

    for ticker, ret, vol in points:
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers+text",
                text=[ticker],
                textposition="top center",
                marker=dict(size=12, opacity=0.8),
                name=ticker,
            )
        )

    fig.update_layout(
        title="Profil risque/rendement",
        xaxis_title="Volatilité annualisée (%)",
        yaxis_title="Rendement annualisé (%)",
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def make_summary_dashboard(summary: pd.DataFrame) -> go.Figure:
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

    country_returns = df.groupby("country")["total_return_%"].mean().sort_values(ascending=False)

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Rendement moyen par pays",
        "Volatilité vs rendement",
        "Volume moyen par bourse",
        "Distribution des rendements",
    ))

    fig.add_trace(
        go.Bar(x=country_returns.index, y=country_returns.values, marker_color="#5C7CFA"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Rendement (%)", row=1, col=1)

    if {"annual_volatility_%", "total_return_%"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df["annual_volatility_%"],
                y=df["total_return_%"],
                mode="markers",
                marker=dict(color="#4ECDC4", size=8, opacity=0.7),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Volatilité annuelle (%)", row=1, col=2)
        fig.update_yaxes(title_text="Rendement total (%)", row=1, col=2)

    if "avg_volume" in df.columns:
        exchange_volume = df.groupby("exchange")["avg_volume"].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=exchange_volume.values,
                y=exchange_volume.index,
                orientation="h",
                marker_color="#FF6B6B",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Volume moyen", row=2, col=1)

    if "total_return_%" in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df["total_return_%"],
                nbinsx=15,
                marker=dict(color="#96CEB4"),
                opacity=0.85,
            ),
            row=2,
            col=2,
        )
        fig.add_vline(
            x=df["total_return_%"].mean(),
            line=dict(color="red", dash="dash"),
            annotation_text="Moyenne",
            annotation_position="top right",
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Rendement (%)", row=2, col=2)
        fig.update_yaxes(title_text="Fréquence", row=2, col=2)

    fig.update_layout(
        title="Tableau de bord des ETF",
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def make_prediction_plot(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    ticker: str,
    history_window: int = 126,
) -> go.Figure:
    work_hist = ensure_price_column(df_hist)
    work_hist = work_hist.sort_values("date").tail(history_window)
    if "date" not in work_hist.columns:
        work_hist = work_hist.reset_index().rename(columns={"index": "date"})
    work_hist["date"] = pd.to_datetime(work_hist["date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work_hist["date"],
            y=work_hist["adj_close"],
            mode="lines",
            name="Historique",
            line=dict(width=2, color="#1E88E5"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_pred.index,
            y=df_pred["adj_close_P025"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="P025",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_pred.index,
            y=df_pred["adj_close_P975"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(92, 124, 250, 0.2)",
            line=dict(width=0),
            name="Intervalle 95%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_pred.index,
            y=df_pred["adj_close_P50"],
            mode="lines",
            line=dict(color="#1E2749", dash="dash", width=2),
            name="Médiane",
        )
    )

    last_hist_date = work_hist["date"].iloc[-1]
    fig.add_vline(x=last_hist_date, line=dict(color="gray", dash="dot"))

    fig.update_layout(
        title=f"{ticker} — LSTM Probability Prediction ",
        xaxis_title="Date",
        yaxis_title="Prix",
        template="plotly_dark",
    )
    return fig


def compute_future_index(last_index: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = last_index + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=horizon)


def show_header() -> None:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=120)
    with col_title:
        st.title("QuantIA — Analyse ETF & Prédictions LSTM")
        st.caption(
            "Interface web interactive propulsée par Streamlit pour explorer les ETF européens,"
            " entraîner un modèle LSTM probabiliste et interroger l'IA pour des questions économiques ou d'analyse financières."
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

    with viz_tab:
        target = st.selectbox("Choisir l'ETF à visualiser", list(prices.keys()), key="single_ticker")
        fig = make_single_candlestick(prices[target], target, log_scale=log_scale)
        st.plotly_chart(fig, use_container_width=True)

    with compare_tab:
        fig = make_multi_performance(prices, normalize=True, log_scale=log_scale)
        st.plotly_chart(fig, use_container_width=True)

    with dist_tab:
        fig = make_returns_distribution(prices)
        st.plotly_chart(fig, use_container_width=True)

    with corr_tab:
        fig = make_correlation_matrix(prices)
        if fig is None:
            st.info("Pas assez d'ETF pour calculer une corrélation.")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with risk_tab:
        fig = make_risk_return(prices)
        if fig is None:
            st.info("Pas assez de données pour tracer le profil risque/rendement.")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with dash_tab:
        fig = make_summary_dashboard(summary)
        st.plotly_chart(fig, use_container_width=True)


def render_lstm_prediction() -> None:
    st.subheader("Prédiction LSTM probabiliste")
    defaults = DEFAULT_LSTM_HP.copy()

    st.markdown("### Recherche Yahoo Finance")
    query = st.text_input(
        "Rechercher un symbole ou un nom",
        value=st.session_state.get("prediction_search_query", "VWCE"),
        key="prediction_search_query",
        help="Saisissez au moins deux caractères pour obtenir des suggestions Yahoo Finance.",
    )
    query = (query or "").strip()

    selected_symbol = st.session_state.get("prediction_last_symbol", "VWCE.DE")
    suggestions: List[Dict[str, str]] = []
    if len(query) >= 2:
        suggestions = yahoo_suggest(query, count=25)

    if suggestions:
        options_map: Dict[str, str] = {}
        option_labels: List[str] = []
        for item in suggestions:
            symbol = (item.get("symbol") or "").upper()
            if not symbol:
                continue
            name = item.get("name") or ""
            exchange = item.get("exchange") or ""
            label_parts = [symbol]
            if name:
                label_parts.append(f"— {name}")
            if exchange:
                label_parts.append(f"({exchange})")
            label = " ".join(label_parts)
            options_map[label] = symbol
            option_labels.append(label)

    if option_labels:
        # 1) Déterminer **avant** le widget quel label doit être présélectionné
        ss_default = st.session_state.get("prediction_selected_label")
        if ss_default in options_map:
            desired_label = ss_default
        else:
            # Essaie d’aligner sur selected_symbol; sinon premier label
            by_symbol = next((lbl for lbl, sym in options_map.items() if sym == selected_symbol), None)
            desired_label = by_symbol or option_labels[0]

        index = option_labels.index(desired_label)

        # 2) Créer le widget avec l'index voulu
        selection_label = st.selectbox(
            "Résultats Yahoo Finance",
            option_labels,
            index=index,
            key="prediction_selected_label",
        )

        # 3) Utiliser la sélection SANS réécrire la clé du widget
        resolved_symbol = options_map.get(selection_label)
        if not resolved_symbol:
            # fallback robuste
            by_symbol = next((sym for lbl, sym in options_map.items() if sym == selected_symbol), None)
            resolved_symbol = by_symbol or options_map[option_labels[0]]

        selected_symbol = resolved_symbol
    elif query:
        st.caption("Aucun résultat trouvé. Le symbole saisi sera utilisé tel quel.")
        selected_symbol = query.upper()
    else:
        st.caption("Saisissez au moins deux caractères pour lancer la recherche Yahoo Finance.")

    st.session_state["prediction_last_symbol"] = selected_symbol
    st.write(f"Actif sélectionné : **{selected_symbol}**")

    with st.form("prediction_form"):
        st.text_input("Ticker sélectionné", value=selected_symbol, disabled=True)
        load_dir = st.text_input("Répertoire du modèle sauvegardé", value="checkpoints")
        period = st.selectbox("Période de téléchargement", ["1y", "5y", "10y", "max"], index=1)
        interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
        horizon = st.number_input("Horizon de prédiction (jours ouvrés)", min_value=1, max_value=90, value=int(defaults["horizon"]))
        submit = st.form_submit_button("Lancer la prédiction")

    if not submit:
        return

    ticker = selected_symbol.strip().upper()
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
        df = collector.get_one_frame(ticker=ticker, period=period, interval=interval)
        df = download_price_history(ticker = ticker, period=period, interval=interval) if df is None else df
        print(df.head())
    except Exception as exc:  # pragma: no cover - runtime safety
        st.error(f"Erreur lors du téléchargement des données: {exc}")
        return

    if df.empty:
        st.error("Aucune donnée téléchargée pour ce ticker.")
        return

    features = build_lstm_features(df)
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
    
    print(preds)

    future_index = compute_future_index(features.index[-1], horizon=horizon)
    preds_df = pd.DataFrame(
        {
        'adj_close_P025': preds[:, 0],
        'adj_close_P50':  preds[:, 1],
        'adj_close_P975': preds[:, 2],
    },
        index=future_index,
    )

    work_hist = df.reset_index().rename(columns={"index": "date"})
    print(work_hist.head())
    work_hist["date"] = pd.to_datetime(work_hist["date"]) if "date" in work_hist.columns else pd.to_datetime(work_hist["Date"])

    fig = make_prediction_plot(work_hist, preds_df, ticker=ticker)
    st.plotly_chart(fig, use_container_width=True)

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
        horizon = st.number_input("Horizon de prédiction", min_value=1, max_value=90, value=int(defaults["horizon"]))
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


def render_etf_analysis_page() -> None:
    st.header("Analyse des ETF européens")

    base_tickers = EuropeanETFCollector().get_tickers()

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

            options_map: Dict[str, str] = {}
            st.session_state["_options_map"] = options_map
            st.session_state["_base_tickers"] = base_tickers

            for item in suggestions:
                symbol = item.get("symbol", "")
                name = item.get("name", "")
                exchange = item.get("exchange", "")
                label_parts = [symbol]
                if name:
                    label_parts.append(f"— {name}")
                if exchange:
                    label_parts.append(f"({exchange})")
                options_map[" ".join(label_parts)] = symbol

            chosen_labels = st.multiselect(
                "Sélectionnez les symboles à ajouter",
                list(options_map.keys()),
                key="suggestion_selection",
            )

            def _on_add_suggestions() -> None:
                options_map = st.session_state.get("_options_map", {})
                base_tickers = st.session_state.get("_base_tickers", [])
                extras = st.session_state.setdefault("extra_tickers", [])
                sel = st.session_state.get("selected_tickers") or []
                chosen = st.session_state.get("suggestion_selection", [])

                added: List[str] = []
                for label in chosen:
                    sym = options_map[label]
                    if sym not in extras and sym not in base_tickers:
                        extras.append(sym)
                    if sym not in sel:
                        sel.append(sym)
                        added.append(sym)

                st.session_state["selected_tickers"] = sel
                st.session_state["suggestion_selection"] = []
                st.session_state["_add_feedback"] = (
                    "success",
                    "Ajouté : " + ", ".join(sorted(added)) if added else "Les symboles sélectionnés étaient déjà présents.",
                )
                st.rerun()

            st.button("Ajouter à la sélection", key="add_suggestion_button", on_click=_on_add_suggestions)

            feedback = st.session_state.pop("_add_feedback", None)
            if feedback:
                kind, msg = feedback
                (st.success if kind == "success" else st.info)(msg)

    st.markdown("---")

    def _on_add_manual() -> None:
        txt = st.session_state.get("manual_ticker_input", "")
        manual_syms = _format_ticker_list([txt])
        if not manual_syms:
            st.session_state["_manual_feedback"] = ("warning", "Veuillez saisir au moins un symbole valide.")
        else:
            extras = st.session_state.setdefault("extra_tickers", [])
            sel = st.session_state.get("selected_tickers") or []
            added: List[str] = []
            for sym in manual_syms:
                if sym not in extras and sym not in base_tickers:
                    extras.append(sym)
                if sym not in sel:
                    sel.append(sym)
                    added.append(sym)
            st.session_state["selected_tickers"] = sel
            st.session_state["_manual_feedback"] = (
                "success" if added else "info",
                "Ajouté : " + ", ".join(sorted(added)) if added else "Les symboles saisis étaient déjà présents.",
            )
        st.session_state["manual_ticker_input"] = ""
        st.rerun()

    manual_col, button_col = st.columns([3, 1])
    with manual_col:
        st.text_input("Ajouter manuellement des symboles (séparés par des virgules)", key="manual_ticker_input")
    with button_col:
        st.button("Ajouter", key="manual_add_button", on_click=_on_add_manual)

    feedback_manual = st.session_state.pop("_manual_feedback", None)
    if feedback_manual:
        kind, msg = feedback_manual
        getattr(st, kind)(msg)

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

    period = st.selectbox("Période", ["6mo", "1y", "5y", "10y", "max"], index=4)
    interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
    use_max = st.checkbox("Ignorer la période et télécharger l'historique complet", value=True)
    log_scale = st.checkbox("Afficher l'échelle logarithmique", value=False)

    if not selected:
        st.info("Sélectionnez au moins un ETF pour lancer la collecte.")
        return

    with st.spinner("Téléchargement des données..."):
        summary, prices = load_etf_data(tuple(selected), period=period, interval=interval, use_max=use_max)

    render_summary_metrics(summary)
    render_visualisations(summary, prices, log_scale)


def render_prediction_page() -> None:
    st.header("Prédictions et entraînement LSTM")
    render_lstm_prediction()
    st.markdown("---")
    render_lstm_training()


def render_assistant_page() -> None:
    st.header("MagistrAssistant")
    render_economic_assistant()


def render_home_page() -> None:
    st.markdown(
        "## Bienvenue sur QuantIA",
    )
    st.write(
        "Explorez l'écosystème des ETF européens, entraînez un modèle LSTM probabiliste et posez vos questions économiques à un assistant IA."
    )

    st.markdown(
        """
        ### Comment naviguer ?
        Utilisez le sélecteur de pages Streamlit situé dans la barre latérale pour accéder aux différents modules :

        * **Analyse ETF** pour explorer les séries historiques, comparer les performances et visualiser les corrélations.
        * **Prédictions & Entraînement** pour lancer des prévisions probabilistes et entraîner votre propre modèle.
        * **MagistrAssistant** pour poser des questions économiques et obtenir des réponses alimentées par Mistral AI.
        """
    )

    st.markdown("### Accès rapide")
    try:
        st.page_link("pages/analyse_graph_app.py", label="📊 Analyse ETF")
        st.page_link("pages/prediction_app.py", label="🧠 Prédictions & Entraînement")
        st.page_link("pages/genai_app.py", label="⚙️ MagistrAssistant")
    except AttributeError:
        st.markdown("- 📊 [Analyse ETF](pages/analyse_graph_app.py)")
        st.markdown("- 🧠 [Prédictions & Entraînement](pages/prediction_app.py)")
        st.markdown("- ⚙️ [MagistrAssistant](pages/genai_app.py)")

    st.markdown(
        """
        ### Conseils d'utilisation
        * Pour tirer parti du cache, laissez l'application ouverte pendant vos explorations : les téléchargements de données seront réutilisés.
        * Les formulaires de prédiction et d'entraînement demandent un modèle LSTM préalablement sauvegardé dans le répertoire `checkpoints`.
        * L'assistant économique nécessite une configuration valide de l'API Mistral AI dans le service `services/genai_service.py`.
        """
    )


__all__ = [
    "init_session_state",
    "load_etf_data",
    "show_header",
    "render_home_page",
    "render_etf_analysis_page",
    "render_prediction_page",
    "render_assistant_page",
    "render_lstm_prediction",
    "render_lstm_training",
    "render_economic_assistant",
]
