from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf



DEFAULT_TRAIN_TICKERS: List[str] = [ "SPY", "ISF.L", "CAC.PA", "EXS1.DE", "IAEX.AS", "1321.T", "XIC.TO", "2800.HK", "STW.AX", "510300.SS"]

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


@dataclass
class PreparedTickerData:
    """Container used to expose both the raw frame and engineered features."""

    ticker: str
    raw: pd.DataFrame
    features: pd.DataFrame


RENAME_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}

def _to_datetime(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts



def download_price_history(
    ticker: str,
    *,
    period: str = "5y",
    interval: str = "1d",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Download OHLCV history from Yahoo Finance and normalise columns."""

    params: Dict[str, object] = dict(
        tickers=ticker,
        interval=interval,
        group_by="column",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=True,
    )

    start_ts = _to_datetime(start)
    end_ts = _to_datetime(end)
    if start_ts is not None or end_ts is not None:
        if start_ts is not None:
            params["start"] = start_ts.to_pydatetime()
        if end_ts is not None:
            params["end"] = (end_ts + pd.Timedelta(days=1)).to_pydatetime()
    else:
        params["period"] = period

    df = yf.download(**params)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2 and len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

    df = df.rename(columns=RENAME_MAP).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]  # guard against duplicate rows

    for col in ["open", "high", "low", "close", "adj_close", "dividends", "stock_splits"]:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            df[col] = pd.to_numeric(series, errors="coerce")
    if "volume" in df.columns:
        series = df["volume"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        df["volume"] = pd.to_numeric(series, errors="coerce")

    return df



def fetch_macro_series(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1d",
    tickers: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Download the auxiliary macro/forex series and align their indices."""

    if start is None or end is None:
        return pd.DataFrame()

    alias_map = FOREX_TICKERS if tickers is None else tickers
    columns: Dict[str, pd.Series] = {}

    for macro_ticker, alias in alias_map.items():
        df = download_price_history(
            macro_ticker,
            interval=interval,
            start=start,
            end=end,
        )
        if df.empty:
            continue
        series = df.get("adj_close")
        if series is None:
            series = df.get("close")
        if series is None:
            continue
        columns[alias] = series.astype(float)

    if not columns:
        return pd.DataFrame()

    macro_df = pd.concat(columns, axis=1)
    macro_df = macro_df.sort_index().ffill()
    return macro_df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base technical indicators required by the LSTM."""

    if df.empty:
        return pd.DataFrame()

    work = pd.DataFrame(index=df.index)
    price = df.get("adj_close")
    if price is None or price.dropna().empty:
        price = df.get("close")
    price = price.astype(float).fillna(method="ffill")
    work["adj_close"] = price

    volume = df.get("volume")
    if volume is None:
        volume = pd.Series(0.0, index=df.index)
    volume = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    work["volume_log"] = np.log1p(np.clip(volume, a_min=0.0, a_max=None))

    log_price = np.log(price.where(price > 0)).replace([-np.inf, np.inf], np.nan)
    work["ret"] = log_price.diff()

    work["ret_mean_5"] = work["ret"].rolling(window=5, min_periods=5).mean()
    work["ret_mean_20"] = work["ret"].rolling(window=20, min_periods=20).mean()
    work["ret_mean_50"] = work["ret"].rolling(window=50, min_periods=50).mean()
    work["ret_std_20"] = work["ret"].rolling(window=20, min_periods=20).std()
    work["ret_std_50"] = work["ret"].rolling(window=50, min_periods=50).std()

    rolling_mean_20 = log_price.rolling(window=20, min_periods=20).mean()
    rolling_std_20 = log_price.rolling(window=20, min_periods=20).std()
    rolling_std_20 = rolling_std_20.replace(0.0, np.nan)
    work["price_zscore_20"] = (log_price - rolling_mean_20) / rolling_std_20

    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    work["RSI_14"] = 100 - (100 / (1 + rs))
    work["RSI_14"] = work["RSI_14"].fillna(50.0)

    return work


def append_macro_indicators(features: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Append macro/forex log returns to the feature matrix."""

    if macro is None or macro.empty:
        enriched = features.copy()
    else:
        macro_aligned = macro.reindex(features.index).ffill()
        enriched = features.copy()
        for alias in macro_aligned.columns:
            col_name = f"{alias}_RET"
            series = macro_aligned[alias].astype(float).replace(0.0, np.nan)
            enriched[col_name] = np.log(series).diff()

    for alias in FOREX_TICKERS.values():
        col_name = f"{alias}_RET"
        if col_name not in enriched.columns:
            enriched[col_name] = 0.0

    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    enriched = enriched.dropna()

    missing = [c for c in LSTM_FEATURE_COLUMNS if c not in enriched.columns]
    if missing:
        for col in missing:
            enriched[col] = 0.0

    base_columns = [c for c in LSTM_FEATURE_COLUMNS if c in enriched.columns]
    result = enriched[base_columns].astype(float)
    return result



def build_lstm_features(
    df: pd.DataFrame,
    *,
    macro: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine technical indicators and macro returns into the LSTM feature grid."""

    base = compute_technical_indicators(df)
    if base.empty:
        return pd.DataFrame()

    features = append_macro_indicators(base, macro)
    if "adj_close" in base.columns:
        adj_close = base["adj_close"].reindex(features.index).astype(float)
        features = features.assign(adj_close=adj_close)
    return features



def load_prepared_ticker(
    ticker: str,
    *,
    period: str = "5y",
    interval: str = "1d",
    macro_tickers: Optional[Dict[str, str]] = None,
    include_macro:bool = True,
) -> PreparedTickerData:
    """Download a ticker and return both raw history and engineered features."""

    raw = download_price_history(ticker, period=period, interval=interval)
    if raw.empty:
        return PreparedTickerData(ticker=ticker, raw=pd.DataFrame(), features=pd.DataFrame())

    macro = None
    if include_macro:
        macro = fetch_macro_series(
            start=raw.index.min(),
            end=raw.index.max(),
            interval=interval,
            tickers=macro_tickers,
        )
    features = build_lstm_features(raw, macro=macro)

    return PreparedTickerData(ticker=ticker, raw=raw, features=features)


def prepare_lstm_training_datasets(
    tickers: Optional[Sequence[str]] = None,
    *,
    period: str = "5y",
    interval: str = "1d",
    window_size: Optional[int] = None,
    macro_tickers: Optional[Dict[str, str]] = None,
    include_forex: bool = True,
    forex_tickers: Optional[Dict[str, object]] = None,
) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, object]]:
    """Prepare datasets for the probabilistic LSTM training loop.
    return list of (ticker, df_features) and metadata about kept/dropped tickers."""

    tickers = list(tickers or DEFAULT_TRAIN_TICKERS)
    datasets: List[Tuple[str, pd.DataFrame]] = []
    kept: List[Tuple[str, int]] = []
    dropped: Dict[str, str] = {}

    for ticker in tickers:
        prepared = load_prepared_ticker(
            ticker,
            period=period,
            interval=interval,
            macro_tickers=macro_tickers,
        )
        if prepared.features.empty:
            dropped[ticker] = "aucune donnée exploitable"
            continue
        if window_size is not None and len(prepared.features) < window_size:
            dropped[ticker] = f"série trop courte ({len(prepared.features)} < {window_size})"
            continue
        datasets.append((ticker, prepared.features))
        kept.append((ticker, len(prepared.features)))

    if include_forex:
        if forex_tickers is not None:
            forex_map = forex_tickers
        elif macro_tickers:
            forex_map = macro_tickers
        else:
            forex_map = FOREX_TICKERS
        for forex_ticker, alias in forex_map.items():
            prepared = load_prepared_ticker(
                forex_ticker,
                period=period,
                interval=interval,
                macro_tickers=forex_map,
            )
            if prepared.features.empty:
                dropped[alias] = "aucune donnée exploitable"
                continue
            if window_size is not None and len(prepared.features) < window_size:
                dropped[alias] = f"série trop courte ({len(prepared.features)} < {window_size})"
                continue
            datasets.append((ticker, prepared.features))
            kept.append((alias, len(prepared.features)))


    metadata = {"kept": kept, "dropped": dropped}
    return datasets, metadata

def resolve_training_universe(env_values: Optional[Iterable[str]] = None) -> List[str]:
    """Resolve the training tickers using environment-provided fallbacks."""

    if env_values:
        for raw in env_values:
            if not raw:
                continue
            tickers = [t.strip() for t in raw.split(",") if t.strip()]
            if tickers:
                return tickers
    return DEFAULT_TRAIN_TICKERS.copy()


__all__ = [
    "PreparedTickerData",
    "DEFAULT_TRAIN_TICKERS",
    "FOREX_TICKERS",
    "LSTM_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "append_macro_indicators",
    "build_lstm_features",
    "compute_technical_indicators",
    "download_price_history",
    "fetch_macro_series",
    "load_prepared_ticker",
    "prepare_lstm_training_datasets",
    "resolve_training_universe",
]