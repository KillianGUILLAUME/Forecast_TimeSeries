# etf_collector.py
from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List



def clean_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2 and len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    rename_map = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
        "Dividends": "dividends", "Stock Splits": "stock_splits",
    }
    df = df.rename(columns=rename_map).sort_index()

    # cast numériques (volume en Int64 pour éviter l’overflow)
    for c in ["open","high","low","close","adj_close","dividends","stock_splits"]:
        if c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[c] = pd.to_numeric(col, errors="coerce")
    if "volume" in df.columns:
        col = df["volume"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df["volume"] = pd.to_numeric(col, errors="coerce").astype("Int64")

    return df


class EuropeanETFCollector:
    def __init__(self, tickers: List[str] | None = None):
        # 3 ETF européens “classiques”
        self.tickers = list(tickers) if tickers is not None else [ "SPY", "ISF.L", "CAC.PA", "EXS1.DE", "IAEX.AS", "1321.T", "XIC.TO", "2800.HK", "STW.AX", "510300.SS"]
        self.frames = None

    def get_tickers(self) -> List[str]:
        """
        Retourne la liste des tickers gérés
        """
        if self.tickers is None:
            return [ "SPY", "ISF.L", "CAC.PA", "EXS1.DE", "IAEX.AS", "1321.T", "XIC.TO", "2800.HK", "STW.AX", "510300.SS"]

        return list(self.tickers)
    

    def frequence_annualization(self, idx : pd.DatetimeIndex) -> float:
        freq = pd.infer_freq(idx)
        if freq in ("B", "C", "D"):          # quotidien / business day
            return 252.0
        if freq and freq.startswith("W"):    # hebdo
            return 52.0
        if freq in ("M", "MS"):              # mensuel
            return 12.0
        if freq in ("Q", "QS"):              # trimestriel
            return 4.0
        if freq in ("A", "AS", "Y"):         # annuel
            return 1.0
        
        deltas = np.diff(idx.values).astype("timedelta64[D]").astype(float)
        step_days = np.nanmean(deltas) if deltas.size else 1.0
        return 365.25 / max(step_days, 1.0)



    def get_etf_frames(self, period: str = "5y", interval: str = "1d", ticker = None) -> List[pd.DataFrame]:
        """
        Télécharge l'historique complet (OHLC, Adj Close, Volume, Dividends, Stock Splits)
        pour tous les tickers, et renvoie un DataFrame 'tidy':
        columns = [date, ticker, open, high, low, close, adj_close, volume, dividends, stock_splits]
        """

        tickers_to_fetch = self.get_tickers() if ticker is None else [ticker]
        kept_tickers : List[str] = []

        # actions=True pour récupérer Dividends / Stock Splits

        frames: List[pd.DataFrame] = []

        for t in tickers_to_fetch:
            print(f"Téléchargement des données pour {t}...")

            df = yf.download(
                tickers=t,
                period=period,
                interval=interval,
                group_by="column",
                actions=True,
                auto_adjust=False,   # on garde Close et Adj Close séparés
                progress=False,
                threads=True
            )

            if df.empty:
                print(f"Aucune donnée trouvée pour {t}. Vérifiez le ticker.")
                continue
            
            df = clean_yf_frame(df)

            df["ticker"] = t  # utile pour filtrer ensuite

            frames.append(df)
            kept_tickers.append(t)

        self.tickers = kept_tickers
        self.frames = frames

        return frames
    

    def extract_price_series(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "adj_close" if "adj_close" in df.columns else "close"

        if df[col].dropna().empty:
            raise ValueError(f"Aucune donnée de prix valide dans la colonne '{col}'.")
        
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            raise ValueError(f"Aucune donnée de prix valide après conversion dans la colonne '{col}'.")
        
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()

        return s


    def build_summary(self) -> pd.DataFrame:

        row_summary = []

        for df in self.frames:
            if df is None or df.empty:
                continue
            
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'UNKNOWN'

            price_series = self.extract_price_series(df)

            if price_series.empty:
                raise ValueError(f"Série de prix vide pour le ticker {ticker}.")
            
            first = float(price_series.iloc[0])
            last = float(price_series.iloc[-1])

            total_return = float(np.nan)
            if np.isfinite(last) and first > 0:
                total_return = (last - first) / first * 100
            else:
                warnings.warn(f"Impossible de calculer le rendement total pour le ticker {ticker}. \nfirst={first}, last={last}")

            ret= price_series.pct_change().dropna()
            ann_vol = np.nan
            if not ret.empty:
                freq = self.frequence_annualization(price_series.index)
                ann_vol = float(np.std(ret) * np.sqrt(freq) * 100)
            
            row_summary.append({'ticker': ticker,
                                'price_series' : price_series,
                                'start': price_series.index[0],
                                'end': price_series.index[-1],
                                'last_price': last,
                                'total_return_%': total_return,
                                'annual_volatility_%': ann_vol}
                                )
            
        summary_df = pd.DataFrame(row_summary)
        return summary_df
    
    def get_indicator(self, liste : List[pd.DataFrame]) -> List[pd.DataFrame]:
        indicators_frames = []

        for df in liste:
            if df is None or df.empty:
                continue
            
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'UNKNOWN'

            price_series = self.extract_price_series(df)

            if price_series.empty:
                raise ValueError(f"Série de prix vide pour le ticker {ticker}.")
            
            df_ind = df.copy()
            df_ind['SMA_5'] = price_series.rolling(window=5).mean()
            df_ind['SMA_50'] = price_series.rolling(window=50).mean()
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_ind['RSI_14'] = 100 - (100 / (1 + rs))
            indicators_frames.append(df_ind)


        return indicators_frames
    

    def get_one_frame(self, ticker: str, period : str ='max', interval: str = '1d'):
        df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                group_by="column",
                actions=True,
                auto_adjust=False,   # on garde Close et Adj Close séparés
                progress=False,
                threads=True
            )
        if df.empty:
            print(f"Aucune donnée trouvée pour {ticker}. Vérifiez le ticker.")
            return None
        
        df = clean_yf_frame(df)
        df['ticker'] = ticker
        return df
    

    
    def get_df_with_tickers(self, ticker: str, period: str = 'max', interval: str = '1d') -> pd.DataFrame:
        """
        renvoie un df particulier avec un ticker donné
        """

        df0 = None
        if getattr(self, 'frames', None) and getattr(self, 'tickers', None):
            if ticker in self.tickers:
                idx = self.tickers.index(ticker)
                df0 = self.frames[idx]

        if df0 is None:
            df0 = self.get_one_frame(ticker, period=period, interval=interval)

        if df0 is None or df0.empty:
            raise ValueError(f"Aucune donnée disponible pour le ticker {ticker}.")


        df_base = df0[['adj_close', 'volume']].copy()
        df_base['ret'] = np.log(df_base['adj_close']).diff()

        indicator_source = df0

        indicator_frames = self.get_indicator([indicator_source]) if indicator_source is not None else []
        if not indicator_frames:
            return df_base

        df_indicator = indicator_frames[0][['SMA_5', 'SMA_50', 'RSI_14']]

        df = pd.concat([df_base, df_indicator], axis=1)

        return df