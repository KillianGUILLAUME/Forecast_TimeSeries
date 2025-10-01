import pandas as pd
import matplotlib.pyplot as plt

def plot_overlay(
    df_hist: pd.DataFrame,
    preds_df: pd.DataFrame,
    feature: str = 'adj_close',
    ticker: str = '',
    history_window: int = 126,
    # options d’affichage
    shade_color: str = 'blue',
    shade_alpha: float = 0.15,
    median_color: str = 'black',
    median_linestyle: str = '--'
):
    """
    Superpose l'historique et les prévisions.
    Si preds_df contient des colonnes de quantiles, on trace un fan chart (P2.5–P97.5) + médiane.
    Sinon, on trace la série 'feature' de preds_df comme avant.

    Colonnes de quantiles attendues (détectées automatiquement) :
      - f'{feature}_P025', f'{feature}_P50', f'{feature}_P975'
    Alternatives acceptées :
      - f'{feature}_Q025', f'{feature}_Q50', f'{feature}_Q975'
      - 'P025'/'P2.5'/'Q025', 'P50'/'Q50', 'P975'/'Q97.5'
    """

    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # --- Historique (index ou colonne 'date')
    if 'date' in df_hist.columns:
        dfh = df_hist[['date', feature]].dropna().copy()
        dfh['date'] = pd.to_datetime(dfh['date'])
        dfh = dfh.iloc[-history_window:]
        x_hist = dfh['date']
        y_hist = dfh[feature].astype(float)
        x_hist_last = dfh['date'].iloc[-1]
    else:
        dfh = df_hist[[feature]].dropna().copy()
        dfh = dfh.iloc[-history_window:]
        x_hist = pd.to_datetime(dfh.index)
        y_hist = dfh[feature].astype(float)
        x_hist_last = pd.to_datetime(df_hist.index[-1])

    # --- Prévisions
    x_pred = pd.to_datetime(preds_df.index)

    # Cherche colonnes de quantiles
    low_col  = _find_col(preds_df, [f'{feature}_P025', f'{feature}_Q025', 'P025', 'P2.5', 'Q025'])
    med_col  = _find_col(preds_df, [f'{feature}_P50',  f'{feature}_Q50',  'P50',  'Q50'])
    high_col = _find_col(preds_df, [f'{feature}_P975', f'{feature}_Q975', 'P975', 'Q97.5'])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Historique
    ax.plot(x_hist, y_hist, label=f'{ticker} - historique (dernier {history_window})', linewidth=2)

    if all(c is not None for c in [low_col, med_col, high_col]):
        # Fan chart + médiane
        y_low  = preds_df[low_col].astype(float)
        y_med  = preds_df[med_col].astype(float)
        y_high = preds_df[high_col].astype(float)

        ax.fill_between(x_pred, y_low, y_high, color=shade_color, alpha=shade_alpha,
                        label='Intervalle 95%')
        ax.plot(x_pred, y_med, median_linestyle, color=median_color, linewidth=2,
                label='Médiane')
    else:
        # Fallback : une seule série de prédiction
        if feature in preds_df.columns:
            y_pred = preds_df[feature].astype(float)
            ax.plot(x_pred, y_pred, '--', marker='o', linewidth=2, label='prévision')
        else:
            raise ValueError(
                f"Colonnes de quantiles non trouvées et '{feature}' absent de preds_df. "
                f"Présentes: {list(preds_df.columns)}"
            )

    # Ligne verticale séparation histo / prévision
    ax.axvline(x_hist_last, color='gray', linestyle=':', alpha=0.6)

    ax.set_title(f"{ticker} — {feature}: historique vs prévisions", fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
