# etf_visualizer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ETFVisualizer:
    """
    Classe pour visualiser les données d'ETF européens
    """
    
    def __init__(self):
        self.etf_short_names = {
        "VWCE.DE": "VWCE",
        "IWDA.AS": "IWDA",
        "EIMI.AS": "EIMI"}


        # Configuration des couleurs par bourse
        self.exchange_colors = {
            '.PA': '#FF6B6B',  # Rouge pour Paris
            '.AS': '#4ECDC4',  # Turquoise pour Amsterdam  
            '.L': '#45B7D1',   # Bleu pour Londres
            '.DE': '#96CEB4',  # Vert pour Francfort
            '.MI': '#FFEAA7',  # Jaune pour Milan
            '.SW': '#DDA0DD'   # Violet pour Suisse
        }
    
    # etf_visualizer.py — remplace juste le début de plot_single_etf par ceci
    def plot_single_etf(self, df: pd.DataFrame, ticker: str, title_suffix: str = "", log = True):
        
        if df.empty:
            print(f"Aucune donnée pour {ticker}")
            return

        # -- robustifier l'entrée --
        df = df.copy()
        # 1) avoir une colonne 'date'
        if 'date' not in df.columns:
            df = df.reset_index()
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            elif 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})
        # 2) s'assurer d'avoir une colonne de prix
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df = df.rename(columns={'close': 'adj_close'})
            elif len([c for c in df.columns if c not in ('date','volume')]) == 1:
                # s'il n'y a qu'une seule série de prix, on la prend
                only_col = [c for c in df.columns if c not in ('date','volume')][0]
                df = df.rename(columns={only_col: 'adj_close'})
            else:
                print(f"{ticker}: colonne de prix introuvable")
                return
        

        etf_name = self.etf_short_names.get(ticker, ticker)

        if 'high' in df.columns and 'low' in df.columns:
            fig_plotly = go.Figure(data=[go.Candlestick(x = df['date'],
                            open = df['adj_close'],
                            high = df['high'],
                            low = df['low'],
                            close = df['adj_close'])])
            fig_plotly.update_layout(
                title = f'{ticker} - {etf_name} Évolution du prix ajusté',
                yaxis_title = 'Prix',
                xaxis_title = 'Date'
            )
            if log:
                fig_plotly.update_yaxes(type="log")

            fig_plotly.show()

 
    
    def plot_multiple_etfs(self, data_dict: Dict[str, pd.DataFrame], 
                          normalize=True, title_suffix: str = "", action = True, log = False):
        """
        Comparaison de plusieurs ETF sur un même graphique
        
        Args:
            data_dict: Dictionnaire {ticker: DataFrame}
            normalize: Si True, normalise à base 100
            title_suffix: Suffixe optionnel pour le titre
        """

       

        fig, ax = plt.subplots(figsize=(12, 8))
        
        for ticker, df in data_dict.items():
            if df.empty:
                continue

            df = df.copy()
            if 'date' not in df.columns:           # date en index -> colonne
                df = df.reset_index()
                if 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'date'})
                elif 'index' in df.columns:
                    df = df.rename(columns={'index': 'date'})

            if 'adj_close' not in df.columns:      # fallback sur 'close' / colonne unique
                if 'close' in df.columns:
                    df = df.rename(columns={'close': 'adj_close'})
                else:
                    value_cols = [c for c in df.columns if c not in ('date', 'volume')]
                    if len(value_cols) == 1:
                        df = df.rename(columns={value_cols[0]: 'adj_close'})
                    else:
                        print(f"{ticker}: colonne de prix introuvable")
                        continue

            df = df.sort_values('date')
                
            etf_name = self.etf_short_names.get(ticker, ticker)
            
            # Normalisation pour comparaison (base 100)
            if normalize:
                prices = (df['adj_close'] / df['adj_close'].iloc[0]) * 100
                ylabel = 'Performance indexée (base 100)'
            else:
                prices = df['adj_close']
                ylabel = 'Prix'
            
            ax.plot(df['date'], prices, linewidth=2.5, 
                   color=self._get_color(ticker),
                   label=f'{ticker} - {etf_name}')
        
        title = 'Comparaison des ETF Européens'
        if title_suffix:
            title += f' {title_suffix}'
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if log:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, data_dict: Dict[str, pd.DataFrame], 
                                 title_suffix: str = ""):
        """
        Distribution des rendements quotidiens
        
        Args:
            data_dict: Dictionnaire {ticker: DataFrame}
            title_suffix: Suffixe optionnel pour le titre
        """
        tickers = list(data_dict.keys())[:4]  # Limite à 4 ETF
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, ticker in enumerate(tickers):
            df = data_dict[ticker]
            if df.empty:
                continue
                
            df_copy = df.copy()
            df_copy['returns'] = df_copy['adj_close'].pct_change() * 100
            
            # Histogramme
            axes[i].hist(df_copy['returns'].dropna(), bins=50, alpha=0.7, 
                       color=self._get_color(ticker), edgecolor='black', linewidth=0.5)
            
            # Statistiques
            mean_return = df_copy['returns'].mean()
            std_return = df_copy['returns'].std()
            
            axes[i].axvline(mean_return, color='red', linestyle='--', 
                          label=f'Moyenne: {mean_return:.2f}%')
            axes[i].set_title(f'{ticker}\nVolatilité: {std_return:.2f}%', 
                            fontweight='bold')
            axes[i].set_xlabel('Rendement quotidien (%)')
            axes[i].set_ylabel('Fréquence')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        title = 'Distribution des rendements quotidiens'
        if title_suffix:
            title += f' {title_suffix}'
            
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame], 
                               title_suffix: str = ""):
        """
        Matrice de corrélation entre les ETF
        
        Args:
            data_dict: Dictionnaire {ticker: DataFrame}
            title_suffix: Suffixe optionnel pour le titre
        """
        # Récupérer les rendements pour tous les ETF
        returns_data = {}
        
        for ticker, df in data_dict.items():
            if df.empty:
                continue
            df_copy = df.copy()
            df_copy['returns'] = df_copy['adj_close'].pct_change()
            returns_data[ticker] = df_copy['returns']
        
        if len(returns_data) < 2:
            print("Pas assez de données pour calculer les corrélations")
            return
        
        # Créer DataFrame des rendements
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculer la matrice de corrélation
        correlation_matrix = returns_df.corr()
        
        # Créer le heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': 'Corrélation'})
        
        title = 'Matrice de corrélation des rendements'
        if title_suffix:
            title += f' {title_suffix}'
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_risk_return_scatter(self, data_dict: Dict[str, pd.DataFrame], 
                                title_suffix: str = ""):
        """
        Graphique risque-rendement (volatilité vs performance)
        
        Args:
            data_dict: Dictionnaire {ticker: DataFrame}
            title_suffix: Suffixe optionnel pour le titre
        """
        risk_return_data = []
        
        for ticker, df in data_dict.items():
            if df.empty or len(df) < 2:
                continue
                
            # Calcul du rendement annualisé
            total_return = ((df['adj_close'].iloc[-1] / df['adj_close'].iloc[0]) ** 
                          (365.25 / len(df)) - 1) * 100
            
            # Calcul de la volatilité annualisée
            df_copy = df.copy()
            df_copy['returns'] = df_copy['adj_close'].pct_change()
            volatility = df_copy['returns'].std() * (252**0.5) * 100
            
            risk_return_data.append({
                'ticker': ticker,
                'return': total_return,
                'volatility': volatility,
                'color': self._get_color(ticker)
            })
        
        if not risk_return_data:
            print("Aucune donnée disponible pour le graphique risque-rendement")
            return
        
        # Créer le scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for data in risk_return_data:
            ax.scatter(data['volatility'], data['return'], 
                      s=100, c=data['color'], alpha=0.7)
            
            # Ajouter les labels des points
            ax.annotate(data['ticker'], 
                       (data['volatility'], data['return']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Volatilité annualisée (%)', fontsize=12)
        ax.set_ylabel('Rendement annualisé (%)', fontsize=12)
        
        title = 'Profil Risque-Rendement des ETF'
        if title_suffix:
            title += f' {title_suffix}'
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_summary_dashboard(self, summary_df: pd.DataFrame):
        """
        Dashboard de résumé avec plusieurs métriques
        
        Args:
            summary_df: DataFrame de résumé depuis EuropeanETFCollector.create_price_summary()
        """
        if summary_df.empty:
            print("Aucune donnée de résumé disponible")
            return
        
        df = summary_df.copy()
        
        if 'exchange' not in df.columns:
            df['exchange'] = df['ticker'].str.extract(r'(\.[A-Z]+)$')[0].fillna('UNKNOWN')
        if 'country' not in df.columns:
            exch_to_country = {
                '.DE': 'Germany', '.AS': 'Netherlands', '.PA': 'France',
                '.L': 'United Kingdom', '.MI': 'Italy', '.SW': 'Switzerland',
                'UNKNOWN': 'Unknown'
            }
            df['country'] = df['exchange'].map(exch_to_country).fillna('Unknown')
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Rendements par pays
        if 'country' in df.columns:
            country_returns = df.groupby('country')['total_return_%'].mean().sort_values(ascending=False)
            ax1.bar(country_returns.index, country_returns.values, 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax1.set_title('Rendement moyen par pays', fontweight='bold')
            ax1.set_ylabel('Rendement (%)')
            ax1.tick_params(axis='x', rotation=45)
        
        else:
            ax1.text(0.5, 0.5, "total_return indisponible", ha='center', va='center')
            ax1.axis('off')

        
        # 2. Volatilité vs Rendement
        if {'annual_volatility_%', 'total_return_%'}.issubset(df.columns):
            scatter = ax2.scatter(summary_df['annual_volatility_%'], summary_df['total_return_%'], 
                                s=60, alpha=0.7, c=range(len(summary_df)), cmap='viridis')
            ax2.set_xlabel('Volatilité annuelle (%)')
            ax2.set_ylabel('Rendement total (%)')
            ax2.set_title('Rendement vs Volatilité', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Volatilité indisponible", ha='center', va='center')
            ax2.axis('off')
            
        # 3. Volume moyen par bourse
        if 'avg_volume'  in df.columns:
            exchange_volume = summary_df.groupby('exchange')['avg_volume'].mean().sort_values(ascending=False)
            ax3.barh(exchange_volume.index, exchange_volume.values, color='lightcoral')
            ax3.set_title('Volume moyen par bourse', fontweight='bold')
            ax3.set_xlabel('Volume moyen')
        
        else:
            ax3.text(0.5, 0.5, "avg_volume indisponible", ha='center', va='center')
            ax3.axis('off')
        
        # 4. Distribution des rendements

        if 'total_return_%' in df.columns:
            ax4.hist(summary_df['total_return_%'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(summary_df['total_return_%'].mean(), color='red', linestyle='--', 
                    label=f'Moyenne: {summary_df["total_return_%"].mean():.1f}%')
            ax4.set_title('Distribution des rendements', fontweight='bold')
            ax4.set_xlabel('Rendement (%)')
            ax4.set_ylabel('Fréquence')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "total return indisponible", ha='center', va='center')
            ax4.axis('off')
        
        plt.suptitle('Dashboard ETF Européens', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _get_color(self, ticker: str) -> str:
        """Retourne une couleur basée sur la bourse"""
        for suffix, color in self.exchange_colors.items():
            if ticker.endswith(suffix):
                return color
        return '#95A5A6'  # Couleur par défaut