#prediction_lstm_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from typing import List, Tuple, Union

import matplotlib.pyplot as plt

import os, json, pickle
from datetime import datetime




def make_sequence_multi_horizon(df: pd.DataFrame, feature: List[str], window_size: int, target_feature: str, H:int = 10) -> Tuple[torch.tensor, torch.tensor]:
    if len(feature) == 1: #feature[0] = target_feature
        data = df[feature].values
        sequences, targets= [],[]
        for i in range(len(data) - window_size - H + 1):
            sequences.append(data[i:i + window_size])
            ret_individual = data[i + window_size:i + window_size + H]
            ret_cumulative = np.cumsum(ret_individual)
            targets.append(ret_cumulative)

        return torch.tensor(sequences), torch.tensor(targets)
    else:
        Xv = df[feature].values
        yv = df[target_feature].values
        sequences, targets = [], []
        for i in range(len(df) - window_size - H + 1):
            sequences.append(Xv[i:i + window_size])
            ret_individual = yv[i + window_size:i + window_size + H]
            ret_cumulative = np.cumsum(ret_individual)
            targets.append(ret_cumulative)
    
    return torch.tensor(sequences), torch.tensor(targets)


def split_data(sequences: torch.tensor, targets: torch.tensor, train_ratio: float=0.8) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    train_size = int(len(sequences) * train_ratio)
    X_train, y_train = sequences[:train_size], targets[:train_size]
    X_test, y_test = sequences[train_size:], targets[train_size:]
    return X_train, y_train, X_test, y_test


def coverage_metrics(y_true, q_low, q_med, q_high, alpha=0.05):
    """
    y_true: (N,H)   réalisations (p.ex. rendements log cumulés)
    q_low:  (N,H)   quantile bas   (p.ex. 2.5%)
    q_med:  (N,H)   médiane        (50%)
    q_high: (N,H)   quantile haut  (97.5%)
    alpha:  0.05 pour un intervalle 95% (2.5–97.5)

    Retourne des métriques par horizon (vecteurs de taille H).
    """
    inside = (y_true >= q_low) & (y_true <= q_high)
    coverage = inside.mean(axis=0)

    lower_tail = (y_true < q_low).mean(axis=0)
    upper_tail = (y_true > q_high).mean(axis=0)

    mpiw = (q_high - q_low).mean(axis=0)  # largeur moyenne

    # Interval Score (Winkler) — plus petit = meilleur
    over_low  = np.clip(q_low  - y_true, a_min=0.0, a_max=None)
    over_high = np.clip(y_true - q_high, a_min=0.0, a_max=None)
    interval_score = (q_high - q_low) + (2/alpha)*over_low + (2/alpha)*over_high
    iscore = interval_score.mean(axis=0)

    return {
        "coverage": coverage,        # idéal ≈ 0.95
        "lower_tail": lower_tail,    # idéal ≈ 0.025
        "upper_tail": upper_tail,    # idéal ≈ 0.025
        "mpiwidth": mpiw,
        "interval_score": iscore
    }

def inverse_transform_quantiles(q_scaled, scaler_y):
    """
    Inverse-transform pour quantiles (N,H,Q) quand scaler_y a été
    fit sur des cibles (N,H). On inverse quantile par quantile.
    """
    N, H, Q = q_scaled.shape
    q = np.empty_like(q_scaled, dtype=np.float32)
    for j in range(Q):
        q[:, :, j] = scaler_y.inverse_transform(q_scaled[:, :, j])
    return q





def quantile_loss(y_pred, y_true, quantiles):
    # y_pred: (B, H, Q), y_true: (B, H)
    B, H, Q = y_pred.shape
    y_true = y_true.unsqueeze(-1).expand(-1, -1, Q)  # (B,H,Q)
    qs = torch.tensor(quantiles, device=y_pred.device).view(1,1,Q)
    e = y_true - y_pred
    return torch.mean(torch.maximum(qs*e, (qs-1)*e))


class LSTMModelProba(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, quantiles=(0.025, 0.5, 0.975),
                 num_layers=3, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.Q = len(quantiles)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.in_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        d_in = hidden_size * self.num_directions
        d_mid = max(64, hidden_size)

        self.attn = nn.Linear(d_in, 1)

        self.head = nn.Sequential(
            nn.Linear(d_in, d_mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_mid, d_mid), nn.GELU(), nn.Dropout(dropout)
        )
        self.proj = nn.Linear(d_in, d_mid) if d_in != d_mid else nn.Identity()
        # → on prédit H * Q sorties (quantiles des rendements cumulés)
        self.out = nn.Linear(d_mid, self.horizon * self.Q)

    def forward(self, x):
        x = self.in_norm(x)
        out,_ = self.lstm(x)                                  # (B,T,2H)
        e = torch.tanh(self.attn(out))                        # (B,T,1)
        w = torch.softmax(e.squeeze(-1), dim=1)               # (B,T)
        h = (out * w.unsqueeze(-1)).sum(dim=1)                # (B,2H)

        z = self.head(h)
        z = z + self.proj(h)

        raw = self.out(z)                                     # (B, H*Q)
        raw = raw.view(-1, self.horizon, self.Q)              # (B,H,Q)

        # Monotonicité des quantiles: q0 libre + cumsum(softplus)
        q0 = raw[:,:, :1]                                     # (B,H,1)
        deltas = torch.nn.functional.softplus(raw[:,:,1:])    # (B,H,Q-1) >= 0
        q = torch.cat([q0, q0 + torch.cumsum(deltas, dim=2)], dim=2)  # (B,H,Q)
        return q
    

class LSTMPredictorProba:
    def __init__(self, feature: List[str], target_feature :str, window_size: int=10, hidden_size: int=50, horizon : int = 10, num_layers: int=3, lr: float=0.001, epochs: int=1000):
        unique_features = []
        for name in feature:
            if name not in unique_features:
                unique_features.append(name)

        self.feature = unique_features
        self.target_feature = target_feature
        self.quantiles = (0.025, 0.5, 0.975)
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_h = horizon
        self.lr = lr
        self.df_index_ = None
        self.epochs = epochs
        self.scaler_x = MinMaxScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, frames: Union[pd.DataFrame, List[pd.DataFrame]]):


        if isinstance(frames, pd.DataFrame):
            iterable =[frames]
        else:
            iterable = list(frames)


        seq_list, targ_list, indices = [], [], []

        for frame in iterable:
            if frame is None or len(frame) == 0:
                continue 
            if frame.columns.duplicated().any():
                frame = frame.loc[:, ~frame.columns.duplicated()].copy() # remove duplicated columns

            missing = [col for col in self.feature if col not in frame.columns]
            if missing:
                print(f"[WARNING] Frame missing features {missing}, skipping")
                continue
        for frame in iterable:
            s, t = make_sequence_multi_horizon(
                frame, feature=self.feature, target_feature=self.target_feature,
                window_size=self.window_size, H=self.output_h
            )
            if len(s) > 0:
                seq_list.append(s)
                targ_list.append(t)
                indices.append(frame.index)

        if not seq_list:
            raise ValueError("Aucune séquence générée (données insuffisantes).")

        sequences = torch.cat(seq_list, dim=0)
        targets   = torch.cat(targ_list, dim=0)

        # --- Aperçu du dataset construit ---
        N, T, F = sequences.shape
        H = targets.shape[1]
        print(f"\n[PREVIEW] sequences: N={N}, T={T}, F={F} | targets: H={H}")
        print(f"Features utilisées (ordre): {self.feature}")

        # 1) stats globales par feature (sur toutes les séquences et toutes les étapes)
        with torch.no_grad():
            feat_mean = sequences.float().mean(dim=(0, 1)).cpu().numpy()
            feat_std  = sequences.float().std(dim=(0, 1)).cpu().numpy()
        print("\n[Stats globales] mean/std par feature:")
        for j, name in enumerate(self.feature):
            print(f" - {name:<10} mean={feat_mean[j]:.6f}  std={feat_std[j]:.6f}")

        # 2) première fenêtre (T x F) sous forme de DataFrame lisible
        try:
            seq0 = sequences[0].detach().cpu().numpy()   # (T, F)
            df_seq0 = pd.DataFrame(seq0, columns=self.feature)
            print("\n[Window #0] premières lignes de la fenêtre (T x F):")
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(df_seq0.head(5))
        except Exception as e:
            print(f"[WARN] impossible d'afficher la première fenêtre: {e}")

        # 3) première cible (H) = rendements log cumulés (ou ce que tu prépares dans make_sequence...)
        try:
            targ0 = targets[0].detach().cpu().numpy()
            print("\n[Target #0] (horizon cumulatif):")
            print(np.round(targ0, 6))
        except Exception as e:
            print(f"[WARN] impossible d'afficher la première cible: {e}")


        # sequences, targets = make_sequence_multi_horizon(df, feature = self.feature, target_feature=self.target_feature, window_size=self.window_size, H =self.output_h)
        X_train, y_train, X_test, y_test = split_data(sequences, targets)



        self.df_index_ = indices[-1] if indices else None


        Ntr, T, F = X_train.shape
        Xtr2d = X_train.detach().cpu().numpy().reshape(-1, F)
        Xte2d = X_test.detach().cpu().numpy().reshape(-1, F) if len(X_test) else np.empty((0, F), np.float32)


        X_train_scaled2d = self.scaler_x.fit_transform(Xtr2d.reshape(-1, len(self.feature))).astype(np.float32)
        X_test_scaled2d = self.scaler_x.transform(Xte2d.reshape(-1, len(self.feature))).astype(np.float32) if len(X_test) else Xte2d



        X_train_scaled_torch = torch.from_numpy(X_train_scaled2d.reshape(Ntr, T, F))
        if len(X_test):
            X_test_scaled_torch = torch.from_numpy(X_test_scaled2d.reshape(X_test.shape[0], T, F))
        else:
            X_test_scaled_torch = X_test


        y_train_scaled = self.scaler_y.fit_transform(y_train.numpy()).astype(np.float32)
        y_train_last = torch.from_numpy(y_train_scaled)


        self.X_test_scaled_torch = X_test_scaled_torch
        if len(X_test):
            y_test_scaled = self.scaler_y.transform(y_test.numpy()).astype(np.float32)
            self.y_test_last = torch.from_numpy(y_test_scaled)
        else:
            self.y_test_last = y_test

        if len(self.X_test_scaled_torch) >0:
            val_ds = torch.utils.data.TensorDataset(self.X_test_scaled_torch.float(), self.y_test_last)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
        else:
            val_dl = None

        assert X_train_scaled_torch.shape[0] == y_train_last.shape[0], f"Shape mismatch: {X_train_scaled_torch.shape[0]} vs {y_train_last.shape[0]}"


        train_ds = torch.utils.data.TensorDataset(X_train_scaled_torch.float(), y_train_last)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

        print(train_dl.dataset, train_dl.dataset.tensors[0].shape, train_dl.dataset.tensors[1].shape)
         
        """         Modele LSTM proba         """

        self.model = LSTMModelProba(
                input_size=len(self.feature),
                hidden_size=self.hidden_size,
                horizon=self.output_h,
                quantiles=self.quantiles,
                num_layers=self.num_layers
            ).to(self.device)
        
        def criterion(pred, targ):
            return quantile_loss(pred, targ, quantiles=self.quantiles)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=30)

        best_loss, patience, bad = float('inf'), 150, 0

        lr_to_plot, loss_plot, val_plot = np.zeros(self.epochs), np.zeros(self.epochs), np.zeros(self.epochs)


        """        Entrainement du modele          """

        print("Starting training...")
        for epoch in range(1, self.epochs +1):
            self.model.train()
            running=0.0
            for seqs, targs in train_dl:
                seqs, targs = seqs.to(self.device).float(), targs.to(self.device).float()
                optimizer.zero_grad()
                outputs = self.model(seqs)
                loss = criterion(outputs, targs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                running+=loss.item()*seqs.size(0)
            

            train_loss = running/len(train_ds)

            if val_dl is not None:
                self.model.eval()
                v_running=0.0
                with torch.no_grad():
                    for vseqs, vtargs in val_dl:
                        vseqs, vtargs = vseqs.to(self.device).float(), vtargs.to(self.device).float()
                        voutputs = self.model(vseqs)
                        loss = criterion(voutputs, vtargs)
                        v_running+=loss.item()*vseqs.size(0)
                val_loss = v_running/len(val_ds)
            else:
                val_loss = float('nan')


            scheduler.step(train_loss)

            monitor = val_loss if val_dl is not None else train_loss
            if monitor < best_loss:
                best_loss = monitor
                bad = 0
                best_state = {k: v.cpu() for k,v in self.model.state_dict().items()}
            else:
                bad +=1
                if bad >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            
            lr_to_plot[epoch-1]=optimizer.param_groups[0]['lr']
            loss_plot[epoch-1]=loss.item()
            val_plot[epoch-1]=val_loss
            if (epoch+1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}]| Train: {train_loss:.4f} | val: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        self.model.load_state_dict(best_state)


        """        Visualisation de l'entrainement  """


        plt.figure(figsize=(12, 8))

        # First subplot for learning rate
        plt.subplot(2, 1, 1)
        plt.plot(lr_to_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')

        # Second subplot for loss
        plt.subplot(2, 1, 2)
        plt.plot(loss_plot, label='Train Loss')
        plt.plot(val_plot, label='Val Loss')
        plt.xlabel('Epochs')
        
        plt.ylabel('training Loss and val Loss')
        plt.title('Losses')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.legend()
        plt.show()



    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")
        
        print("features:", self.feature)
        print("df columns:", df.columns.tolist()) 

        data = df[self.feature].values
        if len(data)< self.window_size:
            raise ValueError(f"Not enough data to make predictions. Require at least {self.window_size} data points.")
        
          
        
        last_window = data[-self.window_size:]
        last_scaled = self.scaler_x.transform(last_window).astype(np.float32)
        input = torch.from_numpy(last_scaled).unsqueeze(0).float().to(self.device)
        
        # preds_scaled = []
        self.model.eval()
        with torch.no_grad():
            q_ret_scaled = self.model(input).cpu().numpy()  # (1, H, Q)
        
        q_ret = inverse_transform_quantiles(q_ret_scaled, self.scaler_y).squeeze(0)

        last_p = df['adj_close'].iloc[-1]

        P_q = last_p * np.exp(q_ret)

        P_low95 = P_q[:, 0]
        P_med50 = P_q[:, 1]
        P_up95 = P_q[:, 2]

        return np.stack([P_low95, P_med50, P_up95], axis=1)
    
    def evaluate_coverage_on_test(self, alpha=0.05):
        """
        Calcule le coverage & métriques sur l'échantillon test
        en espace 'rendements log cumulés'.

        Prérequis:
        - self.model est entraîné et renvoie (B,H,Q) en 'scaled space'
        - self.X_test_scaled_torch (Tensors) et self.y_test_last (scaled)
        - self.scaler_y fit sur y_train (N,H)

        Retourne: dict de vecteurs (H,) + imprime un résumé.
        """
        assert self.model is not None, "Model not trained"
        assert hasattr(self, "X_test_scaled_torch"), "Need test tensors (fit() first)"
        assert len(self.X_test_scaled_torch) > 0, "Empty test set"

        device = self.device
        self.model.eval()

        # 1) Empile les prédictions quantiles sur tout le test
        preds_scaled_list = []
        y_scaled_list = []
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_test_scaled_torch.float(), self.y_test_last.float()),
            batch_size=256, shuffle=False
        )

        with torch.no_grad():
            for xb, yb in dl:
                qb = self.model(xb.to(device)).cpu().numpy()   # (B,H,Q) en scaled
                preds_scaled_list.append(qb)
                y_scaled_list.append(yb.numpy())               # (B,H)  en scaled

        q_scaled = np.concatenate(preds_scaled_list, axis=0)   # (N,H,Q)
        y_scaled = np.concatenate(y_scaled_list, axis=0)       # (N,H)

        # 2) Inverse-transform en échelle d'origine
        q = inverse_transform_quantiles(q_scaled, self.scaler_y)  # (N,H,Q)
        y = self.scaler_y.inverse_transform(y_scaled)             # (N,H)

        # 3) Sépare q_low, q_med, q_high
        q_low  = q[:, :, 0]
        q_med  = q[:, :, 1]
        q_high = q[:, :, 2]

        # 4) Métriques
        m = coverage_metrics(y_true=y, q_low=q_low, q_med=q_med, q_high=q_high, alpha=alpha)

        # Affiche un mini résumé
        def row(v): return " ".join(f"{x:.3f}" for x in v)
        print("Coverage (cible≈{:.3f}):".format(1-alpha), row(m["coverage"]))
        print("Lower tail (cible≈{:.3f}):".format(alpha/2), row(m["lower_tail"]))
        print("Upper tail (cible≈{:.3f}):".format(alpha/2), row(m["upper_tail"]))
        print("Mean PI width:", row(m["mpiwidth"]))
        print("Interval score:", row(m["interval_score"]))
        return m
    

    def save(self, dirpath: str):
        """
        Sauvegarde le modèle + scalers + méta dans un dossier.
        Contenu: model.pt, scaler_x.pkl, scaler_y.pkl, meta.json
        """
        if self.model is None:
            raise ValueError("Model not trained, nothing to save.")
        os.makedirs(dirpath, exist_ok=True)

        # 1) Poids du modèle
        torch.save(self.model.state_dict(), os.path.join(dirpath, "model.pt"))

        # 2) Scalers
        with open(os.path.join(dirpath, "scaler_x.pkl"), "wb") as f:
            pickle.dump(self.scaler_x, f)
        with open(os.path.join(dirpath, "scaler_y.pkl"), "wb") as f:
            pickle.dump(self.scaler_y, f)

        # 3) Méta / config (pour reconstruire l’architecture)
        meta = {
            "created": datetime.now().isoformat(),
            "class": "LSTMPredictorProba",
            "feature": self.feature,
            "target_feature": self.target_feature,
            "window_size": self.window_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "horizon": self.output_h,
            "quantiles": list(self.quantiles),
            "input_size": len(self.feature),
        }
        with open(os.path.join(dirpath, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"💾 Modèle sauvegardé dans: {dirpath}")

    @classmethod
    def load(cls, dirpath: str, device: str | None = None):
        """
        Recharge un prédicteur prêt pour l'inférence (predict).
        """
        # 1) Méta
        meta_path = os.path.join(dirpath, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # 2) Instancie le prédicteur avec la même config
        predictor = cls(
            feature=meta["feature"],
            target_feature=meta["target_feature"],
            window_size=meta["window_size"],
            hidden_size=meta["hidden_size"],
            horizon=meta["horizon"],
            num_layers=meta["num_layers"],
            lr=0.0,      # inutilisé pour l'inférence
            epochs=0,    # idem
        )
        # force les mêmes quantiles
        predictor.quantiles = tuple(meta.get("quantiles", (0.025, 0.5, 0.975)))

        # 3) Recharge les scalers
        with open(os.path.join(dirpath, "scaler_x.pkl"), "rb") as f:
            predictor.scaler_x = pickle.load(f)
        with open(os.path.join(dirpath, "scaler_y.pkl"), "rb") as f:
            predictor.scaler_y = pickle.load(f)

        # 4) Reconstruit le modèle + charge les poids
        if device is not None:
            predictor.device = torch.device(device)
        map_loc = predictor.device if predictor.device.type == "cpu" else None

        predictor.model = LSTMModelProba(
            input_size=len(predictor.feature),
            hidden_size=predictor.hidden_size,
            horizon=predictor.output_h,
            quantiles=predictor.quantiles,
            num_layers=predictor.num_layers
        ).to(predictor.device)

        state = torch.load(os.path.join(dirpath, "model.pt"), map_location=map_loc)
        predictor.model.load_state_dict(state)
        predictor.model.eval()

        print(f"📦 Modèle chargé depuis: {dirpath}")
        return predictor

