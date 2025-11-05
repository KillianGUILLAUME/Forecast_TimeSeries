#prediction_lstm_model.py
import time
import psutil

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import math

from lightgbm import LGBMRegressor

from sklearn.preprocessing import StandardScaler

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Generator

import matplotlib.pyplot as plt

import copy
import re
import os, json, pickle
from datetime import datetime


def _default_plots_dir():
    base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = base / "Forecast" / "artifacts" / "plots" / ts
    d.mkdir(parents=True, exist_ok=True)
    return d



def make_sequence_multi_horizon(
        df: pd.DataFrame, feature: List[str], window_size: int, target_feature: str, H:int = 10
        ) -> Tuple[torch.tensor, torch.tensor]:
    
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


def progressive_time_series_windows(
        sequences: torch.Tensor,
        targets: torch.Tensor,
        n_splits: int,
        test_size: Optional[int] = None
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray], None, None]:
    """Parameters
    ----------
    sequences : torch.Tensor
        Tensor de séquences (N, T, F)
        targets : torch.Tensor (N,H)"""
    
    if len(sequences) == 0:
        return
    
    try:
        splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    except ValueError as exc:
        raise ValueError(f"Error initializing TimeSeriesSplit: {exc}")
    
    full_indices = np.arange(len(sequences))
    for train_idx, val_idx in splitter.split(full_indices):
        train_idx_t = torch.tensor(train_idx, dtype=torch.long)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long)

        X_train, y_train = sequences.index_select(0, train_idx_t), targets.index_select(0, train_idx_t)
        X_val, y_val = sequences.index_select(0, val_idx_t), targets.index_select(0, val_idx_t)
        yield X_train, y_train, X_val, y_val, train_idx, val_idx

    return X_train, y_train, X_val, y_val, train_idx, val_idx


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


def enforce_monotonic_quantiles(q: np.ndarray) -> np.ndarray:
    """Assure la monotonie croissante des quantiles sur le dernier axe."""
    if q.ndim == 0:
        return q
    q_adj = np.array(q, copy=True)
    for j in range(1, q_adj.shape[-1]):
        q_adj[..., j] = np.maximum(q_adj[..., j - 1], q_adj[..., j])
    return q_adj



def quantile_loss(y_pred, y_true, quantiles):
    # y_pred: (B, H, Q), y_true: (B, H)
    B, H, Q = y_pred.shape
    y_true = y_true.unsqueeze(-1).expand(-1, -1, Q)  # (B,H,Q)
    qs = torch.tensor(quantiles, device=y_pred.device).view(1,1,Q)
    e = y_true - y_pred
    return torch.mean(torch.maximum(qs*e, (qs-1)*e))


class LSTMModelProba(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, quantiles=(0.025, 0.5, 0.975),
                 num_layers=3, p_in=0.15, p_post=0.30):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.Q = len(quantiles)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1

        self.in_norm = nn.LayerNorm(input_size)
        self.in_drop  = nn.Dropout1d(p_in)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.0, bidirectional=False)

        d_in = hidden_size * self.num_directions
        d_mid = max(64, hidden_size)

        self.attn = nn.Linear(d_in, 1)

        self.head = nn.Sequential(
            nn.Linear(d_in, d_mid), nn.GELU(), nn.Dropout(p_post),
            nn.Linear(d_mid, d_mid), nn.GELU(), nn.Dropout(p_post)
        )
        self.proj = nn.Linear(d_in, d_mid) if d_in != d_mid else nn.Identity()
        # → on prédit H * Q sorties (quantiles des rendements cumulés)
        self.out = nn.Linear(d_mid, self.horizon * self.Q)

    def forward(self, x):
        x = self.in_norm(x)
        x = x.transpose(1, 2)                         
        x = self.in_drop(x)
        x = x.transpose(1, 2)        
        out,_ = self.lstm(x)                                  # (B,T,num_directionsr*H)
        e = torch.tanh(self.attn(out))                        # (B,T,1)
        w = torch.softmax(e.squeeze(-1), dim=1)               # (B,T)
        h = (out * w.unsqueeze(-1)).sum(dim=1)                # (B,num_directions*H)

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
    def __init__(self, feature: List[str], target_feature :str, window_size: int=10, hidden_size: int=50, horizon : int = 10, num_layers: int=3, lr: float=0.001, epochs: int=1000,
                 *, plot_training: bool = False, plot_dir: Optional[Union[str, os.PathLike[str]]] = None, walkforward_splits: int=2, walkforward_test_size: Optional[int]= None,
                 residual_boosting: bool = False, boosting_params: Optional[Dict[str, Any]] = None):
        unique_features = []
        for name in feature:
            if name not in unique_features:
                unique_features.append(name)

        if target_feature not in unique_features:
            unique_features.append(target_feature)
            print("on append ici")

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
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.walkforward_splits = walkforward_splits
        self.walkforward_test_size = walkforward_test_size
        self.walkforward_metrics_: List[dict] = []
        self.walkforward_summary_: Optional[dict] = None
        self.training_curves_: Optional[dict] = None
        self.best_split_index_: Optional[int] = None
        self.X_test_scaled_torch: Optional[torch.Tensor] = None
        self.y_test_last: Optional[torch.Tensor] = None
        if plot_dir in (None, False):
            self.plot_dir = None
        elif plot_dir is True:
            self.plot_dir = _default_plots_dir()
        elif isinstance(plot_dir, (str, Path)):
            self.plot_dir = Path(plot_dir)
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise TypeError(f"plot_dir doit être None/False/True/str/Path, reçu {type(plot_dir).__name__}")
        self.plot_dir = None
        self.use_residual_boosting = residual_boosting
        self.boosting_params = dict(boosting_params or {})
        self.residual_models_: Dict[float, LGBMRegressor] = {}
        self._boost_feature_dim: Optional[int] = None
        self.asset_id_map_: Dict[str, str] = {}
        self.asset_names_: List[str] = []


    def fit(self, frames: Union[pd.DataFrame,Iterable[Union[pd.DataFrame, Tuple[str, pd.DataFrame]]],]):

        if isinstance(frames, pd.DataFrame):
            iterable =[frames]

        else:
            iterable = list(frames)

        clean_frames: List[pd.DataFrame] = []

        for frame in iterable:
            if frame is None or len(frame) == 0:
                continue 
            if frame[1].columns.duplicated().any():
                frame[1] = frame[1].loc[:, ~frame.columns.duplicated()].copy() # remove duplicated columns
                print(frame[1].head())
            missing = [col for col in self.feature if col not in frame[1].columns]
            if missing:
                print(f" missing columns: {missing}. Skipping.")
                continue

            if self.target_feature not in frame[1].columns:
                print(
                    f"[WARNING] Frame missing target '{self.target_feature}', skipping"
                )
                continue

            clean_frames.append(frame[1].copy())

        if not clean_frames:
            raise ValueError(
                "Aucun jeu de données exploitable après validation des caractéristiques."
            )
        
        all_dates_set = set()

        for frame in clean_frames:
            all_dates_set.update(frame.index)

        all_unique_sorted_dates = pd.Index(sorted(list(all_dates_set)))

        FINAL_TEST_SIZE = 252

        if len(all_unique_sorted_dates) < FINAL_TEST_SIZE *2:
            raise ValueError("not enough data for a final test set of size {FINAL_TEST_SIZE}")
        print(f'data goes from {all_unique_sorted_dates.min().date()} to {all_unique_sorted_dates.max().date()} with {len(all_unique_sorted_dates)} unique dates across all assets.')

        dates_for_cv = all_unique_sorted_dates[:-FINAL_TEST_SIZE]

        dates_for_final_test = all_unique_sorted_dates[-FINAL_TEST_SIZE:]

        self.final_test_start_date = dates_for_final_test.min()
        self.final_test_end_date = dates_for_final_test.max()

        n_splits = self.walkforward_splits 
        test_size = self.walkforward_test_size

        try:
            tscv = TimeSeriesSplit(n_splits=2, test_size=test_size)
        except ValueError as exc:
            raise ValueError(f"Erreur TimeSeriesSplit sur les dates de CV: {exc}")
        date_splits = []
        # On splitte les indices de notre liste de dates
        for train_idx, val_idx in tscv.split(dates_for_cv):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
                
            # On récupère les dates de début et de fin pour ce pli
            train_start_date = dates_for_cv[train_idx[0]]
            train_end_date = dates_for_cv[train_idx[-1]]
            
            val_start_date = dates_for_cv[val_idx[0]]
            val_end_date = dates_for_cv[val_idx[-1]]
            
            train_fold_dates = (train_start_date, train_end_date)
            val_fold_dates = (val_start_date, val_end_date)
            
            date_splits.append( (train_fold_dates, val_fold_dates) )

        if not date_splits:
            raise ValueError("Impossible de créer des splits walk-forward (données/dates insuffisantes).")

        print(f"\n{len(date_splits)} plis de validation croisée (walk-forward) ont été définis.")
        for i, (train_d, val_d) in enumerate(date_splits, 1):
            print(f"  Pli {i}: Train [{train_d[0].date()} -> {train_d[1].date()}], Val [{val_d[0].date()} -> {val_d[1].date()}]")


        best_artifacts = None
        global_best_loss = float("inf")


        oof_feats_list: List[np.ndarray] = []
        oof_residuals_list: List[np.ndarray] = []

        for split_idx, (train_fold_dates, val_fold_dates) in enumerate(date_splits, 1):
            
            print(f"\n--- Démarrage Pli {split_idx}/{len(date_splits)} ---")
            print(f"  Train: {train_fold_dates[0].date()} -> {train_fold_dates[1].date()}")
            print(f"  Val:   {val_fold_dates[0].date()} -> {val_fold_dates[1].date()}")

            # 3a. Rassembler toutes les données 2D d'entraînement de ce pli
            print("  Ajustement des scalers sur les données d'entraînement...")
            mega_train_df = pd.concat([
                frame.loc[train_fold_dates[0]:train_fold_dates[1]] 
                for frame in clean_frames if not frame.loc[train_fold_dates[0]:train_fold_dates[1]].empty
            ])
            
            if mega_train_df.empty:
                print(f"  [AVERTISSEMENT] Pli {split_idx} ignoré: aucune donnée d'entraînement dans cette plage de dates.")
                continue

            # 3b. Ajuster les scalers
            scaler_x = StandardScaler()
            scaler_y = StandardScaler() 

            scaler_x.fit(mega_train_df[self.feature])
            scaler_y.fit(mega_train_df[[self.target_feature]])

            seq_list_train, targ_list_train = [], []
            seq_list_val, targ_list_val = [], []

            for frame in clean_frames:
                
                # 4a. Isoler les données 2D de cet actif pour ce pli
                df_train_asset = frame.loc[train_fold_dates[0]:train_fold_dates[1]]
                df_val_asset = frame.loc[val_fold_dates[0]:val_fold_dates[1]]

                # 4b. Traiter les données d'entraînement de l'actif
                if not df_train_asset.empty:
                    # Scaler les données 2D en utilisant le scaler AJUSTÉ
                    scaled_features = scaler_x.transform(df_train_asset[self.feature])
                    scaled_target = scaler_y.transform(df_train_asset[[self.target_feature]])
                    
                    # Recréer un DataFrame pour 'make_sequence'
                    df_train_scaled = pd.DataFrame(
                        scaled_features, columns=self.feature, index=df_train_asset.index
                    )
                    # Important : 'make_sequence' va chercher la cible par son nom
                    df_train_scaled[self.target_feature] = scaled_target
                    
                    # Créer les fenêtres 3D sur les données SCALÉES
                    s_train, t_train = make_sequence_multi_horizon(
                        df_train_scaled, self.feature, self.window_size, self.target_feature, self.output_h
                    )
                    
                    if len(s_train) > 0:
                        seq_list_train.append(s_train)
                        # NOTE : t_train sera aussi basé sur la cible SCALÉE
                        # C'est ce que nous voulons.
                        targ_list_train.append(t_train)

                # 4c. Traiter les données de validation de l'actif
                if not df_val_asset.empty:
                    # Scaler les données 2D (TOUJOURS avec .transform() !)
                    scaled_features = scaler_x.transform(df_val_asset[self.feature])
                    scaled_target = scaler_y.transform(df_val_asset[[self.target_feature]])
                    
                    df_val_scaled = pd.DataFrame(
                        scaled_features, columns=self.feature, index=df_val_asset.index
                    )
                    df_val_scaled[self.target_feature] = scaled_target
                    
                    s_val, t_val = make_sequence_multi_horizon(
                        df_val_scaled, self.feature, self.window_size, self.target_feature, self.output_h
                    )
                    
                    if len(s_val) > 0:
                        seq_list_val.append(s_val)
                        targ_list_val.append(t_val)

            # 4d. Empiler toutes les séquences pour ce pli
            if not seq_list_train or not seq_list_val:
                print(f"  [AVERTISSEMENT] Pli {split_idx} ignoré: pas assez de données après fenêtrage.")
                continue

            X_train_scaled_torch = torch.cat(seq_list_train, dim=0).float()
            y_train_scaled_torch = torch.cat(targ_list_train, dim=0).float()
            
            X_val_scaled_torch = torch.cat(seq_list_val, dim=0).float()
            y_val_scaled_torch = torch.cat(targ_list_val, dim=0).float()

            print(f"  -> trzining tensor: X={X_train_scaled_torch.shape}, y={y_train_scaled_torch.shape}")
            print(f"  -> val tensor : X={X_val_scaled_torch.shape}, y={y_val_scaled_torch.shape}")
            
            Ntr, T, F = X_train_scaled_torch.shape
            Nval = X_val_scaled_torch.shape[0]


            train_ds = torch.utils.data.TensorDataset(X_train_scaled_torch.float(), y_train_scaled_torch.float())
            val_ds = torch.utils.data.TensorDataset(X_val_scaled_torch.float(), y_val_scaled_torch.float())

            use_cuda = torch.cuda.is_available()
            device    = torch.device("cuda" if use_cuda else "cpu")

            
            #num_workers = max(2, min(4, (os.cpu_count() or 2)//1))
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            use_pin_memory = available_ram_gb > 10

            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True, num_workers =2, pin_memory=use_pin_memory)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False, num_workers =2, pin_memory=use_pin_memory)


            model = LSTMModelProba(
                    input_size=len(self.feature),
                    hidden_size=self.hidden_size,
                    horizon=self.output_h,
                    quantiles=self.quantiles,
                    num_layers=self.num_layers,
                )

            if use_cuda and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)   
            model = model.to(device)

            def criterion(pred, targ):
                return quantile_loss(pred, targ, quantiles=self.quantiles)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-3)
            accum_steps = 4
            steps_per_epoch = math.ceil(len(train_dl) / accum_steps)
            total_steps = self.epochs * steps_per_epoch

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.3,div_factor=25,final_div_factor=100
            )

            best_val_loss_split, patience, bad = float('inf'), 150, 0
            best_state_split = {k: v.cpu() for k, v in model.state_dict().items()}
            
            lr_to_plot, loss_plot, val_plot = np.zeros(self.epochs), np.zeros(self.epochs), np.zeros(self.epochs)


            """        Entrainement du modele          """
            amp_enabled = torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

            def log_cuda_mem(tag=""):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    print(f"[{tag}] alloc={torch.cuda.memory_allocated()/1e9:.2f} GB | "
                        f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

            
            print("Starting training...")
            for epoch in range(1, self.epochs + 1):
                model.train()
                train_sum = 0.0
                seen_train =0
                optimizer.zero_grad(set_to_none=True)
                for i, (seqs, targs) in enumerate(train_dl, 1):
                    seqs  = seqs.to(self.device, non_blocking=True).float()
                    targs = targs.to(self.device, non_blocking=True).float()

                    if i == 1:  # Premier batch seulement pour éviter spam
                        log_cuda_mem(f"TRAIN epoch {epoch} batch {i}: After data.to(device)")

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(seqs)

                        if i == 1:
                            log_cuda_mem(f"TRAIN epoch {epoch} batch {i}: After forward")

                        loss = criterion(outputs, targs) / accum_steps
            
                    scaler.scale(loss).backward()

                    if i == 1:
                        log_cuda_mem(f"TRAIN epoch {epoch} batch {i}: After backward")

                    if i % accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                    if i == accum_steps:  
                        log_cuda_mem(f"TRAIN epoch {epoch} batch {i}: After optimizer.step")

                    train_sum += (loss.detach() * accum_steps) * seqs.size(0)
                    seen_train += seqs.size(0)
                    del seqs, targs, outputs, loss
                if (i % accum_steps) != 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                train_loss = (train_sum / max(1, seen_train)).item()
                model.eval()

                with torch.no_grad():
                    val_sum = 0.0
                    seen_val = 0
                    for vseqs, vtargs in val_dl:
                        vseqs  = vseqs.to(self.device, non_blocking=True).float()
                        vtargs = vtargs.to(self.device, non_blocking=True).float()
                        with torch.cuda.amp.autocast(enabled=amp_enabled):
                            voutputs = model(vseqs)
                            vloss = criterion(voutputs, vtargs)
                        val_sum  += vloss.detach() * vseqs.size(0)
                        seen_val += vseqs.size(0)
                        del vseqs, vtargs, voutputs, vloss
                val_loss = (val_sum / max(1, seen_val)).item()    
                torch.cuda.empty_cache()
                
                if val_loss < best_val_loss_split:
                    best_val_loss_split = val_loss
                    bad = 0
                    best_state_split = {k: v.cpu() for k,v in model.state_dict().items()}
                else:
                    bad +=1
                    if bad >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                
                lr_to_plot[epoch-1]=optimizer.param_groups[0]['lr']
                loss_plot[epoch-1]=train_loss
                val_plot[epoch-1]=val_loss
                log_cuda_mem(f"EPOCH {epoch}: End (train_loss={train_loss:.4f}, val_loss={val_loss:.4f})")
                if (epoch+1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}]| Train: {train_loss:.4f} | val: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            model.load_state_dict(best_state_split)

            model.eval()
            train_preds_scaled_list: List[np.ndarray] = []
            train_y_scaled_list: List[np.ndarray] = []
            train_eval_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False)
            with torch.no_grad():
                for xb, yb in train_eval_dl:
                    qb = model(xb.to(self.device).float()).cpu().numpy()
                    train_preds_scaled_list.append(qb)
                    train_y_scaled_list.append(yb.numpy())

            train_q_scaled = np.concatenate(train_preds_scaled_list, axis=0)
            train_y_scaled = np.concatenate(train_y_scaled_list, axis=0)

            preds_scaled_list, y_scaled_list = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    qb = model(xb.to(self.device).float()).cpu().numpy()
                    preds_scaled_list.append(qb)
                    y_scaled_list.append(yb.numpy())

            q_scaled = np.concatenate(preds_scaled_list, axis=0)
            y_scaled = np.concatenate(y_scaled_list, axis=0)
            q = inverse_transform_quantiles(q_scaled, scaler_y)
            y = scaler_y.inverse_transform(y_scaled)

            
            X_val_seq_scaled = X_val_scaled_torch.cpu().numpy()          # (N_val, T, F)
            X_val_tab = self._build_boost_features(X_val_seq_scaled)     # (N_val*H, D)
            res_val = y[:, :, None] - q                                  # (N_val, H, Q)

            # remodelage pour empilement OOF
            oof_feats_list.append(X_val_tab)                              # (N_val*H, D)
            oof_residuals_list.append(res_val.reshape(-1, res_val.shape[-1]))  # (N_val*H, Q)            

            metrics = coverage_metrics(
                y_true=y,
                q_low=q[:, :, 0],
                q_med=q[:, :, 1],
                q_high=q[:, :, 2],
            )        


            """        Visualisation de l'entrainement  """

            history_entry = {
                "split": split_idx,
                # "train_range": train_range,
                # "val_range": val_range,
                "val_loss": float(best_val_loss_split),
                "coverage": metrics["coverage"].tolist(),
                "winkler": metrics["interval_score"].tolist(),
                "mpiwidth": metrics["mpiwidth"].tolist(),
            }
            self.walkforward_metrics_.append(history_entry)

            print(
                "Validation coverage:",
                " ".join(f"{c:.3f}" for c in metrics["coverage"]),
            )
            print(
                "Validation Winkler score:",
                " ".join(f"{c:.3f}" for c in metrics["interval_score"]),
            )

            if best_val_loss_split < global_best_loss:
                global_best_loss = best_val_loss_split
                best_artifacts = {
                    "state": copy.deepcopy(best_state_split),
                    "scaler_x": copy.deepcopy(scaler_x),
                    "scaler_y": copy.deepcopy(scaler_y),
                    "X_val_scaled": X_val_scaled_torch.clone(),
                    "y_val_scaled": y_val_scaled_torch.clone(),
                    "X_train_scaled": X_train_scaled_torch.clone(),
                    "y_train_scaled": y_train_scaled_torch.clone(),
                    "train_q_scaled": train_q_scaled.copy(),
                    "train_y_scaled": train_y_scaled.copy(),                    
                    "lr_history": lr_to_plot.copy(),
                    "train_history": loss_plot.copy(),
                    "val_history": val_plot.copy(),
                    "split": split_idx,
                }

        if best_artifacts is None:
            raise ValueError("Aucun modèle valide n'a été entraîné lors du walk-forward.")
        print('one st au vrai modele')

        
        self.model = LSTMModelProba(
            input_size=len(self.feature),
            hidden_size=self.hidden_size,
            horizon=self.output_h,
            quantiles=self.quantiles,
            num_layers=self.num_layers,
        ).to(self.device)
        self.model.load_state_dict(best_artifacts["state"])
        self.model.eval()

        self.scaler_x = best_artifacts["scaler_x"]
        self.scaler_y = best_artifacts["scaler_y"]
        self.X_test_scaled_torch = best_artifacts["X_val_scaled"]
        self.y_test_last = best_artifacts["y_val_scaled"]
        self.best_split_index_ = best_artifacts["split"]

        self.training_curves_ = {
            "lr": best_artifacts["lr_history"],
            "train_loss": best_artifacts["train_history"],
            "val_loss": best_artifacts["val_history"],
            "split": best_artifacts["split"],
        }
        print('on est aux résidus')

        if self.use_residual_boosting:
            sequences_scaled = best_artifacts["X_train_scaled"].cpu().numpy()
            y_train_scaled_np = best_artifacts["y_train_scaled"].cpu().numpy()
            train_q_scaled = best_artifacts["train_q_scaled"]
            self._fit_residual_models(sequences_scaled, y_train_scaled_np, train_q_scaled)


        if self.walkforward_metrics_:
            val_losses = np.array([m["val_loss"] for m in self.walkforward_metrics_], dtype=np.float64)
            coverage_mat = np.array([m["coverage"] for m in self.walkforward_metrics_], dtype=np.float64)
            winkler_mat = np.array([m["winkler"] for m in self.walkforward_metrics_], dtype=np.float64)
            mpiwidth_mat = np.array([m["mpiwidth"] for m in self.walkforward_metrics_], dtype=np.float64)

            self.walkforward_summary_ = {
                "splits": len(self.walkforward_metrics_),
                "best_split": int(self.best_split_index_),
                "mean_val_loss": float(np.nanmean(val_losses)),
                "std_val_loss": float(np.nanstd(val_losses)) if len(val_losses) > 1 else 0.0,
                "mean_coverage": coverage_mat.mean(axis=0).tolist(),
                "mean_winkler": winkler_mat.mean(axis=0).tolist(),
                "mean_mpiwidth": mpiwidth_mat.mean(axis=0).tolist(),
            }
        else:
            self.walkforward_summary_ = None
        print('on va plot')

        if self.training_curves_ and "lr" in self.training_curves_ and len(self.training_curves_["lr"]) > 0:
            lr_hist = np.array(self.training_curves_["lr"], dtype=np.float64)
            train_hist = np.array(self.training_curves_["train_loss"], dtype=np.float64)
            val_hist = np.array(self.training_curves_["val_loss"], dtype=np.float64)

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(lr_hist)
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule (best split)')
            plt.yscale('log')

            plt.subplot(2, 1, 2)
            plt.plot(train_hist, label='Train Loss')
            plt.plot(val_hist, label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Losses (best split)')
            plt.yscale('log')

            plt.tight_layout()
            plt.legend()
            plt.show()
        print('on a plot')


    def _build_boost_features(self, sequences_scaled: np.ndarray) -> np.ndarray:
        seq_arr = np.asarray(sequences_scaled, dtype=np.float32)
        if seq_arr.ndim == 2:
            seq_arr = seq_arr[np.newaxis, ...]
        if seq_arr.ndim != 3:
            raise ValueError("Les séquences doivent avoir la forme (N, T, F).")

        N, T, F = seq_arr.shape
        base = seq_arr.reshape(N, T * F)
        base_rep = np.repeat(base, self.output_h, axis=0)
        horizon_idx = np.tile(np.arange(self.output_h, dtype=np.float32), N).reshape(-1, 1)
        features = np.hstack([base_rep, horizon_idx])
        return features

    def _fit_residual_models(self, sequences_scaled: np.ndarray, y_scaled: np.ndarray, q_scaled: np.ndarray) -> None:
        if not self.use_residual_boosting:
            return
        if sequences_scaled.size == 0:
            return
        print('ici fit_res')

        X_tab = self._build_boost_features(sequences_scaled)
        y_unscaled = self.scaler_y.inverse_transform(y_scaled)
        q_unscaled = inverse_transform_quantiles(q_scaled, self.scaler_y)
        residuals = y_unscaled[:, :, None] - q_unscaled

        models: Dict[float, LGBMRegressor] = {}
        for idx, q_level in enumerate(self.quantiles):
            print("Fitting residual model for quantile", q_level)
            model = LGBMRegressor(objective="quantile", alpha=q_level, n_jobs=1, **self.boosting_params)
            target = residuals[:, :, idx].reshape(-1)
            model.fit(X_tab, target)
            models[q_level] = model
            print(model.get_params())

        self.residual_models_ = models
        self._boost_feature_dim = X_tab.shape[1]


    def predict(self, df: pd.DataFrame, asset: Optional[str] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")
        
        use_asset_map = bool(self.asset_id_map_)
        selected_asset = None

        if use_asset_map:
            if asset is None:
                if len(self.asset_names_) == 1:
                    selected_asset = self.asset_names_[0]
                else:
                    raise ValueError(
                        "Veuillez préciser l'actif pour lequel effectuer la prédiction."
                    )
            else:
                selected_asset = str(asset)

            if selected_asset not in self.asset_id_map_:
                raise ValueError(
                    f"Actif inconnu '{selected_asset}'. Actifs disponibles: {self.asset_names_}"
                )
        elif asset is not None:
            selected_asset = str(asset)

        print("features:", self.feature)
        print("df columns:", df.columns.tolist())

        missing = [col for col in self.feature if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes pour la prédiction: {missing}")

        augmented = df.copy() if use_asset_map else df
        if use_asset_map and selected_asset is not None:
            for other_asset, column in self.asset_id_map_.items():
                augmented[column] = float(other_asset == selected_asset)

        data = augmented[self.feature].values

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
        q_ret = enforce_monotonic_quantiles(q_ret)

        if self.use_residual_boosting and self.residual_models_:
            seq_scaled = last_scaled[np.newaxis, ...]
            X_tab = self._build_boost_features(seq_scaled)
            if self._boost_feature_dim is not None and X_tab.shape[1] != self._boost_feature_dim:
                raise ValueError("Dimension de features incohérente pour le modèle de boosting.")
            residual_preds = []
            for q_idx, q_level in enumerate(self.quantiles):
                model = self.residual_models_.get(q_level)
                if model is None:
                    residual_preds.append(np.zeros(self.output_h, dtype=np.float32))
                    continue
                res = model.predict(X_tab).reshape(self.output_h)
                residual_preds.append(res.astype(np.float32))
            residual_matrix = np.stack(residual_preds, axis=-1)
            q_ret = enforce_monotonic_quantiles(q_ret + residual_matrix)
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
        if self.use_residual_boosting and self.residual_models_:
            seq_scaled = self.X_test_scaled_torch.cpu().numpy()
            X_tab = self._build_boost_features(seq_scaled)
            residual_preds = []
            for q_idx, q_level in enumerate(self.quantiles):
                model = self.residual_models_.get(q_level)
                if model is None:
                    residual_preds.append(np.zeros((seq_scaled.shape[0], self.output_h), dtype=np.float32))
                    continue
                preds = model.predict(X_tab).reshape(seq_scaled.shape[0], self.output_h)
                residual_preds.append(preds.astype(np.float32))
            residual_matrix = np.stack(residual_preds, axis=-1)
            q = enforce_monotonic_quantiles(q + residual_matrix)
        else:
            q = enforce_monotonic_quantiles(q)
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
        residual_models_file = None
        if self.use_residual_boosting and self.residual_models_:
            residual_models_file = "residual_models.pkl"
            with open(os.path.join(dirpath, residual_models_file), "wb") as f:
                pickle.dump(self.residual_models_, f)

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
            "use_residual_boosting": self.use_residual_boosting,
            "boosting_params": self.boosting_params,
            "boost_feature_dim": self._boost_feature_dim,
            "residual_models_file": residual_models_file,            
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
            residual_boosting=meta.get("use_residual_boosting", False),
            boosting_params=meta.get("boosting_params"),            
        )
        # force les mêmes quantiles
        predictor.quantiles = tuple(meta.get("quantiles", (0.025, 0.5, 0.975)))

        predictor.feature = list(meta.get("input_features", predictor.feature))
        if not predictor.feature:
            predictor.feature = list(predictor.feature)

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
        predictor._boost_feature_dim = meta.get("boost_feature_dim")
        predictor.boosting_params = dict(meta.get("boosting_params", {}))
        residual_models_file = meta.get("residual_models_file")
        if predictor.use_residual_boosting and residual_models_file:
            models_path = os.path.join(dirpath, residual_models_file)
            if os.path.exists(models_path):
                with open(models_path, "rb") as f:
                    predictor.residual_models_ = pickle.load(f)

        print(f"📦 Modèle chargé depuis: {dirpath}")
        return predictor

