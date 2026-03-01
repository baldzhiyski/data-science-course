from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from ..data.splits import time_fraction_split
from ..evaluation.metrics import regression_metrics, find_best_threshold_f1

from .tuning_utils import (
    create_optuna_study,
    xgb_device_kwargs,
    suggest_xgb_regression_params,
    suggest_xgb_classification_params,
    collect_tuning_artifacts,
    EARLY_STOPPING_ROUNDS,
)

"""
Task: Artist-Trajektorie (Wachstum & Breakout).

Tuning-Strategie:
-----------------
Dieser Trainer hat ZWEI Teilaufgaben mit separaten Tuning-Zielen:

1. Wachstums-Regression:
   - Primäre Metrik: MAE (log-transformiertes Target)
   - XGBoost-Objective: reg:absoluteerror
   - Early Stopping: 300 Runden mit 20k Estimator-Obergrenze

2. Breakout-Klassifikation:
   - Primäre Metrik: PR-AUC (Average Precision)
   - Sekundäre Metrik: Bester F1 bei optimalem Threshold
   - Ungleichgewicht: scale_pos_weight um Klassen-Ratio getuned
   - Early Stopping: 300 Runden mit aucpr eval_metric

Gemeinsame Anti-Overfit-Maßnahmen:
---------------------------------
- max_depth 3-7 (beschränkte Kapazität)
- min_child_weight 3-50 (verhindert kleine Blätter)
- Sinnvolle Regularisierung (reg_lambda >= 1)
- Subsampling 0.6-0.9 für Varianz-Reduktion

Inputs:
artist_panel mit monatlichen Aggregaten pro Artist.

Targets:
- y_growth: Wachstum (Regression; häufig log1p-transformiert)
- y_breakout: Breakout-Event (Klassifikation)

Wichtig:
- Sortierung nach Zeitspalte ist Pflicht, damit der Split zeitlich konsistent ist.
- Drop von IDs/Targets/Time aus Features, um Leakage zu vermeiden.
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def time_split_by_months(df: pd.DataFrame, time_col: str,
                         val_frac=0.10, test_frac=0.15,
                         min_val_months=3, min_test_months=3):
    months = pd.Series(df[time_col].unique()).dropna().sort_values().tolist()
    n_months = len(months)
    if n_months < (min_val_months + min_test_months + 1):
        raise ValueError(f"Not enough unique months: {n_months}")

    n_test = max(min_test_months, int(round(n_months * test_frac)))
    n_val  = max(min_val_months,  int(round(n_months * val_frac)))

    n_test = min(n_test, n_months - 2)
    n_val  = min(n_val,  n_months - n_test - 1)

    test_months = set(months[-n_test:])
    val_months  = set(months[-(n_test + n_val):-n_test])
    train_months = set(months[:-(n_test + n_val)])

    idx_tr = df.index[df[time_col].isin(train_months)].to_numpy()
    idx_va = df.index[df[time_col].isin(val_months)].to_numpy()
    idx_te = df.index[df[time_col].isin(test_months)].to_numpy()
    return idx_tr, idx_va, idx_te


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_breakout_label_per_year(df: pd.DataFrame, year_col: str, growth_col: str,
                                  top_pct: float = 0.10, min_year_n: int = 200) -> pd.Series:
    """
    Breakout=1 wenn growth in Top-X% innerhalb des Jahres liegt.
    Für Jahre mit zu wenig Samples -> NaN (oder 0, je nach Strategie).
    """
    out = pd.Series(np.nan, index=df.index, dtype="float")
    for y, g in df.groupby(year_col):
        if len(g) < min_year_n:
            continue  # zu wenig Daten -> instabil
        thr = g[growth_col].quantile(1 - top_pct)
        out.loc[g.index] = (g[growth_col] >= thr).astype(int)
    return out


@dataclass
class ArtistTrajectoryTrainer:
    seed: int = 42

    time_col: str = "release_month_ts"
    growth_col: str = "y_growth"
    year_col: str = "year"

    # If you already have y_breakout, we'll recompute it safely
    breakout_col: str = "y_breakout"

    drop_cols: tuple = ("artist_id",)

    # breakout config
    breakout_top_pct: float = 0.10
    breakout_min_year_n: int = 200

    def fit_eval(self, artist_panel: pd.DataFrame):
        required = {self.time_col, self.growth_col, self.year_col}
        missing = required - set(artist_panel.columns)
        if missing:
            raise ValueError(f"artist_panel missing columns: {missing}")

        df = artist_panel.sort_values(self.time_col).reset_index(drop=True)

        # --- target: growth (keep negatives)
        y_growth_raw = df[self.growth_col].astype(float).to_numpy()
        y_growth = np.arcsinh(y_growth_raw)

        # --- breakout label: recompute safely (fixes your 'max=1.0 per year' issue)
        y_break_series = build_breakout_label_per_year(
            df, year_col=self.year_col, growth_col=self.growth_col,
            top_pct=self.breakout_top_pct, min_year_n=self.breakout_min_year_n
        )

        # drop rows where breakout undefined (years too small)
        keep_mask = y_break_series.notna()
        df = df.loc[keep_mask].reset_index(drop=True)
        y_growth = y_growth[keep_mask.to_numpy()]
        y_break = y_break_series.loc[keep_mask].astype(int).to_numpy()

        # --- features (numeric/bool)
        X = df.select_dtypes(include=["number", "bool"]).copy()

        # drop targets/time/id
        X = X.drop(columns=[self.growth_col, self.breakout_col, self.time_col], errors="ignore")
        X = X.drop(columns=list(self.drop_cols), errors="ignore")

        # CRITICAL: drop any future columns
        X = X.drop(columns=[c for c in X.columns if c.lower().startswith("future_") or "future" in c.lower()],
                   errors="ignore")

        X = X.fillna(0)

        # split by months
        idx_tr, idx_va, idx_te = time_split_by_months(df, self.time_col, val_frac=0.10, test_frac=0.15)

        Xtr, Xva, Xte = X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]
        ytr_g, yva_g, yte_g = y_growth[idx_tr], y_growth[idx_va], y_growth[idx_te]
        ytr_b, yva_b, yte_b = y_break[idx_tr], y_break[idx_va], y_break[idx_te]

        # --- growth model
        reg = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
        )
        reg.fit(Xtr, ytr_g, eval_set=[(Xva, yva_g)], verbose=False)

        pred_g = reg.predict(Xte)
        g_m = regression_metrics(yte_g, pred_g)
        growth_metrics = {
            "MAE_asinh": g_m["MAE"],
            "RMSE_asinh": g_m["RMSE"],
            "R2_asinh": g_m["R2"],
            "n_test": int(len(yte_g)),
            "n_features": int(X.shape[1]),
        }

        # --- breakout model
        pos_rate = float(np.mean(ytr_b))
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
        )
        clf.fit(Xtr, ytr_b, eval_set=[(Xva, yva_b)], verbose=False)

        proba = clf.predict_proba(Xte)[:, 1]
        breakout_metrics = {
            "ROC_AUC": float(roc_auc_score(yte_b, proba)) if len(np.unique(yte_b)) > 1 else float("nan"),
            "PR_AUC": float(average_precision_score(yte_b, proba)) if len(np.unique(yte_b)) > 1 else float("nan"),
            "breakout_rate_test": float(np.mean(yte_b)),
            "n_test": int(len(yte_b)),
            "scale_pos_weight": float(scale_pos_weight),
            "n_features": int(X.shape[1]),
            "breakout_min_year_n": int(self.breakout_min_year_n),
        }

        plot_pack = {
            "feature_cols": list(X.columns),
            "year_counts": df.groupby(self.year_col).size().to_dict(),
            "y_growth_raw_stats": {
                "mean": float(np.mean(df[self.growth_col])),
                "std": float(np.std(df[self.growth_col])),
                "min": float(np.min(df[self.growth_col])),
                "max": float(np.max(df[self.growth_col])),
            },
        }

        return {"growth_model": reg, "breakout_model": clf}, growth_metrics, breakout_metrics, plot_pack
    def tune(self, artist_panel: "pd.DataFrame", n_trials: int = 40, device: str = "cpu"):
        """
        Hyperparameter-Tuning für Artist-Trajektorie (Wachstum + Breakout).

        Optimierungs-Strategie:
        ----------------------
        Zwei separate Studies werden sequentiell ausgeführt:

        1. Wachstum (Regression):
           - Ziel: Minimiere MAE auf log-transformiertem Target
           - XGBoost-Objective: reg:absoluteerror
           - Early Stopping: 300 Runden Geduld

        2. Breakout (Klassifikation):
           - Ziel: Maximiere PR-AUC auf Validierung
           - Sekundär: F1 + optimalen Threshold tracken
           - Ungleichgewicht: scale_pos_weight um Klassen-Ratio getuned
           - Early Stopping: 300 Runden mit aucpr eval_metric

        Anti-Overfit-Maßnahmen:
        ----------------------
        - n_estimators=20000 mit Early Stopping (nicht getuned)
        - max_depth 3-7 (beschränkte Kapazität)
        - min_child_weight 3-50 (verhindert kleine Blätter)
        - Sinnvolle Regularisierung (reg_lambda >= 1)
        - Subsampling 0.6-0.9 für Varianz-Reduktion
        - Reproduzierbar: Fester TPE-Sampler-Seed
        """
        # Erforderliche Spalten validieren
        time_col, growth_col, breakout_col = self.time_col, self.growth_col, self.breakout_col
        req = {time_col, growth_col, breakout_col}
        missing = req - set(artist_panel.columns)
        if missing:
            raise ValueError(f"artist_panel fehlende Spalten: {missing}")

        df = artist_panel.sort_values(time_col).reset_index(drop=True)

        # Targets vorbereiten
        y_growth_raw = df[growth_col].astype(float).values
        y_growth = np.log1p(np.clip(y_growth_raw, a_min=0, a_max=None))
        y_break = df[breakout_col].astype(int).values

        # Features vorbereiten (Targets, Zeit, Identifikatoren droppen)
        X = df.select_dtypes(include=["number", "bool"]).drop(
            columns=[time_col, growth_col, breakout_col] + list(self.drop_cols),
            errors="ignore"
        ).fillna(0)

        # Zeitbasierter Split
        idx_tr, idx_va, idx_te = time_fraction_split(
            len(df), val_frac=0.10, test_frac=0.15, min_val=30, min_test=30
        )
        Xtr, Xva = X.iloc[idx_tr], X.iloc[idx_va]
        ytr_g, yva_g = y_growth[idx_tr], y_growth[idx_va]
        ytr_b, yva_b = y_break[idx_tr], y_break[idx_va]

        # Klassen-Ungleichgewicht für Breakout berechnen
        pos = float(np.sum(ytr_b == 1))
        neg = float(np.sum(ytr_b == 0))
        base_spw = (neg / pos) if pos > 0 else 1.0

        # Sekundäre Metriken tracken
        growth_best_r2 = {"r2": float("-inf"), "trial": -1}
        breakout_best_f1 = {"f1": 0.0, "threshold": 0.5, "trial": -1}

        # --- Wachstums-Objective (MAE minimieren) ---
        def obj_growth(trial):
            nonlocal growth_best_r2

            params = suggest_xgb_regression_params(trial)

            reg = XGBRegressor(
                random_state=self.seed,
                n_jobs=4,
                objective="reg:absoluteerror",
                eval_metric="mae",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                **xgb_device_kwargs(device),
                **params,
            )
            reg.fit(Xtr, ytr_g, eval_set=[(Xva, yva_g)], verbose=False)

            if getattr(reg, "best_iteration", None) is not None:
                pred = reg.predict(Xva, iteration_range=(0, reg.best_iteration + 1))
            else:
                pred = reg.predict(Xva)

            metrics = regression_metrics(yva_g, pred)
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            if r2 > growth_best_r2["r2"]:
                growth_best_r2 = {"r2": r2, "trial": trial.number}

            trial.set_user_attr("r2", r2)
            trial.set_user_attr("rmse", metrics["RMSE"])
            trial.set_user_attr("best_iteration", getattr(reg, "best_iteration", None))

            return mae

        # --- Breakout-Objective (PR-AUC maximieren) ---
        def obj_breakout(trial):
            nonlocal breakout_best_f1

            params = suggest_xgb_classification_params(trial, base_scale_pos_weight=base_spw)

            clf = XGBClassifier(
                random_state=self.seed,
                n_jobs=4,
                eval_metric="aucpr",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                **xgb_device_kwargs(device),
                **params,
            )
            clf.fit(Xtr, ytr_b, eval_set=[(Xva, yva_b)], verbose=False)

            if getattr(clf, "best_iteration", None) is not None:
                proba = clf.predict_proba(Xva, iteration_range=(0, clf.best_iteration + 1))[:, 1]
            else:
                proba = clf.predict_proba(Xva)[:, 1]

            pr_auc = average_precision_score(yva_b, proba)

            thr, f1 = find_best_threshold_f1(yva_b, proba)
            if f1 > breakout_best_f1["f1"]:
                breakout_best_f1 = {"f1": f1, "threshold": thr, "trial": trial.number}

            trial.set_user_attr("best_f1", f1)
            trial.set_user_attr("best_threshold", thr)
            trial.set_user_attr("best_iteration", getattr(clf, "best_iteration", None))

            return pr_auc

        # Wachstums-Tuning ausführen
        study_g = create_optuna_study(
            direction="minimize",
            seed=self.seed,
            study_name="artist_trajectory_growth",
        )
        n_growth_trials = max(10, n_trials // 2)
        study_g.optimize(obj_growth, n_trials=n_growth_trials, show_progress_bar=False)

        # Breakout-Tuning ausführen
        study_b = create_optuna_study(
            direction="maximize",
            seed=self.seed + 1,
            study_name="artist_trajectory_breakout",
        )
        n_breakout_trials = max(10, n_trials // 2)
        study_b.optimize(obj_breakout, n_trials=n_breakout_trials, show_progress_bar=False)

        # Artefakte sammeln
        growth_result = collect_tuning_artifacts(
            study=study_g,
            metric_name="mae",
            device=device,
            best_iteration=study_g.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "best_r2_across_trials": growth_best_r2["r2"],
                "best_trial_rmse": study_g.best_trial.user_attrs.get("rmse"),
                "target_transform": "log1p",
            },
        )

        breakout_result = collect_tuning_artifacts(
            study=study_b,
            metric_name="pr_auc",
            device=device,
            best_iteration=study_b.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "best_f1_across_trials": breakout_best_f1["f1"],
                "best_f1_threshold": breakout_best_f1["threshold"],
                "class_imbalance_ratio": base_spw,
            },
        )

        return {
            "growth": growth_result.to_dict(),
            "breakout": breakout_result.to_dict(),
            "device": device,
        }
