from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
from ..data.splits import cohort_time_split
from ..data.preprocess import TabularPreprocessor

from .tuning_utils import (
    create_optuna_study,
    xgb_device_kwargs,
    suggest_xgb_ranker_params,
    collect_tuning_artifacts,
    mean_ndcg_at_k,
    pct_to_relevance,
    EARLY_STOPPING_ROUNDS,
)


def _to_relevance_0_31(y_pct):
    # Einfache Abbildung: Perzentil -> Relevanz-Bin 0..31
    y = np.asarray(y_pct)
    rel = np.floor(y / (100.0 / 32.0)).astype(int)
    return np.clip(rel, 0, 31).astype(np.int32)


"""
Task: Erfolgs-Perzentil-Ranking (Learning-to-Rank).

Tuning-Strategie:
-----------------
- Primäre Metrik: mittlerer NDCG@k auf Validierungs-Split
- XGBoost-Objective: rank:ndcg (LambdaMART-ähnlich)
- Early Stopping: 300 Runden mit n_estimators=20000 Obergrenze
- Query-Gruppen: Kohortenbasiert (Tracks derselben Kohorte konkurrieren)

Warum NDCG@k?
- Standard-Ranking-Metrik, die Position und Relevanz berücksichtigt
- Gewichtet niedrigere Positionen logarithmisch ab
- k kontrolliert Tiefe der Ranking-Qualitäts-Optimierung

Gruppen-Behandlung:
------------------
- Gruppen definiert durch cohort_ym (im selben Zeitraum veröffentlicht)
- Gruppen müssen zusammenhängend sein (nach Kohorte sortiert vor Training)
- Gruppengrößen werden validiert um Datenlänge zu entsprechen

Relevanz-Binning:
----------------
- Perzentile (0-100) werden auf Relevanz-Labels (0-4) abgebildet
- 5-Bin-Schema: <50=0, <80=1, <95=2, <99=3, >=99=4
- Erfasst Erfolgsverteilung (Top 1% bekommt höchste Relevanz)

Ziel:
Modelliert die Rangordnung von Tracks innerhalb ihrer Kohorte basierend auf Erfolgperzentilen.
Hinweise:
- Nutzt Learning-to-Rank-Methoden (z.B. LambdaMART) für kohortenbasierte Rangvorhersagen.
- Kohortenbasierte Zeit-Splits sind wichtig, um Daten-Leakage zu vermeiden.
- Relevanz-Binning wird verwendet, um kontinuierliche Perzentile in diskrete Relevanzstufen umzuwandeln.
"""


@dataclass
class RankerTrainer:
    seed: int = 42
    k_eval: int = 10

    def fit_eval(self, ds_success_pct, params: dict | None = None):
        y_rel = _to_relevance_0_31(ds_success_pct.y)

        idx_tr, idx_va, idx_te = cohort_time_split(ds_success_pct.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr = ds_success_pct.X.iloc[idx_tr]
        Xva = ds_success_pct.X.iloc[idx_va]
        Xte = ds_success_pct.X.iloc[idx_te]

        ytr = y_rel[idx_tr]
        yva = y_rel[idx_va]
        yte = y_rel[idx_te]

        meta_tr = ds_success_pct.meta.iloc[idx_tr].copy()
        meta_va = ds_success_pct.meta.iloc[idx_va].copy()
        meta_te = ds_success_pct.meta.iloc[idx_te].copy()

        # OPTIONAL (aber empfohlen): innerhalb jedes Splits nach cohort_ym sortieren,
        # damit group sizes sicher zur Reihenfolge passen
        def sort_by_cohort(X, y, meta):
            order = meta["cohort_ym"].astype(str).argsort()
            return X.iloc[order], y[order], meta.iloc[order].reset_index(drop=True)

        Xtr, ytr, meta_tr = sort_by_cohort(Xtr, ytr, meta_tr)
        Xva, yva, meta_va = sort_by_cohort(Xva, yva, meta_va)
        Xte, yte, meta_te = sort_by_cohort(Xte, yte, meta_te)

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)
        Xte_p = ct.transform(Xte)

        def group_sizes(meta):
            # Reihenfolge passt jetzt, weil wir sortiert haben
            return meta.groupby("cohort_ym").size().astype(int).tolist()

        gtr = group_sizes(meta_tr)
        gva = group_sizes(meta_va)

        model = XGBRanker(
            objective="rank:ndcg",
            random_state=self.seed,
            n_jobs=4,
            **(params or dict(
                learning_rate=0.05,
                max_depth=6,
                n_estimators=800,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
            ))
        )

        model.fit(Xtr_p, ytr, group=gtr, eval_set=[(Xva_p, yva)], eval_group=[gva], verbose=False)

        # Scores auf Test
        scores = model.predict(Xte_p)

        # NDCG@K pro Kohorte
        ndcg_rows = []
        for cohort, idxs in meta_te.groupby("cohort_ym").groups.items():
            idxs = np.asarray(list(idxs))
            if idxs.size < self.k_eval:
                continue  # zu kleine Kohorte -> kein ndcg@10 sinnvoll

            y_true = yte[idxs].reshape(1, -1)
            y_score = scores[idxs].reshape(1, -1)

            nd = ndcg_score(y_true, y_score, k=self.k_eval)
            ndcg_rows.append({
                "cohort_ym": cohort,
                "n": int(idxs.size),
                f"ndcg@{self.k_eval}": float(nd),
            })

        ndcg_df = pd.DataFrame(ndcg_rows).sort_values("cohort_ym").reset_index(drop=True)

        metrics = {
            f"mean_ndcg@{self.k_eval}": float(ndcg_df[f"ndcg@{self.k_eval}"].mean()) if not ndcg_df.empty and f"ndcg@{self.k_eval}" in ndcg_df.columns else float("nan"),
            "k": int(self.k_eval),
            "n_test": int(len(meta_te)),
            "n_cohorts_test": int(ndcg_df.shape[0]),
        }

        # Feature-Namen (falls verfügbar)
        feat_names = None
        if hasattr(ct, "get_feature_names_out"):
            try:
                feat_names = list(ct.get_feature_names_out())
            except Exception:
                feat_names = None

        plot_pack = {
            "scores": np.asarray(scores),
            "y_true_rel": np.asarray(yte),
            "meta_te": meta_te,           # enthält cohort_ym
            "ndcg_by_cohort": ndcg_df,
            "feature_names": feat_names
        }
        return model, metrics, plot_pack

    def tune(self, ds, n_trials: int = 30, device: str = "cpu", k: int = 10):
        """
        Hyperparameter-Tuning für Ranking-Modell (XGBRanker).

        Optimierungs-Strategie:
        ----------------------
        - Primäres Ziel: Maximiere mittleren NDCG@k auf Validierungs-Split
        - XGBoost-Objective: rank:ndcg (LambdaMART-ähnlich)
        - eval_metric: ndcg@k (für Early Stopping verwendet)
        - Reproduzierbar: Fester TPE-Sampler-Seed

        Gruppen-Behandlung:
        ------------------
        - Gruppen definiert durch cohort_ym (zusammenhängend nach Sortierung)
        - Gruppengrößen vor Training validiert
        - Per-Gruppen NDCG@k berechnet, dann gemittelt

        Anti-Overfit-Maßnahmen:
        ----------------------
        - n_estimators=20000 mit 300 Runden Early Stopping
        - max_depth 3-8 (leicht höhere Kapazität für Ranking erlaubt)
        - max_leaves 16-256 (Hist-Regularisierer)
        - Sinnvolle Regularisierung (reg_lambda >= 1)
        - Subsampling 0.6-0.9 für Varianz-Reduktion
        """
        def build_group_sizes(meta_df):
            return meta_df.groupby("cohort_ym", sort=False).size().to_list()

        # ---- Split (zeitbasiert nach Kohorte) ----
        idx_tr, idx_va, _ = cohort_time_split(ds.meta, cohort_col="cohort_ym", n_val=3, n_test=6)

        Xtr, ytr, mtr = ds.X.iloc[idx_tr], ds.y.iloc[idx_tr], ds.meta.iloc[idx_tr]
        Xva, yva, mva = ds.X.iloc[idx_va], ds.y.iloc[idx_va], ds.meta.iloc[idx_va]

        # ---- Gruppen zusammenhängend machen (nach Kohorte sortieren) ----
        tr_order = mtr.sort_values("cohort_ym").index
        va_order = mva.sort_values("cohort_ym").index

        Xtr, ytr, mtr = ds.X.loc[tr_order], ds.y.loc[tr_order], ds.meta.loc[tr_order]
        Xva, yva, mva = ds.X.loc[va_order], ds.y.loc[va_order], ds.meta.loc[va_order]

        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)

        group_tr = build_group_sizes(mtr)
        group_va = build_group_sizes(mva)

        # Gruppengrößen validieren
        if sum(group_tr) != len(Xtr):
            raise ValueError(f"group_tr sum {sum(group_tr)} != len(Xtr) {len(Xtr)}")
        if sum(group_va) != len(Xva):
            raise ValueError(f"group_va sum {sum(group_va)} != len(Xva) {len(Xva)}")

        # Perzentile zu Relevanz-Labels konvertieren mit shared Utility
        ytr_rel = pct_to_relevance(ytr.to_numpy(), n_bins=5)
        yva_rel = pct_to_relevance(yva.to_numpy(), n_bins=5)

        # Zusätzliche Metriken über Trials tracken
        best_trial_info = {"ndcg": 0.0, "best_iteration": None, "trial": -1}

        def objective(trial):
            nonlocal best_trial_info

            # Suchraum von unified utilities holen
            params = suggest_xgb_ranker_params(trial)

            ranker = XGBRanker(
                objective="rank:ndcg",
                random_state=self.seed,
                n_jobs=4,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                eval_metric=f"ndcg@{k}",
                **xgb_device_kwargs(device),
                **params,
            )

            ranker.fit(
                Xtr_p, ytr_rel,
                group=group_tr,
                eval_set=[(Xva_p, yva_rel)],
                eval_group=[group_va],
                verbose=False,
            )

            # Mit bester Iteration vorhersagen wenn Early Stopping ausgelöst
            best_iter = getattr(ranker, "best_iteration", None)
            if best_iter is not None:
                y_score = ranker.predict(Xva_p, iteration_range=(0, best_iter + 1))
            else:
                y_score = ranker.predict(Xva_p)

            # Mittleren NDCG@k über Gruppen berechnen mit shared Utility
            ndcg = mean_ndcg_at_k(yva_rel, y_score, group_va, k=k)

            # Bestes Ergebnis tracken
            if ndcg > best_trial_info["ndcg"]:
                best_trial_info = {"ndcg": ndcg, "best_iteration": best_iter, "trial": trial.number}

            # In Trial für Analyse speichern
            trial.set_user_attr("best_iteration", best_iter)
            trial.set_user_attr("n_groups_val", len(group_va))

            return ndcg  # NDCG@k maximieren

        # Study mit reproduzierbarem Seeding erstellen
        study = create_optuna_study(
            direction="maximize",
            seed=self.seed,
            study_name="ranker_tuning",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Artefakte sammeln
        result = collect_tuning_artifacts(
            study=study,
            metric_name=f"ndcg@{k}",
            device=device,
            best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            extra_metrics={
                "k": k,
                "n_groups_train": len(group_tr),
                "n_groups_val": len(group_va),
                "relevance_bins": 5,
            },
        )

        return result.to_dict()
