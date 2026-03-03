from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error

from ..data.splits import cohort_time_split
from ..data.preprocess import TabularPreprocessor


def _to_relevance_0_31(y):
    """
    Mapping: 0..100 (Popularity oder Percentile) -> Relevanz 0..31
    (identisch zur Idee im Ranker)
    """
    y = np.asarray(y, dtype=float)
    rel = np.floor(y / (100.0 / 32.0)).astype(int)
    return np.clip(rel, 0, 31).astype(np.int32)


@dataclass
class PopularityRegressorTrainer:
    seed: int = 42
    k_eval: int = 10

    def fit_eval(self, ds_popularity, params: dict | None = None):
        """
        Trainiert Regressor auf absolute Popularität.
        Eval: Regression-as-Ranking via NDCG@K innerhalb von Kohorten.
        """
        # ---- Split (gleiches Schema wie RankerTrainer) ----
        idx_tr, idx_va, idx_te = cohort_time_split(ds_popularity.meta, "cohort_ym", n_val=3, n_test=6)

        Xtr = ds_popularity.X.iloc[idx_tr]
        Xva = ds_popularity.X.iloc[idx_va]
        Xte = ds_popularity.X.iloc[idx_te]

        ytr = ds_popularity.y.iloc[idx_tr].to_numpy(dtype=float)
        yva = ds_popularity.y.iloc[idx_va].to_numpy(dtype=float)
        yte = ds_popularity.y.iloc[idx_te].to_numpy(dtype=float)

        meta_tr = ds_popularity.meta.iloc[idx_tr].copy()
        meta_va = ds_popularity.meta.iloc[idx_va].copy()
        meta_te = ds_popularity.meta.iloc[idx_te].copy()

        # ---- Optional, aber konsistent: sort by cohort_ym (wie bei dir) ----
        def sort_by_cohort(X, y, meta):
            order = meta["cohort_ym"].astype(str).argsort()
            return X.iloc[order], y[order], meta.iloc[order].reset_index(drop=True)

        Xtr, ytr, meta_tr = sort_by_cohort(Xtr, ytr, meta_tr)
        Xva, yva, meta_va = sort_by_cohort(Xva, yva, meta_va)
        Xte, yte, meta_te = sort_by_cohort(Xte, yte, meta_te)

        # ---- Preprocessing (wie bei Ranker) ----
        pre = TabularPreprocessor(model_kind="tree", text_cols=[])
        ct = pre.build(Xtr)

        Xtr_p = ct.fit_transform(Xtr)
        Xva_p = ct.transform(Xva)
        Xte_p = ct.transform(Xte)

        # ---- Model (ohne Tuning) ----
        model = XGBRegressor(
            random_state=self.seed,
            n_jobs=4,
            **(params or dict(
                learning_rate=0.05,
                max_depth=6,
                n_estimators=800,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                reg_lambda=1.0,
            ))
        )

        model.fit(Xtr_p, ytr, eval_set=[(Xva_p, yva)], verbose=False)

        # ---- Predict ----
        y_pred = model.predict(Xte_p)

        # ---- NDCG@K pro Kohorte (Regression-as-Ranking) ----
        # Ground truth als Relevanz-Bins, damit ndcg_score sinnvoll ist.
        yte_rel = _to_relevance_0_31(yte)

        ndcg_rows = []
        for cohort, idxs in meta_te.groupby("cohort_ym").groups.items():
            idxs = np.asarray(list(idxs))
            if idxs.size < self.k_eval:
                continue

            y_true = yte_rel[idxs].reshape(1, -1)
            y_score = y_pred[idxs].reshape(1, -1)

            nd = ndcg_score(y_true, y_score, k=self.k_eval)
            ndcg_rows.append({
                "cohort_ym": cohort,
                "n": int(idxs.size),
                f"ndcg@{self.k_eval}": float(nd),
            })

        ndcg_df = pd.DataFrame(ndcg_rows).sort_values("cohort_ym").reset_index(drop=True)

        # ---- Zusätzliche Regressionsmetriken (optional) ----
        rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
        mae = float(mean_absolute_error(yte, y_pred))

        metrics = {
            f"mean_ndcg@{self.k_eval}": float(ndcg_df[f"ndcg@{self.k_eval}"].mean()) if not ndcg_df.empty else float("nan"),
            "rmse": rmse,
            "mae": mae,
            "k": int(self.k_eval),
            "n_test": int(len(meta_te)),
            "n_cohorts_test": int(ndcg_df.shape[0]),
        }

        feat_names = None
        if hasattr(ct, "get_feature_names_out"):
            try:
                feat_names = list(ct.get_feature_names_out())
            except Exception:
                feat_names = None

        plot_pack = {
            "y_pred": np.asarray(y_pred),
            "y_true_pop": np.asarray(yte),
            "y_true_rel": np.asarray(yte_rel),
            "meta_te": meta_te,
            "ndcg_by_cohort": ndcg_df,
            "feature_names": feat_names,
        }

        return model, metrics, plot_pack