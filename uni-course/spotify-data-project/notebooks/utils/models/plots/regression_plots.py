# utils/viz/regression_plots.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re

import numpy as np
import matplotlib.pyplot as plt


# ---------- Theme ----------
def apply_dark_blue_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2b2b2b",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "#444444",
    })


@dataclass(frozen=True)
class RegressionPlotConfig:
    dpi: int = 220
    color_points: str = "#339af0"
    color_diag: str = "#4ea8de"
    alpha: float = 0.12
    point_size: int = 8


def _map_fx_to_names(items: List[Tuple[str, float]], feature_names: Optional[List[str]]) -> List[Tuple[str, float]]:
    if not feature_names:
        return items
    mapped = []
    for k, v in items:
        m = re.match(r"f(\d+)", k)
        if m:
            idx = int(m.group(1))
            name = feature_names[idx] if idx < len(feature_names) else k
            mapped.append((name, v))
        else:
            mapped.append((k, v))
    return mapped


# ---------- Plots ----------
def save_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    cfg: RegressionPlotConfig,
    subtitle: Optional[str] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(y_true, y_pred, alpha=cfg.alpha, s=cfg.point_size, color=cfg.color_points)

    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], linewidth=2, color=cfg.color_diag)

    ax.set_title(title if not subtitle else f"{title}\n{subtitle}")
    ax.set_xlabel("Actual (0–100)")
    ax.set_ylabel("Predicted (0–100)")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def save_xgb_feature_importance(
    model,
    out_path: Path,
    cfg: RegressionPlotConfig,
    feature_names: Optional[List[str]] = None,
    top_n: int = 15,
    importance_type: str = "gain",
    title: Optional[str] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    score = booster.get_score(importance_type=importance_type)
    items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    items = _map_fx_to_names(items, feature_names)

    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color=cfg.color_points)
    ax.set_title(title or f"XGBoost Feature Importance ({importance_type}) – Top {top_n}")
    ax.set_xlabel(importance_type.capitalize())
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def save_success_pct_report_plots(
    model,
    metrics: Dict[str, Any],
    plot_pack: Dict[str, Any],
    out_dir: Path,
    cfg: Optional[RegressionPlotConfig] = None,
) -> Dict[str, Path]:
    """
    One-call entrypoint:
      - predicted vs actual (test)
      - feature importance top 15
    """
    cfg = cfg or RegressionPlotConfig()
    apply_dark_blue_theme()

    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(plot_pack["y_true"])
    y_pred = np.asarray(plot_pack["y_pred"])

    subtitle = None
    # if you have MAE/R2 in metrics, show them
    mae = metrics.get("mae") or metrics.get("MAE")
    r2 = metrics.get("r2") or metrics.get("R2")
    if mae is not None or r2 is not None:
        parts = []
        if mae is not None: parts.append(f"MAE={float(mae):.2f}")
        if r2 is not None: parts.append(f"R²={float(r2):.2f}")
        subtitle = " | ".join(parts)

    p1 = save_pred_vs_actual(
        y_true, y_pred,
        out_path=out_dir / "pct_pred_vs_actual.png",
        title="Success Percentile: Predicted vs Actual (Test)",
        subtitle=subtitle,
        cfg=cfg
    )

    p2 = save_xgb_feature_importance(
        model,
        out_path=out_dir / "pct_feature_importance_top15.png",
        cfg=cfg,
        feature_names=plot_pack.get("feature_names"),
        top_n=15,
        importance_type="gain",
        title="XGBoost Feature Importance (Gain) – Top 15"
    )

    return {"pred_vs_actual": p1, "feature_importance": p2}