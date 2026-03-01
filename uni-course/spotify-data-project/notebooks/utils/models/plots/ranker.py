import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# ---------- Theme ----------
def apply_dark_blue_theme() -> None:
    """Global plotting theme for presentation-ready dark charts."""
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2b2b2b",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "#444444",
        "axes.edgecolor": "#aaaaaa",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.facecolor": "#2b2b2b",
        "legend.edgecolor": "#2b2b2b",
    })


@dataclass(frozen=True)
class RankerPlotConfig:
    k: int = 10
    max_cohorts: int = 24
    min_cohort_n: int = 5
    color_main: str = "#339af0"
    color_mean: str = "#4ea8de"
    dpi: int = 260
    top_n: int = 12                   # <<< Top 10/12 wirkt meistens besser als 15
    max_label_len: int = 26           # <<< damit Labels nicht zu lang werden


# ---------- Helpers ----------
def _map_fx_to_names(items: List[Tuple[str, float]], feature_names: Optional[List[str]]) -> List[Tuple[str, float]]:
    """Map XGBoost feature ids f0..fN to real feature names if available."""
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


def _clean_feature_name(name: str, max_len: int = 26) -> str:
    """
    Make feature names slide-friendly:
    - remove pipeline prefixes: num__, cat__, bool__, remainder__
    - simplify double underscores
    - shorten long names
    """
    # remove common prefixes
    name = name.replace("num__", "").replace("cat__", "").replace("bool__", "").replace("remainder__", "")
    # remove accidental triple underscores from your CT naming
    name = name.replace("___", "__")
    # compact
    name = name.replace("__", "_")

    # small cosmetic replacements
    name = name.replace("mood_", "mood:")
    name = name.replace("_", " ")

    # trim
    name = name.strip()
    if len(name) > max_len:
        name = name[: max_len - 1] + "…"
    return name


def _format_cohort_label(x) -> str:
    """
    If cohort_ym is like 202102 -> '2021-02'.
    Otherwise return as str.
    """
    s = str(x)
    if len(s) == 6 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s


# ---------- Plots ----------
def save_ndcg_by_cohort_plot(
    ndcg_by_cohort,
    out_dir: Path,
    cfg: RankerPlotConfig,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ndcg_by_cohort.copy()
    col = f"ndcg@{cfg.k}"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in ndcg_by_cohort.")

    # 1) Filter small cohorts if 'n' exists (THIS fixes ugly drops)
    if "n" in df.columns:
        df = df[df["n"] >= cfg.min_cohort_n].copy()

    if df.empty:
        raise ValueError("ndcg_by_cohort is empty after filtering. Lower min_cohort_n or check data.")

    # 2) Show only last N cohorts (readability)
    df_show = df.tail(cfg.max_cohorts) if len(df) > cfg.max_cohorts else df

    # nicer x labels
    x_labels = [_format_cohort_label(v) for v in df_show["cohort_ym"]]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    # line + markers (markers make it easier to read)
    ax.plot(x_labels, df_show[col], linewidth=2.6, color=cfg.color_main, marker="o", markersize=5)

    mean_val = float(df[col].mean())
    ax.axhline(mean_val, linewidth=2.0, color=cfg.color_mean, alpha=0.9, label=f"Mean {col}: {mean_val:.3f}")

    ax.set_title(f"NDCG@{cfg.k} pro Kohorte (Test)")
    ax.set_xlabel("Release-Kohorte (YYYY-MM)")
    ax.set_ylabel(f"NDCG@{cfg.k}")
    ax.set_ylim(0.0, 1.02)

    ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=30)

    # small note if we filtered
    if "n" in ndcg_by_cohort.columns:
        ax.text(
            0.01, 0.02,
            f"Filter: n ≥ {cfg.min_cohort_n} Tracks/Kohorte",
            transform=ax.transAxes,
            fontsize=10,
            color="#cfcfcf"
        )

    ax.legend(loc="upper right")
    fig.tight_layout()

    out = out_dir / f"rank_ndcg_at_{cfg.k}_by_cohort.png"
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def save_feature_importance_plot(
    model,
    out_dir: Path,
    cfg: RankerPlotConfig,
    feature_names: Optional[List[str]] = None,
    importance_type: str = "gain",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    score = booster.get_score(importance_type=importance_type)
    items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[: cfg.top_n]
    items = _map_fx_to_names(items, feature_names)

    # clean names + reverse for barh
    names = [_clean_feature_name(k, max_len=cfg.max_label_len) for k, _ in items][::-1]
    vals  = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(names, vals, color=cfg.color_main)

    ax.set_title(f"Learning-to-Rank: Wichtigste Features ({importance_type}) – Top {cfg.top_n}")
    ax.set_xlabel(importance_type.capitalize())
    ax.grid(alpha=0.18)

    # value labels (makes it much easier to interpret)
    xmax = max(vals) if vals else 1.0
    ax.set_xlim(0, xmax * 1.12)
    for b in bars:
        w = b.get_width()
        ax.text(w + xmax * 0.02, b.get_y() + b.get_height()/2, f"{w:.2f}", va="center", fontsize=10, color="#eaeaea")

    fig.tight_layout()
    out = out_dir / "rank_feature_importance_top.png"
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def save_ranker_report_plots(
    rank_model,
    rank_metrics: Dict[str, Any],
    rank_plot: Dict[str, Any],
    out_dir: Path,
    cfg: Optional[RankerPlotConfig] = None,
) -> Dict[str, Path]:
    """One-call entrypoint from notebooks."""
    cfg = cfg or RankerPlotConfig(k=int(rank_metrics.get("k", 10)))
    apply_dark_blue_theme()

    ndcg_path = save_ndcg_by_cohort_plot(rank_plot["ndcg_by_cohort"], out_dir, cfg)
    imp_path = save_feature_importance_plot(
        rank_model,
        out_dir,
        cfg,
        feature_names=rank_plot.get("feature_names"),
        importance_type="gain",
    )
    return {"ndcg_by_cohort": ndcg_path, "feature_importance": imp_path}