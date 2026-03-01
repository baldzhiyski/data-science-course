# utils/models/plots/track_similarity.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple

import numpy as np
import pandas as pd



def apply_dark_blue_theme() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#151a23",
        "axes.edgecolor":   "#2a3242",
        "axes.labelcolor":  "#e6e6e6",
        "xtick.color":      "#cfd3dc",
        "ytick.color":      "#cfd3dc",
        "text.color":       "#e6e6e6",
        "grid.color":       "#2a3242",

        "font.family":      "DejaVu Sans",
        "axes.titlesize":   16,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.frameon":   True,
        "legend.facecolor": "#151a23",
        "legend.edgecolor": "#2a3242",
    })
@dataclass(frozen=True)
class TrackSimPlotConfig:
    top_k: int = 5
    sample_points: int = 4000             # for the embedding map
    dpi: int = 260
    title_col: str = "name"               # track title column
    key_cols: Tuple[str, ...] = ("energy", "danceability", "valence", "tempo", "loudness")
    color_main: str = "#339af0"           # points
    color_query: str = "#ffffff"          # query highlight
    color_neighbors: str = "#4ea8de"      # neighbor highlight
    alpha_points: float = 0.10


def _safe_title(df: pd.DataFrame, key, title_col: str) -> str:
    try:
        if title_col in df.columns:
            t = df.loc[key, title_col]
            if isinstance(t, pd.Series):
                t = t.iloc[0]
            return str(t)[:80] if pd.notna(t) else "(no title)"
    except Exception:
        pass
    return "(no title)"


def build_similarity_table(
    track_df: pd.DataFrame,
    query_key,
    similar: List[Tuple[Any, float]],
    title_col: str = "name",
    key_cols: Sequence[str] = ("energy", "danceability", "valence")
) -> pd.DataFrame:
    """
    Returns a compact dataframe for slides:
      role | title | sim | energy | danceability | ...
    """
    rows = []

    q_title = _safe_title(track_df, query_key, title_col)
    q_row = {"role": "QUERY", "title": q_title, "sim": 1.0, "track_key": query_key}
    for c in key_cols:
        if c in track_df.columns:
            q_row[c] = track_df.loc[query_key, c]
    rows.append(q_row)

    for k, s in similar:
        title = _safe_title(track_df, k, title_col)
        row = {"role": "SIMILAR", "title": title, "sim": float(s), "track_key": k}
        for c in key_cols:
            if c in track_df.columns:
                row[c] = track_df.loc[k, c]
        rows.append(row)

    out = pd.DataFrame(rows)

    # clean formatting
    out["sim"] = out["sim"].map(lambda v: f"{v:.3f}")
    for c in key_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

    # drop key in final slide table (keep only if you want)
    return out.drop(columns=["track_key"], errors="ignore")


def save_similarity_table_png(
    table_df: pd.DataFrame,
    out_path: Path,
    title: str = "Track Similarity – Example Query",
    cfg: Optional[TrackSimPlotConfig] = None
) -> Path:
    import matplotlib.pyplot as plt

    cfg = cfg or TrackSimPlotConfig()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # kompaktere Höhe
    fig_h = 0.55 * len(table_df) + 1.2
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    # Titel oben links
    ax.text(0.01, 1.08, title, transform=ax.transAxes, fontsize=15, fontweight="bold")

    # Table fills area via bbox
    tbl = ax.table(
        cellText=table_df.values.tolist(),
        colLabels=list(table_df.columns),
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.92],   # <-- KEY: nutzt fast die ganze Fläche
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.35)

    # Styling
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a3242")

        if r == 0:  # header
            cell.set_facecolor("#1f2a44")
            cell.set_text_props(color="white", weight="bold")
        else:
            role = str(table_df.iloc[r-1]["role"])
            if role == "QUERY":
                cell.set_facecolor("#152233")
                cell.set_text_props(color="white", weight="bold")
            else:
                # zebra striping
                cell.set_facecolor("#151a23" if r % 2 == 0 else "#121620")
                cell.set_text_props(color="#e6e6e6")

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def save_embedding_map_png(
    embeddings: np.ndarray,
    track_df: pd.DataFrame,
    out_path: Path,
    query_key,
    neighbor_keys: Sequence,
    cfg: Optional[TrackSimPlotConfig] = None,
    color_by: Optional[str] = None,
) -> Path:
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    cfg = cfg or TrackSimPlotConfig()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = embeddings.shape[0]
    rng = np.random.RandomState(42)
    idx_all = np.arange(n)

    # sample points for readability
    idx_plot = rng.choice(idx_all, size=min(cfg.sample_points, n), replace=False)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    emb2 = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Base scatter
    if color_by and color_by in track_df.columns:
        cvals = pd.to_numeric(track_df.iloc[idx_plot][color_by], errors="coerce").fillna(0).to_numpy()
        sc = ax.scatter(
            emb2[idx_plot, 0], emb2[idx_plot, 1],
            c=cvals,
            s=9,
            cmap="turbo",
            alpha=0.22,           # etwas höher -> besser sichtbar
            linewidths=0
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(color_by, color="#e6e6e6")
        cb.ax.yaxis.set_tick_params(color="#cfd3dc")
        for t in cb.ax.get_yticklabels():
            t.set_color("#cfd3dc")
    else:
        ax.scatter(
            emb2[idx_plot, 0], emb2[idx_plot, 1],
            s=9,
            alpha=0.10,
            color="#3b82f6",
            linewidths=0
        )

    # Highlight query
    handles = []
    labels = []

    if query_key in track_df.index:
        qi = track_df.index.get_loc(query_key)
        hq = ax.scatter(
            emb2[qi, 0], emb2[qi, 1],
            s=220, facecolors="white", edgecolors="black", linewidths=1.5,
            zorder=5
        )
        handles.append(hq); labels.append("Query")

    # Highlight neighbors
    neighbor_pts = []
    for nk in neighbor_keys:
        if nk in track_df.index:
            ni = track_df.index.get_loc(nk)
            neighbor_pts.append((emb2[ni, 0], emb2[ni, 1]))

    if neighbor_pts:
        xs, ys = zip(*neighbor_pts)
        hn = ax.scatter(
            xs, ys,
            s=140, facecolors="#60a5fa", edgecolors="black", linewidths=1.2,
            zorder=4
        )
        handles.append(hn); labels.append("Top-K Neighbors")

    # cosmetics
    ax.set_title("Track Similarity – Audio Embedding Map (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path

def save_track_similarity_slide_assets(
    runner,
    sim_models: Dict[str, Any],
    sim_artifact: Dict[str, Any],
    track_df: pd.DataFrame,
    out_dir: Path,
    cfg: Optional[TrackSimPlotConfig] = None,
    query_key=None,
    color_by: Optional[str] = "energy",
) -> Dict[str, Path]:
    """
    One-call entrypoint from notebook:
      - picks a query track
      - finds top-k neighbors
      - saves (1) similarity table png (2) embedding map png
    """
    cfg = cfg or TrackSimPlotConfig()
    apply_dark_blue_theme()
    out_dir.mkdir(parents=True, exist_ok=True)

    q = query_key if query_key is not None else sim_artifact.get("example_key", None)
    if q is None:
        raise ValueError("No query_key available. Provide query_key or ensure sim_artifact['example_key'] exists.")

    similar = runner.get_similar(
        track_key=q,
        track_index=track_df.index,
        embeddings=sim_models["embeddings"],
        top_k=cfg.top_k
    )

    neighbor_keys = [k for k, _ in similar]

    # Table image
    tdf = build_similarity_table(
        track_df=track_df,
        query_key=q,
        similar=similar,
        title_col=cfg.title_col,
        key_cols=[c for c in cfg.key_cols if c in track_df.columns]
    )
    table_path = save_similarity_table_png(
        tdf,
        out_path=out_dir / "track_similarity_example_table.png",
        title="Track Similarity – Example Query (Top-K Neighbors)",
        cfg=cfg
    )

    # Embedding map image
    map_path = save_embedding_map_png(
        embeddings=sim_models["embeddings"],
        track_df=track_df,
        out_path=out_dir / "track_similarity_embedding_map.png",
        query_key=q,
        neighbor_keys=neighbor_keys,
        cfg=cfg,
        color_by=color_by
    )

    return {"table_png": table_path, "embedding_map_png": map_path}