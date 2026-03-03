import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc

def save_hit_plots(y_true, proba, threshold, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Confusion Matrix
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Hit Prediction – Confusion Matrix")
    p1 = out_dir / "hit_confusion_matrix.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Precision–Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Hit Prediction – Precision-Recall Curve")
    p2 = out_dir / "hit_precision_recall_curve.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) ROC Curve (optional)
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Hit Prediction – ROC Curve")
    ax.legend()
    p3 = out_dir / "hit_roc_curve.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"confusion_matrix": p1, "pr_curve": p2, "roc_curve": p3}

def save_xgb_feature_importance(model, feature_names, out_path, top_n=25):
    # XGBoost Booster importance (gain)
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")  # dict: f0, f1, ...

    # Map f{i} -> feature name
    items = []
    for k, v in score.items():
        if k.startswith("f"):
            idx = int(k[1:])
            name = feature_names[idx] if feature_names and idx < len(feature_names) else k
            items.append((name, v))

    items = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in items][::-1]
    vals = [x[1] for x in items][::-1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.barh(names, vals)
    ax.set_title("Feature Importance (Gain) – Top Features")
    ax.set_xlabel("Gain")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path