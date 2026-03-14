import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm: np.ndarray, normalize: bool, save_dir: str):
    label = ""
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        label = "norm"
    labels = ["Zero", "Non-zero"]
    display = ConfusionMatrixDisplay(cm, display_labels=labels)
    display.plot()
    plt.savefig(f"{save_dir}/confusion_matrix_{label}", dpi=300)
    plt.close()


def plot_attention_heatmap(weights: np.ndarray, features: list[str], save_dir: str):
    features = _rename_features(features)
    n_heads, n_features, _ = weights.shape
    assert len(features) == n_features

    vmin, vmax = weights.min(), weights.max()
    for i in range(n_heads):
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(weights[i], cmap="jet", vmin=vmin, vmax=vmax)

        ax.set_title(f"Attention Heatmap - Head {i + 1}")
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_yticklabels(features)

        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/attn_h{i+1}.png", dpi=300)
        plt.close()

    # Average attention (mean)
    attn_mean = weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(attn_mean, cmap="jet", vmin=vmin, vmax=vmax)

    ax.set_title("Attention Heatmap - Mean Across Heads")
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/attn_mean.png", dpi=300)
    plt.close()

    # Attention variability (std)
    attn_std = weights.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(attn_std, cmap="magma")

    ax.set_title("Attention Heatmap - Std Across Heads")
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/attn_std.png", dpi=300)
    plt.close()


def _rename_features(features: list[str]):
    return [f.replace("_", " ").title() for f in features]
