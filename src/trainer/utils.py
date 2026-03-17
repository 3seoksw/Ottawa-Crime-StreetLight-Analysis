import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import ConfusionMatrixDisplay
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    ScalarEvent,
)
from scipy.interpolate import make_interp_spline


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


def plot_training_results(path: str):
    ea = EventAccumulator(path)
    ea.Reload()
    plt.rcParams.update({"font.size": 14})
    _plot_loss_results(ea, "cls", path)
    _plot_loss_results(ea, "count", path)


def _plot_loss_results(ea: EventAccumulator, loss_type: str, path: str):
    plt.figure(figsize=(10, 6))
    sets = ["train", "val"]
    for dset in sets:
        loss = ea.Scalars(f"{dset}/loss_{loss_type}")
        values, steps = _event_to_val_step(loss)
        steps = np.array(steps)

        interp_spline = make_interp_spline(steps, values)
        X = np.linspace(steps.min(), steps.max(), 150)
        Y = interp_spline(X)

        n = 4
        plt.plot(X[::n], Y[::n], label=f"{dset.title()}")
    loss = ea.Scalars(f"test/loss_{loss_type}")
    values, steps = _event_to_val_step(loss)
    test_loss = values[0]
    plt.axhline(test_loss, color="red", linestyle="--", label="Test")

    if loss_type == "count":
        loss_fn_name = "Poisson NLL Loss"
        loss_fn = r"$\lambda - y \log(\lambda)$"
    else:
        loss_fn_name = "Binary Cross-Entropy Loss"
        loss_fn = r"$-y\log(p) - (1 - y)\log(1 - p)$"

    plt.legend()
    plt.title(f"{loss_fn_name}", fontsize=18)
    plt.xlabel("Steps", fontsize=16)
    plt.ylabel(rf"{loss_fn}")
    plt.tight_layout()
    plt.savefig(f"{path}/loss_{loss_type}.png", dpi=300)
    plt.close()


def plot_performance(path: str):
    plt.figure(figsize=(10, 6))

    ea = EventAccumulator(path)
    ea.Reload()
    sets = ["train", "val"]
    for dset in sets:
        acc = ea.Scalars(f"{dset}/acc")
        values, steps = _event_to_val_step(acc)

        plt.plot(steps, values, label=f"{dset.title()}")

    acc = ea.Scalars("test/acc")
    values, steps = _event_to_val_step(acc)
    test_acc = values[0]
    plt.axhline(test_acc, color="red", linestyle="--", label="Test")

    plt.legend()
    plt.title("Crime Count Accuracy", fontsize=18)
    plt.xlabel("Steps", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{path}/acc.png", dpi=300)
    plt.close()


def plot_shap(
    test_samples: torch.Tensor,
    explainer: shap.GradientExplainer,
    features: list[str],
    path: str,
    pred_type: str,
):
    fig = plt.figure(figsize=(12, 8))

    indices = torch.randperm(len(test_samples))[:50]
    samples = test_samples[indices]
    feature_values = samples.detach().numpy()

    feature_values_scaled = feature_values.copy()
    for i in range(feature_values.shape[1]):
        col = feature_values[:, i]
        p5, p95 = np.percentile(col, [5, 95])
        col = np.clip(col, p5, p95)
        col = (col - p5) / (p95 - p5 + 1e-8)
        feature_values_scaled[:, i] = col

    shap_values = explainer.shap_values(samples)
    shap_values = shap_values.squeeze()
    p1, p99 = np.percentile(shap_values, [1, 99])
    shap_values_clipped = np.clip(shap_values, p1, p99)

    shap.plots.violin(
        shap_values_clipped,
        features=feature_values_scaled,
        feature_names=_rename_features(features),
        plot_type="violin",
        sort=False,
    )
    fig.savefig(f"{path}/shap_{pred_type}", dpi=300)
    plt.close()


def _event_to_val_step(events: list[ScalarEvent]):
    values = [e.value for e in events]
    steps = [e.step for e in events]
    return values, steps


def _rename_features(features: list[str]):
    return [f.replace("_", " ").title() for f in features]


if __name__ == "__main__":
    # plot_training_results("runs/20260316_154859")
    plot_performance("runs/20260316_154859")
