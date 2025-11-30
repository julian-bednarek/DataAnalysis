import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile, gaussian_kde
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
)

# Local imports
from const import *
from models import *
from dataset import WaldoImageDataset, TEST_TRANSFORM


MODEL_PATH_DEFAULT = "waldo_model.pth"


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def load_model(model_path, device):
    print(f"Loading model from {model_path} ...")
    model = ShallowConvAutoencoder(latent_dim=2048, image_size=IMAGE_SIZE, channels=3)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_mse_scores(model, loader, device):
    scores = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            outputs = model(images)

            diff = (images - outputs) ** 2
            mse_per_image = diff.view(diff.size(0), -1).mean(dim=1)
            scores.extend(mse_per_image.cpu().numpy().tolist())

    return np.array(scores)


def compute_per_image_maps(model, loader, device, max_images=None):
    maps = []
    with torch.no_grad():
        count = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            outputs = model(images)
            batch_size = images.shape[0]
            for i in range(batch_size):
                orig = images[i].detach().cpu().numpy()
                recon = outputs[i].detach().cpu().numpy()

                # Math must match get_mse_scores
                se = (orig - recon) ** 2
                err_map = np.mean(se, axis=0)
                mse_scalar = float(np.mean(se))

                maps.append((orig, recon, err_map, mse_scalar))
                count += 1
                if max_images is not None and count >= max_images:
                    return maps
    return maps


def save_single_map(data_tuple, save_path, title_desc):
    orig, recon, err_map, mse_scalar = data_tuple
    orig_img = np.transpose(orig, (1, 2, 0))
    recon_img = np.transpose(recon, (1, 2, 0))

    em = err_map
    em_norm = (em - em.min()) / (em.max() - em.min() + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.clip(orig_img, 0, 1))
    axes[0].set_title(f"Input Image\n({title_desc})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.clip(recon_img, 0, 1))
    axes[1].set_title("Autoencoder Reconstruction", fontsize=11)
    axes[1].axis("off")

    im = axes[2].imshow(em_norm, cmap="jet")
    axes[2].set_title(f"Error Heatmap\nMSE: {mse_scalar:.6f}", fontsize=11, color="red")
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046)
    cbar.set_label("Relative Reconstruction Error", rotation=270, labelpad=15)

    plt.suptitle(f"{title_desc}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_plot(
    model_path,
    normal_path,
    anomaly_path,
    device,
    out_dir="eval_plots",
    percentile_threshold=95,
):
    ensure_dir(out_dir)
    model = load_model(model_path, device)

    print(f"Loading Normal data: {normal_path}")
    normal_set = WaldoImageDataset(normal_path, transform=TEST_TRANSFORM)
    normal_loader = DataLoader(normal_set, batch_size=32, shuffle=False)

    print(f"Loading Anomaly data: {anomaly_path}")
    anomaly_set = WaldoImageDataset(anomaly_path, transform=TEST_TRANSFORM)
    anomaly_loader = DataLoader(anomaly_set, batch_size=32, shuffle=False)

    print("Calculating scalar reconstruction errors (Consistency Guaranteed)...")
    normal_errors = get_mse_scores(model, normal_loader, device)
    anomaly_errors = get_mse_scores(model, anomaly_loader, device)

    if len(anomaly_errors) > len(normal_errors) * 10:
        anomaly_errors_vis = anomaly_errors[: len(normal_errors) * 10]
    else:
        anomaly_errors_vis = anomaly_errors

    print(f"Counts -> Normal: {len(normal_errors)}, Anomaly: {len(anomaly_errors)}")
    print(
        f"Mean MSE -> Normal: {np.mean(normal_errors):.6f}, Anomaly: {np.mean(anomaly_errors):.6f}"
    )

    thr_percentile = scoreatpercentile(normal_errors, percentile_threshold)

    y_true_all = np.concatenate(
        [np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))]
    )
    y_scores_all = np.concatenate([normal_errors, anomaly_errors])

    # Youden J Statistic for optimal threshold (sensitivity + specificity - 1)
    fpr, tpr, roc_thresholds = roc_curve(y_true_all, y_scores_all)
    youden_j = tpr - fpr
    idx = np.argmax(youden_j)
    thr_youden = roc_thresholds[idx]

    print(f"Thr (Percentile-{percentile_threshold}): {thr_percentile:.6f}")
    print(f"Thr (Youden J): {thr_youden:.6f}")

    def make_preds(errors, thr):
        return (errors > thr).astype(int)

    y_pred_youden = np.concatenate(
        [make_preds(normal_errors, thr_youden), make_preds(anomaly_errors, thr_youden)]
    )

    # --- PLOT 1: ROC Curve ---
    roc_auc = roc_auc_score(y_true_all, y_scores_all)
    plt.figure(figsize=(8, 7))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.scatter(
        fpr[idx],
        tpr[idx],
        c="red",
        s=100,
        zorder=10,
        label=f"Optimal Thr ({thr_youden:.4f})",
    )
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title(
        "ROC Curve: Can the model rank Waldo higher than Background?", fontsize=14
    )
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve_enhanced.png"), dpi=150)
    plt.close()

    # --- PLOT 2: Precision-Recall ---
    precision_vals, recall_vals, _ = precision_recall_curve(y_true_all, y_scores_all)
    ap = average_precision_score(y_true_all, y_scores_all)
    baseline = len(anomaly_errors) / len(y_true_all)

    plt.figure(figsize=(8, 7))
    plt.plot(
        recall_vals,
        precision_vals,
        color="purple",
        lw=2,
        label=f"PR Curve (AP = {ap:.3f})",
    )
    plt.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        label=f"No Skill (Baseline={baseline:.2f})",
    )
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title("Precision-Recall: Performance on Imbalanced Data", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve_enhanced.png"), dpi=150)
    plt.close()

    # --- PLOT 3: Descriptive Histogram + KDE ---
    plt.figure(figsize=(12, 7))
    bins = 60
    plt.hist(
        normal_errors,
        bins=bins,
        density=True,
        alpha=0.3,
        color="green",
        label="Normal (Background)",
    )
    plt.hist(
        anomaly_errors_vis,
        bins=bins,
        density=True,
        alpha=0.3,
        color="red",
        label="Anomaly (Waldo)",
    )

    xs = np.linspace(min(y_scores_all), max(y_scores_all), 500)
    try:
        kde_norm = gaussian_kde(normal_errors)
        plt.plot(xs, kde_norm(xs), color="green", lw=2)
        kde_anom = gaussian_kde(anomaly_errors_vis)
        plt.plot(xs, kde_anom(xs), color="red", lw=2)
    except:
        pass

    plt.axvline(
        thr_youden,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary (Youden)",
    )

    max_dens = (
        max(kde_norm(xs).max(), kde_anom(xs).max()) if "kde_norm" in locals() else 1.0
    )
    plt.text(
        thr_youden,
        max_dens * 1.05,
        f" Threshold\n {thr_youden:.4f}",
        ha="center",
        va="bottom",
        color="black",
    )

    plt.title("Separation Analysis: Distribution of Reconstruction Errors", fontsize=15)
    plt.xlabel(
        "Reconstruction Error (MSE)",
        fontsize=12,
    )
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_distribution_descriptive.png"), dpi=150)
    plt.close()

    # --- PLOT 4: Boxplot + Jitter ---
    plt.figure(figsize=(9, 6))

    bp = plt.boxplot(
        [normal_errors, anomaly_errors_vis],
        labels=["Normal", "Anomaly"],
        patch_artist=True,
        showfliers=False,
    )

    colors = ["lightgreen", "salmon"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    def add_jitter(data, index, color):
        x = np.random.normal(index, 0.04, size=len(data))
        plt.scatter(x, data, alpha=0.2, s=5, color=color)

    add_jitter(normal_errors, 1, "green")
    add_jitter(anomaly_errors_vis, 2, "red")

    plt.title("Statistical Spread: Normal vs Anomaly MSE", fontsize=14)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplot_with_jitter.png"), dpi=150)
    plt.close()

    # --- PLOT 6: Confusion Matrix with Metrics ---
    report = classification_report(
        y_true_all, y_pred_youden, target_names=["Normal", "Waldo"], output_dict=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    cm = confusion_matrix(y_true_all, y_pred_youden)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Waldo"]
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d")
    disp.ax_.set_title(f"Confusion Matrix (Threshold = {thr_youden:.4f})", fontsize=13)

    norm_stats = report["Normal"]
    waldo_stats = report["Waldo"]

    stats_text = (
        "PER-CLASS METRICS\n"
        "-------------------\n\n"
        "NORMAL (Background):\n"
        f"  Precision : {norm_stats['precision']:.3f}\n"
        f"  Recall    : {norm_stats['recall']:.3f}\n"
        f"  F1-Score  : {norm_stats['f1-score']:.3f}\n\n"
        "WALDO (Anomaly):\n"
        f"  Precision : {waldo_stats['precision']:.3f}\n"
        f"  Recall    : {waldo_stats['recall']:.3f}\n"
        f"  F1-Score  : {waldo_stats['f1-score']:.3f}"
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    plt.figtext(
        0.72,
        0.5,
        stats_text,
        fontsize=11,
        verticalalignment="center",
        bbox=props,
        fontfamily="monospace",
    )

    plt.subplots_adjust(right=0.65)
    plt.savefig(os.path.join(out_dir, "confusion_matrix_youden.png"), dpi=150)
    plt.close()

    # --- Image Reconstructions ---
    print("\nGenerating descriptive reconstruction maps...")

    # 1. Standard summary plots
    maps_normal = compute_per_image_maps(
        model, DataLoader(normal_set, batch_size=8), device, max_images=100
    )
    maps_normal_sorted = sorted(maps_normal, key=lambda t: t[3], reverse=True)

    # 2. Get ALL anomaly maps for detection saving
    print("Computing maps for ALL anomaly images...")
    maps_anom = compute_per_image_maps(
        model, DataLoader(anomaly_set, batch_size=8), device, max_images=None
    )

    maps_anom_sorted = sorted(maps_anom, key=lambda t: t[3], reverse=True)
    maps_anom_low = sorted(maps_anom, key=lambda t: t[3], reverse=False)

    for i in range(min(3, len(maps_normal_sorted))):
        save_single_map(
            maps_normal_sorted[i],
            os.path.join(out_dir, f"Normal_Hardest_{i+1}.png"),
            "Hardest Normal Image",
        )

    for i in range(min(3, len(maps_anom_sorted))):
        save_single_map(
            maps_anom_sorted[i],
            os.path.join(out_dir, f"Waldo_Detected_{i+1}.png"),
            "Waldo (Easiest to Spot)",
        )

    for i in range(min(3, len(maps_anom_low))):
        save_single_map(
            maps_anom_low[i],
            os.path.join(out_dir, f"Waldo_Missed_{i+1}.png"),
            "Waldo (Hardest to Spot)",
        )

    # 3. SAVE ALL DETECTED WALDOS
    detected_dir = os.path.join(out_dir, "detected_waldos")
    ensure_dir(detected_dir)
    print(
        f"\nSaving ALL detected Waldo images (MSE > {thr_youden:.4f}) to '{detected_dir}'..."
    )

    detected_count = 0
    for i, item in enumerate(maps_anom):
        mse = item[3]
        if mse > thr_youden:
            save_name = os.path.join(
                detected_dir, f"detected_waldo_{i}_mse_{mse:.6f}.png"
            )
            save_single_map(item, save_name, f"Detected Waldo #{i}")
            detected_count += 1

    print(f"Successfully saved {detected_count} detected Waldo images.")
    print(f"Evaluation Complete. Results in: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    model_path = MODEL_PATH_DEFAULT
    normal_path = TRAIN_DATA_PATH
    anomaly_path = TEST_DATA_PATH
    device = DEVICE

    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        normal_path = sys.argv[2]
    if len(sys.argv) >= 4:
        anomaly_path = sys.argv[3]
    if len(sys.argv) >= 5:
        device = sys.argv[4]

    evaluate_and_plot(model_path, normal_path, anomaly_path, device)
