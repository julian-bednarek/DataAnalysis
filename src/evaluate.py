#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from scipy.stats import scoreatpercentile
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Local imports
from const import *
from models import ShallowConvAutoencoder
from dataset import WaldoImageDataset, calculate_errors, TEST_TRANSFORM

MODEL_PATH = "waldo_model.pth"


def evaluate(model_path, normal_path, anomaly_path, device):
    print(f"Initializing model and loading weights from {model_path}...")
    model = ShallowConvAutoencoder(latent_dim=2048, image_size=IMAGE_SIZE, channels=3)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading Normal (NotWaldo) data from: {normal_path}")
    normal_set = WaldoImageDataset(normal_path, transform=TEST_TRANSFORM)
    normal_loader = DataLoader(normal_set, batch_size=32, shuffle=False)

    print(f"Loading Anomaly (Waldo) data from: {anomaly_path}")
    anomaly_set = WaldoImageDataset(anomaly_path, transform=TEST_TRANSFORM)
    anomaly_loader = DataLoader(anomaly_set, batch_size=32, shuffle=False)

    print("\nCalculating Reconstruction Errors...")
    normal_errors = calculate_errors(model, normal_loader, device)
    anomaly_errors = calculate_errors(model, anomaly_loader, device)

    if len(anomaly_errors) > len(normal_errors) * 5:
        anomaly_errors = anomaly_errors[: len(normal_errors) * 5]

    print(f"Normal (Not Waldo) Mean Error: {np.mean(normal_errors):.2f}")
    print(f"Anomaly (Waldo) Mean Error:    {np.mean(anomaly_errors):.2f}")

    threshold = scoreatpercentile(normal_errors, 95)
    print(f"\nDetermined Threshold (95th percentile of normal data): {threshold:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(
        normal_errors,
        bins=50,
        alpha=0.7,
        label="Normal (Trained)",
        density=True,
        color="tab:blue",
    )
    plt.hist(
        anomaly_errors,
        bins=50,
        alpha=0.6,
        label="Anomaly (Waldo)",
        density=True,
        color="orange",
    )
    plt.axvline(
        threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.0f})"
    )

    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.legend()
    plt.show()

    y_true = np.concatenate(
        [np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))]
    )

    normal_preds = (normal_errors > threshold).astype(int)
    anomaly_preds = (anomaly_errors > threshold).astype(int)
    y_pred = np.concatenate([normal_preds, anomaly_preds])

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- Confusion Matrix ---")
    print(f"True Negatives (Normal correctly ignored):  {tn}")
    print(f"False Positives (Normal flagged as Waldo):  {fp}")
    print(f"False Negatives (Waldo missed):             {fn}")
    print(f"True Positives (Waldo detected):            {tp}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Waldo"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    y_scores = np.concatenate([normal_errors, anomaly_errors])
    auc = roc_auc_score(y_true, y_scores)
    print(f"\nAUC-ROC Score: {auc:.4f}")


if __name__ == "__main__":
    evaluate(MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, DEVICE)
