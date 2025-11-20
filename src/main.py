#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import scoreatpercentile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from torchvision.utils import save_image

from const import *
from models import ShallowConvAutoencoder

# ==========================================
# 1. CONFIGURATION
# ==========================================

Model = ShallowConvAutoencoder

REAL_TRAIN_PATH = f"../datasets/Hey-Waldo/{IMAGE_SIZE}{IMAGE_TYPE['file']}/waldo"
REAL_TEST_PATH = f"../datasets/Hey-Waldo/{IMAGE_SIZE}{IMAGE_TYPE['file']}/notwaldo"

BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 200
WEIGHT_DECAY: float = 1e-4

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# ==========================================
# 2. DATASET
# ==========================================


class WaldoImageDataset(Dataset):
    def __init__(self, root_dir, file_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if file_list:
            self.image_paths = file_list
        else:
            self.image_paths = [
                os.path.join(root_dir, fname)
                for fname in os.listdir(root_dir)
                if fname.endswith(IMAGE_EXTENSION)
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return dummy if broken
            return torch.zeros(3, 64, 64), torch.zeros(3, 64, 64), img_path

        if self.transform:
            image = self.transform(image)

        # Return Path for debugging
        return image, image, img_path


# ==========================================
# 3. TRAINING
# ==========================================


def train(model, data_root, epochs, device):
    print(f"Preparing data from {data_root}...")
    all_files = [
        os.path.join(data_root, f)
        for f in os.listdir(data_root)
        if f.endswith(IMAGE_EXTENSION)
    ]
    random.shuffle(all_files)

    val_count = max(2, int(0.1 * len(all_files)))
    train_files = all_files[val_count:]
    val_files = all_files[:val_count]

    print(
        f"Training on {len(train_files)} images. Validating on {len(val_files)} images."
    )

    train_dataset = WaldoImageDataset(
        data_root, file_list=train_files, transform=TRAIN_TRANSFORM
    )
    val_dataset = WaldoImageDataset(
        data_root, file_list=val_files, transform=TEST_TRANSFORM
    )

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=2000)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, _, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, _, _ in val_loader:
                val_images = val_images.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_images).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        saved_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_waldo_model.pth")
            saved_msg = "[Saved Best]"

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} {saved_msg}"
            )

    print("Training complete. Loading best model.")
    model.load_state_dict(torch.load("best_waldo_model.pth"))
    return model


# ==========================================
# 4. EVALUATION (DEBUG MODE)
# ==========================================


def calculate_errors_with_filenames(model, dataloader, device):
    model.eval()
    errors = []
    filenames = []
    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for images, _, paths in dataloader:
            images = images.to(device)
            reconstructed = model(images)
            pixel_loss = criterion(reconstructed, images)
            loss = pixel_loss.sum(dim=[1, 2, 3])

            errors.extend(loss.cpu().numpy())
            filenames.extend(paths)

    return np.array(errors), np.array(filenames)


def evaluate(model, waldo_path, crowd_path, device):
    waldo_set = WaldoImageDataset(waldo_path, transform=TEST_TRANSFORM)
    crowd_set = WaldoImageDataset(crowd_path, transform=TEST_TRANSFORM)

    waldo_loader = DataLoader(waldo_set, batch_size=32, shuffle=False)
    crowd_loader = DataLoader(crowd_set, batch_size=32, shuffle=True)

    print("\nCalculating Errors...")
    waldo_errors, waldo_files = calculate_errors_with_filenames(
        model, waldo_loader, device
    )
    crowd_errors, _ = calculate_errors_with_filenames(model, crowd_loader, device)

    if len(crowd_errors) > len(waldo_errors) * 5:
        crowd_errors = crowd_errors[: len(waldo_errors) * 5]

    print(f"Waldo Mean Error: {np.mean(waldo_errors):.2f}")
    print(f"Crowd Mean Error: {np.mean(crowd_errors):.2f}")

    # --- FIND THE BAD WALDO ---
    print("\n--- TOP 5 WORST WALDO IMAGES (Highest Error) ---")
    bad_indices = np.argsort(waldo_errors)[::-1][:5]
    for i in bad_indices:
        print(
            f"Error: {waldo_errors[i]:.2f} | File: {os.path.basename(waldo_files[i])}"
        )
    print("------------------------------------------------\n")

    # --- PLOTS & METRICS ---
    plt.figure(figsize=(10, 6))
    plt.hist(
        waldo_errors,
        bins=50,
        alpha=0.7,
        label="Waldo (Trained)",
        density=True,
        color="orange",
    )
    plt.hist(
        crowd_errors,
        bins=50,
        alpha=0.6,
        label="Crowd (Unknown)",
        density=True,
        color="tab:blue",
    )

    threshold = scoreatpercentile(waldo_errors, 95)
    plt.axvline(
        threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.0f})"
    )
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.show()

    y_true = np.concatenate([np.ones(len(waldo_errors)), np.zeros(len(crowd_errors))])
    y_scores = np.concatenate([-waldo_errors, -crowd_errors])

    waldo_preds = (waldo_errors < threshold).astype(int)
    crowd_preds = (crowd_errors < threshold).astype(int)
    y_pred = np.concatenate([waldo_preds, crowd_preds])

    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

    auc = roc_auc_score(y_true, y_scores)
    print(f"\nAUC-ROC Score: {auc:.4f}")


if __name__ == "__main__":
    model = ShallowConvAutoencoder(latent_dim=64, image_size=IMAGE_SIZE, channels=3)

    # CRITICAL FIX: Move model to device explicitly
    model.to(DEVICE)

    # Load the model
    try:
        model.load_state_dict(torch.load("best_waldo_model.pth", map_location=DEVICE))
        print("Loaded model from disk.")
    except FileNotFoundError:
        print("No saved model found. Training from scratch...")
        model = train(model, REAL_TRAIN_PATH, NUM_EPOCHS, DEVICE)

    evaluate(model, REAL_TRAIN_PATH, REAL_TEST_PATH, DEVICE)
