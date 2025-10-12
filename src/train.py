import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from const import *
from models import *
from dataset import WaldoImageDataset, TRAIN_TRANSFORM, TEST_TRANSFORM

Model = ShallowConvAutoencoder
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
NUM_EPOCHS = 350
WEIGHT_DECAY = 5e-6
SAVE_PATH = "waldo_model.pth"


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

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=3500)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5
    )

    model.to(device)
    print(f"Running on device: {device}")

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:
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
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_images).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        saved_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            saved_msg = "[Saved Best]"

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} {saved_msg}"
            )

    print(f"Training complete. Best model weights saved to {SAVE_PATH}.")


if __name__ == "__main__":
    model = ShallowConvAutoencoder(latent_dim=2048, image_size=IMAGE_SIZE, channels=3)
    train(model, TRAIN_DATA_PATH, NUM_EPOCHS, DEVICE)
