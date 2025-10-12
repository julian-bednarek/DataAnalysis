import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from const import IMAGE_SIZE, IMAGE_EXTENSION

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
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
            return torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)

        if self.transform:
            image = self.transform(image)

        return image, image


def calculate_errors(model, dataloader, device):
    """
    Computes reconstruction error (MSE summed over dimensions) for a dataset.
    Returns a numpy array of errors.
    """
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reconstructed = model(images)

            # Compute MSE per image
            pixel_loss = criterion(reconstructed, images)
            # Sum error over C, H, W dimensions
            loss = pixel_loss.sum(dim=[1, 2, 3])

            errors.extend(loss.cpu().numpy())

    return np.array(errors)
