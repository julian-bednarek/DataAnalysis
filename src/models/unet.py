from torch import nn
import torch

class UNetAutoencoder(nn.Module):
    """Skip connections preserve spatial details"""
    def __init__(self, latent_dim: int, image_size: int, channels: int):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoded_spatial_dim = image_size // 8
        self.flat_size = 64 * self.encoded_spatial_dim * self.encoded_spatial_dim

        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, channels, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.enc1(x)
        x = self.pool1(x)
        e1 = x

        x = self.enc2(x)
        x = self.pool2(x)
        e2 = x

        x = self.enc3(x)
        x = self.pool3(x)

        x = x.view(batch_size, -1)
        x = self.fc_encoder(x)

        x = self.fc_decoder(x)
        x = x.view(batch_size, 64, self.encoded_spatial_dim, self.encoded_spatial_dim)
        x = self.dec3(x)
        x = torch.cat([x, e2], dim=1)  # Skip connection

        x = self.dec2(x)
        x = torch.cat([x, e1], dim=1)  # Skip connection

        x = self.dec1(x)

        return x
