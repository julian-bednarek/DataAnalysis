import torch.nn as nn

class ThreePoolLayerConvAutoencoder(nn.Module):
    """3 Pooling layers. Works perfectly for 256x256 images."""
    def __init__(self, latent_dim: int, image_size: int, channels: int):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.encoded_spatial_dim = image_size // 8
        self.flat_size = 64 * self.encoded_spatial_dim * self.encoded_spatial_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, self.channels, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc_encoder(x)
        
        x = self.fc_decoder(x)
        x = x.view(batch_size, 64, self.encoded_spatial_dim, self.encoded_spatial_dim)
        x = self.decoder(x)
        
        return x