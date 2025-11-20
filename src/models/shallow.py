from torch import nn


class ShallowConvAutoencoder(nn.Module):
    """
    Balanced Architecture v3:
    - Latent Dim: Flexible (we will use 64).
    - Dropout: Added Mild Dropout (0.1) to fight overfitting.
    """

    def __init__(self, latent_dim: int, image_size: int, channels: int):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoded_spatial_dim = image_size // 4
        # Encoder output is 64 filters x (H/4) x (W/4)
        self.flat_size = 64 * self.encoded_spatial_dim * self.encoded_spatial_dim

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),  # Mild dropout
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),  # Mild dropout
        )

        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.channels, kernel_size=2, stride=2),
            nn.Tanh(),
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
