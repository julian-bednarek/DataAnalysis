from torch import nn

class ShallowConvAutoencoder(nn.Module):
    """Less aggressive downsampling - better for larger features"""
    def __init__(self, latent_dim: int, image_size: int, channels: int):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.encoded_spatial_dim = image_size // 4
        self.flat_size = 64 * self.encoded_spatial_dim * self.encoded_spatial_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.channels, kernel_size=2, stride=2),
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
