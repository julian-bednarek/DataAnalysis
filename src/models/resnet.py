from torch import nn
from torchvision import models


class ResNetTeacher(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )  # Use modern weights API

        # Adapt first conv layer for grayscale
        original_conv1 = self.resnet.conv1
        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if channels == 1 and original_conv1.in_channels == 3:
            # Sum weights across the input channel dim to go from (64, 3, 7, 7) to (64, 1, 7, 7)
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
        else:
            self.conv1.weight.data = original_conv1.weight.data

        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3  # We will extract features from here

        # Freeze the teacher
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Get features from layer 3
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# --- The Student Model (Trainable) ---
class ResNetAnomalyStudent(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        # --- 1. Encoder (mirrors the Teacher) ---
        # We load a *new* resnet18 to ensure we have the right architecture
        # but we will *not* use its pretrained weights (it will be trained from scratch)
        student_base = models.resnet18(weights=None)  # No pretrained weights

        # Adapt first conv layer
        self.encoder_conv1 = nn.Conv2d(
            channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder_bn1 = student_base.bn1
        self.encoder_relu = student_base.relu
        self.encoder_maxpool = student_base.maxpool
        self.encoder_layer1 = student_base.layer1
        self.encoder_layer2 = student_base.layer2
        self.encoder_layer3 = student_base.layer3

        # --- 2. Decoder (reverses the encoder) ---
        # After layer3, output is (B, 256, H/16, W/16) -> (B, 256, 16, 16) for 256x256 input
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, channels, kernel_size=4, stride=2, padding=1
            ),  # -> (B, C, 256, 256)
            nn.Tanh(),  # To match the -1 to 1 normalization
        )

    def get_features(self, x):
        # Encoder path
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        return x

    def forward(self, x):
        # The full autoencoder path (used for a potential reconstruction loss)
        features = self.get_features(x)
        reconstruction = self.decoder(features)
        return reconstruction
