"""Project-wide constants."""

from enum import Enum
import torch


class ImageTypes(Enum):
    """Types of images in our dataset."""

    NORMAL = {"file": "", "no_channels": 3}
    BW = {"file": "-bw", "no_channels": 1}
    GRAY = {"file": "-gray", "no_channels": 1}


IMAGE_SIZE = 64
IMAGE_TYPE = ImageTypes.GRAY.value
IMAGE_EXTENSION = ".jpg"
TRAIN_DATA_PATH = f"../datasets/Hey-Waldo/{IMAGE_SIZE}{IMAGE_TYPE['file']}/notwaldo"
TEST_DATA_PATH = f"../datasets/Hey-Waldo/{IMAGE_SIZE}{IMAGE_TYPE['file']}/waldo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
