"""Project-wide constants."""

from enum import Enum
import torch


class ImageTypes(Enum):
    """Types of images in our dataset."""

    NORMAL = ""
    BW = "-bw"
    GRAY = "-gray"


IMAGE_SIZE = 64
IMAGE_TYPE = ImageTypes.NORMAL.value
IMAGE_EXTENSION = ".jpg"
TRAIN_DATA_PATH = f"../datasets/Hey-Waldo/{IMAGE_SIZE}{IMAGE_TYPE}/notwaldo"
TEST_DATA_PATH = f"../datasets/Hey-Waldo/{IMAGE_SIZE}{IMAGE_TYPE}/waldo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
