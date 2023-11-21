from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torch.utils.data import Dataset
from PIL import Image
import torch as th
from torch.utils.data import DataLoader, Dataset

IMAGE_PATH = Path("/home/jakob/data/small-imagenet/train.X1/n01440764/n01440764_12971.JPEG")


class SingleImage(Dataset):
    def __init__(self, img_size: int):
        transform = Compose(
            [
                Resize(size=[img_size, img_size], antialias=True),
                # Turn into tensor (scales [0, 255] to (0, 1))
                ToTensor(),
                # Map data to (-1, 1)
                Lambda(lambda x: (x * 2) - 1),
            ],
        )
        image = Image.open(str(IMAGE_PATH))
        self.sample = transform(image)

    def __len__(self):
        # Dummy dataset size, to artifically prolong epochs.
        return 100

    def __getitem__(self, _idx):
        return self.sample, "dummy"


def get_single_image_dataloader(img_size: int):
    dataset_train = SingleImage(img_size)
    dataset_val = SingleImage(img_size)
    batch_size = 1

    # create dataloader
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=dummy_collate_fn,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dummy_collate_fn,
    )
    return dataloader_train, dataloader_val


def dummy_collate_fn(batch):
    return {
        "pixel_values": th.stack([x for x, _ in batch]),
        "labels": None,
    }
