from pathlib import Path
import json
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from src.data.utils import collate_fn
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torchvision import transforms


def test():
    root = Path("/home/jakob/data/cifar10/")
    dataset = load_dataset("cifar10")


def get_cifar10_data_loaders(batch_size: int):
    dataset = load_dataset("cifar10")

    # define image transformations
    def transforms_f(train=True):
        img_size = 32
        transform = Compose(
            [
                Resize(size=[img_size, img_size], antialias=True),
                # Turn into tensor (scales [0, 255] to (0, 1))
                ToTensor(),
                # Map data to (-1, 1)
                Lambda(lambda x: (x * 2) - 1),
            ],
        )

        def f(examples):
            examples["pixel_values"] = [transform(image) for image in examples["img"]]
            del examples["img"]

            return examples

        return f

    # transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
    transformed_dataset_train = dataset.with_transform(transforms_f(train=True))
    transformed_dataset_val = dataset.with_transform(transforms_f(train=False))

    # create dataloader
    dataloader_train = DataLoader(
        transformed_dataset_train["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    dataloader_val = DataLoader(transformed_dataset_val["test"], batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader_train, dataloader_val
    # self.transform = Compose(
    #     [
    #         Resize(size=[img_size, img_size], antialias=True),
    #         # Turn into tensor (scales [0, 255] to (0, 1))
    #         ToTensor(),
    #         # Map data to (-1, 1)
    #         Lambda(lambda x: (x * 2) - 1),
    #     ],
    # )
