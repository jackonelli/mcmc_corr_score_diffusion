from pathlib import Path
import json
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, RandomHorizontalFlip
from datasets import load_dataset

CIFAR_IMAGE_SIZE = 32
CIFAR_NUM_CHANNELS = 3
CIFAR_100_NUM_CLASSES = 100


def collate_fn(batch):
    print(f"Type: {type(batch)}")
    new = {}
    new["x"] = batch["pixel_values"]
    new["labels"] = batch["fine_label"]
    return new


def get_cifar100_class_map(json_path: Path = Path.cwd() / "static/cifar100_class_map.json"):
    with open(json_path, "r") as ff:
        class_map = json.load(ff)
    class_map = {int(k): v for k, v in class_map.items()}
    return class_map


def get_cifar100_data_loaders(batch_size: int, data_root: Path):
    dataset = load_dataset("cifar100", cache_dir=str(data_root))

    # define image transformations
    def transforms_f(train=True):
        img_size = 32
        compose = [
                Resize(size=[img_size, img_size], antialias=True),
                # Turn into tensor (scales [0, 255] to (0, 1))
                ToTensor(),
                # Map data to (-1, 1)
                Lambda(lambda x: (x * 2) - 1),
            ]
        if train:
            compose = [RandomHorizontalFlip()] + compose
        transform = Compose(compose)

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
    dataloader_val = DataLoader(
        transformed_dataset_val["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return dataloader_train, dataloader_val


def get_cifar10_data_loaders(batch_size: int, data_root: Path):
    dataset = load_dataset("cifar10", cache_dir=str(data_root))

    # define image transformations
    def transforms_f(train=True):
        img_size = 32
        compose = [
            Resize(size=[img_size, img_size], antialias=True),
            # Turn into tensor (scales [0, 255] to (0, 1))
            ToTensor(),
            # Map data to (-1, 1)
            Lambda(lambda x: (x * 2) - 1),
        ]
        if train:
            compose = [RandomHorizontalFlip()] + compose
        transform = Compose(compose)

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
