from pathlib import Path
import json
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, RandomHorizontalFlip
from datasets import load_dataset
from typing import Optional

CIFAR_IMAGE_SIZE = 32
CIFAR_NUM_CHANNELS = 3
CIFAR_100_NUM_CLASSES = 100
CIFAR_10_NUM_CLASSES = 10


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


def get_cifar100_data_loaders(batch_size: int, data_root: Optional[str] = None):
    dataset = load_dataset("cifar100", cache_dir=data_root)

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


def get_cifar10_data_loaders(batch_size: int, data_root: Optional[str] = None):
    dataset = load_dataset("cifar10", cache_dir=data_root)

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


label2str = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    3: 'bear',
    4: 'beaver',
    5: 'bed',
    6: 'bee',
    7: 'beetle',
    8: 'bicycle',
    9: 'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'cra',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}
str2label = {v: k for k, v in label2str.items()}