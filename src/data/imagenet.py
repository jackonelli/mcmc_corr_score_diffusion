from pathlib import Path
import json
from copy import deepcopy
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    CenterCrop,
    Normalize,
    ToTensor,
    Resize,
    InterpolationMode,
    Lambda,
)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from src.data.utils import construct_collate_fn


REGNET_TRANSFORM = Compose(
    [
        ToTensor(),
        Resize(size=256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        CenterCrop(size=224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

CLASSIFIER_TRANSFORM = Compose(
    [
        Lambda(lambda x: (x + 1) / 2),
        Resize(size=256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        CenterCrop(size=224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# TRAINING_TRANSFORMS = Compose(
#     [
#         Resize(size=[img_size, img_size], antialias=True),
#         # Turn into tensor (scales [0, 255] to (0, 1))
#         ToTensor(),
#         # Map data to (-1, 1)
#     ],
# )


class ImageNet100(Dataset):
    def __init__(self, root: Path, img_size: int, train=True, transforms: Compose = REGNET_TRANSFORM):
        self.root = root
        self.id_to_int_map = _parse_label_maps(root / "Labels.json", Path.cwd() / "models/imagenet.json")
        self.samples = self._parse_samples(train)
        self.transform = REGNET_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_path = self.root / img_path
        # image = read_image(str(img_path))
        image = Image.open(str(img_path))
        # Apparently there are some grayscale and RGBA images in ImageNet100.
        if image.mode != "RGB":
            rgb = Image.new("RGB", image.size)
            rgb.paste(image)
            image = rgb
        if self.transform:
            image = self.transform(image)
        return image, label

    def _get_numeric_label(self, class_name):
        return self.id_to_int_map[class_name]

    def _parse_samples(self, train: bool):
        if train:
            sub_dirs = self.root.glob("train.X*")
        else:
            sub_dirs = self.root.glob("val.X")

        samples = []
        for dir_ in sub_dirs:
            for class_dir in dir_.iterdir():
                label = self._get_numeric_label(class_dir.stem)
                tmp = [(path, label) for path in class_dir.glob("*.JPEG")]
                samples.extend(tmp)
        return samples


def get_imagenet_data_loaders(root: Path, img_size: int, batch_size: int):
    dataset_train = ImageNet100(root, img_size, train=True)
    dataset_val = ImageNet100(root, img_size, train=False)

    # create dataloader
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=construct_collate_fn({"x": "x", "y": "y"}),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=construct_collate_fn({"x": "x", "y": "y"}),
    )
    return dataloader_train, dataloader_val


def test():
    root = Path.home() / "data/small-imagenet"
    _parse_label_maps(root / "Labels.json", Path.cwd() / "models/imagenet.json")


def _parse_label_maps(id_to_name_map_path: Path, int_to_name_map_path):
    """Generate Imagenet id's to int map

    We have two maps:
        - imagenet id: name. E.g.: "n01698640": "American alligator, Alligator mississipiensis"
        - str(int): name. E.g.: "50": "American_alligator"
    For the data loader which only has access to the imagenet id,
    we generate an:
        - imagenet id: int map.
    """
    # Load the two maps
    with open(id_to_name_map_path) as json_file:
        id_to_name_map = json.load(json_file)
    with open(int_to_name_map_path) as json_file:
        int_to_name_map = json.load(json_file)

    # Reverse int: name map
    name_to_int_map = {v: int(k) for k, v in int_to_name_map.items()}

    # Generate id: int map
    id_to_int_map = {}
    for id, long_name in id_to_name_map.items():
        name = _transf_long_name(long_name)
        id_to_int_map[id] = name_to_int_map[name]
    return id_to_int_map


def _transf_long_name(long_name: str) -> str:
    """Transform long Imagenet class name
    E.g.,
        American alligator, Alligator mississipiensis -> American_alligator
    """
    return long_name.split(sep=",")[0].replace(" ", "_")


def plot_imagenet(dataloader: DataLoader):
    images = next(iter(dataloader))
    _, axs = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(images[i * 10 + j].squeeze(), cmap="gray")
            axs[i, j].axis("off")
    plt.show()


if __name__ == "__main__":
    test()
