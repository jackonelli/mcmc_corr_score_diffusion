from pathlib import Path
import json
from copy import deepcopy
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
import matplotlib.pyplot as plt
import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image


def test():
    root = Path("/home/jakob/data/small-imagenet/")
    img_size = 224
    inet = ImageNet100(root, img_size)
    print(len(inet.samples))
    # Dataloader
    train, val = get_imagenet_data_loaders(root, img_size, batch_size=100)


class ImageNet100(Dataset):
    def __init__(self, root: Path, img_size: int, train=True):
        self.root = root
        self.id_to_num_map, self.id_to_name_map = _parse_labels_map(root / "Labels.json")
        self.samples = self._parse_samples(train)
        self.transform = Compose([Resize(size=[img_size, img_size], antialias=True), ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_path = self.root / img_path
        # image = read_image(str(img_path))
        image = Image.open(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, label

    def _get_numeric_label(self, class_name):
        return self.id_to_num_map[class_name]

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
    )
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader_train, dataloader_val


def _parse_labels_map(json_path: Path):
    with open(json_path) as json_file:
        id_to_name_map = json.load(json_file)
    id_to_num_map = deepcopy(id_to_name_map)
    id_to_num_map = {id: num for (num, id) in enumerate(id_to_num_map.keys())}
    return id_to_num_map, id_to_name_map


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
