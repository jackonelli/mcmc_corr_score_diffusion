from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import torch as th


class NoiseDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        x = self.examples['pixel_values'][index]
        y = self.examples['label'][index]

        return {'pixel_values': x, 'label': y}

    def __len__(self):
        return len(self.examples['label'])


def get_noise_mnist_data_loader(dataset_size: int, batch_size: int):
    examples = dict()
    examples["pixel_values"] = th.randn((dataset_size, 1, 28, 28))
    examples['label'] = th.randint(0, 10, (dataset_size, ))
    dataset = NoiseDataset(examples)

    # create dataloader
    dataloader_train = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    return dataloader_train



def get_mnist_data_loaders(batch_size: int):
    dataset = load_dataset("mnist")

    # define image transformations
    def transforms_f(train=True):
        transform = Compose([transforms.ToTensor()])

        def f(examples):
            examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
            del examples["image"]

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


def plot_mnist(dataloader: DataLoader):
    images = next(iter(dataloader))["pixel_values"][:100]
    _, axs = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(images[i * 10 + j].squeeze(), cmap="gray")
            axs[i, j].axis("off")
    plt.show()


if __name__ == "__main__":
    train, _ = get_mnist_data_loaders(100)
    plot_mnist(train)
