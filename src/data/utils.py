"""Dataset utilities"""
import torch as th
from torchvision.transforms import Compose, Lambda, ToPILImage
import numpy as np
from torch.utils.data import Dataset, DataLoader


def collate_fn(batch):
    return {
        "x": th.stack([x for x, _ in batch]),
        "labels": th.tensor([y for _, y in batch]),
    }


def reverse_transform(tensor):
    transf = Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )
    return transf(tensor)


def get_full_sample_data_loaders(dataset, num_samples: int, batch_size: int, num_val_samples=500):
    train = FullSampleDataset(dataset, num_samples)
    val = FullSampleDataset(dataset, num_val_samples)

    dataloader_train = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collate_fn,
    )

    dataloader_val = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    return dataloader_train, dataloader_val


class FullSampleDataset(Dataset):
    def __init__(self, dataset, num_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.samples, self.labels = dataset.sample(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]
