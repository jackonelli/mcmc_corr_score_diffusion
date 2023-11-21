"""Dataset utilities"""
import torch as th
from torchvision.transforms import Compose, Lambda, ToPILImage
import numpy as np


def collate_fn(batch):
    return {
        "pixel_values": th.stack([x for x, _ in batch]),
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
