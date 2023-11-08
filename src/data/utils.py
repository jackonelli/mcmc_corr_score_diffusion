"""Dataset utilities"""
import torch as th


def collate_fn(batch):
    return {
        "pixel_values": th.stack([x for x, _ in batch]),
        "labels": th.tensor([y for _, y in batch]),
    }
