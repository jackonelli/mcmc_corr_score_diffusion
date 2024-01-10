from functools import partial
from pathlib import Path
import torch.nn as nn
from src.model.comp_two_d.diffusion import Block


def load_classifier(params_path, num_classes, device, x_dim=2, num_diff_steps=100):
    class_ = Classifier(x_dim=x_dim, num_classes=num_classes, num_diff_steps=num_diff_steps)
    class_.load_state_dict(th.load(params_path, map_location="cpu"))
    class_.to(device)
    class_.eval()
    return class_


class Classifier(nn.Module):
    """Resnet score model.

    Adds embedding for each scale after each linear layer.
    """

    def __init__(
        self,
        x_dim,
        num_classes,
        num_diff_steps,
        n_layers=2,
        h_dim=64,
        emb_dim=16,
        widen=2,
        emb_type="learned",
    ):
        assert emb_type in ("learned", "sinusoidal")
        super().__init__()
        self._n_steps = num_diff_steps
        self._n_layers = n_layers
        self._x_dim = x_dim
        self._num_classes = num_classes
        self._h_dim = h_dim
        self._emb_dim = emb_dim
        self._widen = widen

        # Modules
        if emb_type == "learned":
            self.emb = nn.Embedding(self._n_steps, self._emb_dim)
        else:
            self.emb = partial(timestep_embedding, dim=self._emb_dim)
        self.first = nn.Linear(self._x_dim, self._h_dim)

        self.blocks = nn.ModuleList([])
        for _ in range(self._n_layers):
            self.blocks.append(Block(self._h_dim, self._emb_dim, self._widen))

        self.out = nn.Linear(self._h_dim, self._num_classes)

    def forward(self, x, t):
        emb = self.emb(t)
        x = self.first(x)
        for block in self.blocks:
            x = block(x, emb)
        x = self.out(x)
        return x


import torch as th


def test():
    diff_model = Classifier(x_dim=2, num_classes=8, num_diff_steps=100)
    B = 10
    x_test = th.randn((B, 2))
    t_test = th.ones((B,)).long()
    logits = diff_model(x_test, t_test)
    print(logits.size())


if __name__ == "__main__":
    test()
