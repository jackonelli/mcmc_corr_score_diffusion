from copy import deepcopy

import torch
from pytorch_lightning import Callback
import pytorch_lightning as pl


class EMACallback(Callback):
    def __init__(self, decay=0.995):
        self.decay = decay
        self.module_pair_list = []

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        def forward_wrapper(module, org, ema):
            def forward(*args, **kwargs):
                return org(*args, **kwargs) if module.training else ema(*args, **kwargs)
            return forward

        modules = list(filter(lambda x: len(list(x[1].parameters())) > 0, pl_module.named_children()))

        for name, module in modules:
            ema_module = deepcopy(module)
            self.module_pair_list.append((ema_module, module))
            pl_module.add_module(f'EMA_{name}', ema_module)
            module.forward_bc = module.forward
            module.forward = forward_wrapper(module, module.forward_bc, ema_module.forward)

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for ema_module, module in self.module_pair_list:
            self._update(ema_module, module, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def _update(self, ema_module, module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(ema_module.state_dict().values(), module.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))