from copy import deepcopy

import torch
from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.utils.net import load_params_from_file
from pathlib import Path
from collections import OrderedDict


class EMACallback(Callback):
    def __init__(self, decay=0.995, path_ckpt: Path=None):
        self.decay = decay
        self.module_pair_list = []
        self.path_ckpt = path_ckpt

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        def forward_wrapper(module, org, ema):
            def forward(*args, **kwargs):
                return org(*args, **kwargs) if module.training else ema(*args, **kwargs)
            return forward

        modules = list(filter(lambda x: len(list(x[1].parameters())) > 0, pl_module.named_children()))
        params = None

        if self.path_ckpt is not None:
            params = load_params_from_file(self.path_ckpt)

            if 'ema_model' in params.keys():
                params = params['ema_model']
            else:
                all_keys = [k for k in params.keys()]
                ema_keys = all_keys[int(len(all_keys) / 2):]
                keys = all_keys[:int(len(all_keys) / 2)]
                params_ = OrderedDict()
                for key, ema_key in zip(keys, ema_keys):
                    params_[key] = params[ema_key]
                params = params_

        for name, module in modules:
            ema_module = deepcopy(module)
            if self.path_ckpt is not None:
                ema_module.load_state_dict(params)
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



def load_non_ema(params):
    params_ = OrderedDict()
    all_keys = [k for k in params.keys()]
    org_keys = all_keys[:int(len(all_keys) / 2)]
    for org_key in org_keys:
        params_[org_key] = params[org_key]
    params = params_
    return params

def load_ema(params):
    if 'ema_model' in params.keys():
        params = params['ema_model']
    else:
        all_keys = [k for k in params.keys()]
        ema_keys = all_keys[int(len(all_keys) / 2):]
        keys = all_keys[:int(len(all_keys) / 2)]
        params_ = OrderedDict()
        for key, ema_key in zip(keys, ema_keys):
            params_[key] = params[ema_key]
        params = params_
    return params
