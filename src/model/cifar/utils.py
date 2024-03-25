from collections import OrderedDict
from pathlib import Path
from typing import Optional
import numpy as np
import torch as th

from src.data.cifar import CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS, CIFAR_100_NUM_CLASSES
from src.model.cifar.class_t import load_unet_classifier_t as load_unet_classifier_t, ClassifierHoDrop, \
    load_unet_classifier_t
from src.model.cifar.unet import UNetEnergy, UNet
from src.model.cifar.unet_ho import UNetEnergy_Ho, Unet_Ho
from src.model.cifar.unet_ho_drop import UnetDropEnergy, Unet_drop
from src.model.guided_diff.classifier import load_guided_classifier as load_guided_diff_classifier_t
from src.model.resnet import load_classifier_t as load_resnet_classifier_t
from src.utils.net import load_params_from_file
from src.model.cifar.standard_class import load_standard_class, vgg11_bn, vgg13_bn, vgg19_bn, vgg16_bn
from src.utils.callbacks import load_ema, load_non_ema


def get_diff_model(name, diff_model_path, device, energy_param, image_size, num_steps, dropout=0.,
                   org_model=False, make_compile=False):
    if "small" in name:
        diff_model = load_unet_diff_model(
            diff_model_path, device, image_size=image_size, energy_param=energy_param
        )
    elif "large2" in name:
        diff_model = load_unet_ho_drop_diff_model(
            diff_model_path,
            device,
            energy_param=energy_param,
            T = num_steps,
            dropout=dropout,
            org_model=org_model
        )
    elif "large" in name:
        diff_model = load_unet_ho_diff_model(
            diff_model_path, device, energy_param=energy_param
        )
    else:
        raise ValueError("Not specified model size")

    if make_compile:
        diff_model = th.compile(diff_model, mode="reduce-overhead")
        th.set_float32_matmul_precision('high')
    return diff_model


def load_unet_diff_model(
    model_path: Optional[Path],
    device,
    time_emb_dim: int = 112,
    image_size: int = CIFAR_IMAGE_SIZE,
    num_channels: int = CIFAR_NUM_CHANNELS,
    energy_param: bool = False,
):
    """Load UNET diffusion model from state dict

    The model_path can be a standalone '.th' file or part of a pl checkpoint '.ckpt' file.
    If model_path is None, a new model is initialised.
    """
    if energy_param:
        unet = UNetEnergy(dim=image_size, time_emb_dim=time_emb_dim, channels=num_channels).to(device)
    else:
        unet = UNet(dim=image_size, time_emb_dim=time_emb_dim, channels=num_channels)
    if model_path is not None:
        params = load_params_from_file(model_path)
        if 'ema' in model_path.stem:
            params = load_ema(params)
        unet.load_state_dict(params)
    unet.to(device)
    unet.eval()
    return unet


def load_unet_ho_diff_model(
    model_path: Optional[Path],
    device,
    energy_param: bool = False,
):
    """Load UNET Ho diffusion model from state dict

    The model_path can be a standalone '.th' file or part of a pl checkpoint '.ckpt' file.
    If model_path is None, a new model is initialised.
    """
    if energy_param:
        unet = UNetEnergy_Ho(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False)
    else:
        unet = Unet_Ho(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=False)
    if model_path is not None:
        params = load_params_from_file(model_path)
        if 'ema' in model_path.stem:
            params = load_ema(params)
        unet.load_state_dict(params)
    unet.to(device)
    unet.eval()
    return unet


def load_unet_ho_drop_diff_model(
    model_path: Optional[Path],
    device,
    energy_param: bool = False,
    T: int = 1000,
    dropout: float = 0.,
    org_model: bool = False
):
    """Load UNET Ho diffusion model from state dict

    The model_path can be a standalone '.th' file or part of a pl checkpoint '.ckpt' file.
    If model_path is None, a new model is initialised.
    """
    if energy_param:
        unet = UnetDropEnergy(T=T, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                              num_res_blocks=2, dropout=dropout)
    else:
        unet = Unet_drop(T=T, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                         num_res_blocks=2, dropout=dropout)

    if model_path is not None:
        params = load_params_from_file(model_path)
        if 'ema' in model_path.stem:
            if org_model:
                params = load_non_ema(params)
            else:
                params = load_ema(params)

        unet.load_state_dict(params)
    unet.to(device)
    unet.eval()
    return unet


def select_cifar_classifier(model_path: Path, dev, num_steps):
    arch = parse_arch(model_path)
    num_classes = parse_class(model_path)
    return select_classifier_t(arch=arch, dev=dev, num_classes=num_classes,
                               num_channels=CIFAR_NUM_CHANNELS, img_size=CIFAR_IMAGE_SIZE,
                               dropout=0., num_diff_steps=num_steps, model_path=model_path)

def best_match(s: str, options: list[str]) -> str:
    matches = [opt for opt in options if opt in s]
    return matches[np.argmax([len(match) for match in matches])]

def parse_class(model_path: Path) -> int:
    name = model_path.name
    available_sizes = ["cifar100", "cifar10"]
    return int(best_match(name, available_sizes)[5:])

def parse_arch(model_path: Path):
    """Get model architecture from model name

    Models are (supposed) to be stored as <dataset>_<arch>_class_t.{ckpt,th}
    """
    name = model_path.name
    available_archs = ["unet", "resnet", "guided_diff", "unet_ho_drop"]
    return best_match(name, available_archs)


def load_unet_ho_drop_classifier_t(
    model_path: Optional[Path], dev, num_diff_steps, num_classes, x_size, ch=128, ch_mult=None,
        attn=None, num_res_blocks=2, dropout=0.1
):
    if ch_mult is None:
        ch_mult = [1, 2, 2, 2]

    if attn is None:
        attn = [1]
    print("Loading unet dropout model")
    model = ClassifierHoDrop(T=num_diff_steps, ch=ch, ch_mult=ch_mult, attn=attn,
                             num_res_blocks=num_res_blocks, dropout=dropout, num_classes=num_classes,
                             x_size=x_size)
    if model_path is not None:
        params = load_params_from_file(model_path)

        if 'ema' in model_path.stem:
            params = load_ema(params)
        model.load_state_dict(params)
    model.to(dev)
    return model


def select_classifier_t(arch, dev, num_classes, num_channels, img_size, dropout=0., num_diff_steps=1000,
                        model_path=None):
    if arch == "unet":
        class_t = load_unet_classifier_t(model_path, dev)
    elif arch == "resnet":
        class_t = load_resnet_classifier_t(
            model_path=model_path,
            dev=dev,
            emb_dim=256,
            num_classes=num_classes,
            num_channels=num_channels,
            dropout=dropout,
        ).to(dev)
    elif arch == "guided_diff":
        class_t = load_guided_diff_classifier_t(
            model_path=model_path, dev=dev, image_size=img_size, num_classes=num_classes,
            dropout=dropout,
        ).to(dev)
    elif arch == "unet_ho_drop":
        x_size = (num_channels, img_size, img_size)
        class_t = load_unet_ho_drop_classifier_t(model_path=model_path, dev=dev, dropout=dropout,
                                                 num_diff_steps=num_diff_steps, num_classes=num_classes,
                                                 x_size=x_size).to(dev)
    else:
        raise ValueError(f"Incorrect model arch: {arch}")
    return class_t


def select_classifier(arch, dev, num_classes, num_channels, dataset):
    if arch == "simple_resnet":
        class_ = load_standard_class(
            model_path=None,
            device=dev,
            num_channels=num_channels,
            num_classes=num_classes,
        ).to(dev)
    elif arch == "vgg11_bn":
        class_ = vgg11_bn(dataset=dataset).to(dev)
    elif arch == "vgg13_bn":
        class_ = vgg13_bn(dataset=dataset).to(dev)
    elif arch == "vgg16_bn":
        class_ = vgg16_bn(dataset=dataset).to(dev)
    elif arch == "vgg19_bn":
        class_ = vgg19_bn(dataset=dataset).to(dev)
    else:
        raise ValueError(f"Incorrect model arch: {arch}")
    return class_
