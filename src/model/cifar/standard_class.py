"""UNet model for time-dep classification"""
from typing import Optional
from pathlib import Path
import torch.nn as nn
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS
from src.utils.net import load_params_from_file
from src.utils.datasets import dataset_image_description
from src.utils.callbacks import load_ema
import torch


def load_standard_class(
    model_path: Optional[Path],
    device,
    image_size: int = CIFAR_IMAGE_SIZE,
    num_channels: int = CIFAR_NUM_CHANNELS,
    num_classes: int = CIFAR_100_NUM_CLASSES,
):
    """Load classifier model from state dict

    The model_path can be a standalone '.th' file or part of a pl checkpoint '.ckpt' file.
    If model_path is None, a new model is initialised.
    """
    class_t = StandardClassifier(in_channels=num_channels, num_classes=num_classes)
    if model_path is not None:
        class_t.load_state_dict(load_params_from_file(model_path))
    class_t.to(device)
    class_t.eval()
    return class_t


NUM_FEATURES = 8192


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class StandardClassifier(nn.Module):
    def __init__(self, in_channels=CIFAR_NUM_CHANNELS, num_classes=CIFAR_100_NUM_CLASSES):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(cfg, batch_norm, model_path, device, dataset='cifar10', **kwargs):
    num_classes, _, _ = dataset_image_description(dataset)
    if model_path is not None:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes, **kwargs)
    if model_path is not None:
        params = load_params_from_file(model_path)
        if 'ema' in model_path.stem:
            params = load_ema(params)
        model.load_state_dict(params)
    return model


def vgg11_bn(model_path=None, dataset='cifar10', device="cpu", **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    return _vgg("A", True, model_path, device, dataset, **kwargs)


def vgg13_bn(model_path=None, dataset='cifar10', device="cpu", **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""

    return _vgg("B", True, model_path, device, dataset, **kwargs)


def vgg16_bn(model_path=None, dataset='cifar10', device="cpu", **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return _vgg("D", True, model_path, device, dataset, **kwargs)


def vgg19_bn(model_path=None, dataset='cifar10', device="cpu", **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return _vgg("E", True, model_path, device, dataset, **kwargs)
