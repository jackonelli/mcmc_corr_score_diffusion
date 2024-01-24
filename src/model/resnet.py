"""Classification model

Used for approximating the likelihood p(y | x_t)

Adapted from https://github.com/blt2114/twisted_diffusion_sampler, which in turn is adapted from
https://github.com/RobustBench/robustbench
"""

from pathlib import Path
from typing import Optional
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src.model.unet import SinusoidalPositionEmbeddings
from einops import rearrange
from abc import ABC
from src.utils.net import load_params_from_file


def load_mnist_classifier(class_path: Path, device):
    classifier = load_classifier(class_path, True)
    classifier.to(device)
    return classifier


def load_classifier(resnet_model_path: Path, time_emb=False):
    """Helper function to load default classifier with pre-trained weights"""
    if time_emb:
        model = ResNetTimeEmbedding(BottleneckTimeEmb, [3, 4, 6, 3], 112, num_classes=10, num_channels=1)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, num_channels=1)
    model.load_state_dict(th.load(resnet_model_path))
    return model


# TODO: Merge with fn above
def load_classifier_t(
    model_path: Optional[Path], dev, num_blocks=[3, 4, 6, 3], emb_dim=112, num_classes=10, num_channels=1
):
    print("Loading resnet model")
    model = ResNetTimeEmbedding(
        block=BottleneckTimeEmb,
        num_blocks=num_blocks,
        emb_dim=emb_dim,
        num_classes=num_classes,
        num_channels=num_channels,
    )
    if model_path is not None:
        model.load_state_dict(load_params_from_file(model_path))
    model.to(dev)
    return model


class ResNetBase(nn.Module, ABC):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, in_planes=64):
        super(ResNetBase, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(num_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.linear = nn.Linear(self.in_planes * 2 ** (len(num_blocks) - 1) * block.expansion, num_classes)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ResNet(ResNetBase):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, in_planes=64):
        super(ResNet, self).__init__(block, num_blocks, num_classes, num_channels, in_planes)
        self.in_planes = in_planes
        dims = [self.in_planes, self.in_planes * 2, self.in_planes * 4, self.in_planes * 8]

        self.layer1 = self._make_layer(block, dims[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, dims[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, dims[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, dims[3], num_blocks[3], stride=2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # import pdb;
        # pdb.set_trace()
        return out

    def p_y_given_x(self, x: th.Tensor) -> th.Tensor:
        logits = self.forward(x)
        return logits_to_prob(logits)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


def logits_to_prob(logits: th.Tensor) -> th.Tensor:
    return F.softmax(logits, dim=1)


class ResNetTimeEmbedding(ResNetBase):
    def __init__(self, block, num_blocks, emb_dim, num_classes=10, num_channels=3, in_planes=64):
        super(ResNetTimeEmbedding, self).__init__(block, num_blocks, num_classes, num_channels, in_planes)
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        dims = [self.in_planes * 2**i for i in range(len(num_blocks))]
        strides = [1] + [2] * (len(num_blocks) - 1)
        self.layers = list()
        for i, num_block in enumerate(num_blocks):
            self.layers.append(self._make_layer(block, dims[i], num_block, stride=strides[i]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, t):
        x_ = F.relu(self.bn1(self.conv1(x)))
        time_emb = self.time_mlp(t)
        for j, layer in enumerate(self.layers):
            for block_ in layer:
                x_ = block_(x_, time_emb)
        x_ = F.avg_pool2d(x_, 4)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.linear(x_)
        return x_

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.emb_dim, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckTimeEmb(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, time_emb_dim, stride=1):
        super(BottleneckTimeEmb, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_planes * 2))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckChen2020AdversarialNet(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckChen2020AdversarialNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        pre = F.relu(self.bn0(x))
        out = F.relu(self.bn1(self.conv1(pre)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        if len(self.shortcut) == 0:
            out += self.shortcut(x)
        else:
            out += self.shortcut(pre)
        return out


class CustomResNet(ResNet):
    """
    Replacing avg_pool with a adaptive_avg_pool. Now this model can be used much
    resolution beyond cifar10.
    Note: ResNet models in RobustBench are cifar10 style, thus geared to 32x32. These
    models are slightly different than original ResNets (224x224 resolution).
    """

    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super(CustomResNet, self).__init__(block, num_blocks, num_classes, num_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
