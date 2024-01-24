"""UNet model for time-dep classification"""
from typing import Optional
from pathlib import Path
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS
from src.model.cifar.common import (
    load_params_from_file,
    SinusoidalPositionEmbeddings,
    ResnetBlock,
    Attention,
    LinearAttention,
    Residual,
    PreNorm,
    downsample,
)


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
    class_t = StandardClassifier(dim=image_size, channels=num_channels, num_classes=num_classes)
    if model_path is not None:
        class_t.load_state_dict(load_params_from_file(model_path))
    class_t.to(device)
    class_t.eval()
    return class_t


NUM_FEATURES = 8192


class StandardClassifier(nn.Module):
    """UNet based classifier
    @param dim: dimension of input x (assumes square (dim x dim) img for now)
    @param time_emb_dim: dimension of time embedding.
    @param channels: number of channels the data have (e.g. 3 for RGB images).
    """

    def __init__(self, num_classes=CIFAR_100_NUM_CLASSES):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    # Take a non-used time step t to conform to the API
    def forward(self, x, _t):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)

        # Flatten before fully connected layers
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class StandardClassifier(nn.Module):
#     """UNet based classifier
#     @param dim: dimension of input x (assumes square (dim x dim) img for now)
#     @param time_emb_dim: dimension of time embedding.
#     @param channels: number of channels the data have (e.g. 3 for RGB images).
#     """
#
#     def __init__(
#         self,
#         dim: int,
#         num_classes: int,
#         channels: int,
#     ):
#         super().__init__()
#
#         self.channels = channels
#         self.dim = dim
#         self.num_classes = num_classes
#
#         # First & Final Layers
#         self.init_conv = nn.Conv2d(channels, dim, 1, padding=0)
#         self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_emb_dim)
#         self.final_conv = nn.Conv2d(dim, channels, 1)
#
#         # Layers
#         self.downs = nn.ModuleList([])
#         # self.ups = nn.ModuleList([])
#
#         # Encoder
#         # Block 1
#         self.downs.append(
#             nn.ModuleList(
#                 [
#                     ResnetBlock(dim, dim, time_emb_dim),
#                     ResnetBlock(dim, dim, time_emb_dim),
#                     Residual(PreNorm(dim, LinearAttention(dim))),
#                     downsample(dim, dim),
#                 ]
#             )
#         )
#         # Block 2
#         self.downs.append(
#             nn.ModuleList(
#                 [
#                     ResnetBlock(dim, dim, time_emb_dim),
#                     ResnetBlock(dim, dim, time_emb_dim),
#                     Residual(PreNorm(dim, LinearAttention(dim))),
#                     downsample(dim, 2 * dim),
#                 ]
#             )
#         )
#         # Block 3
#         self.downs.append(
#             nn.ModuleList(
#                 [
#                     ResnetBlock(2 * dim, 2 * dim, time_emb_dim),
#                     ResnetBlock(2 * dim, 2 * dim, time_emb_dim),
#                     Residual(PreNorm(2 * dim, LinearAttention(2 * dim))),
#                     nn.Conv2d(2 * dim, 4 * dim, kernel_size=3, padding=1),
#                 ]
#             )
#         )
#
#         # Latent Layer
#         self.middle = nn.ModuleList(
#             [
#                 ResnetBlock(4 * dim, 4 * dim, time_emb_dim),
#                 Residual(PreNorm(4 * dim, Attention(4 * dim))),
#                 ResnetBlock(4 * dim, 4 * dim, time_emb_dim),
#             ]
#         )
#
#         # Decoder
#         # Block 1
#         # self.ups.append(
#         #     nn.ModuleList(
#         #         [
#         #             ResnetBlock(6 * dim, 4 * dim, time_emb_dim),
#         #             ResnetBlock(6 * dim, 4 * dim, time_emb_dim),
#         #             Residual(PreNorm(4 * dim, LinearAttention(4 * dim))),
#         #             upsample(4 * dim, 2 * dim),
#         #         ]
#         #     )
#         # )
#         # # Block 2
#         # self.ups.append(
#         #     nn.ModuleList(
#         #         [
#         #             ResnetBlock(3 * dim, 2 * dim, time_emb_dim),
#         #             ResnetBlock(3 * dim, 2 * dim, time_emb_dim),
#         #             Residual(PreNorm(2 * dim, LinearAttention(2 * dim))),
#         #             upsample(2 * dim, dim),
#         #         ]
#         #     )
#         # )
#         # # Block 3
#         # self.ups.append(
#         #     nn.ModuleList(
#         #         [
#         #             ResnetBlock(2 * dim, dim, time_emb_dim),
#         #             ResnetBlock(2 * dim, dim, time_emb_dim),
#         #             Residual(PreNorm(dim, LinearAttention(dim))),
#         #             nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#         #         ]
#         #     )
#         # )
#         self.log_reg = nn.Linear(NUM_FEATURES, self.num_classes)
#
#     def forward(self, x: th.Tensor, time: th.Tensor):
#         t = self.time_mlp(time)
#         return self._unet_forward(x, t)
#
#     def _unet_forward(self, x: th.Tensor, t: th.Tensor):
#         x = self.init_conv(x)
#         r = x.clone()
#
#         h = []
#
#         # Down-sampling
#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             h.append(x)
#
#             x = block2(x, t)
#             x = attn(x)
#             h.append(x)
#
#             x = downsample(x)
#
#         # Latent Space
#         for block in self.middle:
#             x = block(x, t) if type(block) is ResnetBlock else block(x)
#
#         # Up-sampling
#         # for block1, block2, attn, upsample in self.ups:
#         #     x = th.cat((x, h.pop()), dim=1)
#         #     x = block1(x, t)
#
#         #     x = th.cat((x, h.pop()), dim=1)
#         #     x = block2(x, t)
#         #     x = attn(x)
#
#         #     x = upsample(x)
#
#         # x = th.cat((x, r), dim=1)
#
#         # x = self.final_res_block(x, t)
#
#         x = th.flatten(x, start_dim=1)
#         x = self.log_reg(x)
#         return x
