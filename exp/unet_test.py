from typing import Tuple
import io
import os
import blobfile as bf
import torch as th
from src.model.unet import UNetModel


def dev(device):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def main():
    T = 10
    image_channels = 1
    image_size = 28
    # model = UNetModel(
    #     attention_resolutions=(1, 2, 4),
    #     channel_mult=(1, 2, 2, 2),
    #     diffusion_steps=1000,
    #     dropout=0,
    #     image_size=28,
    #     in_channels=1,
    #     model_channels=64,
    #     num_classes=None,
    #     num_head_channels=-1,
    #     num_heads=4,
    #     num_heads_upsample=-1,
    #     num_res_blocks=3,
    #     out_channels=2,
    #     resblock_updown=True,
    #     use_checkpoint=False,
    #     use_fp16=False,
    #     use_new_attention_order=False,
    #     use_scale_shift_norm=True,
    # )
    model = UNetModel(
        image_size=image_size,
        in_channels=image_channels,
        model_channels=64,
        out_channels=2 * image_channels,  # Times 2 if learned sigma
        num_res_blocks=3,
        attention_resolutions=attention_down_sampling((28, 14, 7), image_size),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        diffusion_steps=T,
    )
    model.load_state_dict(
        load_state_dict(
            os.path.expanduser("models/model060000.pt"),
            map_location="cpu",
        )
    )

    image_channels = 1
    image_size = 28
    batch_size = 2
    x = th.rand((batch_size, image_channels, image_size, image_size))
    ts = th.randint(low=1, high=T, size=(batch_size,))
    print(x.size(), ts.size())
    model(x, ts)


def attention_down_sampling(resolutions: Tuple[int, int, int], image_size: int):
    """Map downsampled image size to downsampling factor"""
    return tuple(map(lambda res: image_size // res, resolutions))


if __name__ == "__main__":
    main()
