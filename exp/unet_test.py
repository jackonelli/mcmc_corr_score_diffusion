from src.model.unet import UNetModel


def main():
    image_channels = 3  # From RGB, I guess
    num_channels = 64  # Do not know what this means
    unet = UNetModel(
        image_size=28,
        in_channels=image_channels,
        model_channels=num_channels,
        out_channels=image_channels,
        num_res_blocks=3,
        attention_resolutions=(28, 14, 7),
    )


if __name__ == "__main__":
    main()
