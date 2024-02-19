from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import torch as th
from torchvision.models import regnet_x_8gf, RegNet_X_8GF_Weights
from src.data.imagenet import get_imagenet_data_loaders, guided_diff_transf
from src.utils.metrics import accuracy, hard_label_from_logit, top_n_accuracy
from src.utils.net import get_device, Device
from src.model.guided_diff.classifier import load_guided_classifier


@th.no_grad()
def main():
    inet_dir = Path.home() / "data/small-imagenet"
    device = get_device(Device.GPU)
    img_size = 256  # With Resize transf removed, I don't think this actually affects size
    batch_size = 25
    _, val = get_imagenet_data_loaders(inet_dir, img_size, batch_size, transform=guided_diff_transf(img_size))

    # class_ = regnet_x_8gf(weights=RegNet_X_8GF_Weights.IMAGENET1K_V2)
    # class_.eval()
    classifier_path = Path.cwd() / "models/256x256_classifier.pt"
    class_ = load_guided_classifier(model_path=classifier_path, dev=device, image_size=img_size)
    class_.eval()
    class_.to(device)
    ts = th.ones((batch_size,)).long().to(device)
    class_fn = partial(class_, timesteps=ts)
    acc = 0.0
    for batch_idx, batch in enumerate(val, 1):
        if batch_idx % 10 == 0:
            print(f"{batch_idx} / {len(val)}")
        x, y_true = batch["x"], batch["y"]
        x, y_true = x.to(device), y_true.to(device)
        logits = class_fn(x)
        top_n_acc = top_n_accuracy(logits, y_true, 10).cpu().item()
        batch_acc = accuracy(hard_label_from_logit(logits), y_true).cpu().item()
        print("batch acc: ", batch_acc)
        acc += batch_acc
        # print(batch_acc, top_n_acc)
    acc /= len(val)
    print("Accuracy: ", acc)


def parse_args():
    parser = ArgumentParser(prog="Test classifier accuracy at t=0")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "imagenet"], help="Dataset selection")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument(
        "--arch", default="unet", type=str, choices=["unet", "resnet", "guided_diff"], help="Model architecture to use"
    )
    parser.add_argument(
        "--sim_batch", type=int, default=0, help="Simulation batch index, indexes parallell simulations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
