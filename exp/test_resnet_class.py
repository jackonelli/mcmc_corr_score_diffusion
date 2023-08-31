"""Prototype script for testing ResNet MNIST classifier"""


from pathlib import Path
import torch as th
from src.model.resnet import load_classifier
from src.data.mnist import get_mnist_data_loaders
from src.utils.net import Device, get_device


@th.no_grad()
def main():
    model_path = Path.cwd() / "models/resnet.pth.tar"
    model = load_classifier(model_path)
    device = get_device(Device.GPU)
    model.to(device)
    model.eval()

    batch_size = 256
    _, val_loader = get_mnist_data_loaders(batch_size)
    accs = []
    for batch_idx, batch in enumerate(val_loader):
        print(f"Batch {batch_idx}/{len(val_loader)}")
        x, y = batch["pixel_values"].to(device), batch["label"].to(device)
        x = mnist_transform(x)
        logits = model(x)
        y_pred = logits_to_label(logits)
        batch_acc = accuracy(y, y_pred)
        accs.append(batch_acc)
    accs = th.Tensor(accs)
    print(f"Accuracy: {accs.mean()}+/-{accs.std()}")


def mnist_transform(data):
    """Map MNIST pixel values from [-1,1] to [0, 1]"""
    return (data + 1) / 2


@th.no_grad()
def accuracy(y_true: th.Tensor, y_pred: th.Tensor) -> float:
    """Compute accuracy of class predictions

    Args:
        y_true: (batch_size,)
        y_pred: (batch_size,)

    Returns:
        acc: average accuracy
    """
    accs = y_true == y_pred
    return accs.float().mean().item()


@th.no_grad()
def logits_to_label(logits):
    """Convert logits to hard label

    Args:
        logits: (batch_size, num_classes)

    Returns:
        labels: (batch_size, num_classes)
    """
    return th.argmax(logits, dim=1)


if __name__ == "__main__":
    main()
