"""Prototype script for testing ResNet MNIST classifier"""


from pathlib import Path
import torch as th
from src.model.resnet import load_classifier, load_classifier_t
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.data.mnist import get_mnist_data_loaders
from src.model.resnet import load_classifier
from src.data.mnist import get_mnist_data_loaders
from src.utils.net import Device, get_device
import matplotlib.pyplot as plt


@th.no_grad()
def test_reconstruction_classifier():
    model_path = Path.cwd() / "models/resnet_reconstruction_classifier_mnist.pt"
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
        logits = model(x)
        y_pred = logits_to_label(logits)
        batch_acc = accuracy(y, y_pred)
        accs.append(batch_acc)
    accs = th.Tensor(accs)
    print(f"Accuracy: {accs.mean()}+/-{accs.std()}")


@th.no_grad()
def test_classifier_t():
    model_path = Path.cwd() / "models/resnet_classifier_t_mnist.pt"
    model = load_classifier_t(model_path)
    device = get_device(Device.GPU)
    model.to(device)
    model.eval()

    num_diff_steps = 1000
    noise_scheduler = DiffusionSampler(improved_beta_schedule, num_diff_steps)
    n_points = 10
    ts = th.linspace(0, 999, n_points).type(th.int).numpy().tolist()
    batch_size = 256
    _, val_loader = get_mnist_data_loaders(batch_size)
    accs = {t: list() for t in ts}
    for t in ts:
        print(t)
        for batch_idx, batch in enumerate(val_loader):
            # print(f"Batch {batch_idx}/{len(val_loader)}")
            x, y = batch["pixel_values"].to(device), batch["label"].to(device)
            bs = x.shape[0]

            t_tensor = (t * th.ones(bs, dtype=th.int)).to(device)
            noise = th.randn_like(x)
            x_noisy = noise_scheduler.q_sample(x_0=x, ts=t_tensor.long(), noise=noise)

            logits = model(x_noisy, t_tensor)
            y_pred = logits_to_label(logits)
            batch_acc = accuracy(y, y_pred)
            accs[t].append(batch_acc)
        accs[t] = th.Tensor(accs[t])
        print(f"t={t}, Accuracy: {accs[t].mean()}+/-{accs[t].std()}")
        accs[t] = th.Tensor(accs[t].mean())
    plt.plot(ts, accs.values())
    plt.show()


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
    test_reconstruction_classifier()
