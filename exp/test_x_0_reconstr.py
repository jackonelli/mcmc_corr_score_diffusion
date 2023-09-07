"""Script for sampling with reconstruction guidance"""
from pathlib import Path
import torch as th
import torch.nn.functional as F
from src.guidance.reconstruction import ReconstructionGuidance, ReconstructionSampler
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.vis import every_nth_el, plot_reconstr_diff_seq, plot_accs
from src.utils.classification import accuracy, logits_to_label


@th.no_grad()
def main():
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet_reconstruction_classifier_mnist.pt", device)
    T = 1000
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps=T)
    diff_sampler.to(device)
    num_samples = 100
    print("Sampling x_0:T")
    x_0, samples = diff_sampler.sample(uncond_diff, num_samples, device, th.Size((1, 28, 28)))

    guidance = ReconstructionGuidance(uncond_diff, classifier, diff_sampler.alphas_bar.clone(), F.cross_entropy)
    print("Reconstructing x_0")
    # Dummy samples
    # samples = 10 * [(12, th.randn((num_samples, 1, 28, 28)).to(device))]
    # x_0 = th.randn((num_samples, 1, 28, 28)).to(device)

    single_samples = [(t, x_t_batch[0, :, :, :].reshape((1, 1, 28, 28))) for (t, x_t_batch) in samples]
    thinned_samples = every_nth_el(single_samples, every_nth=100)
    pred_samples = predict_x_0(thinned_samples, guidance, device)
    plot_reconstr_diff_seq(thinned_samples, pred_samples)

    print("Classifying x_0:T")
    # Assume we can classify x_0 correctly.
    pred_class = logits_to_label(classifier(x_0))
    accs = reconstr_accuracy(samples, classifier, guidance, pred_class)
    plot_accs(accs)


def reconstr_accuracy(samples, classifier, guidance, ys):
    accs = []
    for t, x_t_batch in samples:
        x_t_batch = x_t_batch.to(th.device("cuda"))
        base_pred_ys = logits_to_label(classifier(x_t_batch))
        base_acc_t = accuracy(ys, base_pred_ys)
        x_0_hat = guidance.predict_x_0(x_t_batch, t)
        rec_pred_ys = logits_to_label(classifier(x_0_hat))
        rec_acc_t = accuracy(ys, rec_pred_ys)
        accs.append((t, base_acc_t, rec_acc_t))
    return accs


def predict_x_0(samples, guidance, device):
    predicted_samples = []
    for t, x_t in samples:
        x_t = x_t.to(device)
        x_0_hat = guidance.predict_x_0(x_t, t)
        predicted_samples.append((t, x_0_hat))
    return predicted_samples


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path)
    classifier.to(device)
    classifier.eval()
    return classifier


if __name__ == "__main__":
    main()
