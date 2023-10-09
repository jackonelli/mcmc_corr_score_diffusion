"""Script for sampling with classifier-full guidance with MCMC"""
from argparse import ArgumentParser
from pathlib import Path
import torch as th
from src.guidance.base import MCMCGuidanceSampler
from src.guidance.classifier_full import ClassifierFullGuidance
from src.samplers.mcmc import AnnealedHMCScoreSampler
from src.model.resnet import load_classifier
from src.utils.net import Device, get_device
from src.diffusion.base import DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.unet import load_mnist_diff
from src.utils.vis import plot_samples_grid


def main():
    args = parse_args()
    device = get_device(Device.GPU)
    models_dir = Path.cwd() / "models"
    uncond_diff = load_mnist_diff(models_dir / "uncond_unet_mnist.pt", device)
    classifier = _load_class(models_dir / "resnet_classifier_t_mnist.pt", device)
    T = 1000
    diff_sampler = DiffusionSampler(improved_beta_schedule, num_diff_steps=T)
    diff_sampler.to(device)
    mcmc_steps = 4
    # step_sizes = diff_sampler.betas * 0.005
    # step_sizes[70:400] = diff_sampler.betas[70:400] * 0.001
    # step_sizes[:70] = diff_sampler.betas[:70] * 0.0001
    # a = 0.1
    # b = 1.8
    a = 0.05
    b = 1.6
    step_sizes = a * diff_sampler.betas**b
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.semilogy(step_sizes.cpu())
    # plt.semilogy(a * diff_sampler.betas.cpu() ** b)
    # plt.title(r'$a \beta^b, \; a={}, b={}$'.format(a, b))
    # plt.show()
    mcmc_sampler = AnnealedHMCScoreSampler(mcmc_steps, step_sizes, 0.9, diff_sampler.betas, 3, None)
    guidance = ClassifierFullGuidance(classifier, lambda_=args.guid_scale)
    guided_sampler = MCMCGuidanceSampler(
        diff_model=uncond_diff,
        diff_proc=diff_sampler,
        guidance=guidance,
        mcmc_sampler=mcmc_sampler,
        reverse=True,
        verbose=True,
    )
    num_samples = args.num_samples
    th.manual_seed(0)
    classes = th.randint(10, (num_samples,), dtype=th.int64)
    # classes = th.ones((num_samples,), dtype=th.int64)
    samples, _ = guided_sampler.sample(num_samples, classes, device, th.Size((1, 28, 28)))
    import pickle

    run = 4
    data = dict()
    data["samples"] = samples.detach().cpu()
    data["accepts"] = guided_sampler.mcmc_sampler.accepts
    data["parameters"] = {"stepsizes": step_sizes.detach().cpu(), "a": a, "b": b}
    data["classes"] = classes
    pickle.dump(data, open("data_run{}.p".format(str(run)), "wb"))
    # plot_samples_grid(samples.detach().cpu())


def _load_class(class_path: Path, device):
    classifier = load_classifier(class_path, True)
    classifier.to(device)
    classifier.eval()
    return classifier


def parse_args():
    parser = ArgumentParser(prog="Sample with classifier-full guidance")
    parser.add_argument("--guid_scale", default=1.0, type=float, help="Guidance scale")
    parser.add_argument("--num_samples", default=100, type=int, help="Num samples (batch size to run in parallell)")
    return parser.parse_args()


if __name__ == "__main__":
    # import pickle
    # data0 = pickle.load(open("data_run1.p", "rb"))
    # data1 = pickle.load(open("samples_mnist_reverse_lambda1_run2.p", "rb"))
    # data1 = pickle.load(open("data_run1.p", "rb"))
    # device = get_device(Device.GPU)
    # models_dir = Path.cwd() / "models"
    # classifier = _load_class(models_dir / "resnet_classifier_t_mnist.pt", device)
    # sm = classifier(data0['samples'].cuda(), th.full((data0['samples'].shape[0],), 0, device=device))
    # sm = classifier(data1.cuda(), th.full((data1.shape[0],), 0, device=device))
    # pred = th.argmax(sm.detach().cpu(), axis=1)
    # print(th.sum(pred == 1)/100)
    main()
