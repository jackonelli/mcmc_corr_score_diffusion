# Diff models
from src.model.cifar.utils import get_diff_model, select_cifar_classifier
from src.model.guided_diff.unet import load_pretrained_diff_unet
from src.model.unet import load_mnist_diff

# Beta Schedules
from src.diffusion.beta_schedules import (
    improved_beta_schedule,
    linear_beta_schedule,
)

# Classifiers
from src.model.resnet import load_classifier_t as load_resnet_classifier_t
from src.model.guided_diff.classifier import load_guided_classifier as load_guided_diff_classifier_t

from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS


def load_models(config, device, MODELS_DIR):
    diff_model_name = f"{config.diff_model}"
    diff_model_path = MODELS_DIR / f"{diff_model_name}"
    assert diff_model_path.exists(), f"Model '{diff_model_path}' does not exist."
    energy_param = "energy" in diff_model_name

    assert not (config.class_cond and "uncond" in config.diff_model)
    classifier_name = f"{config.classifier}"
    classifier_path = MODELS_DIR / f"{classifier_name}"
    assert classifier_path.exists(), f"Model '{classifier_path}' does not exist."
    if "mnist" in diff_model_name:
        dataset_name = "mnist"
        beta_schedule, post_var = improved_beta_schedule, "beta"
        image_size, num_classes, num_channels = (28, 10, 1)
        diff_model = load_mnist_diff(diff_model_path, device)
        diff_model.eval()
        classifier = load_resnet_classifier_t(
            model_path=classifier_path,
            dev=device,
            num_channels=num_channels,
            num_classes=num_classes,
        )
        classifier.eval()
    elif "cifar100" in diff_model_name:
        dataset_name = "cifar100"
        if "cos" in diff_model_name:
            beta_schedule, post_var = improved_beta_schedule, "beta"
        else:
            beta_schedule, post_var = linear_beta_schedule, "beta"
        image_size, num_classes, num_channels = (CIFAR_IMAGE_SIZE, CIFAR_100_NUM_CLASSES, CIFAR_NUM_CHANNELS)
        diff_model = get_diff_model(name=diff_model_name,
                                    diff_model_path=diff_model_path,
                                    device=device,
                                    energy_param=energy_param,
                                    image_size=CIFAR_IMAGE_SIZE,
                                    num_steps=config.num_diff_steps)
        diff_model.eval()
        classifier = select_cifar_classifier(model_path=classifier_path, dev=device, num_steps=config.num_diff_steps)
        classifier.eval()
    elif 'cifar10' in diff_model_name:
        dataset_name = "cifar10"
        if "cos" in diff_model_name:
            beta_schedule, post_var = improved_beta_schedule, "beta"
        else:
            beta_schedule, post_var = linear_beta_schedule, "beta"
        image_size, num_classes, num_channels = (CIFAR_IMAGE_SIZE, 10, CIFAR_NUM_CHANNELS)
        diff_model = get_diff_model(diff_model_name, diff_model_path, device, energy_param, CIFAR_IMAGE_SIZE,
                                    config.num_diff_steps)
        diff_model.eval()
        classifier = select_cifar_classifier(model_path=classifier_path, dev=device, num_steps=config.num_diff_steps)
        classifier.eval()
    elif f"{config.image_size}x{config.image_size}_diffusion" in diff_model_name:
        dataset_name = "imagenet"
        beta_schedule, post_var = linear_beta_schedule, "learned"
        image_size, num_classes, num_channels = (config.image_size, 1000, 3)
        diff_model = load_pretrained_diff_unet(
            model_path=diff_model_path, dev=device, class_cond=config.class_cond, image_size=image_size
        )
        diff_model.eval()
        classifier = load_guided_diff_classifier_t(model_path=classifier_path, dev=device, image_size=image_size)
        classifier.eval()
    else:
        print(f"Incorrect model '{diff_model_name}'")
        raise ValueError
    return (
        diff_model,
        classifier,
        (dataset_name, image_size, num_classes, num_channels),
        beta_schedule,
        post_var,
        energy_param,
    )
