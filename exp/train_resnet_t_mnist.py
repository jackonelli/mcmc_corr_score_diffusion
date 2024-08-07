"""Script for training a ResNet-based classifier for x_t on MNIST

For classifier-full guidance, we assume access to a likelihood function
p(y | x_t, t), which we estimate by a ResNet-based classifier which takes the diffusion step t as a parameter.
"""


from pathlib import Path
import torch as th
import pytorch_lightning as pl
from src.diffusion.base import DiffusionClassifier, DiffusionSampler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.resnet import ResNetTimeEmbedding, BottleneckTimeEmb
from src.utils.net import get_device, Device
from src.data.mnist import get_mnist_data_loaders


def main():
    n_classes = 10
    time_emb_dim = 112
    channels = 1
    num_diff_steps = 1000
    batch_size = 225
    model_path = Path.cwd() / "models" / "mnist_classifier_t.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)

    resnet = ResNetTimeEmbedding(
        block=BottleneckTimeEmb,
        num_blocks=[3, 4, 6, 3],
        emb_dim=time_emb_dim,
        num_classes=n_classes,
        num_channels=channels,
    ).to(dev)
    resnet.train()

    betas = improved_beta_schedule(num_timesteps=num_diff_steps)
    time_steps = th.tensor([i for i in range(num_diff_steps)])
    noise_scheduler = DiffusionSampler(betas, time_steps)

    diff_classifier = DiffusionClassifier(
        model=resnet, loss_f=th.nn.CrossEntropyLoss(), noise_scheduler=noise_scheduler
    )
    diff_classifier.to(dev)

    trainer = pl.Trainer(max_epochs=20, num_sanity_val_steps=0, accelerator="gpu", devices=1)

    dataloader_train, dataloader_val = get_mnist_data_loaders(batch_size)
    trainer.fit(diff_classifier, dataloader_train, dataloader_val)

    print(f"Saving model to {model_path}")
    th.save(resnet.state_dict(), model_path)


if __name__ == "__main__":
    main()
