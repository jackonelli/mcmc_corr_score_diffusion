"""Script for training a ResNet-based classifier for x_t on MNIST"""


from pathlib import Path
import torch as th
import pytorch_lightning as pl
from src.diffusion.base import DiffusionClassifier, NoiseScheduler
from src.diffusion.beta_schedules import improved_beta_schedule
from src.model.resnet import ResNetTimeEmbedding, BottleneckTimeEmb
from src.utils.net import get_device, Device
from src.data.mnist import get_mnist_data_loaders


def main():
    n_classes = 10
    time_emb_dim = 112
    channels = 1
    num_diff_steps = 1000
    batch_size = 256
    model_path = Path.cwd() / "models" / "resnet_classifier_t_mnist.pt"
    if not model_path.parent.exists():
        print(f"Save dir. '{model_path.parent}' does not exist.")
        return

    dev = get_device(Device.GPU)

    resnet = ResNetTimeEmbedding(block=BottleneckTimeEmb,
                                 num_blocks=[3, 4, 6, 3],
                                 emb_dim=time_emb_dim,
                                 num_classes=n_classes,
                                 num_channels=channels).to(dev)
    resnet.train()

    noise_scheduler = NoiseScheduler(improved_beta_schedule, num_diff_steps)

    diff_classifier = DiffusionClassifier(model=resnet, loss_f=th.nn.CrossEntropyLoss(), noise_scheduler=noise_scheduler)
    diff_classifier.to(dev)

    trainer = pl.Trainer(max_epochs=20, num_sanity_val_steps=0, accelerator="gpu", devices=1)

    dataloader_train, dataloader_val = get_mnist_data_loaders(batch_size)
    trainer.fit(diff_classifier, dataloader_train, dataloader_val)

    print("Saving model")
    th.save(resnet.state_dict(), model_path)


if __name__ == '__main__':
    main()
