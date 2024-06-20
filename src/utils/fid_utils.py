from PIL import Image
import pathlib
import torch as th
import numpy as np
from pytorch_fid.inception import InceptionV3
from src.utils.image_utils import convert_to_transformed_data
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as TF
import os
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


class PILDataset(th.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def compute_fid_statistics_dataloader(model, dataloader, device, dims):
    pred_arr = np.empty((len(dataloader.dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with th.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def get_model(device, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def dataset_thfiles(path_folder, data_name='samples', num_samples=None):
    files = os.listdir(path_folder)
    files = [os.path.join(path_folder, file) for file in files if file.split('.')[-1] == 'th' and data_name in file]
    images_ = [th.load(file) for file in files]
    dim = list(images_[0].shape)
    dim[0] = sum([im.shape[0] for im in images_])
    images = th.empty(dim)
    i = 0
    for im in images_:
        images[i:i+im.shape[0], ...] = im
        i += im.shape[0]
        if num_samples is not None and i > num_samples:
            break

    if num_samples is not None and i > num_samples:
        images = images[:num_samples]
    images = convert_to_transformed_data(images)
    return images


def dataset_jpeg(path_folder):
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                        'tif', 'tiff', 'webp'}
    path = pathlib.Path(path_folder)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])
    dataset = PILDataset(files, transforms=TF.ToTensor())
    return dataset


def compute_fid_statistics_thfiles(path_folder, device, data_name='samples', batch_size=50, dims=2048, num_workers=1):
    """
    Compute fid statistics for all .th-files in a given folder
    @param path_folder: path to folder with .th-files
    @param device: utilization of CPU or GPU for model
    @param data_name: assumes that the .th-files are stored with certain names
    @param batch_size: batch size for the model
    @param dims: Dimensionality of Inception features to use. By default, uses pool3 features'
    @param num_workers: Number of processes to use for data loading. Defaults to `min(8, num_cpus)`
    @return:
    """
    files = os.listdir(path_folder)
    files = [os.path.join(path_folder, file) for file in files if file.split('.')[-1] == 'th' and data_name in file]
    images_ = [th.load(file) for file in files]
    dim = list(images_[0].shape)
    dim[0] = sum([im.shape[0] for im in images_])
    images = th.empty(dim)
    i = 0
    for im in images_:
        images[i:i+im.shape[0], ...] = im
        i += im.shape[0]
    images = (images + 1).clamp(0., 1.)
    dataloader = th.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, drop_last=False,
                                          num_workers=num_workers)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    pred_arr = np.empty((images.shape[0], dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with th.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma, model



def compute_fid_statistics_thfiles_split(model, device, dims, path_folder, data_name='samples', num_samples=None):
    model.eval()

    files = os.listdir(path_folder)
    files = [os.path.join(path_folder, file) for file in files if file.split('.')[-1] == 'th' and data_name in file]

    if len(files) > 0:
        images = th.load(files[0])
        n = len(files) * images.shape[0]

        pred_arr = np.empty((n, dims))

        start_idx = 0
        for file in files:
            images = th.load(file)
            images = (images + 1).clamp(0., 1.)

            images = images.to(device)

            with th.no_grad():
                pred = model(images)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

            if num_samples is not None and start_idx >= num_samples - 1:
                pred_arr = pred_arr[:num_samples]
                break

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
    else:
        mu = None
        sigma = None
    return mu, sigma
