import sys
sys.path.append(".")
from src.utils.fid_utils import get_model, dataset_thfiles, dataset_jpeg, compute_fid_statistics_dataloader, PILDataset
from argparse import ArgumentParser
from pytorch_fid.fid_score import (save_fid_stats, calculate_fid_given_paths, compute_statistics_of_path,
                                   calculate_frechet_distance)
import torchvision.transforms as TF
from pytorch_fid.inception import InceptionV3
from datasets import load_dataset
import torch as th
import numpy as np
import torch
import os


def get_statistics(model, device, batch_size, dims, path_dataset, type_dataset, num_workers, path_save_stats,
                   num_samples=None):
    if type_dataset == 'th':
        dataset = dataset_thfiles(path_dataset, num_samples=num_samples)
        if th.any(dataset.isnan()).item():
            m, s = None, None
        else:
            dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                                  num_workers=num_workers)
            m, s = compute_fid_statistics_dataloader(model, dataloader, device, dims)
    elif type_dataset == 'cifar100_train':
        if path_dataset is not None:
            path_dataset = str(path_dataset)
        dataset = PILDataset(load_dataset("cifar100", cache_dir= path_dataset)['train']['img'], TF.ToTensor())
        dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)
        m, s = compute_fid_statistics_dataloader(model, dataloader, device, dims)
    elif type_dataset == 'cifar100_val':
        if path_dataset is not None:
            path_dataset = str(path_dataset)
        dataset = PILDataset(load_dataset("cifar100", cache_dir=path_dataset)['test']['img'], TF.ToTensor())
        dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)
        m, s = compute_fid_statistics_dataloader(model, dataloader, device, dims)
    elif type_dataset == 'cifar10_train':
        if path_dataset is not None:
            path_dataset = str(path_dataset)
        dataset = PILDataset(load_dataset("cifar10", cache_dir= path_dataset)['train']['img'], TF.ToTensor())
        dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)
        m, s = compute_fid_statistics_dataloader(model, dataloader, device, dims)
    elif type_dataset == 'cifar10_val':
        if path_dataset is not None:
            path_dataset = str(path_dataset)
        dataset = PILDataset(load_dataset("cifar10", cache_dir=path_dataset)['test']['img'], TF.ToTensor())
        dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)
        m, s = compute_fid_statistics_dataloader(model, dataloader, device, dims)
    elif type_dataset == 'stats':
        with np.load(path_dataset) as f:
            m, s = f['mu'][:], f['sigma'][:]
    elif type_dataset == 'jpeg':
        dataset = dataset_jpeg(path_dataset)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)
        m, s = compute_fid_statistics_dataloader(model, dataloader, device, dims)
    else:
        raise ValueError

    if path_save_stats is not None and type_dataset != 'stats':
        np.savez_compressed(os.path.join(path_save_stats, type_dataset), mu=m, sigma=s)
    return m, s


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    # Get Model
    model = get_model(device, dims=args.dims)

    # Dataset 1
    m1, s1 = get_statistics(model, device, args.batch_size, args.dims, args.path_dataset1, args.type_dataset1, num_workers, args.path_save_stats_1)

    # Dataset 2
    m2, s2 = get_statistics(model, device, args.batch_size, args.dims, args.path_dataset2, args.type_dataset2, num_workers, args.path_save_stats_2)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    print('FID: ', fid_value)


def parse_args():
    parser = ArgumentParser(prog="Compute FID score")
    parser.add_argument("--path_dataset1", type=str, default=None,
                        help='Path to folder with data. If type is choosen to a dataset (e.g., cifar100), '
                             'then no path is needed')
    parser.add_argument("--type_dataset1", type=str, default=None, choices= ['th', 'jpeg', 'stats', 'cifar100_train',
                                                                             'cifar100_val', 'cifar10_train', 'cifar10_val'],
                        help=('Type of data. If th is chosen then all files that include samples in the name are '
                              'used. The choice stats assumes that the path is a npz file with statistics.'))
    parser.add_argument("--path_save_stats_1", default=None, type=str,
                        help='Path to save statistics of dataset 1. If no one is given then no save.')
    parser.add_argument("--path_dataset2", type=str, default=None,
                        help=('Path to folder with data. If type is choosen to a dataset (e.g., cifar100), '
                             'then no path is needed')
    )
    parser.add_argument("--type_dataset2", type=str, default=None, choices=['th', 'jpeg', 'stats', 'cifar100_train',
                                                                            'cifar100_val', 'cifar10_train', 'cifar10_val'],
                        help=('Type of data. If th is chosen then all files that include samples in the name are '
                              'used. The choice stats assumes that the path is a npz file with statistics.'))
    parser.add_argument("--path_save_stats_2", default=None, type=str,
                        help='Path to save statistics of dataset 2. If no one is given then no save.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size to use')
    parser.add_argument('--num_workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    return parser.parse_args()


if __name__ == '__main__':
    main()
