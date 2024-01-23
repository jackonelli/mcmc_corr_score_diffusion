import sys
sys.path.append(".")
from src.utils.fid_utils import compute_fid_statistics_thfiles
from argparse import ArgumentParser
from pathlib import Path
from pytorch_fid.fid_score import (save_fid_stats, calculate_fid_given_paths, compute_statistics_of_path,
                                   calculate_frechet_distance)
from pytorch_fid.inception import InceptionV3
import torch
import os


def main():
    args = parse_args()
    fid_generate = Path(args.path_generated)

    fid_real = Path(args.path_real)
    if fid_real.parts[-1][-3:] == 'npz':
        statistic_file = True
    else:
        statistic_file = False

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

    if args.save_stats and not statistic_file:
        fid_real_save = os.path.join(fid_real.absolute().as_posix(), fid_real.stem + '_statistics.npz')
        paths = [fid_real.absolute().as_posix(), fid_real_save]
        save_fid_stats(paths, args.batch_size, device, args.dims, num_workers)
        fid_real = Path(fid_real_save)

    if args.file_type == 'th':
        m1, s1, model = compute_fid_statistics_thfiles(fid_generate, device, batch_size=args.batch_size, dims=args.dims,
                                                       num_workers=num_workers)
        m2, s2 = compute_statistics_of_path(fid_real.absolute().as_posix(), model, args.batch_size, args.dims, device,
                                            num_workers)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    else:
        paths = [fid_real.absolute().as_posix(), fid_generate.absolute().as_posix()]
        fid_value = calculate_fid_given_paths(paths,
                                              args.batch_size,
                                              device,
                                              args.dims,
                                              num_workers)
    print('FID: ', fid_value)


def parse_args():
    parser = ArgumentParser(prog="Compute FID score")
    parser.add_argument("--path_generated", type=str,
                        help="Path to folder with generated samples")
    parser.add_argument("--file_type", choices=['jpeg', 'th'], help="Type of file to look for in folder "
                                                                   "- assumes that .th-files are include "
                                                                   "samples in the name")
    parser.add_argument("--path_real", type=str,
                        help="Either path to folder of real samples or path to FID statistics file (.npz)")
    parser.add_argument('--batch_size', type=int, default=50,
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
    parser.add_argument('--save_stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples given by path_real.'))
    return parser.parse_args()


if __name__ == '__main__':
    main()
