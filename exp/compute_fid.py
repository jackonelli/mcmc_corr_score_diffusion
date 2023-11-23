from src.utils.file_mangement import npz_images_to_png
from argparse import ArgumentParser
from pathlib import Path
from pytorch_fid.fid_score import save_fid_stats, calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
import torch
import os


def main():
    args = parse_args()
    images_dir = Path.cwd() / "images"

    if args.name_generated.split('.')[-1] == 'npz':
        fid_generate = npz_images_to_png(images_dir/args.name_generated, images_dir)
    else:
        fid_generate = images_dir / args.name_generated

    fid_real = images_dir / args.name_real

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

    if args.save_stats:
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    fid_value = calculate_fid_given_paths([fid_generate.absolute().as_posix(), fid_real.absolute().as_posix()],
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID: ', fid_value)


def parse_args():
    parser = ArgumentParser(prog="Compute FID score")
    parser.add_argument("--name_generated", type=str,
                        help="Either name to folder with generated samples or name of npz-file")
    parser.add_argument("--name_real", type=str,
                        help="Either name to folder of real samples or name of FID statistics file")
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--save-stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples. '
                              'The first path is used as input and the second as output.'))
    return parser.parse_args()


if __name__ == '__main__':
    main()
