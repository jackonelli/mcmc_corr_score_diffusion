import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


def npz_images_to_png(file_path: Path, dir_path: Path, key: str = 'arr_0'):
    """
    Converts a npz-file with data on the format (num_images, x_dim, y_dim, 3) to png images
    and places them in a folder created at 'dir_path' (with the name as the file)

    @param file_path: Path to file.npz
    @param dir_path: Path to directory where the folder with the images is placed
    @param key: The key to the data (assumes data is a dict)
    """
    images = np.load(file_path)[key]
    n = images.shape[0]
    make_dir = True
    text = ''
    counter = 0
    new_dir_path = None
    while make_dir:
        new_dir_path = dir_path / (file_path.stem + text)
        if not new_dir_path.is_dir():
            make_dir = False
            os.makedirs(new_dir_path)
        counter += 1
        text = '_' + str(counter)

    for i in range(n):
        plt.imsave(new_dir_path / '{}.png'.format(str(i)), images[i])

    return new_dir_path


def find_num_trained_steps(name):
    parts = name.split('.')[0].split('_')
    n_trained = [int(part[5:]) for part in parts if 'step=' in part][0]
    return n_trained
