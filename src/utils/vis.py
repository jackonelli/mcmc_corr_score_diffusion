"""Visualisation"""
from typing import Tuple
import matplotlib.pyplot as plt


def plot_samples_grid(samples, grid=(10, 10)):
    """Plot grid of images"""
    num_rows, num_cols = grid
    _, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(samples[i * 10 + j].squeeze(), cmap="gray")
            axes[i, j].axis("off")
    plt.show()


def plot_accs(accs: list[Tuple[int, float, float]]):
    """Plot comparison of classification accuracy for reconstruction

    base: classifier p(y | x_t)
    rec: classifer p(y | x_0_hat(x_t))
    """
    t, base, rec = zip(*accs)
    _, ax = plt.subplots()
    ax.plot(t, base, "-b", label="diff")
    ax.plot(t, rec, "-r", label="rec")
    ax.legend()
    plt.show()


def plot_reconstr_diff_seq(orig, reconstr):
    """Plot list of diffused images"""
    num_cols = len(orig)
    assert len(reconstr) == num_cols

    _, axes = plt.subplots(2, num_cols, figsize=(8, 8))
    for j in range(num_cols):
        t, x_t = orig[j]
        x_t = x_t.cpu()
        axis = axes[0, j]
        axis.imshow(x_t.squeeze(), cmap="gray")
        axis.axis("off")
        axis.set_title(str(t))

    for j in range(num_cols):
        t, x_t = reconstr[j]
        x_t = x_t.cpu()
        axis = axes[1, j]
        axis.imshow(x_t.squeeze(), cmap="gray")
        axis.axis("off")
        axis.set_title(str(t))
    plt.show()


def every_nth_el(seq, every_nth):
    """Take every n'th element and the last in sequence"""
    final = seq[-1]
    seq = seq[::every_nth]
    seq += [final]
    return seq


def plot_diff_seq(diff_seq, every_nth=1):
    """Plot list of diffused images"""
    diff_seq = every_nth_el(diff_seq)
    num_cols = len(diff_seq)
    _, axes = plt.subplots(1, num_cols, figsize=(8, 8))
    for j in range(num_cols):
        t, x_t = diff_seq[j]
        axis = axes[j]
        axis.imshow(x_t.squeeze(), cmap="gray")
        axis.axis("off")
        axis.set_title(str(t))
    plt.show()
