"""Visualisation"""
import matplotlib.pyplot as plt


def plot_samples(samples, grid=(10, 10)):
    """Plot grid of images"""
    num_rows, num_cols = grid
    _, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(samples[i * 10 + j].squeeze(), cmap="gray")
            axes[i, j].axis("off")
    plt.show()
