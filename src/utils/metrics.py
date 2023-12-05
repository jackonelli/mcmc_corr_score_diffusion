from pathlib import Path
import csv
import numpy as np
import torch as th


def hard_label_from_logit(logit):
    return logit.argmax(dim=1)


def accuracy(pred_class, true_class):
    """Accuracy from hard labels"""
    return (pred_class == true_class).float().mean()


def top_n_accuracy(logits, true_class, n):
    """Top-n accuracy"""
    true_class = true_class.clone().reshape((true_class.size(0), 1))
    top_n = logits.sort(descending=True, dim=1).indices[:, :n]
    exists = th.any(th.eq(top_n, true_class), dim=1)
    return exists.float().mean()


def mahalanobis_diagonal(u, v, cov):
    """Mahalanobis distance for diagonal cov. matrices"""
    delta = u - v
    inv_diag = 1 / cov
    m = th.sum(inv_diag * delta * delta)
    return th.sqrt(m)


def mahalanobis(u, v, cov):
    """Mahalanobis distance for general cov. matrices"""
    delta = u - v
    m = th.dot(delta, th.matmul(th.inverse(cov), delta))
    return th.sqrt(m)


def parse_diff_metrics(path: Path):
    """Parse metrics from lightning logs"""
    train_losses, val_losses = [], []
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        _ = next(reader)
        for row in reader:
            train_l, _, step, val_l = row
            if train_l:
                train_losses.append((int(step), float(train_l)))
            if val_l:
                val_losses.append((int(step), float(val_l)))

    return np.array(train_losses), np.array(val_losses)
