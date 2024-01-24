from pathlib import Path
import csv
import numpy as np
import torch as th
import torch.nn.functional as F


def hard_label_from_logit(logit):
    return logit.argmax(dim=1)


def prob_vec_from_logit(logit):
    return F.softmax(logit, dim=1)


def r3_accuracy(prob_vec, true_class):
    """Compute accuracy with the condition that the prediction is correct only when p(y_true | x) > 50%"""
    probs_y_true = prob_vec[th.arange(prob_vec.size(0)), true_class]
    return (probs_y_true > 0.5).float().mean()


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
    lrs, train_losses, val_losses = [], [], []
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        _ = next(reader)
        for row in reader:
            lr, step, train_l, _, val_l, acc = row
            if train_l:
                train_losses.append((int(step), float(train_l)))
            if lr:
                lrs.append((int(step), float(lr)))

            if val_l:
                val_losses.append((int(step), float(val_l)))

    return np.array(lrs), np.array(train_losses), np.array(val_losses)
