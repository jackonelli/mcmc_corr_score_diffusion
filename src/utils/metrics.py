from pathlib import Path
import csv
import numpy as np


def hard_label_from_logit(logit):
    return logit.argmax()


def accuracy(pred_class, true_class):
    """Accuracy from hard labels"""
    return (pred_class == true_class).float().mean()


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
