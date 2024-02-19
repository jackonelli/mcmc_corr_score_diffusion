import re
from typing import Tuple
import torch as th
from src.utils.metrics import accuracy, hard_label_from_logit, prob_vec_from_logit, r3_accuracy, top_n_accuracy


PATTERN = re.compile(r".*_(\d+)_(\d+).th")


def compute_acc(classifier_fn, classes_and_samples, transform, batch_size, device) -> Tuple[float, float, float, int]:
    pred_logits = []
    true_classes = []
    for _, (classes_path, samples_path) in enumerate(classes_and_samples):
        samples = th.load(samples_path)
        num_samples = samples.size(0)
        if num_samples > batch_size:
            for batch in th.chunk(samples, num_samples // batch_size):
                batch = batch.to(device)
                batch = transform(batch)
                pred_logits.append(classifier_fn(batch).detach().cpu())
        else:
            batch = transform(samples.to(device))
            pred_logits.append(classifier_fn(batch).detach().cpu())

        classes = th.load(classes_path).detach().cpu()
        true_classes.append(classes)
    # print(len(pred_logits), pred_logits[0].size())
    pred_logits = th.cat(pred_logits, dim=0)
    true_classes = th.cat(true_classes, dim=0)
    simple_acc = accuracy(hard_label_from_logit(pred_logits), true_classes)
    r3_acc = r3_accuracy(prob_vec_from_logit(pred_logits), true_classes)
    top_5_acc = top_n_accuracy(pred_logits, true_classes, 5)
    return simple_acc.item(), r3_acc.item(), top_5_acc.item(), true_classes.size(0)
