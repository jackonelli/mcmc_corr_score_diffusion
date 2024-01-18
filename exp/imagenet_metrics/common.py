import re
import torch as th
from src.utils.metrics import accuracy, hard_label_from_logit, prob_vec_from_logit, r3_accuracy


PATTERN = re.compile(r".*_(\d+)_(\d+).th")


def compute_acc(classifier, classes_and_samples, batch_size, device):
    pred_logits = []
    true_classes = []
    for i, (classes_path, samples_path) in enumerate(classes_and_samples):
        print(f"File {i+1}/{len(classes_and_samples)}")
        samples = th.load(samples_path)
        num_samples = samples.size(0)
        for batch in th.chunk(samples, num_samples // batch_size):
            batch = batch.to(device)
            ts = th.zeros((batch.size(0),)).to(device)
            pred_logits.append(classifier(batch, ts).detach().cpu())

        classes = th.load(classes_path).detach().cpu()
        true_classes.append(classes)
    pred_logits = th.cat(pred_logits, dim=0)
    true_classes = th.cat(true_classes, dim=0)
    simple_acc = accuracy(hard_label_from_logit(pred_logits), true_classes)
    r3_acc = r3_accuracy(prob_vec_from_logit(pred_logits), true_classes)
    return simple_acc, r3_acc
