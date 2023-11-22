"""Classification utils"""
import torch as th
import torch.nn.functional as F


@th.no_grad()
def accuracy(y_true: th.Tensor, y_pred: th.Tensor) -> float:
    """Compute accuracy of class predictions

    Args:
        y_true: (batch_size,)
        y_pred: (batch_size,)

    Returns:
        acc: average accuracy
    """
    accs = y_true == y_pred
    return accs.float().mean().item()


@th.no_grad()
def entropy(prob_vec: th.Tensor) -> float:
    """Compute entropy of a probability vector

    Args:
        prob_vec: (batch_size, num_classes)

    Returns:
        average entropy
    """
    entropy = -th.sum(prob_vec * th.log(prob_vec), dim=1)
    return entropy.mean().item()


def logits_to_log_prob(logits):
    """Convert logits to log probability vector

    Args:
        logits: (batch_size, num_classes) in R^n

    Returns:
        log_prob_vec: (batch_size, num_classes)
    """
    return F.log_softmax(logits, dim=1)


@th.no_grad()
def logits_to_label(logits):
    """Convert logits to hard label

    Args:
        logits: (batch_size, num_classes)

    Returns:
        labels: (batch_size, num_classes)
    """
    return th.argmax(logits, dim=1)


def logits_to_log_prob_mean(logits):
    """
    Convert logits to log mean of ensemble probabilities
    Multiple logsum-trick

    Args:
        logits: (batch_size, num_classes, num_classifiers) in R^d

    Returns:
        log_prob_vec: (batch_size, num_classes)
    """

    b = logits.shape[0]
    d = logits.shape[1]
    n = logits.shape[2]

    c_l = th.max(logits, dim=1).values
    c_l_expand = c_l.reshape(b, 1, n).expand(b, d, n)
    sum_alpha = (logits - c_l_expand).exp().sum(dim=1)
    alpha = sum_alpha.prod(dim=1)
    alpha_k = alpha.reshape(b, 1).expand(b, n) / sum_alpha
    beta = c_l.sum(dim=1)
    beta_k = beta.reshape(b, 1).expand(b, n) - c_l
    exp_ = logits + beta_k.reshape(b, 1, n)
    c = exp_.max(dim=2).values
    base = (exp_ - c.reshape(b, d, 1).expand(b, d, n)).exp()
    logsum_base = (base * alpha_k.reshape(b, 1, n).expand(b, d, n)).sum(dim=2).log()
    nominator = c + logsum_base
    denominator = (beta + alpha.log()).reshape(b, 1).expand(b, d)
    frac = nominator - denominator
    return frac - th.log(th.tensor(n))
