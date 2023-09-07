"""Classification utils"""
import torch as th


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
def logits_to_label(logits):
    """Convert logits to hard label

    Args:
        logits: (batch_size, num_classes)

    Returns:
        labels: (batch_size, num_classes)
    """
    return th.argmax(logits, dim=1)
