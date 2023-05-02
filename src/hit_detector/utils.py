import numpy as np
import torch


def fix_seeds(seed):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Evaluation metrics provided by the original MonoTrack paper
def eval_metrics(y_pred, y_true):
    """Compute evaluation metrics for the given predictions and ground truth."""
    true_hits = set([(i, label) for i, label in enumerate(y_true) if label != 0])
    pred_hits = set([(i, pred) for i, pred in enumerate(y_pred) if pred != 0])

    num_intersect = len(true_hits & pred_hits)
    num_union = len(true_hits | pred_hits)

    acc = num_intersect / num_union if num_union != 0 else 0
    recall = num_intersect / len(true_hits) if len(true_hits) != 0 else 0
    prec = num_intersect / len(pred_hits) if len(pred_hits) != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0

    return acc, recall, prec, f1
