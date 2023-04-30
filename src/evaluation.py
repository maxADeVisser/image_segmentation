import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchmetrics import JaccardIndex

from _types import CLASS_COUNT


def mIoU(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the mean intersection over union (mIoU) score.
    See: https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html"""
    jac = JaccardIndex(num_classes=CLASS_COUNT, task="multiclass")
    return float(jac(torch.from_numpy(y_pred), torch.from_numpy(y_true)))