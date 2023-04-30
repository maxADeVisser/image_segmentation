import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchmetrics.classification import MulticlassJaccardIndex

from _types import CLASS_COUNT

SMOOTH = 1e-6  # To avoid division by zero


def mIoU(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """See: https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html
    """
    jaccard = MulticlassJaccardIndex(num_classes=CLASS_COUNT)
    total_iou = 0
    
    for i in range(len(y_pred)):
        total_iou += jaccard(y_pred[i], y_true[i])
    
    return total_iou / len(y_pred)