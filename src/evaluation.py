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

def evaluate_pixel_basis(
    y_pred: np.ndarray, y_true: np.ndarray
) -> tuple[float, float, float, float]:
    """Evaluate the model on pixel basis.
    Returns accuracy, f1, precision, recall in that order"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(
        f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
    )

    return acc, f1, precision, recall
