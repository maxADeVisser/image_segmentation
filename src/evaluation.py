import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SMOOTH = 1e-6  # To avoid division by zero


def iou(model_output: torch.Tensor, label: torch.Tensor) -> float:
    """Calculate the IoU for a single model output and label pair."""
    # FIXME: function does not work correctly i think. Needs to be validated.
    intersection = (
        (model_output & label).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (model_output | label).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # smooth division to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.7), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


def evaluate_pixel_basis(
    y_pred: np.array, y_true: np.array
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
