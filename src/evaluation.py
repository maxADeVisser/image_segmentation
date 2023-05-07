import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
from torch import Tensor

from _types import CLASS_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH
from src.utils import extract_class_masks


def mIoU(output: Tensor, sample: Tensor, batch_size: int = 2) -> float:
    """Calculates the mIoU score for a given output and sample."""
    jacc_scores = []
    for i in range(CLASS_COUNT):
        out = extract_class_masks(output.argmax(1).numpy())[i]  # 2D
        label = extract_class_masks(sample.unsqueeze(0).numpy()).reshape(
            CLASS_COUNT * batch_size, IMAGE_HEIGHT, IMAGE_WIDTH
        )[i]
        
        if label.sum() == 0:
            # skip the class if it is not present in the current image
            continue
        else:
            score = jaccard_score(label, out, average="micro")
            jacc_scores.append(score)

    return np.mean(jacc_scores)


def plot_class_prediction(label: np.ndarray, prediction: np.ndarray) -> None:
    """
    Converts the inputs into an RGB images where all the values with 0.0 are displayed
    as black and all the values with 1.0 are displayed as white so we can see the
    specific class predictions of the model.

    Args:
        label: A 2D numpy array with float values of 0.0 and 1.0 indicating where in
        the image the class is present

        prediction: A 2D numpy array with float values of 0.0 and 1.0 indicating where in
        the image the class is present
    """
    jacc = jaccard_score(label, prediction, average="micro")
    # Ensure that the input array has the correct dtype and shape
    label = np.array(label, dtype=np.float32)
    if label.ndim == 2:
        label = np.expand_dims(label, axis=-1)
    assert (
        label.ndim == 3 and label.shape[-1] == 1 or label.shape[-1] == 3
    ), f"Invalid shape {label.shape} for input array"

    # Scale the values to the range [0, 255]
    label *= 255

    # Convert the array to an RGB image
    label = label.astype(np.uint8)
    if label.shape[-1] == 1:
        label = np.tile(label, (1, 1, 3))

    prediction = np.array(prediction, dtype=np.float32)
    if prediction.ndim == 2:
        prediction = np.expand_dims(prediction, axis=-1)
    assert (
        prediction.ndim == 3 and prediction.shape[-1] == 1 or prediction.shape[-1] == 3
    ), f"Invalid shape {prediction.shape} for input arr1ay"

    # Scale the values to the range [0, 255]
    prediction *= 255

    # Convert the arr1ay to an RGB image
    prediction = prediction.astype(np.uint8)
    if prediction.shape[-1] == 1:
        prediction = np.tile(prediction, (1, 1, 3))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))
    f = 14

    fig.suptitle(f"Jaccard Score: {jacc}", fontsize=f)

    ax1.set_title("Original Images", fontsize=f)
    ax1.imshow(label)
    ax2.set_title("Predicted Images", fontsize=f)
    ax2.imshow(prediction)
    fig.tight_layout()
    plt.show()
