import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from _types import CLASS_COLOR_LABELS, CLASS_COUNT, RGB


def locate_data(
    data_path: Path,
    show: bool = False,
) -> tuple[list, list, list, list, list, list]:
    """Returns the data as ordered paths, so we can load the images when we need them.
    set @show=True to plot the number of images in each dataset split."""
    X_train = sorted([x for x in data_path.glob("train/*")])
    y_train = sorted([x for x in data_path.glob("train_labels/*")])
    X_test = sorted([x for x in data_path.glob("test/*")])
    y_test = sorted([x for x in data_path.glob("test_labels/*")])
    X_val = sorted([x for x in data_path.glob("val/*")])
    y_val = sorted([x for x in data_path.glob("val_labels/*")])

    if show:
        data_counts = [X_train, X_test, X_val]
        n = len(data_counts)
        w = 0.5
        x = np.arange(n)

        _, ax = plt.subplots(figsize=(4, 3), layout="constrained")
        for i in range(n):
            value = len(data_counts[i])
            rect = ax.bar(x=x[i], height=value, width=w)
            ax.bar_label(rect, padding=3)
            ax.set_title("Dataset Split")
        labels = ["Training", "Test", "Validation"]
        plt.ylim(0, 420)
        plt.xticks(x, labels)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
    )


def load_data(file_paths: list[Path]) -> np.ndarray:
    """Load images into memory with uint8 dtype (dtype PyTorch prefers)"""
    print("Loading images into memory...")
    return_array = []
    for i in tqdm(range(len(file_paths))):
        return_array.append(np.array(Image.open(file_paths[i]), dtype=np.uint8))

    return np.array(return_array)


def count_classes(images: list[np.ndarray]) -> dict:
    """Counts the number of pixels for each class in the dataset.
    The classes are defined in @CLASS_COLOR_LABELS from _types.py module."""
    class_counts = {c: 0 for c in CLASS_COLOR_LABELS.keys()}

    for i in tqdm(images):
        for c, v in CLASS_COLOR_LABELS.items():
            # creates a boolean mask of current class on current image,
            # and sums the number of pixels in the mask:
            mask = np.all(i == v, axis=2)
            class_pixel_count = np.sum(mask)

            class_counts[c] += class_pixel_count

    # pickle class_counts
    if Path("dev_output/y_train_class_count.pkl").exists():
        with open("dev_output/y_train_class_count.pkl", "wb") as f:
            pickle.dump(class_counts, f)

    return class_counts


def verify_dimensions(
    data: list[np.ndarray],
    height: int,
    width: int,
) -> bool:
    """Verify that all images in @data have the provided dimensions (@height, @width)"""
    return all([x.shape[:-1] == (height, width) for x in data])


def search_for_classes(classes: list[RGB], data: np.ndarray) -> dict[str, list[int]]:
    """Searches for classes in the provided data and returns,
    the indices of the images that contain them."""
    # IMPROVE THIS FUNCTION
    assert (
        data.ndim == 4
    ), "Data must be 4-dimensional: (n_images, image_height, image_width, color_channels)"

    # imageIdxs_containing_classes: list[int] = []
    imageIdxs_containing_classes: dict[str, list[int]] = {}

    for i in tqdm(range(len(data))):  # for each image
        for c in classes:  # check if the classes in is the image
            contains_class = (
                np.sum(np.all(data[i] == c, axis=2)) > 0
            )  # does the image contain the class?

            if contains_class:
                imageIdxs_containing_classes[c.get_label()].append(i)

    return imageIdxs_containing_classes


def show_imageNlabel(
    image_idx: list[int], X_train: np.ndarray, y_train: np.ndarray
) -> None:
    """Plots images indexed by @image_idx from the training/test/val data.
    Requires that images are loaded into memory in the same order (both for labels and raw)
    """
    n = len(image_idx)  # number of images
    n_cols = 2
    fig, ax = plt.subplots(nrows=n, ncols=n_cols, figsize=(12, n * 5))
    f = 16

    ax[0, 0].set_title("Original Images", fontsize=f)
    ax[0, 1].set_title("Labeled Images", fontsize=f)

    for i in range(0, n):  # for each image
        for j in range(0, n_cols):  # for each column
            if j == 0:
                ax[i, j].imshow(X_train[image_idx[i]])
            elif j == 1:
                ax[i, j].imshow(y_train[image_idx[i]])

    plt.tight_layout()


def extract_class_masks(image: np.ndarray) -> np.ndarray:
    """Extract each class in the image as a mask from input.

    Args:
        image: shape -> (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH)

    Returns:
        np.ndarray of shape (CLASS_COUNT * BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH)
    """
    all_class_masks = []
    for _class in image:
        class_mask = np.array(
            [(_class == i).astype(np.int8) for i in range(CLASS_COUNT)]
        )

        all_class_masks.append(class_mask)

    return np.concatenate(all_class_masks, axis=0)
