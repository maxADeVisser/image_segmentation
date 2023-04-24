from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def locate_data(
    data_path: Path,
    show: bool = False,
) -> tuple[pd.DataFrame, list, list, list, list, list, list]:
    """Returns the data as ordered paths, so we can load the images when we need them"""
    class_lables = pd.read_csv(data_path / "class_dict.csv").set_index("name")
    X_train = sorted([x for x in data_path.glob("train/*")])
    y_train = sorted([x for x in data_path.glob("train_labels/*")])
    X_test = sorted([x for x in data_path.glob("test/*")])
    y_test = sorted([x for x in data_path.glob("test_labels/*")])
    X_val = sorted([x for x in data_path.glob("val/*")])
    y_val = sorted([x for x in data_path.glob("val_labels/*")])

    if show:
        data_counts = [X_train, y_train, X_test, y_test, X_val, y_val]
        n = len(data_counts)
        multiplier = 0
        w = 0.5
        x = np.arange(n)  # x positions

        _, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        for i in range(n):
            value = len(data_counts[i])
            rect = ax.bar(
                x=x[i], height=value, width=w
            )
            ax.bar_label(rect, padding=3)
            #ax.bar(x=labels, height=[len(y_train), len(y_test), len(y_val)], width=w)
            ax.set_title("Data Distribution")
        labels = ["Train", "Test", "Validation"]
        #plt.xticks(x, labels)

    return (
        class_lables,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
    )


def load_data(file_paths: list[Path]) -> np.ndarray:
    """Load images into memory with uint8 dtype (format PyTorch likes"""
    return np.array([np.array(Image.open(x), dtype=np.uint8) for x in file_paths])
