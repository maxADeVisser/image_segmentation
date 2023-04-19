from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def locate_data(
    path: Path,
) -> tuple[pd.DataFrame, list, list, list, list, list, list]:
    """Returns the data as ordered paths, so we can load the images when we need them"""
    class_lables = pd.read_csv(path / "class_dict.csv")
    X_train = sorted([x for x in path.glob("train/*")])
    y_train = sorted([x for x in path.glob("train_labels/*")])
    X_test = sorted([x for x in path.glob("test/*")])
    y_test = sorted([x for x in path.glob("test_labels/*")])
    X_val = sorted([x for x in path.glob("val/*")])
    y_val = sorted([x for x in path.glob("val_labels/*")])

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
    """Load images into memory with uint8 dtype (format PyTorch likes)"""
    return np.array(
        [np.array(Image.open(x), dtype=np.uint8) for x in file_paths]
    )
