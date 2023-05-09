import numpy as np
from pathlib import Path

data_path = Path("data/CamVid")



print("images_crop")
images_crop = np.concatenate((np.load(data_path / "train.npy"),np.load(data_path / "train_crop.npy")))
print("saving")
np.save(data_path / "train_crop.npy", images_crop)

print("mask_crop")
masks_crop = np.concatenate((np.load(data_path / "train_labels.npy"),np.load(data_path / "train_labels_crop.npy")))
print("saving")
np.save(data_path / "train_labels_crop.npy", masks_crop)


print("images_jitter")
images_jitter = np.concatenate((np.load(data_path / "train.npy"),np.load(data_path / "train_jitter.npy")))
print("saving")
np.save(data_path / "train_jitter.npy", images_jitter)

print("mask_jitter")
masks_jitter = np.concatenate((np.load(data_path / "train_labels.npy"),np.load(data_path / "train_labels_jitter.npy")))
print("saving")
np.save(data_path / "train_labels_jitter.npy", masks_jitter)

print("images_perspective")
images_perspective = np.concatenate((np.load(data_path / "train.npy"),np.load(data_path / "train_perspective.npy")))
print("saving")
np.save(data_path / "train_perspective.npy", images_perspective)

print("mask_perspective")
masks_perspective= np.concatenate((np.load(data_path / "train_labels.npy"),np.load(data_path / "train_labels_perspective.npy")))
print("saving")
np.save(data_path / "train_labels_perspective.npy", masks_perspective)