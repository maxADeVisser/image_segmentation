from typing import Tuple
import torch.utils.data as data
import torchvision.transforms as T
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
from max import locate_data

class SemanticSegmentationDataset(data.Dataset):
    def __init__(self, data_path: str, image_size: Tuple[int, int] = (720, 960)) -> None:
        self.data_path = Path(data_path)
        (
            self.X_train_paths,
            self.y_train_paths,
            self.X_test_paths,
            self.y_test_paths,
            self.X_val_paths,
            self.y_val_paths,
        ) = locate_data(self.data_path)

        self.class_labels = self.load_class_labels()
        # load original training set
        self.X_train  = self.load_data(self.X_train_paths)
        self.y_train = self.load_data(self.y_train_paths)

        # transforms
        self.flip_transform = T.RandomHorizontalFlip(p=1)
        self.crop_transform = T.RandomCrop(size=(360, 480))
        self.perspective_transform = T.Compose(
            [T.RandomPerspective(distortion_scale=0.5, p=1),
            T.CenterCrop(size=(360, 480))])
        self.jitter_transform = T.ColorJitter(brightness=(1.2,2),hue=(-0.5,0.5))
        self.resize_transform = T.Resize(image_size)



    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        
        org_idx = self.original_index(index)
        
        X = self.X_train[org_idx]
        y = self.y_train[org_idx]

        if index // self.X_train.shape[0] == 1:
            X = self.flip_transform(X)
            y = self.flip_transform(y)
        elif index // self.X_train.shape[0] == 2:
            joint_tensor = torch.cat([X.unsqueeze(0), y.unsqueeze(0)], dim=0)
            joint_tensor = self.crop_transform(joint_tensor)
            X,y = joint_tensor[0], joint_tensor[1]
        elif index // self.X_train.shape[0] == 3:
            X = self.jitter_transform(X)
        elif index // self.X_train.shape[0] == 4:
            joint_tensor = torch.cat([X.unsqueeze(0), y.unsqueeze(0)], dim=0)
            joint_tensor = self.perspective_transform(joint_tensor)
            X,y = joint_tensor[0], joint_tensor[1]

        X = self.resize_transform(X)
        y = self.resize_transform(y)
        
        
        y = np.array(y.permute(1,2,0))
        y = self.adjust_mask(y, self.class_labels)
        y = torch.tensor(y)
        y = torch.squeeze(y)

        return X, y

    def __len__(self) -> int:
        return len(self.X_train) * 5

    def load_data(self,file_paths: list[Path]) -> np.ndarray:
        """Load images into memory with uint8 dtype (format PyTorch likes"""
        array= np.array([np.array(Image.open(x), dtype=np.uint8) for x in file_paths])
        tensor = torch.from_numpy(array).permute(0,3,1,2)
        return tensor
    def load_class_labels(self) -> pd.DataFrame:
        """Load class labels"""
        return pd.read_csv(self.data_path / "class_dict.csv")
    
    def original_index(self,index: int) -> int:
        """Get original index of augmented image"""
        return index % self.X_train.shape[0]
    
    def adjust_mask(self,mask: np.array, class_labels: pd.DataFrame) -> torch.Tensor:
        """Adjust mask to be in range 0-11"""
        label_dict = class_labels.iloc[:, 1:].to_dict(orient='index')
        label_dict ={ k: list(v.values()) for k, v in label_dict.items()}
        segmentation_map_list = []
        for x,color in enumerate(label_dict.values()):
            segmentation_map = (mask==color).all(axis=-1)
            segmentation_map=(segmentation_map*1)
            segmentation_map*=x
            segmentation_map_list.append(segmentation_map)
            
            
        return np.amax(np.stack(segmentation_map_list,axis=-1),axis=-1)
    

def mask_to_rgb(mask: np.array, class_labels: pd.DataFrame) -> np.array:
    """Transforms mask to RGB image using class_labels"""
    label_dict = class_labels.iloc[:, 1:].to_dict(orient='index')
    label_dict = {k: list(v.values()) for k, v in label_dict.items()}
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            label = mask[i, j]
            color = label_dict[label]
            rgb_image[i, j] = color
    return rgb_image

def compare_images(images,image_titles = ""):
    
    if image_titles == "":
        image_titles = tuple([f'Image {i}' for i in range(len(images))])

    no_images = len(images)
    fig, axs = plt.subplots(no_images,1,  figsize=(5, 5*no_images))
    for i,ax in enumerate(axs):

        ax.set_title(image_titles[i])
        ax.imshow(images[i])

    plt.show()
