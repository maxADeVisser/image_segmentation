from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class SegmentationDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 augments: list = list(),
                 transforms: Optional[Callable] = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "rgb") -> None: 
        
        super().__init__(root, transforms)
        image_array_path = Path(self.root) / image_folder
        mask_array_path  = Path(self.root) / mask_folder

        self.images = np.load(image_array_path)
        self.masks = np.load(mask_array_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Any:
        image = self.images[index]
        image = torch.tensor(image)


        mask = self.masks[index]
        mask = torch.tensor(mask)
        mask = torch.squeeze(mask)        

        #sample = {"image": image, "mask": mask}
        #return sample

        return image, mask