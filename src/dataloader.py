from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from CamVidData import SemanticSegmentationDataset

from segmentationData import SegmentationDataset

def get_dataloader(data_dir: str= "data/CamVid",
                   image_folder: str = "",
                   mask_folder: str = "_labels",
                   batch_size: int = 4) -> dict:
    """ 
    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir),
                               transforms=data_transforms,
                               image_folder=x+image_folder,
                               mask_folder=x+mask_folder)
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['train', 'val']
    }

    return dataloaders
