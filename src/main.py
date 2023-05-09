import os
import sys
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils import data

from dataloader import get_dataloaders
from evaluation import mIoU
from model import createDeepLabv3, fit_deeplabv3

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000"

def main(
    data_dir: str = "data/CamVid/",
    out_dir: str = "out",
    epochs: int = 25,
    batch_size: int = 3,
    learn_rate: float = 0.001,
    augments: list = list(),
    output_name: str = "/deeplabv3_no_transforms_weights.pt"
) -> None:
    model = createDeepLabv3()  # instantiate model
    model.train()  # set model to train mode
    criterion = torch.nn.CrossEntropyLoss()  # define loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # define optimizer

    # define metrics to track
    metrics = {
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
#        "mIoU": mIoU
    }

    dataloaders = get_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        augments=augments
        )

    # fit the model using the dataloaders
    fit_deeplabv3(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders=dataloaders,
        out_dir=out_dir,
        metrics=metrics,
        num_epochs=epochs,
    )

    torch.save(model, out_dir + output_name)


if __name__ == "__main__":
    flip = ["flip"]
    crop = ["crop"]
    perspective = ["perspective"]
    jitter = ["jitter"]
    all = [
        "flip",
        "crop",
        "perspective",
        "jitter"
    ]
    main(augments=all, output_name="/deeplabv3_all_weights.pt")
    # augment = sys.argv[1]
    # match augment:
    #     case "flip":
    #         main(augments=flip, output_name="/deeplabv3_flip_weights.pt")
    #     # case "crop":
    #     #     main(augments=crop, output_name="/deeplabv3_crop_weights.pt")
    #     # case "perspective":
    #     #     main(augments=perspective, output_name="/deeplabv3_perspective_weights.pt")
    #     case "jitter":
    #         main(augments=jitter, output_name="/deeplabv3_jitter_weights.pt")
    #     case "all":
    #         main(augments=all, output_name="/deeplabv3_all_weights.pt")
