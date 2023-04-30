import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils import data

from dataloader import get_dataloaders
from evaluation import mIoU
from model import createDeepLabv3, fit_deeplabv3


def main(
    data_dir: str = "data/CamVid/",
    out_dir: str = "out",
    epochs: int = 25,
    batch_size: int = 4,
    learn_rate: float = 0.001,
) -> None:
    model = createDeepLabv3()  # instantiate model
    model.train()  # set model to train mode
    criterion = torch.nn.CrossEntropyLoss()  # define loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # define optimizer

    # define metrics to track
    metrics = {
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        #    "mIoU": mIoU
    }

    dataloaders = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

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

    torch.save(model, out_dir + "/weights.pt")


if __name__ == "__main__":
    main()
