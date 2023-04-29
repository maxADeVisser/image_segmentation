import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils import data

from dataloader import get_dataloader


import sys
 
# setting path

from evaluation import mIoU


from model import createDeepLabv3, train_model

def main(data_dir: str = "data/CamVid/", 
         out_dir: str = "out", 
         epochs: int = 25, 
         batch_size: int = 4, 
         learn_rate=0.001):

    model = createDeepLabv3()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    metrics = {
        "accuracy_score":accuracy_score,
        "f1_score": f1_score,
        "mIoU": mIoU
    }

    dataloaders = get_dataloader(data_dir=data_dir,
                                 batch_size=batch_size)

    _ = train_model(model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    dataloaders=dataloaders,
                    bpath =out_dir,
                    metrics=metrics,
                    num_epochs=epochs)
    
    torch.save(model, out_dir +  "/weights.pt")

if __name__ == "__main__":
    main()
