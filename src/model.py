import copy
import csv
import os
import time
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm

from _types import CLASS_COUNT


def createDeepLabv3(
    outputchannels: int = CLASS_COUNT,
) -> models.segmentation.DeepLabV3:
    """Create DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 32.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(
        2048, outputchannels
    )  # 2048 is the number of ResNet101 output channels
    model.train()  # Set the model in training mode

    return model


def fit_deeplabv3(
    model: models.segmentation.DeepLabV3,
    dataloaders: dict[str, DataLoader],
    criterion,
    optimizer,
    metrics: dict[str, Callable],
    num_epochs: int,
    out_dir: str,
) -> models.segmentation.DeepLabV3:
    """Fit the DeepLabv3 model using the dataloaders."""

    since = time.time()

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )  # initialise variable to store the best model weights
    best_loss = 1e10

    # Use gpu if available (else cpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # move model to GPU if available

    # Create the log file for training and testing loss and metrics
    fieldnames = (
        ["epoch", "train_loss", "val_loss"]
        + [f"train_{m}" for m in metrics.keys()]
        + [f"val_{m}" for m in metrics.keys()]
    )
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    ### TRAINING LOOP ###
    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        batchsummary = {field: [0] for field in fieldnames}  # Initialize batch summary

        # Each epoch has a training and validation phase. Set model mode accordingly:
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Iterate over batches for current epoch:
            for batch in tqdm(iter(dataloaders[phase])):
                # get the raw and label images:
                inputs = batch["image"].to(device)
                masks = batch["mask"].to(device)

                # zero the gradient parameters
                optimizer.zero_grad()

                # track history if model is in train mode
                with torch.set_grad_enabled(phase == "train"):
                    # forward pass:
                    output = model(
                        inputs
                    )  # output['out] has shape (batch_size, n_classes, height, width)
                    loss = criterion(output["out"], masks)  # calculate loss
                    y_pred = (
                        torch.argmax(output["out"], dim=1).data.cpu().numpy().ravel()
                    )
                    y_true = masks.data.cpu().numpy().ravel()

                    # Calculate batch summary
                    for name, metric in metrics.items():
                        if name == "f1_score":
                            classification_threshold = 0.1
                            batchsummary[f"{phase}_{name}"].append(
                                metric(y_true > 0, y_pred > classification_threshold)
                            )
                        elif name == "mIoU":
                            batchsummary[f"{phase}_{name}"].append(
                                metric(output["out"], masks)
                            )
                        else:
                            batchsummary[f"{phase}_{name}"].append(
                                metric(y_true.astype("uint8"), y_pred)
                            )

                    # backwards pass:
                    if phase == "train":
                        loss.backward()  # compute gradients
                        optimizer.step()  # update parameters

            # Calculate epoch summary
            batchsummary["epoch"] = epoch
            epoch_loss = loss
            batchsummary[f"{phase}_loss"] = epoch_loss.item()
            print("{} Loss: {:.4f}".format(phase, loss))

        # Write @batchsummary to log file and stdout for each epoch
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        with open(os.path.join(out_dir, "log.csv"), "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model if it's the best one yet:
            if phase == "val" and loss < best_loss:
                best_loss = loss
                best_model_weights = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Lowest Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model
