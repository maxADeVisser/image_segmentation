import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from matplotlib import pyplot as plt
import torch

def compare_images(images,image_titles = ""):
    
    if image_titles == "":
        image_titles = tuple([f'Image {i}' for i in range(len(images))])

    no_images = len(images)
    fig, axs = plt.subplots(1, no_images, figsize=(5*no_images, 5))
    for i,ax in enumerate(axs):

        ax.set_title(image_titles[i])
        ax.imshow(images[i])

    plt.show()


def augment_images_plain(X_tensor,y_tensor,augmenter, no_augmentations = 2):

    samples = np.random.randint(X_tensor.shape[0], size = no_augmentations)
    X_changes = augmenter(X_tensor[samples].permute(0,3,1,2))
    X_changes = X_changes.permute(0,2,3,1)
    y_changes = y_tensor[samples]

    
    return X_changes , y_changes 

def augment_five_crop(X_tensor,y_tensor,crop_size = (360, 480), no_augmentations = 2):
    
    five_cropper = T.FiveCrop(size=crop_size)
    resizer = T.Resize((720, 960))
    samples = np.random.randint(X_tensor.shape[0], size = no_augmentations)
    X_changes = five_cropper(X_tensor[samples].permute(0,3,1,2))
    y_changes = five_cropper(y_tensor[samples].permute(0,3,1,2))
    X_changes = torch.cat( X_changes,0)
    y_changes = torch.cat( y_changes,0)
    X_changes = resizer(X_changes)
    y_changes = resizer(y_changes)
    X_changes = X_changes.permute(0,2,3,1)
    y_changes = y_changes.permute(0,2,3,1)
    
    return X_changes , y_changes

def augment_perspective(X_tensor,y_tensor,distortions_scale= 0.6,p = 1.0, no_augmentations = 2):

    perspective = T.RandomPerspective(distortion_scale=distortions_scale, p=1.0)
    samples = np.random.randint(X_tensor.shape[0], size = no_augmentations)
    joint_tensor = torch.cat(
        (X_tensor.permute(0,3,1,2)[samples],y_tensor.permute(0,3,1,2)[samples]),0)
    X = perspective(joint_tensor)
    X_changes = X[:no_augmentations]
    y_changes = X[no_augmentations:]
    
    return X_changes , y_changes

