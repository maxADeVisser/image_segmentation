import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import layers

from model import DeeplabV3Plus
from data import data_generator

NUM_CLASSES = 32
IMAGE_HEIGHT=720
IMAGE_WIDTH=960

def main():
    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)



    model = DeeplabV3Plus(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, num_classes=NUM_CLASSES)
    model.summary()


    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy", 
                 tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)],
    )

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=25)

    model.save("output")