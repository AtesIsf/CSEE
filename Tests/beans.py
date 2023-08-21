import os
import sys
import nengo_dl
import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import Models.models as models
import Data.beans as beans

from scipy.ndimage import zoom
from tensorflow.keras.utils import to_categorical
from time import time

BATCH_SIZE = 16
N_EPOCHS = 6
SCALE_FACTOR = 0.3
TARGET_SHAPE = (int(beans.SHAPE[0] * SCALE_FACTOR), int(beans.SHAPE[1] * SCALE_FACTOR), 3)

def condense_image(image):
    reduced_image = zoom(image, (SCALE_FACTOR, SCALE_FACTOR, 1))
    min_value = np.min(reduced_image)
    max_value = np.max(reduced_image)
    reduced_image = ((reduced_image - min_value) / (max_value - min_value))
    return reduced_image

def get_data():
    x_train = np.zeros((len(beans.ds["train"]), *TARGET_SHAPE))
    for i in range(len(beans.ds["train"])):
        x_train[i] = condense_image(np.array(beans.ds["train"][i]["image"]))

    x_test = np.zeros((len(beans.ds["test"]), *TARGET_SHAPE))
    for i in range(len(beans.ds["test"])):
        x_test[i] = condense_image(np.array(beans.ds["test"][i]["image"]))

    y_train = np.zeros(len(beans.ds["train"]))
    for i in range(len(beans.ds["train"])):
        y_train[i] = beans.ds["train"][i]["labels"]
    
    y_test = np.zeros(len(beans.ds["test"]))
    for i in range(len(beans.ds["test"])):
        y_test[i] = beans.ds["test"][i]["labels"]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_data()

y_train = to_categorical(y_train, len(beans.labels))
y_test = to_categorical(y_test, len(beans.labels))

# Add a time axis
snn_x_train = x_train.reshape((x_train.shape[0], 1, -1))
snn_x_test = x_test.reshape((x_test.shape[0], 1, -1))
snn_y_train = y_train.reshape((y_train.shape[0], 1, -1))
snn_y_test = y_test.reshape((y_test.shape[0], 1, -1))

# 2 time steps
snn_x_train = np.tile(snn_x_train, (1, 2, 1))
snn_x_test = np.tile(snn_x_test, (1, 2, 1))
snn_y_train = np.tile(snn_y_train, (1, 2, 1))
snn_y_test = np.tile(snn_y_test, (1, 2, 1))

converter, inp, out = models.get_models(len(beans.labels), TARGET_SHAPE)

# Change to load params instead
do_training = True

# -Train
if do_training:

    # SNN train
    with nengo_dl.Simulator(converter.net, minibatch_size=BATCH_SIZE, progress_bar=True) as sim:
        sim.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        snn_start = time()
        sim.fit(
            {converter.inputs[inp]: snn_x_train},
            {converter.outputs[out]: snn_y_train},
            validation_data=(
                {converter.inputs[inp]: snn_x_test},
                {converter.outputs[out]: snn_y_test},
            ),
            epochs=N_EPOCHS, verbose=1
        )
        snn_end = time()

        sim.save_params("Params/beans_snn")

    # ANN Train
    converter.model.compile(
        optimizer="adam", 
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        metrics=["accuracy"]
    )
    ann_start = time()
    converter.model.fit(
        x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
        validation_data=(x_test, y_test), verbose=1
    )
    ann_end = time()

    converter.model.save("Params/beans_ann.keras")

else:
    converter.model = tf.keras.models.load_model("Params/beans_ann.keras")

# SNN Predict
with nengo_dl.Simulator(converter.net, minibatch_size=16, progress_bar=True) as nengo_sim:
    nengo_sim.load_params("Params/beans_snn")
    # repeat inputs for some number of timesteps
    data = nengo_sim.predict({converter.inputs[inp]: x_test})

# compute accuracy on test data, using output of network on the last timestep
snn_predictions = np.argmax(data[converter.outputs[out]][:, -1], axis=-1)
snn_accuracy = (snn_predictions == y_test[:len(x_test)[0]]).mean()

with open("Results/beans.txt", "a") as file:
    file.write(f"SNN Test Accuracy: {100 * snn_accuracy:.2f}%")
    if do_training:
        file.write(f"SNN Training Time: {snn_end-snn_start:.2f}s")

# ANN Predict
ann_loss, ann_accuracy = converter.model.evaluate(x_test, y_test)
with open("Results/beans.txt", "a") as file:
    file.write(f"ANN Test Accuracy: {100 * ann_accuracy:.2f}%")
    if do_training:
        file.write(f"ANN Training Time: {ann_end-ann_start:.2f}s")
