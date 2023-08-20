import os
import sys
import nengo_dl
import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import Models.models as models
import Data.beans as beans

from time import time

BATCH_SIZE = 16
N_EPOCHS = 1

x_train = np.zeros((len(beans.ds["train"]), *beans.SHAPE))
for i in range(len(beans.ds["train"])):
    x_train[i] = beans.ds["train"][i]["image"]

x_test = np.zeros((len(beans.ds["test"]), *beans.SHAPE))
for i in range(len(beans.ds["test"])):
    x_test[i] = beans.ds["test"][i]["image"]

y_train = np.zeros(len(beans.ds["train"]))
for i in range(len(beans.ds["train"])):
    y_train[i] = beans.ds["train"][i]["labels"]
    
y_test = np.zeros(len(beans.ds["test"]))
for i in range(len(beans.ds["test"])):
    y_test[i] = beans.ds["test"][i]["labels"]

# Add a time axis
snn_x_train = x_train.reshape((x_train.shape[0], 1, -1))
snn_x_test = x_test.reshape((x_test.shape[0], 1, -1))
snn_y_train = y_train.reshape((y_train.shape[0], 1, -1))
snn_y_test = y_test.reshape((y_test.shape[0], 1, -1))

# 10 time steps
snn_x_train = np.tile(snn_x_train, (1, 2, 1))
snn_x_test = np.tile(snn_x_test, (1, 2, 1))
snn_y_train = np.tile(snn_y_train, (1, 2, 1))
snn_y_test = np.tile(snn_y_test, (1, 2, 1))

converter, inp, out = models.get_models(len(beans.labels), beans.SHAPE)

do_training = True

# -Train
if do_training:

    # SNN train
    with nengo_dl.Simulator(converter.net, minibatch_size=BATCH_SIZE, progress_bar=True) as sim:
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],
        )

        snn_start = time()
        sim.fit(
            {converter.inputs[inp]: snn_x_train},
            {converter.outputs[out]: snn_y_train},
            validation_data=(
                {converter.inputs[inp]: snn_x_test},
                {converter.outputs[out]: snn_y_test},
            ),
            epochs=N_EPOCHS,
        )
        snn_end = time()

        sim.save_params("Params/beans_snn")

    # ANN Train
    converter.model.compile(
        optimizer=tf.optimizers.RMSprop(0.001), 
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=[tf.metrics.sparse_categorical_accuracy]
    )
    ann_start = time()
    converter.model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
    ann_end = time()

# SNN Predict
with nengo_dl.Simulator(converter.net, minibatch_size=16, progress_bar=True) as nengo_sim:
    nengo_sim.load_params("Params/beans_snn")
    # repeat inputs for some number of timesteps
    data = nengo_sim.predict({converter.inputs[inp]: x_test})

# compute accuracy on test data, using output of network on the last timestep
snn_predictions = np.argmax(data[converter.outputs[out]][:, -1], axis=-1)
snn_accuracy = (snn_predictions == y_test[:len(x_test)[0]]).mean()
print(f"SNN Test Accuracy: {100 * snn_accuracy:.2f}%\nSNN Training Time: {snn_end-snn_start}")

# ANN Predict
ann_loss, ann_accuracy = converter.model.evaluate(x_test, y_test)
print(f"ANN Test Accuracy: {100 * ann_accuracy:.2f}%\nANN Training Time: {ann_end-ann_start}")
