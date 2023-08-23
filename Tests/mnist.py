import os
import sys
import nengo_dl
import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import Models.models as models
import Data.mnist as mnist

from tensorflow.keras.utils import to_categorical

BATCH_SIZE = 8
N_EPOCHS = 5
SCALE_FACTOR = 1
TARGET_SHAPE = (int(mnist.SHAPE[0] * SCALE_FACTOR), int(mnist.SHAPE[1] * SCALE_FACTOR), 1)
N_TEST_TIME_STEPS = 20
N_TRAIN_TIME_STEPS = 4

def get_data():
    x_train = np.zeros((len(mnist.ds["train"]), *TARGET_SHAPE))
    for i in range(len(mnist.ds["train"])):
        x_train[i] = np.array(mnist.ds["train"][i]["image"]).reshape(*TARGET_SHAPE)

    x_test = np.zeros((len(mnist.ds["test"]), *TARGET_SHAPE))
    for i in range(len(mnist.ds["test"])):
        x_test[i] = np.array(mnist.ds["test"][i]["image"]).reshape(*TARGET_SHAPE)

    y_train = np.zeros(len(mnist.ds["train"]))
    for i in range(len(mnist.ds["train"])):
        y_train[i] = mnist.ds["train"][i]["label"]
    
    y_test = np.zeros(len(mnist.ds["test"]))
    for i in range(len(mnist.ds["test"])):
        y_test[i] = mnist.ds["test"][i]["label"]
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_data()

y_train = to_categorical(y_train, mnist.N_CLASSES)
y_test = to_categorical(y_test, mnist.N_CLASSES)

# Add a time axis
nengo_x_train = x_train.reshape((x_train.shape[0], 1, -1))
nengo_x_test = x_test.reshape((x_test.shape[0], 1, -1))
nengo_y_train = y_train.reshape((y_train.shape[0], 1, -1))
nengo_y_test = y_test.reshape((y_test.shape[0], 1, -1))

nengo_x_train = np.tile(nengo_x_train, (1, N_TRAIN_TIME_STEPS, 1))
nengo_x_test = np.tile(nengo_x_test, (1, N_TRAIN_TIME_STEPS, 1))
nengo_y_train = np.tile(nengo_y_train, (1, N_TRAIN_TIME_STEPS, 1))
nengo_y_test = np.tile(nengo_y_test, (1, N_TRAIN_TIME_STEPS, 1))

tiled_x_test = np.tile(nengo_x_test, (1, N_TEST_TIME_STEPS, 1))
tiled_y_test = np.tile(y_test, N_TEST_TIME_STEPS)

# inp->Node, out->KerasTensor, converter.outputs[out]->Probe
converter, inp, out = models.get_models(mnist.N_CLASSES, TARGET_SHAPE)

# Equ nengo ANN Train & Test
with nengo_dl.Simulator(converter.net, minibatch_size=BATCH_SIZE, progress_bar=True) as sim:
    sim.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    sim.fit(
        {converter.inputs[inp]: nengo_x_train},
        {converter.outputs[out]: nengo_y_train},
        validation_data=(
            {converter.inputs[inp]: nengo_x_test},
            {converter.outputs[out]: nengo_y_test},
        ),
        epochs=N_EPOCHS, verbose=1
    )
    
    sim.save_params("Params/mnist")

    ann_data = sim.predict({converter.inputs[inp]: tiled_x_test})

    # compute accuracy on test data, using output of network on the last timestep
    nengo_ann_predictions = np.argmax(ann_data[converter.outputs[out]][:, -1], axis=-1)
    nengo_ann_accuracy = models.get_test_acc(tiled_y_test, nengo_ann_predictions)

    with open("Results/mnist.txt", "a") as file:
        file.write(f"ANN Test Accuracy: {100 * nengo_ann_accuracy:.2f}%\n")
    
# Convert to SNN and Predict
# out2-> snn_converter.outputs[out2]->
snn_converter, inp2, out2 = models.get_models(mnist.N_CLASSES, TARGET_SHAPE, make_SNN=True)

snn_inp = snn_converter.inputs[inp2] # Type-> Node
snn_out = snn_converter.outputs[out2] # Type-> Probe

with nengo_dl.Simulator(snn_converter.net, minibatch_size=BATCH_SIZE, progress_bar=True) as nengo_sim:
    nengo_sim.load_params("Params/mnist")

    # repeat inputs for some number of timesteps
    snn_data = nengo_sim.predict({snn_inp: tiled_x_test})

    # compute accuracy on test data, using output of network on the last timestep
    snn_predictions = np.argmax(snn_data[snn_out][:, -1], axis=-1)
    snn_accuracy = models.get_test_acc(tiled_y_test, snn_predictions)

    with open("Results/mnist.txt", "a") as file:
        file.write(f"SNN Test Accuracy: {100 * snn_accuracy:.2f}%\n")

