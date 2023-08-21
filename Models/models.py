import nengo_dl
import nengo
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, AveragePooling2D

# Define the model architecture
def get_models(n_labels, img_shape):

    inp = Input(shape=img_shape)
    conv1 = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu, padding="same") (inp)
    pool1 = AveragePooling2D((2, 2)) (conv1)
    conv2 = Conv2D(48, kernel_size=(5, 5), activation=tf.nn.relu, padding="same") (pool1)
    pool2 = AveragePooling2D((2, 2)) (conv2)
    flat = Flatten() (pool2)
    dense1 = Dense(64, activation=tf.nn.relu) (flat)
    out = Dense(n_labels, activation=tf.nn.softmax) (dense1)

    model = Model(inputs=[inp], outputs=[out])

    # converter.model->ann, converter.net->snn
    converter = nengo_dl.Converter(
        model, swap_activations={
            tf.nn.relu: nengo.SpikingRectifiedLinear(),
            tf.nn.softmax: nengo.LIF()
        }, 
        scale_firing_rates=10, synapse=0.02 # Reduces noise and makes the output clearer
    )

    return converter, inp, out
