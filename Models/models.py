import nengo_dl
import nengo
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input

#TODO: FIX THIS
# https://www.nengo.ai/nengo-dl/v3.2.0/examples/keras-to-snn.html

# Define the model architecture
def get_models(n_labels, img_shape):

    inp = Input(shape=img_shape)
    conv1 = Conv2D(128, kernel_size=(5, 5),activation=tf.nn.relu, padding="same") (inp)
    conv2 = Conv2D(128, kernel_size=(5, 5), activation=tf.nn.relu, padding="same") (conv1)
    flat = Flatten() (conv2)
    out = Dense(n_labels, activation=tf.nn.relu) (flat)

    model = Model(inputs=[inp], outputs=[out])

    # converter.model->ann, converter.net->snn
    converter = nengo_dl.Converter(
        model, swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()},
        scale_firing_rates=10, synapse=0.01 # Reduces noise and makes the output clearer
    )

    return converter, inp, out


# Training the model on a dataset
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
