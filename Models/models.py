import nengo_dl
import nengo
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, AveragePooling2D
from scipy.ndimage import zoom

# Define the model architecture
def get_models(n_labels, img_shape, make_SNN=False):

    inp = Input(shape=img_shape)
    conv1 = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu, padding="same") (inp)
    pool1 = AveragePooling2D((2, 2)) (conv1)
    conv2 = Conv2D(32, kernel_size=(5, 5), activation=tf.nn.relu, padding="same") (pool1)
    pool2 = AveragePooling2D((2, 2)) (conv2)
    flat = Flatten() (pool2)
    dense1 = Dense(48, activation=tf.nn.relu) (flat)
    out = Dense(n_labels, activation=tf.nn.softmax) (dense1)
    model = Model(inputs=[inp], outputs=[out])

    # converter.model->ann, converter.net->snn
    if not make_SNN:
        converter = nengo_dl.Converter(model)
    else:
        converter = nengo_dl.Converter(
            model, swap_activations={
                tf.nn.relu: nengo.SpikingRectifiedLinear()
            }, 
            scale_firing_rates=20, synapse=0.01 # Reduces noise and makes the output clearer
        )

    return converter, inp, out

def condense_image(image, scale_factor):
    reduced_image = zoom(image, (scale_factor, scale_factor, 1))
    min_value = np.min(reduced_image)
    max_value = np.max(reduced_image)
    reduced_image = ((reduced_image - min_value) / (max_value - min_value))
    return reduced_image

def get_test_acc(eval_set, preds):
    preds = preds.reshape(-1)
    n_correct = 0
    for i in range(len(eval_set)):
        if int(preds[i]) == np.argmax(eval_set[i]):
            n_correct+=1
    return n_correct/len(eval_set)
