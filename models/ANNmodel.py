import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers.pooling import MaxPool2D
import numpy as np

ann_model = Sequential()