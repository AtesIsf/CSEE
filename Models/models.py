from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import nengo_dl
import matplotlib.pyplot

#TODO: FIX THIS
# https://www.nengo.ai/nengo-dl/v3.2.0/examples/keras-to-snn.html

# Define the model architecture
def models(n_labels, shape):
    model = Sequential()
    model.add(Conv2D(128, activation='relu'))
    model.add(Conv2D(128, activation='relu'))
    model.add(Flatten(shape))
    model.add(Dense(n_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    converter = nengo_dl.converter(model)

    return model

# Training the model on a dataset
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
