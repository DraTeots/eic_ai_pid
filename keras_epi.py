# first neural network with keras tutorial
import pickle
import time
from datetime import datetime

import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D

from data_parser import import_training_data
import tensorflow as tf
import math
import seaborn as sns
import pandas as pd

# load the dataset
parse_start = time.time()
print(f"Start preparing events...")

# Load data from 3 different files:
electron_data, electron_hit_xye, electron_hists = import_training_data("data/3x3_e_1GeV_10000ev.root") # , process_events=100)
pim_data, pim_hit_xye, pim_hists = import_training_data("data/3x3_pim_1GeV_10000ev.root") # , process_events=100)
mum_data, mum_xye, mum_hists = import_training_data("data/3x3_mum_1GeV_10000ev.root") # , process_events=100)

# We have to build our training data from data per particle
training_data = []
true_categories = []

# We will do dump mix here. Take one event from each data one by one
min_events = min(len(electron_data), len(pim_data)) #, len(mum_data))

for i in range(min_events):
    training_data.append(electron_data[i])
    training_data.append(pim_data[i])
    #training_data.append(mum_data[i])

#    true_categories.append([1, 0, 0])
#    true_categories.append([0, 1, 0])

    true_categories.append([1, 0])
    true_categories.append([0, 1])
    #true_categories.append([0, 0, 1])

# Convert to numpy so keras could use training/validation split
training_data = numpy.array(training_data)
true_categories = numpy.array(true_categories)

# We save training data to .npy files for FPGA load and optimization
# divide:
ratio_index = int(len(training_data)*0.8)
x_train = training_data[:ratio_index]
x_test = training_data[ratio_index:]

y_train = true_categories[:ratio_index]
y_test = true_categories[ratio_index:]

# save:
numpy.save('x_train.npy', x_train)
numpy.save('x_test.npy', x_test)
numpy.save('y_train.npy', y_train)
numpy.save('y_test.npy', y_test)


# Print some preliminary results
print(f"len(training_data) = {len(training_data)}")
print(f"len(true_categories) = {len(true_categories)}")
parse_end = time.time()
print(f"Total events prepare time = {parse_end - parse_start}")
#df = pd.DataFrame(data = training_data)


# Define the Keras model
model = Sequential()
model.add(Dense(10, input_dim=9, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

# compile the keras model
model.compile(loss='binary_crossentropy', metrics=['acc', 'mse', 'mae', 'categorical_accuracy'])


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# fit the keras model on the dataset

# history = model.fit(training_data,
#                     true_categories,
#                     validation_split=0.1,
#                     epochs=10,
#                     batch_size=32,
#                     verbose=1,
#                     callbacks=[tensorboard_callback])

#history = model.fit(df, true_categories, validation_split=0.1, epochs=5, batch_size=64, verbose=1)


# Save everything !
name = "deep_dense_epi"

# Saving history
with open(name + "-history.pickle", 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Saving the model
model.save(name + ".h5")

# Try plotting what we have
try:
    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.savefig('loss.png')
    plt.show()
    plt.plot(history.history['acc'])
    plt.title("acc")
    plt.show()
    plt.plot(history.history['mse'])
    plt.title("mse")
    plt.show()
    plt.plot(history.history['mae'])
    plt.title("mae")
    plt.show()

    if 'cosine' in history.history:
        plt.plot(history.history['cosine'])
        plt.show()
    plt.plot(history.history['categorical_accuracy'])
    plt.show()
except Exception as ex:
    print("(!) Error building plots ", ex)

print(len(x_test))


