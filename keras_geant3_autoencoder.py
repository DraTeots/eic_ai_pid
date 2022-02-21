# first neural network with keras tutorial
import pickle
import time
import os
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
from geant3_parser import Geant3DataFile
from geant3_parser import build_train_set


file_name = os.path.join('data', 'shower_geant3_new.dat')


# file_name = 'sample_data.txt'
data_file = Geant3DataFile(file_name, skip_lines=3)

# split into input (X) and output (y) variables
parse_start = time.time()
print(f"Start preparing events...")

add_real_xy = False
inputs, true_e, sum_e = build_train_set(data_file, 50000, add_real_xy=add_real_xy, normalize=True)
parse_end = time.time()

print(f"Total events prepare time = {parse_end - parse_start}")
print(f"max hit value = {np.max(inputs)}")
print(f"max e = {np.max(true_e)}")


from keras.models import Sequential
from keras.layers import Dense

# # define the keras model
model = Sequential()

# model.add(Dense(123, input_dim=123, activation='gelu'))
# model.add(Dense(90, activation='gelu'))
# model.add(Dense(80, activation='gelu'))
# model.add(Dense(70, activation='gelu'))
# model.add(Dense(50, activation='gelu'))
# model.add(Dense(70, activation='gelu'))
# model.add(Dense(80, activation='gelu'))
# model.add(Dense(90, activation='gelu'))
# model.add(Dense(123, activation='linear'))

# More extensive model
# model.add(Dense(123, input_dim=123, activation='selu'))
# model.add(Dense(90, activation='selu'))
# model.add(Dense(60, activation='selu'))
# model.add(Dense(40, activation='selu'))
# model.add(Dense(30, activation='selu'))
# model.add(Dense(10, activation='selu'))
# model.add(Dense(1, activation='linear'))

input_dims = 123
middle_layer_size = 40
hidden_layers_num = 3
neuron_type = 'gelu'

# go from 20 to 123 in 5 steps (or from 123 to 20)

delta_dim = int((input_dims - middle_layer_size) / hidden_layers_num)

model.add(Dense(input_dims, input_dim=123, activation=neuron_type))

for i in range(1, hidden_layers_num):
    neurons_num =  input_dims - i*delta_dim
    model.add(Dense(neurons_num, activation=neuron_type))

for i in range(hidden_layers_num):
    neurons_num =  middle_layer_size + i*delta_dim
    model.add(Dense(neurons_num, activation=neuron_type))

model.add(Dense(input_dims, activation='linear'))


# compile the keras model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['acc', 'mse', 'mae'])
# model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc', 'mse', 'mae'])

# fit the keras model on the dataset
history = model.fit(inputs, inputs, validation_split=0.05, epochs=20, batch_size=32, verbose=1)

# Save everything
name = "g3_autoencoder_with_xy" if add_real_xy else "g3_autoencoder_no_xy"

# Saving history
with open(name + "-history.pickle", 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Saving the model
model.save(name + ".hd5")

print(history.history)

try:
    plt.plot(history.history['loss'])
    plt.show()
    plt.plot(history.history['acc'])
    plt.show()
    plt.plot(history.history['mse'])
    plt.show()
    plt.plot(history.history['mae'])
    plt.show()
    # plt.plot(history.history['cosine'])
    #plt.show()
except Exception as ex:
    print("(!) Error building plots ", ex)
