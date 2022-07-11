# first neural network with keras tutorial
from ossaudiodev import SOUND_MIXER_BASS
import sys, os

from event_display import gdml_read_ecal_info, print_tabled_event

print(os.path.dirname(sys.executable))

import pickle
import time
import os
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
from geant3_parser import Geant3DataFile
from geant3_parser import build_train_set

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, UpSampling2D, Cropping2D, Input, Conv2DTranspose


file_name = os.path.join('data', 'shower_geant3_new.dat')


# file_name = 'sample_data.txt'
data_file = Geant3DataFile(file_name, skip_lines=3)

# split into input (X) and output (y) variables
parse_start = time.time()
print(f"Start preparing events...")

add_real_xy = False
dir_path = os.path.dirname(os.path.realpath("__file__"))
# Load geometry file
ecal_info = gdml_read_ecal_info(f"{dir_path}/data/ecce.gdml")

# ecal_info is of EcalGeoInfo class, which is a helper holding information
# about all needed ecal geometries.
# Print what information it holds:
ecal_info.print()

# Data file name
#data_file_name = f"{dir_path}/data/test_gun.edm4hep.root"
data_file_name = f"{dir_path}/data/circ_gun_gamma_100-ev.edm4hep.root"

events = ecal_info.read_events_from_file(data_file_name, 0, 100)
print(f"events shape original = {np.shape(events)}")

events = np.reshape(events, (len(events), ecal_info.num_modules_x, ecal_info.num_modules_y, 1))
print(f"events shape reshaped = {np.shape(events)}")
# Pad with 1 row and column of zeroes, so it divides by 2
events = np.pad(events, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
print(f"events shape padded = {np.shape(events)}")
#%%

#print_tabled_event(events[0])


# print(f"Total events prepare time = {parse_end - parse_start}")
# print(f"max hit value = {np.max(inputs)}")
#print(f"max e = {np.max(true_e)}")


#inputs = np.reshape(inputs, (len(inputs), 11, 11, 1))  # -1 => autodetermine



#print(f"Inputs shape new = {np.shape(inputs)}")



model = Sequential()
model.add(Input(shape=(62, 62, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(6, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
model.summary()

exit(0)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
# output layer
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc', 'mse', 'mae'])
#model.compile(optimizer= 'adam', loss = 'binary_crossentropy')
history = model.fit(events, events,
                epochs=25,
                batch_size=32,
                validation_split=0.2)


# compile the keras model

# model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc', 'mse', 'mae'])

# fit the keras model on the dataset
#history = model.fit(inputs, inputs, validation_split=0.05, epochs=20, batch_size=32, verbose=1)

# Save everything
name = "g3__with_xy" if add_real_xy else "g3_autoencoder_conv_no_xy"

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
