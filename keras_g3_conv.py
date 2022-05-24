# Keras Geant3 Events to True table convolutional autoencoder
import sys, os
print(os.path.dirname(sys.executable))

import pickle
import time
import os
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
from geant3_parser import Geant3DataFile
from geant3_parser import build_true_answers_train_set

from event_display import print_tabled_event

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, UpSampling2D, Cropping2D, Input, Conv2DTranspose


file_name = os.path.join('data', 'shower_geant3_new.dat')

def norm_func(e):
    return np.float64(np.log(e) / 11)


# file_name = 'sample_data.txt'
data_file = Geant3DataFile(file_name, skip_lines=3)

# split into input (X) and output (y) variables
parse_start = time.time()
print(f"Start preparing events...")

inputs, answers, values = build_true_answers_train_set(data_file, 20000, norm_func=norm_func, rnd_shift=((-2,2), (-2,2)) )
parse_end = time.time()
print(f"Inputs shape original = {np.shape(inputs)}")
print(f"Total events prepare time = {parse_end - parse_start}")
print(f"max hit value = {np.max(inputs)}")
# print(f"max e = {np.max(true_e)}")


inputs = np.reshape(inputs, (len(inputs), 11, 11, 1))  # -1 => autodetermine
answers = np.reshape(answers, (len(answers), 121))  # -1 => autodetermine
# # Pad with 1 row and column of zeroes, so it divides by 2
# inputs = np.pad(inputs, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
# answers = np.pad(answers, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
print(f"Inputs shape new = {np.shape(inputs)}")
print(f"Answers shape new = {np.shape(answers)}")

print_tabled_event(inputs[0])
print(answers[0])
#print_tabled_event(answers[0]*11)
#print("-----------------------------------")
#print_tabled_event(inputs[1]*11)
#print_tabled_event(answers[1]*11)


model = Sequential()
model.add(Input(shape=(12, 12, 1)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(6, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(6, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(16, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(1, kernel_size=(2, 2), activation='sigmoid', padding='same'))
model.summary()


#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
# output layer
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mse', 'mae'])
#model.compile(optimizer= 'adam', loss = 'binary_crossentropy')
history = model.fit(inputs, answers, epochs=25, batch_size=32, validation_split=0.2)


# compile the keras model

# model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc', 'mse', 'mae'])

# fit the keras model on the dataset
#history = model.fit(inputs, inputs, validation_split=0.05, epochs=20, batch_size=32, verbose=1)

# Save everything
name = "g3_conv"

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
