

# example of creating a CNN with an efficient inception module
from gc import callbacks
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.utils import plot_model

# function for creating a projected inception module
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
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, UpSampling2D, Cropping2D, Input, Conv2DTranspose, Dropout
from keras.callbacks import ModelCheckpoint

file_name = os.path.join('data', 'shower_geant3_new.dat')

def norm_func(e):
    return np.float64(np.log(e) / 11)


# file_name = 'sample_data.txt'
data_file = Geant3DataFile(file_name, skip_lines=3)

# split into input (X) and output (y) variables
parse_start = time.time()
print(f"Start preparing events...")

num_events = 200000
inputs, answers, values = build_true_answers_train_set(data_file, num_events, norm_func=norm_func, rnd_shift=((-2,2), (-2,2)) )
parse_end = time.time()
print(f"Inputs shape original = {np.shape(inputs)}")
print(f"Total events prepare time = {parse_end - parse_start}")
print(f"max hit value = {np.max(inputs)}")
# print(f"max e = {np.max(true_e)}")


inputs = np.reshape(inputs, (len(inputs), 11, 11, 1))  # -1 => autodetermine
answers = np.reshape(answers, (len(answers), 11, 11, 1))  # -1 => autodetermine
#answers = np.reshape(answers, (len(answers), 121))  # -1 => autodetermine
# # Pad with 1 row and column of zeroes, so it divides by 2
#inputs = np.pad(inputs, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
#answers = np.pad(answers, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
print(f"Inputs shape new = {np.shape(inputs)}")
print(f"Answers shape new = {np.shape(answers)}")

print_tabled_event(inputs[0])
print(answers[0])
#print_tabled_event(answers[0]*11)
#print("-----------------------------------")
#print_tabled_event(inputs[1]*11)
#print_tabled_event(answers[1]*11)


#model = Sequential()
#model.add(Input(shape=(11, 11, 1)))
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# define model input
visible = Input(shape=(11, 11, 1))
# add inception block 1
layer = inception_module(visible, 64, 32, 32, 6, 6, 6)
# add inception block 1
layer = inception_module(layer, 128, 128, 192, 32, 121, 121)
# create model
model = Model(inputs=visible, outputs=layer)
plot_model(model, show_shapes=True, to_file='inception_module.png')
model.summary()


#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
# output layer
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mse', 'mae'])
#model.compile(optimizer= 'adam', loss = 'binary_crossentropy')
name = "inception_1"
filepath = os.path.join('trained_models', "g3_" + name + "_{}".format(num_events) + ".hdf5")
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks = [checkpoint]
history = model.fit(inputs, answers, epochs=5, batch_size=32, validation_split=0.2, callbacks=callbacks)


# compile the keras model

# model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc', 'mse', 'mae'])

# fit the keras model on the dataset
#history = model.fit(inputs, inputs, validation_split=0.05, epochs=20, batch_size=32, verbose=1)

# Save everything
# Saving history
with open(os.path.join('trained_models', "g3_" + name + "_{}".format(num_events) + "-history.pickle"), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Saving the model
model.save(os.path.join('trained_models', "g3_" + name + "_{}".format(num_events) + ".hd5"))

print(history.history)

try:
    plt.plot(history.history['loss'])
    plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_loss.png"))
    plt.clf()
    plt.plot(history.history['acc'])
    plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_acc.png"))
    plt.clf()
    plt.plot(history.history['mse'])
    plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_mse.png"))
    plt.clf()
    plt.plot(history.history['mae'])
    plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_mae.png"))
except Exception as ex:
    print("(!) Error building plots ", ex)
