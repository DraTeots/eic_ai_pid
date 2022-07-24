# Keras Geant3 Events convolutional multiple answers

import datetime
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, Input, Dropout
from keras.models import Sequential

from event_display import print_tabled_event
from geant3_parser import Geant3DataFile
from geant3_parser import build_true_answers_train_set

print(os.path.dirname(sys.executable))
datetime_now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(datetime_now_str)

def norm_func(e):
    """A function used for cell values normalization
    The purpose is to de-exponentize it and bring to 0-1 range"""
    return np.float64(np.log(e) / 11)


# Kind of config:
output_name = "conv_bce-sgm_fpga_2dense_smbtch"
logdir = f"logs/{datetime_now_str}_dcnn_smbtch_700k"
input_name = os.path.join('data', 'shower_geant3_new.dat')
num_events = 700000

# Open shower library file
print(f"Preparing events...")
parse_start = time.time()
data_file = Geant3DataFile(input_name, skip_lines=3)
inputs, answers, values = build_true_answers_train_set(data_file, num_events, norm_func=norm_func, rnd_shift=((-2, 2), (-2, 2)))
parse_end = time.time()

# Print some debug info
print(f"Inputs shape original = {np.shape(inputs)}")
print(f"Total events prepare time = {parse_end - parse_start}")
print(f"max hit value = {np.max(inputs)}")

# Reshape (and pad if needed) data to be image like
inputs = np.reshape(inputs, (len(inputs), 11, 11, 1))  # -1 => autodetermine
answers = np.reshape(answers, (len(answers), 121))  # -1 => autodetermine
# Pad with 1 row and column of zeroes, so it divides by 2
# inputs = np.pad(inputs, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
# answers = np.pad(answers, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
print(f"Inputs shape new = {np.shape(inputs)}")
print(f"Answers shape new = {np.shape(answers)}")

# Print sample event (left and right part)
print_tabled_event(inputs[0])
print("-----------------------------------")
print(answers[0])
print("-----------------------------------")

#
# Create model
model = Sequential()
model.add(Input(shape=(11, 11, 1)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(6, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
#model.add(Dropout(0.1))
#model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Flatten())
## 128 -> 216
## 64  -> 294
model.add(Dense(294, activation='relu'))
#model.add(Dropout(0.5))
# model.add(Dense(284, activation='relu'))
# model.add(Dense(274, activation='relu'))
# model.add(Dense(264, activation='relu'))
# model.add(Dense(254, activation='relu'))
# model.add(Dense(242, activation='relu'))
# model.add(Dense(222, activation='relu'))
# model.add(Dense(202, activation='relu'))
# model.add(Dense(182, activation='relu'))
# model.add(Dense(162, activation='relu'))
#model.add(Dense(220, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(180, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(142, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(121, activation='sigmoid'))
#model.add(Dense(121))
#model.add(Conv2D(1, kernel_size=(2, 2), activation='sigmoid', padding='same'))
model.summary()
#plot_model(model, show_shapes=True, to_file='conv_module.png')

# Tensorboard log callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, update_freq='epoch', profile_batch=0)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
# output layer
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mse', 'mae'])
#model.compile(optimizer= 'adam', loss = 'binary_crossentropy')
history = model.fit(inputs, answers, epochs=40, batch_size=2, validation_split=0.25, callbacks=[tensorboard_callback])


# compile the keras model

# model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc', 'mse', 'mae'])

# fit the keras model on the dataset
#history = model.fit(inputs, inputs, validation_split=0.05, epochs=20, batch_size=32, verbose=1)

# Save everything

# Saving history
with open(os.path.join('trained_models', "g3_" + output_name + "_{}".format(num_events) + "-history.pickle"), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Saving the model
model.save(os.path.join('trained_models', "g3_" + output_name + "_{}".format(num_events) + ".hd5"))

print(history.history)

# try:
#     plt.plot(history.history['loss'])
#     plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_loss.png"))
#     plt.clf()
#     plt.plot(history.history['acc'])
#     plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_acc.png"))
#     plt.clf()
#     plt.plot(history.history['mse'])
#     plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_mse.png"))
#     plt.clf()
#     plt.plot(history.history['mae'])
#     plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_mae.png"))
# except Exception as ex:
#     print("(!) Error building plots ", ex)
