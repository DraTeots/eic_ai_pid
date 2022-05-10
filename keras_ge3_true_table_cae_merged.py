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

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, UpSampling2D, Cropping2D, Input, Conv2DTranspose

from merge_function import filter_data


file_name = os.path.join('data', 'shower_geant3_new.dat')

def norm_func(e):
    return np.float64(np.log(e) / 11)


# file_name = 'sample_data.txt'
data_file = Geant3DataFile(file_name, skip_lines=3)

# split into input (X) and output (y) variables
parse_start = time.time()
print(f"Start preparing events...")

events_to_read = 5000

add_real_xy = False
inputs, answers, values = build_true_answers_train_set(data_file, events_to_read * 2, norm_func=norm_func, rnd_shift=((-4,4), (-4,4)) )
parse_end = time.time()
print(f"Inputs shape original = {np.shape(inputs)}")
print(f"Total events prepare time = {parse_end - parse_start}")
print(f"max hit value = {np.max(inputs)}")
# print(f"max e = {np.max(true_e)}")


inputs = np.reshape(inputs, (len(inputs), 11, 11, 1))  # -1 => autodetermine
answers = np.reshape(answers, (len(answers), 11, 11, 1))  # -1 => autodetermine
# # Pad with 1 row and column of zeroes, so it divides by 2
inputs = np.pad(inputs, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
answers = np.pad(answers, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
# print(f"Inputs shape new = {np.shape(inputs)}")

# input_first = inputs[:events_to_read]
# input_second = inputs[events_to_read:]
# input_merged = np.add(input_first, input_second)

# answers_first = answers[:events_to_read]
# answers_second = answers[events_to_read:]
# answers_merged = np.add(answers_first, answers_second)

# inputs = input_merged
# answers = answers_merged

# # filtering events that are within 1 square of each other
# tf = []
# for i in range(len(answers_first)):
#     col1 = np.argmax(np.argmax(answers_first[i], axis=1)) - 1
#     row1 = np.argmax(np.argmax(answers_first[i], axis=0)) - 1

#     col2 = np.argmax(np.argmax(answers_second[i], axis=1)) - 1
#     row2 = np.argmax(np.argmax(answers_second[i], axis=0)) - 1

#     if np.abs(col1 - col2) < 2 and np.abs(row1 - row2) < 2:
#         tf.append(False)
#     else:
#         tf.append(True)
# tf = np.array(tf)

# inputs = inputs[tf]
# answers = answers[tf]

temp = filter_data(inputs, answers, events_to_read)
inputs = temp[0]
answers = temp[1]

# Prints 11x11 cells event
def print_event(table):    
    if not len(table):
        print("EMPTY TABLE")
        return
    
    split_line = ""
    for irow, row in enumerate(table):
        if irow == 0:
            # First row => making title
            
            col_names = "ROW   " +  " ".join([f"{column_num:<5}" for column_num in range(len(row))])
            spaces = int((len(col_names) - len("COLUMNS"))/2)
            header = "{0}COLUMNS{0}".format(spaces*" ")
            split_line = "-"*len(col_names)
            print()            
            print(header)
            print(col_names)
            print(split_line)
        cells = f"{irow:<4}| " + " ".join([f"{cell[0]:<5.2}" for cell in row])
        print(cells)

    # Footer
    print(split_line)


print_event(inputs[0]*11)
print_event(answers[0]*11)
print("-----------------------------------")
print_event(inputs[1]*11)
print_event(answers[1]*11)



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
name = "g3__with_xy" if add_real_xy else "g3_true_table_cae"

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
