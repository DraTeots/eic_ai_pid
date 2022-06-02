# first neural network with keras tutorial
import pickle
import time
import os
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
from geant3_parser import Geant3DataFile
from geant3_parser import build_train_set_xy
from event_display import print_tabled_event


file_name = os.path.join('data', 'shower_geant3_new.dat')


# file_name = 'sample_data.txt'
data_file = Geant3DataFile(file_name, skip_lines=3)

# split into input (X) and output (y) variables
parse_start = time.time()
print(f"Start preparing events...")


cells, true_values = build_train_set_xy(data_file, 5000, normalize=True)
parse_end = time.time()

print(f"Total events prepare time = {parse_end - parse_start}")
print(f"max hit value = {np.max(cells)}")
print(f"max e = {np.max(true_values[:,0])}")
print(f"Inputs shape original = {np.shape(cells)}")

# Reshape it to 11x11x1
cells = np.reshape(cells, (len(cells), 11, 11, 1))  # -1 => autodetermine

# Pad with 1 row and column of zeroes, so it divides by 2
# inputs = np.pad(inputs, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
print(f"Inputs shape new = {np.shape(cells)}")


energies = true_values[:, 0]        # True energy of the particle
cell_sums = true_values[:, 1]       # Sum of all cells (you don't need it)
incident_x = true_values[:, 2]      # X of incidence inside the cell 
incident_y = true_values[:, 3]      # Y 

event_number = 0

print_tabled_event(cells[event_number])
print("True data:")
print(f"  energy [GeV]          : {energies[event_number]}")
print(f"  incident_x [cell_size]: {incident_x[event_number]} ")
print(f"  incident_y [cell_size]: {incident_y[event_number]}")