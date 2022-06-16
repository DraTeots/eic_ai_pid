# first neural network with keras tutorial
import pickle
import time
import os
from sys import platform

import numpy as np
import matplotlib.pyplot as plt
from geant3_parser import Geant3DataFile
from geant3_parser import build_train_set_xy


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

energies = true_values[:, 0]        # True energy of the particle
cell_sums = true_values[:, 1]       # Sum of all cells (you don't need it)
incident_x = true_values[:, 2]      # X of incidence inside the cell 
incident_y = true_values[:, 3]      # Y 

arr = np.array([[]])
for i in range(4999, 2):
    # print(f"Event #{i}")
    # print(f"  energy {energies[i]}")
    # print(f"  cell_sum {cell_sums[i]}")
    # print(f"  incident_x {incident_x[i]}")
    # print(f"  incident_y {incident_y[i]}")
    # print(cells[i])
	cells[i] = np.reshape(cells[i], 1, -1)
	cells[i+1] = np.reshape(cells[i+1], 1, -1)
	arr = np.hstack(cells[i], cells[i+1])
	
print(arr.shape)
print(cells[0].shape)