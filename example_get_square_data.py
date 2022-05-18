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

print_event(cells[0])