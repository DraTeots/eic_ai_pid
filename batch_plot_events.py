import os
import numpy as np

from event_display import table_display
from geant3_parser import Geant3DataFile, build_train_set


file_name = os.path.join('data', 'shower_geant3_new.dat')
data_file = Geant3DataFile(file_name, skip_lines=3)
input_data, true_e, sum_e = build_train_set(data_file, 10000, add_real_xy=False, normalize=True)

pics_prefix = os.path.join("event_plots", "single", "v1_")

# Remove x,y
input_data = input_data[:,2:]

# Reshape to events x 11 x 11 x 1 form
input_data = np.reshape(input_data, (len(input_data), 11, 11, 1))  # -1 => autodetermine

# Pad with 1 row and column of zeroes, so it divides by 2, events x 12 x 12 x 1 now
input_data = np.pad(input_data, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)

# printout shape
print(f"Inputs shape new = {np.shape(input_data)}")

for i in range(100):
    true_event = np.exp(input_data[i]*11)
    fig, ax = table_display(true_event)
    fig.savefig(f"{pics_prefix}{i}", transparent=False)
    del fig