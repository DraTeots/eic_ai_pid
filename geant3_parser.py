import imp
import time
import itertools
from itertools import zip_longest  # for Python 3.x
import os
import numpy as np
import math
import random



class Geant3DataFile:
    max_cell_index = 32
    cell_width = 2.077  # cm
    cell_height = 2.075  # cm
    hole_size = 5  # number of cells (odd)

    def __init__(self, file_name, skip_lines=0):

        time_start = time.time()

        with open(file_name, 'r') as content_file:
            content = content_file.read()
        time_opened = time.time()
        print(f"Time open {time_opened - time_start}")

        self.file_lines = content.split("\n")
        self.line_iter = skip_lines
        self.skip_lines = skip_lines
        self.reco_hist_bins = {0: 0, 1: 0, 2: 0, 3: 0}
        time_split = time.time()
        print(f"Time split {time_split - time_opened}")

        del content

    @staticmethod
    def get_empty_event():
        return "", "", "", [], []  # hits

    def next_event(self):

        def grouper(iterable, n, padvalue=None):
            """grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"""
            return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)

        for line_index, line in enumerate(self.file_lines):
            if line_index < self.skip_lines or not line:
                continue

            tokens = line.split()

            # x, y, e, N cells, 3xcell (at least 1 cell)
            if len(tokens) < 7:
                continue

            dx = float(tokens[0])
            dy = float(tokens[1])
            e = float(tokens[2])
            tokens = tokens[4:]
            hits = [(int(str_x), int(str_y), float(str_e)) for str_x, str_y, str_e in grouper(tokens, 3)]

            yield dx, dy, e, hits


def process_events(data_file):
    for event in itertools.islice(data_file.next_event(), 5):
        real_x, real_y, real_e, hits = event
        print(f"Real: x={real_x}, y={real_y}, energy={real_e}")
        print("   x    y       energy")
        for hit in hits:
            hit_x, hit_y, hit_e = hit
            print(f"{hit_x:>4} {hit_y:>4} {hit_e:>9}")
        print()
        

def build_train_set(data_file, events_to_process=40000, add_real_xy=True, normalize=True):
    assert isinstance(data_file, Geant3DataFile)

    # Some constants    
    max_cell_index = 5
    total_columns = max_cell_index * 2 + 1
    total_rows = total_columns              # square calorimeter
    event_size = total_columns * total_rows + 2  # 2 for real_x, real_y

    
    input_values = np.zeros((events_to_process, event_size))
    true_values = np.zeros((events_to_process, 1))
    sum_values = np.zeros((events_to_process, 1))

    data = itertools.islice(data_file.next_event(), events_to_process)
    
    for event_index, event in enumerate(data):
        real_x, real_y, real_e, hits = event

        input_values[event_index, 0] = real_x if add_real_xy else 0
        input_values[event_index, 1] = real_y if add_real_xy else 0
        sum_e = 0
        for col, row, e in hits:
            if e == 0:
                continue
            rowcol_shift = int(total_columns/2)
            data_index = int((row + rowcol_shift)*total_columns + col + rowcol_shift + 2)
            norm_e = math.log(e) / 11 if normalize else e
            # >oO debug print(f"{data_index:>4} {row:>4} {col:>4} {norm_e:>8}")
            input_values[event_index, data_index] = norm_e
            sum_e += e

        sum_values[event_index][0] = sum_e
        true_values[event_index][0] = real_e

    return input_values, true_values, sum_values


def build_train_set_xy(data_file, events_to_process=10000, normalize=True):
    assert isinstance(data_file, Geant3DataFile)

    # Some constants    
    max_cell_index = 5
    total_columns = max_cell_index * 2 + 1
    total_rows = total_columns              # square calorimeter
    event_size = total_columns * total_rows # 2 for real_x, real_y
    
    input_values = np.zeros((events_to_process, event_size))
    true_values = np.zeros((events_to_process, 4))

    data = itertools.islice(data_file.next_event(), events_to_process)
    
    for event_index, event in enumerate(data):
        real_x, real_y, real_e, hits = event
        
        sum_e = 0
        for col, row, e in hits:
            if e == 0:
                continue
            rowcol_shift = int(total_columns/2)
            data_index = int((row + rowcol_shift)*total_columns + col + rowcol_shift)
            norm_e = math.log(e) / 11 if normalize else e
            # >oO debug print(f"{data_index:>4} {row:>4} {col:>4} {norm_e:>8}")
            input_values[event_index, data_index] = norm_e
            sum_e += e

        true_values[event_index][0] = real_e
        true_values[event_index][1] = sum_e
        true_values[event_index][2] = real_x
        true_values[event_index][3] = real_y

    return input_values, true_values


def build_true_answers_train_set(data_file, events_to_process=40000, norm_func=None, rows=11, cols=11, rnd_shift=None):
    """
    rnd_shift = ((row_min, row_max), (col_min, col_max)) where row,col=(0,0) is a center
    """
    assert isinstance(data_file, Geant3DataFile)
    input_values = np.zeros((events_to_process, rows, cols))
    true_values = np.zeros((events_to_process, rows, cols))
    e_values = np.zeros((events_to_process, 2))
    central_row = int(rows/2)
    central_col = int(cols/2)

    data = itertools.islice(data_file.next_event(), events_to_process)
    

    for event_index, event in enumerate(data):
        real_x, real_y, real_e, hits = event

        #input_values[event_index, 0] = real_x if add_real_xy else 0
        #input_values[event_index, 1] = real_y if add_real_xy else 0
        sum_e = 0

        # Position
        row_shift = 0
        col_shift = 0
        if rnd_shift:
            row_shift = random.randint(rnd_shift[0][0], rnd_shift[0][1])
            col_shift = random.randint(rnd_shift[1][0], rnd_shift[1][1])

        # Going for each hit
        for col, row, e in hits:
            # Energy
            if e == 0:
                continue
            
            norm_e = norm_func(e) if norm_func else e
            sum_e += e

            row_index = central_row + row_shift + row
            col_index = central_col + col_shift + col

            # >oO
            # print(f"row_index({row_index}) = central_row({central_row}) + row_shift({row_shift}) + row({row})")
            # print(f"col_index({col_index}) = central_col({central_col}) + col_shift({col_shift}) + col({col})")
            # print(f"norm_e={norm_e} e={e}")

            if row_index >= 0 and row_index < rows and col_index >= 0 and col_index < cols:
                input_values[event_index, row_index, col_index] = norm_e
            
        
        true_values[event_index, central_row + row_shift, central_col + col_shift] = norm_func(sum_e) if norm_func else real_e
        
        e_values[event_index][0] = real_e
        e_values[event_index][1] = sum_e

    return input_values, true_values, e_values

def test_data_file():
    file_name = os.path.join('data', 'shower_geant3_new.dat')
    # file_name = 'sample_data.txt'

    data_file = Geant3DataFile(file_name, skip_lines=0)

    parse_start = time.time()
    process_events(data_file)

    print(f"Clusters in one event found by reconstruction = {data_file.reco_hist_bins}")

    parse_end = time.time()
    print(f"Time parse {parse_end - parse_start}")


if __name__ == '__main__':
    #test_data_file()
    file_name = os.path.join('data', 'shower_geant3_new.dat')
    data_file = Geant3DataFile(file_name, skip_lines=3)
    input_data, true_e, sum_e = build_train_set(data_file, 2, add_real_xy=False, normalize=False)
