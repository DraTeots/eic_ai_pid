import imp
import time
import itertools
from itertools import zip_longest  # for Python 3.x
import os
import numpy as np
import math

max_cell_index = 5
total_columns = max_cell_index * 2 + 1
total_rows = total_columns              # square calorimeter
event_size = total_columns * total_rows + 2  # 2 for real_x, real_y

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
        

def build_train_set(data_file, events_to_process=40000, add_real_xy=True):
    assert isinstance(data_file, Geant3DataFile)
    input_values = np.zeros((events_to_process, event_size))
    true_values = np.zeros((events_to_process, 1))
    sum_values = np.zeros((events_to_process, 1))

    for event_index, event in enumerate(itertools.islice(data_file.next_event(), events_to_process)):
        real_x, real_y, real_e, hits = event

        input_values[event_index, 0] = real_x if add_real_xy else 0
        input_values[event_index, 1] = real_y if add_real_xy else 0
        sum_e = 0
        for col, row, e in hits:
            if e == 0:
                continue
            data_index = row * total_columns + col + 2
            input_values[event_index, data_index] = math.log(e) / 11
            sum_e += e

        sum_values[event_index][0] = sum_e
        true_values[event_index][0] = real_e

    return input_values, true_values, sum_values        


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
    test_data_file()
