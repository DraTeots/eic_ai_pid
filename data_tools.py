import numpy as np


def find_max_valxy(event):
    """ Returns maximum value, row and col indexes"""

    max_row, max_col, max_val = 0, 0, 0
    for row_index, row in enumerate(event):
        for col_index, col_vals in enumerate(row):
            if col_vals > max_val:
                max_val = col_vals[0]
                max_row = col_index
                max_col = row_index

    return max_val, max_row, max_col


def reduce_9cells(event):
    """Reduces a cluster to a 9cells block"""

    max_val, max_row, max_col = find_max_valxy(event)

    # sets box to be around max val
    max_row = max_row + 1
    min_row = max_row - 1
    max_col = max_col + 1
    min_col = max_col - 1

    # TODO if max_col+1 is outside of the box? 

    # gets outside box energy
    outside_e = 0
    for row_i, row in enumerate(event):
        for col_i, col_vals in enumerate(row):
            if row_i < min_col or row_i > max_col or col_i < min_row or col_i > max_row:
                outside_e += col_vals
                event[row_i][col_i] = 0

    added_e = float(outside_e / 9) # outside energy / num cells left 
    # adds extra energy to all cells left
    for row_i, row in enumerate(event):
        for col_i, col_vals in enumerate(row):
            if min_col <= row_i <= max_col and min_row <= col_i <= max_row:
                event[row_i][col_i] += added_e


def merge_clusters(inputs, answers, minimal_dist=2):
    """Filters merged cluster data"""

    events_to_read = int(len(inputs)/2)

    input_first = inputs[:events_to_read]
    input_second = inputs[events_to_read:]
    input_merged = np.add(input_first, input_second)

    answers_first = answers[:events_to_read]
    answers_second = answers[events_to_read:]
    answers_merged = np.add(answers_first, answers_second)

    inputs = input_merged
    answers = answers_merged

    # filtering events that are within 1 square of each other
    is_far_enough = []
    for i in range(len(answers_first)):
        col1 = np.argmax(np.argmax(answers_first[i], axis=1)) - 1
        row1 = np.argmax(np.argmax(answers_first[i], axis=0)) - 1

        col2 = np.argmax(np.argmax(answers_second[i], axis=1)) - 1
        row2 = np.argmax(np.argmax(answers_second[i], axis=0)) - 1

        # The distance between clusters. Is it good? Far enough?
        if np.abs(col1 - col2) < minimal_dist and np.abs(row1 - row2) < minimal_dist:
            is_far_enough.append(False)
        else:
            is_far_enough.append(True)
    is_far_enough = np.array(is_far_enough)

    inputs = inputs[is_far_enough]
    answers = answers[is_far_enough]
    return inputs, answers
