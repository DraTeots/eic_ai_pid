
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
            if row_i >= min_col and row_i <= max_col and col_i >= min_row and col_i <= max_row:
                event[row_i][col_i] += added_e

