import uproot4
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
import awkward1 as ak
import numpy as np
from lxml import etree as ET


def get_module_positions(base_name, tree):

    # <position  name="pos_397" x=... y=...  z=... ... />
    """
    <physvol copynumber="372" name="crystal_module_3720x55600cea9670">
        <volumeref ref="crystal_module0x556007d83b50"/>
        <position name="crystal_module_3720x55600cea9670_pos" unit="mm" x="-71.7499999999999" y="-399.75" z="175"/>
      </physvol>
    """
    
    # We grep xml_phys_volumes and xml_positions this way because it is order of magnitude faster
    xml_phys_volumes = tree.xpath(f"//physvol[starts-with(@name,'{base_name}')]")
    xml_positions = tree.xpath(f"//position[starts-with(@name,'{base_name}')]")

    positions_by_id = {}
    for i, xml_physvol in enumerate(xml_phys_volumes):
        xml_position = xml_positions[i]        
        x = float(xml_position.attrib['x'])
        y = float(xml_position.attrib['y'])
        name = xml_physvol.attrib['name']
        copynumber = xml_physvol.attrib['copynumber']
        module_id = name[len(base_name) + len(copynumber)+1:]     # +1 for '_'
        module_id = int(module_id, 16)
        positions_by_id[module_id] = (x,y)
    return positions_by_id


def get_module_geometry(base_name, tree):
    update = tree.xpath(f"//box[starts-with(@name,'{base_name}')]")[0]
    unit = update.attrib['lunit']
    size_x = float(update.attrib['x'])
    size_y = float(update.attrib['y'])
    size_z = float(update.attrib['z'])
    return size_x, size_y, size_z, unit

def build_calorimeter_section(ax, positions, size_x, size_y):
    dx = size_x / 2.0
    dy = size_y / 2.0

    module_rects = []
    for position in positions:
        x, y = position
        patch = patches.Rectangle((x-dx, y-dy), width=size_x, height=size_y, edgecolor='black', facecolor='gray')
        module_rects.append(patch)

    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)

    ax.autoscale()
    ax.axis('equal')

    return ax


def plot_calorimeter_hits(root_file, ax, pos_by_id, size_x, size_y, start_event, process_events=1):

    tree = root_file["events"]

    entry_start=start_event
    entry_stop = start_event + process_events
    events = tree.arrays(['ce_emcal_id', 'ce_emcal_adc'],
                         library="ak", how="zip", entry_start=entry_start, entry_stop=entry_stop)
    print(events.type)
    ids = ak.flatten(events.ce_emcal.id)
    weights = ak.flatten(events.ce_emcal.adc)

    build_calorimeter_section(ax, pos_by_id.values(), size_x, size_y)

    norm = LogNorm()
    norm.autoscale(weights)
    cmap = cm.get_cmap('inferno')

    weights_by_id = {}

    for id, weight in zip(ids, weights):
        if id in weights_by_id.keys():
            weights_by_id[id] += weight
        else:
            weights_by_id[id] = weight

    dx = size_x / 2.0
    dy = size_y / 2.0

    module_rects = []
    for id, weight in weights_by_id.items():
        if id>1000000:
            continue
        x,y = pos_by_id[id]
        patch = patches.Rectangle((x-dx, y-dy), size_x, size_y, edgecolor='black', facecolor=cmap(norm(weight)))
        module_rects.append(patch)

    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)
    return norm, cmap, ax


def table_display(event_data, fig=None, ax=None, cal_size=12, cell_size=1):
    """
    event_data should be cal_size X cal_size array
    """
    # constants
    size_x = cell_size
    size_y = cell_size
    dx = size_x / 2.0
    dy = size_y / 2.0

    # go through all cells and calculate their centers
    centers = np.arange(-cal_size/2.0 + cell_size/2, cal_size/2 + cell_size/2, 1)
    positions = []
    for y_iter in range(cal_size):
        for x_iter in range(cal_size):
            positions.append((centers[x_iter], centers[y_iter]))

    # plot calorimeter with empty cells
    if not fig or not ax:
        fig, ax = plt.subplots()
    build_calorimeter_section(ax, positions, 1, 1)

    # Create a heat map
    norm = LogNorm()
    norm.autoscale(event_data)
    cmap = cm.get_cmap('inferno')

    # Convert data to rectangular patches    
    module_rects = []
    for y_iter in range(cal_size):
        for x_iter in range(cal_size):
            x = centers[x_iter]
            y = centers[y_iter]
            weight = event_data[x_iter][y_iter][0]
            #print(x,y,weight)
            patch = patches.Rectangle((x-dx, y-dy), size_x, size_y, edgecolor='black', facecolor=cmap(norm(weight)))
            module_rects.append(patch)

    # plot rectangles with data
    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)

    # plot heatmap legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(event_data)
    fig.colorbar(sm)
    
    return fig, ax


def table_display_compare(left_data, right_data, fig=None, cal_size=12, cell_size=1):
    """
    event_data should be cal_size X cal_size array
    """
    # constants
    size_x = cell_size
    size_y = cell_size
    dx = size_x / 2.0
    dy = size_y / 2.0

    # go through all cells and calculate their centers
    centers = np.arange(-cal_size/2.0 + cell_size/2, cal_size/2 + cell_size/2, 1)
    positions = []
    for y_iter in range(cal_size):
        for x_iter in range(cal_size):
            positions.append((centers[x_iter], centers[y_iter]))

    # plot calorimeter with empty cells
    if not fig:
        fig = plt.figure(figsize=(12,9))

    ax_left, ax_right = fig.subplots(1, 2)
    build_calorimeter_section(ax_left, positions, 1, 1)
    build_calorimeter_section(ax_right, positions, 1, 1)

    # Create a heat map
    norm = LogNorm()    
    norm.autoscale(np.vstack((left_data,right_data)))
    cmap = cm.get_cmap('inferno')

    def display_event_values(data, ax):
        # Convert data to rectangular patches    
        module_rects = []
        for y_iter in range(cal_size):
            for x_iter in range(cal_size):
                x = centers[x_iter]
                y = centers[y_iter]
                weight = data[x_iter][y_iter][0]
                #print(x,y,weight)
                patch = patches.Rectangle((x-dx, y-dy), size_x, size_y, edgecolor='black', facecolor=cmap(norm(weight)))
                module_rects.append(patch)

        # plot rectangles with data
        col = PatchCollection(module_rects, match_original=True)
        ax.add_collection(col)

    display_event_values(left_data, ax_left)
    display_event_values(right_data, ax_right)

    # plot heatmap legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.vstack((left_data,right_data)))
    fig.colorbar(sm, orientation="horizontal", ax=[ax_left,ax_right], extend="both")

    ax_left.set_aspect('equal', 'box')
    ax_right.set_aspect('equal', 'box')
    
    return fig, ax_left, ax_right


def get_bin_centers(bins):
    """Calculates bin centers out of bin boundaries"""
    assert len(bins) >= 2
    return bins[:-1] + (bins[1:] - bins[:-1]) / 2


# Prints 11x11 cells event
def print_tabled_event(table):
    """
    Print calorimeter data as a [rows[column_values]] table
    """
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
        cells = f"{irow:<4}| " + " ".join([f"{cell[0]*11:<5.2}" for cell in row])
        print(cells)

    # Footer
    print(split_line)


if __name__ == "__main__":
    # using the variable ax for single a Axes
    fig, ax = plt.subplots()
    positions = [
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.5, -0.5),
        (-0.5, -0.5),
    ]
    build_calorimeter_section(ax, positions, 1, 1)
    #fig.plot()
    #plt.plot()
    ax.plot()