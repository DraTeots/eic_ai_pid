import uproot
import uproot4
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
import awkward as ak
import numpy as np
from lxml import etree


class EcalGeomInfo:
    """
    Holds geometry information about the calorimeter
    """

    def __init__(self, msize_x, msize_y, msize_z, mpositions, unit):
        self.module_size_x = msize_x
        self.module_size_y = msize_y
        self.module_size_z = msize_z        
        self.module_positions = mpositions        
        self.modules_x, self.modules_y, self.modules_z = zip(*mpositions)

        self.min_x = np.min(self.modules_x)
        self.max_x = np.max(self.modules_x)
        self.min_y = np.min(self.modules_y)
        self.max_y = np.max(self.modules_y)
        self.min_z = np.min(self.modules_z)
        self.max_z = np.max(self.modules_z)

        self.num_modules_x = int(np.around(1 + (self.max_x - self.min_x)/self.module_size_x))
        self.num_modules_y = int(np.around(1 + (self.max_y - self.min_y)/self.module_size_y))

        self.border_left = self.min_x - self.module_size_x/2
        self.border_right = self.max_x + self.module_size_x / 2
        self.border_top = self.max_y + self.module_size_y / 2
        self.border_bottom = self.min_y - self.module_size_x / 2

        self.unit = unit

    def __repr__(self):
        return f"EcalInfo: mod.size: " \
               f"{self.module_size_x:.2f}x{self.module_size_y:.2f}x{self.module_size_z:.2f}[{self.unit}]  " \
               f"num.modules: {self.num_modules_x}x{self.num_modules_y}"

    def print(self):
        """Prints all info about calorimeter"""

        print(f"module_size_x    : {self.module_size_x}")
        print(f"module_size_y    : {self.module_size_y}")
        print(f"module_size_z    : {self.module_size_z}")
        print(f"total_modules    : {len(self.module_positions)}")
        print(f"num_modules_x    : {self.num_modules_x}")
        print(f"num_modules_y    : {self.num_modules_y}")
        print(f"min_x            : {self.min_x}")
        print(f"max_x            : {self.max_x}")
        print(f"min_y            : {self.min_y}")
        print(f"max_y            : {self.max_y}")
        print(f"min_z            : {self.min_z}")
        print(f"max_z            : {self.max_z}")
        print(f"border_left      : {self.border_left}")
        print(f"border_right     : {self.border_right}")
        print(f"border_top       : {self.border_top}")
        print(f"border_bottom    : {self.border_bottom}")
        print(f"unit             : {self.unit}")

    def coord_to_index(self, x, y):
        """Returns an index in modules array from a coordinate

            @coordinate: coordinate in real units (usually mm)
            :least_position: Real coordinates of the most left or the most bottom module
            :module_size: Size of a module (and distance between module centers)
            """
        index_x = coord_to_index(x, self.min_x, self.module_size_x)
        index_y = coord_to_index(y, self.min_y, self.module_size_y)
        return index_x, index_y

    def arrays_to_event(self, energies, hits_x, hits_y):
        data = np.zeros((self.num_modules_x, self.num_modules_y))
        index_x, index_y = self.coord_to_index(hits_x, hits_y)
        indices = list(zip(index_x, index_y))
        data[index_x, index_y] = energies
        return data

    def get_modules_mask(self, edge_width=2):
        """Returns a numpy array of size (num_modules_x, num_modules_y) that represents ecal modules layout
           Array values: 0 - no module here, 1 or 2 - module is here, 2 - modules on the edge of ecal
        """
        print(self.num_modules_x, self.num_modules_y)
        mod_mask = np.zeros((self.num_modules_x, self.num_modules_y), dtype=np.int)
        for x, y, _ in self.module_positions:
            index_x = coord_to_index(x, self.min_x, self.module_size_x)
            index_y = coord_to_index(y, self.min_y, self.module_size_y)
            mod_mask[index_x][index_y] = 1

        # Go over the array and mark it as edges
        for index_x in range(self.num_modules_x):
            for index_y in range(self.num_modules_y):
                # Skip empty cells
                if mod_mask[index_x][index_y] == 0:
                    continue

                # Check if this module is on the edge of the array
                is_edge = False
                if index_x <= edge_width - 1 or index_x >= self.num_modules_x - edge_width or \
                        index_y <= edge_width - 1 or index_y >= self.num_modules_y - edge_width:
                    is_edge = True
                else:
                    # Check around the module
                    for area_step_x in range(edge_width + 1):
                        for area_step_y in range(edge_width + 1):
                            if area_step_x == 0 or area_step_x == 0:
                                continue
                            if mod_mask[index_x - area_step_x][index_y - area_step_y] == 0:
                                is_edge = True
                            if mod_mask[index_x - area_step_x][index_y + area_step_y] == 0:
                                is_edge = True
                            if mod_mask[index_x + area_step_x][index_y - area_step_y] == 0:
                                is_edge = True
                            if mod_mask[index_x + area_step_x][index_y + area_step_y] == 0:
                                is_edge = True

                # Mark if this is an edge
                if is_edge:
                    mod_mask[index_x][index_y] = 2
        return mod_mask

    def read_events_from_file(self, data_file_name, entry_start=0, entry_stop=1):
        # Open root file and get "events" tree from it
        tree = uproot.open(data_file_name)["events"]

        # At this point we have `hits_x` and `hits_y` a cell centers and corresponding `energies` arrays.
        # We can draw how such data looks
        events_to_process = entry_stop - entry_start
        rows = self.num_modules_x
        cols = self.num_modules_y
        input_values = np.zeros((events_to_process, rows, cols))
        true_values = np.zeros((events_to_process, 1))

        current_event = 0
        for i in range(entry_start, entry_stop):
            # Read energies, x and y positions, flatten arrays for simplicity
            energies = ak.flatten(tree['EcalEndcapNHits/EcalEndcapNHits.energy']
                                  .array(entry_start=i, entry_stop=i+1)).to_numpy()
            hits_x = ak.flatten(tree['EcalEndcapNHits/EcalEndcapNHits.position.x']
                                .array(entry_start=i, entry_stop=i+1)).to_numpy()
            hits_y = ak.flatten(tree['EcalEndcapNHits/EcalEndcapNHits.position.y']
                                .array(entry_start=i, entry_stop=i+1)).to_numpy()
            hits_y = ak.flatten(tree['EcalEndcapNHits/EcalEndcapNHits.position.y']
                                .array(entry_start=i, entry_stop=i + 1)).to_numpy()
            input_values[current_event] = self.arrays_to_event(energies, hits_x, hits_y)
            current_event+=1
        return input_values

    def mpl_plot_event(self, event, ax, mask=None):
        if not mask:
            mask = self.get_modules_mask()
        mpl_plot_mask(mask, ax)
        mpl_plot_event_array(event, ax)


def decode_id(id):
    # <id>system:8,sector:4,module:20</id>

    # 1 500 422  = 0x16E506 = 0x16 0xE5 0x06
    # 0x16 0xE5 0x06  ||  0x06 0xE5 0x16

    # our id (ALL values in HEX below)
    # FF B9 FF 91 01 03 0A 65
    #
    # system =  FF B9 FF 91 01 03 0A 65   &   00 00 00 00 00 00 00 FF = 65
    # module
    # FF B9 FF 91 01 03 0A 65 >> 8  =  00 FF B9 FF 91 01 03 0A
    # module = 00 FF B9 FF 91 01 03 0A   &   00 00 00 00 00 00 00 FF = 0A

    system = id & 0xFF
    sector = (id >> 8) & 0x0F
    module = (id >> 4) & 0xFFFFF

    return system, sector, module


def gdml_read_module_positions(base_name, tree):
    """
    Each module position is given like below.
    In this example base_name should be 'crystal_module'
    tree is lxml etree object
    <physvol copynumber="372" name="crystal_module_3720x55600cea9670">
        <volumeref ref="crystal_module0x556007d83b50"/>
        <position name="crystal_module_3720x55600cea9670_pos" unit="mm" x="-71.7499999999999" y="-399.75" z="175"/>
    </physvol>
    """

    # We grep xml_phys_volumes and xml_positions this way because it is order of magnitude faster
    xml_phys_volumes = tree.xpath(f"//physvol[starts-with(@name,'{base_name}')]")
    xml_positions = tree.xpath(f"//position[starts-with(@name,'{base_name}')]")

    positions = []
    for i, xml_physvol in enumerate(xml_phys_volumes):
        xml_position = xml_positions[i]
        x = float(xml_position.attrib['x'])
        y = float(xml_position.attrib['y'])
        z = float(xml_position.attrib['z'])
        positions.append((x, y, z))

    return positions


def gdml_read_module_geometry(base_name, tree):
    update = tree.xpath(f"//box[starts-with(@name,'{base_name}')]")[0]
    unit = update.attrib['lunit']
    size_x = float(update.attrib['x'])
    size_y = float(update.attrib['y'])
    size_z = float(update.attrib['z'])
    return size_x, size_y, size_z, unit


def coord_to_index(coordinate, least_position, module_size):
    """Returns an index in modules array from a coordinate

    @coordinate: coordinate in real units (usually mm)
    :least_position: Real coordinates of the most left or the most bottom module
    :module_size: Size of a module (and distance between module centers)
    """
    float_index = np.around((coordinate - least_position) / module_size)
    if isinstance(float_index, ak.highlevel.Array):
        return ak.values_astype(float_index, "int64")
    if isinstance(float_index, np.ndarray):
        return float_index.astype(int)
    else:
        return int(float_index)



def gdml_read_ecal_info(file_name):
    """
    Reads ecal geometry information from file_name and returns in form of EcalInfo
    """

    # Open file
    geometry_xml = etree.parse(file_name)

    # Read single module geometry
    msize_x, msize_y, msize_z, unit = gdml_read_module_geometry('wrapper_vol', geometry_xml)

    # Read module positions
    mpositions = gdml_read_module_positions('wrapper_vol', geometry_xml)

    return EcalGeomInfo(msize_x, msize_y, msize_z, mpositions, unit)


def build_calorimeter_section(ax, positions, size_x, size_y):
    dx = size_x / 2.0
    dy = size_y / 2.0

    module_rects = []
    for xyz in positions:
        if len(xyz) == 3:
            x, y, _ = xyz
        else:
            x, y = xyz
        patch = patches.Rectangle((x-dx, y-dy), width=size_x, height=size_y, edgecolor='black', facecolor='gray')
        module_rects.append(patch)

    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)

    ax.autoscale()
    ax.axis('equal')

    return ax


def mpl_plot_mask(mask, ax, show_legend=True):
    num_modules_x, num_modules_y = mask.shape
    module_rects = []

    # Go over array creating squares for each cell
    for index_x in range(num_modules_x):
        for index_y in range(num_modules_y):

            # Select cell colors
            facecolor = 'gray' if mask[index_x, index_y] > 0.1 else 'white'
            if mask[index_x, index_y] > 1:
                facecolor = 'dimgray'

            # Create & add Rectangle
            patch = patches.Rectangle((index_x, index_y), width=1, height=1, edgecolor='black', facecolor=facecolor)
            module_rects.append(patch)

    # Add rectangles to plot
    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)
    ax.autoscale()
    ax.axis('equal')

    # Should we create a legend?
    if show_legend:
        edge_legend_patch = patches.Patch(color='dimgray', label='Edge modules')
        inside_legend_patch = patches.Patch(color='gray', label='Inside modules')
        ax.legend(handles=[edge_legend_patch, inside_legend_patch])

    return ax


def mpl_plot_hits(energies, x_positions, y_positions, ax, size_x, size_y):
    """
    Plots in a real coordinates: energies by x and y positions
    """

    norm = LogNorm()
    norm.autoscale(energies)
    cmap = cm.get_cmap('inferno')

    dx = size_x / 2.0
    dy = size_y / 2.0

    module_rects = []
    for x, y, e in zip(x_positions, y_positions, energies):
        patch = patches.Rectangle((x - dx, y - dy), size_x, size_y, edgecolor='black', facecolor=cmap(norm(e)))
        module_rects.append(patch)

    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)
    ax.autoscale()
    ax.axis('equal')
    return norm, cmap, ax


def mpl_plot_event_array(data, ax):
    """
    Accepts 2 dimensional numpy array as
    """
    assert isinstance(data, np.ndarray)
    norm = LogNorm()
    norm.autoscale(data.flat)
    cmap = cm.get_cmap('inferno')

    dx = 1 / 2.0
    dy = 1 / 2.0

    module_rects = []
    num_modules_x, num_modules_y = data.shape
    for index_x in range(num_modules_x):
        for index_y in range(num_modules_y):
            e = data[index_x, index_y]
            patch = patches.Rectangle((index_x, index_y), width=1, height=1, edgecolor='black', facecolor=cmap(norm(e)))
            module_rects.append(patch)

    col = PatchCollection(module_rects, match_original=True)
    ax.add_collection(col)
    ax.autoscale()
    ax.axis('equal')
    return norm, cmap, ax


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