import uproot4
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
import awkward1 as ak

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

