import uproot4
import awkward1 as ak
from hist import Hist
import hist
import matplotlib.pyplot as plt


def import_training_data(file_name, cal_size=3, start_event=0, process_events=None):
    """
    @param file_name: The filename with data
    @cal_size: Calorimeter size cal_size=3 - 3x3 size
    @start_event: Number of events to start processing
    @process_events: number of events to process
    @returns: (training_data, histograms)
    """

    root_file = uproot4.open(file_name)    
    tree = root_file["events"]
    #print(tree.keys())

    entry_start = start_event
    
    entry_stop = start_event + process_events if process_events else None
    events = tree.arrays(['EcalEndcapNHits.cellID',                          
                          'EcalEndcapNHits.position.x', 
                          'EcalEndcapNHits.position.y', 
                          'EcalEndcapNHits.position.z',
                          'EcalEndcapNHits.energyDeposit', 
                          'EcalEndcapNHits.truth.pdgID'],
                         aliases={"hits": "hit"},
                         library="ak",
                         how=tuple,
                         entry_start=entry_start, 
                         entry_stop=entry_stop)



    # Create histograms
    hist_hit_xy = Hist(hist.axis.Regular(50, -50, 50, name="X", label="Hits x [mm]", flow=False),
                       hist.axis.Regular(50, -50, 50, name="Y", label="Hits y [mm]", flow=False))

    hist_hit_z = Hist(hist.axis.Regular(50, 100, 105, name="Z", label="Hits z [mm]", flow=False))

    hist_total_dep = Hist(hist.axis.Regular(200, 0, 2000, name="Z", label="Cal total cells E deposit [mm]", flow=False))
    hist_true_e = Hist(hist.axis.Regular(50, 0, 10, name="E", label="Energy [GeV]", flow=False))

    # Events loop
    train_data = []     # Resulting data for each event
    inc_hits = []       # Cal incidence x,y,z for each event
    event_count = 0
    for event in zip(*events):
        arr_id, arr_hit_x, arr_hit_y, arr_hit_z, arr_de, arr_true_e = event

        # Print what event is being processed
        event_count += 1
        if (event_count % 1000) == 0:
            print(f"Events processed: {event_count}")
        
        # Skip event if we don't have hits or generated particles
        if not len(arr_hit_z) or not len(arr_true_e):
            continue            

        # in a simple scenario (1 particle gun particle per event) 1-st hit is an incidence hit
        hit_x, hit_y, hit_z = arr_hit_x[0], arr_hit_y[0], arr_hit_z[0]

        total_deposit = 0

        # Form a flat data training array. Indexes are like:
        # [0] [1] [2]
        # [3] [4] [5]
        # [6] [7] [8]
        event_train_data = cal_size*cal_size*[0]

        for col, row, adc in zip(arr_ce_emcal_col, arr_ce_emcal_row, arr_ce_emcal_adc) :
            total_deposit += adc
            index = col + row*cal_size
            event_train_data[index] = adc

            #print(cell.col, cell.row, cell.adc)
        train_data.append(event_train_data)
        inc_hits.append((hit_x, hit_y, hit_z))

        # Fill some histograms
        hist_hit_xy.fill(hit_x, hit_y)
        hist_hit_z.fill(hit_z)
        hist_total_dep.fill(total_deposit)
        hist_true_e.fill(arr_true_e[0])

    histograms = {
        'hit_xy': hist_hit_xy,
        'hit_z': hist_hit_z,
        'total_dep': hist_total_dep,
        'true_e': hist_true_e
    }

    return train_data, inc_hits, histograms


if __name__ == '__main__':
    import_training_data("/home/pipi/eicml/caloml/data/3x3_e_1GeV_10000ev.root")

