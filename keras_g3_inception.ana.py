from __future__ import print_function

import uproot4
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
import os
import numpy as np
import pandas as pd
import itertools

from event_display import  table_display, table_display_compare, get_bin_centers

from geant3_parser import build_true_answers_train_set
from geant3_parser import Geant3DataFile

from keras.models import load_model

from statistics import NormalDist
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

pics_prefix = os.path.join("plots", "v1_")

def norm_func(e):
    return np.float64(np.log(e) / 11)

file_name = os.path.join('data', 'shower_geant3_new.dat')
data_file = Geant3DataFile(file_name, skip_lines=3)
input_data, answers, values = build_true_answers_train_set(data_file, 50000, norm_func=norm_func, rnd_shift=((-2,2), (-2,2)))
print(np.shape(answers))
print(np.shape(input_data))

# Remove x,y

# Reshape to events x 11 x 11 x 1 form
input_data = np.reshape(input_data, (len(input_data), 11, 11, 1))  # -1 => autodetermine
answers = np.reshape(answers, (len(answers), 11, 11, 1))  # -1 => autodetermine
print(np.shape(answers))
print(np.shape(input_data))


# Pad with 1 row and column of zeroes, so it divides by 2, events x 12 x 12 x 1 now
#input_data = np.pad(input_data, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)
#answers = np.pad(answers, ((0,0), (0,1), (0,1), (0,0)), mode='constant', constant_values=0)

# printout shape
print(f"Inputs shape new = {np.shape(input_data)}")

# Load NN model
num_events = 200000
name = "inception"
model = load_model(os.path.join("trained_models", "g3_" + name + "_{}".format(num_events) + ".hd5"))

# Run ML on data
ml_events = model.predict(input_data)
# reshape events
ml_events = np.reshape(ml_events, (len(ml_events), 11, 11, 1))

def data_to_energies(data):
    # We took log(hit) to make hits linear. /11 as a scale. Now reverting it    
    denorm_data = np.exp(data*11)

    # Get rid of array form of cells
    # For ML you have arrays like 
    # [num_pictures][x_pixels][y_pixels][colors]  where colors=3 for RGB,  colors=1 for BW
    # We use colors=1 (we just have an energy deposit in a cell) and we have num_pictures=num_events, [xy]_pixels=[xy]_cells
    # No we want to remove this useless [1]
    # (len(data), 12, 12, 1) => (len(data), 12, 12)
    reshaped = np.reshape(denorm_data, (len(data), 11, 11))

    # Sum along (x_pixels, y_pixels) axis => basicly summ everything in one event
    sums = np.sum(reshaped, axis=(1,2))
    return sums

sr_e = data_to_energies(input_data)             # SR = simple reco = energy from original data, summed together
ml_e = data_to_energies(ml_events)              # Energy from ML (sum everything)
true_e = 1000*np.reshape(values[:,0], len(input_data))    # True energy, 1000 to convert from GeV to

# create pandas table... for fun
pd = pd.DataFrame({"sr_e": sr_e, "ml_e": ml_e, "true_e":true_e})

# using tuple unpacking for multiple Axes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,6))

ax1.hist(sr_e, bins=10)
ax2.hist(ml_e, bins=100, range=(0,10000))
ax3.hist2d(sr_e, ml_e/sr_e, bins=(15, 100))
#plt.savefig(pics_prefix + "e_2dhist.png")
plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_hists.png"))
plt.clf()

# Build numpy 2d histogram
err_bins = np.arange(0.4, 1.4, 0.05)
e_bins = np.arange(0, 6000, 500)
err_bin_centers = get_bin_centers(err_bins)
e_bin_centers = get_bin_centers(e_bins)
de = ml_e/sr_e
dee_hist, de_bin_edges, e_bin_edges = np.histogram2d(sr_e, de, bins=[e_bins, err_bins])
e_bins_num, dee_bins_num = dee_hist.shape
e_bins_num, dee_bins_num, len(err_bin_centers)

plot_columns = 4
plot_rows = int(e_bins_num / plot_columns) + 1
(int(e_bins_num/plot_columns) + 1)*plot_columns
print(f"Total subplots: {e_bins_num} plot rows={plot_rows} columns={plot_columns}")
fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(plot_columns*4, plot_rows*4), sharey=True)

# Fit fragment is taken from here
# https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

# axes is a 2d array consisting of rows and colums [[...], [...], ...]
# flatten it as we need to feel it one by one
axes = list(itertools.chain.from_iterable(axes))
sigmas = []
dee_peaks = []
for i in range(e_bins_num):
    hist1d = dee_hist[i]
    ax = axes[i]
    # (!) using ax.step() for histograms
    # But commented ax.plot() to check ax.step() centers correctness
    # ax.plot(err_bin_centers, hist1d, "-o")    
    ax.step(err_bins[1:], hist1d)

    # Fit using Gaussian function
    H, A, x0, sigma = gauss_fit(err_bin_centers, hist1d)
    std = np.std(hist1d)    

    # Plot gauss
    smooth_gauss_x = np.arange(err_bins[0], err_bins[-1], (err_bins[-1]-err_bins[0])/1000)
    smooth_gauss_y = gauss(smooth_gauss_x, H, A, x0, sigma)
    ax.plot(smooth_gauss_x, smooth_gauss_y, '--r', label='fit')

    # titles
    ax.set_title(f"Eml/E for E={e_bins[i]}-{e_bins[i+1]}\n sgm={sigma:.3f} peak={x0:.4f}", wrap=True)  

    # Add our error
    sigmas.append(sigma)
    dee_peaks.append(x0)
    

plt.tight_layout()
#plt.savefig(pics_prefix + "ml_div_true_fit")
plt.plot()
plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_guass.png"))
plt.clf()

plt.plot(e_bin_centers, sigmas, "--o")
plt.title("Sigma by energy")
plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_sig_by_energy.png"))
plt.clf()

#plt.savefig(pics_prefix + "errors")

sigma_div_e = sigmas/e_bin_centers

plt.plot(e_bin_centers, sigma_div_e, "--o")
plt.xlabel("Energy, MeV")
plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_errors_div_energy.png"))
plt.clf()


def calo_error(x, a, b, c):
    return a/np.sqrt(x) + b/x + c

popt, pcov = curve_fit(calo_error, e_bin_centers, sigma_div_e, p0=[1, 1, 0])
a, b, c = popt
print(f"a={a} b={b} c={c}")

# Plot gauss
smooth_cal_x = np.arange(e_bin_centers[0], e_bin_centers[-1], (e_bin_centers[-1]-e_bin_centers[0])/1000)
smooth_cal_y = calo_error(smooth_cal_x, a, b, c)

fig, ax = plt.subplots()
ax.plot(e_bin_centers, sigma_div_e, "o")
ax.plot(smooth_cal_x, smooth_cal_y, '--r', label='fit')
plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_smooth.png"))
plt.clf()

plt.plot(e_bin_centers, dee_peaks , "--o")
plt.title("Mean MLE/E by energy")
plt.ylim(0.4,1.4)

plt.savefig(os.path.join('plots', "g3_" + name + "_{}".format(num_events), name +"_mean_mle.png"))
plt.clf()