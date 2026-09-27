# -*- coding: utf-8 -*-
"""
NpxRateMap.py

Script version of NpxRateMap.ipynb (all cells up to the "Ignore the rest" markdown).
Based on the NeuroPyxels quickstart (UCL Neuropixels course, October 2022).
Github repo: https://github.com/m-beau/NeuroPyxels
Install with `pip install npyx`.

Cells are kept in the same order as the notebook; all hardcoded paths and
parameters are collected in the USER SETTINGS block below.

"""
"""

Settings block at the top

Paths: DP (the Neuropixels data folder) and BEHAVIOUR_CSV (the tracking file).
Units: which quality to load ("mua", and later "good") and which unit indices to use (10 and 7).
Analysis parameters: frame rate 30, moving-average window 15, spike-matching tolerance 0.04 s, 
the filter thresholds (1000, 0.9, 0.2 and 30), spatial bin 5 cm, smoothing k=3 and h=5, and the 
heading settings (30 histogram bins, 30 Hz, 10° bins, smoothing 1).
Plotting: the raster x-limit (0–3000 s) and the gap between rows (1.5).
Each value is used exactly where the old hardcoded number was, so nothing else about the processing changed.

Changes needed for it to work as a script (none change the results)

The notebook-only commands (%config, %reload_ext, %autoreload) are kept as comments.
Lines that only displayed a value in the notebook (onsets, good_units) are now print(...).
I added plt.show() to the four figures that didn't have one, so each still appears on its own.
I added import os and gathered the repeated imports at the top.
The cell marked "ONLY FOR BATCH" is now behind a switch, RUN_BATCH_LOOP, which is on by default. 
Leaving it on runs all the cells in order as-is. That means the batch loop replaces u, t and the 
matched spike times sp_t with the last unit's, and the later rate-map cells use those. Set it to 
False to keep working with unit 7, as you probably did when running cells selectively.

Two likely bugs left unchanged

In the centre-proximity filter section, the spike-index bound uses len(x_filtered), 
which comes from the previous filter, instead of len(x_fin).
In the last heading section, sp_t holds spike times in seconds but is converted to 
integers and used as array positions (heading_deg[sp_t], x_fin[sp_t]). So those plots 
pick the wrong samples, not the positions at spike times.
"""

#%%############################################################################
# USER SETTINGS ###############################################################
###############################################################################

# --- Directories / files -----------------------------------------------------
# Spike-sorted Neuropixels dataset (kilosort + phy). dp stands for datapath
DP = "F:/Fa1680378B/1Cntrl/ks4phy_autocurated"
# Behavioural tracking csv (columns: 'time' [ms], 'x', 'y')
BEHAVIOUR_CSV = r'E:/Runita/TraackingWithCorrectTS/TSforConcatData/1680378B/Day5/Fa1680378B_Day5_1Cntrl.csv'

# --- Unit selection ------------------------------------------------------------
UNIT_QUALITY_FIRST = "mua"     # quality used in sections 1.2, 1.3 and the single-unit sync
UNIT_QUALITY_SECOND = "good"   # quality used for the raster after the batch loop
UNIT_IDX_EXAMPLE = 10          # unit index for the example spike train (section 1.3)
UNIT_IDX_ANALYSIS = 7          # unit index used for the spike/tracking sync and rate maps

# --- Batch loop ----------------------------------------------------------------
# Notebook cell marked "ONLY FOR BATCH". NOTE: when True it overwrites t, u and sp_t
# with the last unit of the loop (same as running the cell in the notebook).
RUN_BATCH_LOOP = True

# --- Raster plots --------------------------------------------------------------
RASTER_XLIM = [0, 3000]        # s
RASTER_Y_OFFSET = 1.5          # vertical offset between units in raster plots

# --- Tracking --------------------------------------------------------------------
FPS = 30                       # video recording rate (Hz)
MOVING_AVG_WINDOW = 15         # moving average window size (frames)

# --- Spike / behaviour syncing -----------------------------------------------------
SPIKE_MATCH_TOLERANCE = 0.04   # s, max distance between a spike and a behaviour timestamp

# --- Tracking filters ---------------------------------------------------------------
SPEED_THRESHOLD = 1000                 # speed filter threshold
CENTER_THRESHOLD_FACTOR = 0.9          # x median radius -> centre proximity threshold
MAX_JUMP_FACTOR_CIRCULAR = 0.2         # x median radius -> max allowed jump (circular filter)
MAX_ALLOWED_JUMP_PIXEL = 30            # max allowed jump (pixel-value filter)

# --- Rate map ---------------------------------------------------------------------
SPATIAL_BIN_SIZE = 5           # bin size in cm
GAUSS_K = 3                    # gaussian smoothing kernel
GAUSS_H = 5                    # max distance to avoid extrapolation error

# --- Heading direction ------------------------------------------------------------
HEADING_HIST_BINS = 30         # bins for the polar histogram of spike headings
HEADING_SAMPLING_RATE = 30     # Hz
HEADING_BIN_SIZE = 10          # degrees
HEADING_SMOOTH_SIGMA = 1       # gaussian_filter1d sigma


#%%############################################################################
# Imports #####################################################################
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
# (notebook: %config InlineBackend.figure_format = 'retina')

# (notebook: %reload_ext autoreload / %autoreload 2 -- ensures npyx is reimported)
from npyx import *

import pandas as pd
import math as m
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d


#%%############################################################################
# 1 - Loading Neuropixels data ################################################
###############################################################################

dp = DP  # dp stands for datapath
print(os.path.exists(dp))

#%% 1.1 - Load Neuropixels sync channel, metadata ------------------------------

# Threshold crosses of the sync channel acquired with the SMA port on the acquisition board
onsets, offsets = get_npix_sync(dp)
print(onsets)

# blah.ap.meta file contents and more
meta = read_metadata(dp)
print(f"Neuropixels probe {meta['probe_version']} acquired with {meta['acquisition_software']}.")
fs = meta['NP']['sampling_rate']
print(f"Highpass filtered data at {meta['NP']['binary_relative_path']} was acquired at {fs} Hz.")

#%% 1.2 Load units and unit qualities (good, mua, noise...) --------------------

all_units = get_units(dp)
good_units = get_units(dp, UNIT_QUALITY_FIRST)
print(good_units)

#%% 1.3 - Load spike trains ----------------------------------------------------

# pick unit
u = good_units[UNIT_IDX_EXAMPLE]

# Load spikes
t = trn(dp, u)
print(f"Neuron {u} has {t.shape[0]} spikes.")
t = t / fs # convert from samples to seconds

# Plot
plt.scatter(t, t*0, marker="|")
fig = mplp(xlim = RASTER_XLIM, ylabel = f"Spikes of neuron {u}\n between 0 and 1s", xlabel = "Time (s)")
plt.show()

#%%----------------------------------------------------------------------------
all_units = get_units(dp)
good_units = get_units(dp, UNIT_QUALITY_FIRST)

ts_list = []  # List to store spike times for each unit

for i, u in enumerate(good_units):
    t = trn(dp, u) / fs  # Convert from samples to seconds
    ts_list.append((t, i * RASTER_Y_OFFSET))  # Store spike times with vertical offset
    print(f"Neuron {u} has {t.shape[0]} spikes.")

# Plot raster
plt.figure(figsize=(10, 6))
for t, y_offset in ts_list:
    plt.scatter(t, [y_offset] * len(t), marker="|", color="black")

plt.xlim(*RASTER_XLIM)
plt.xlabel("Time (s)")
plt.ylabel("Neuron index")
plt.title("Raster plot of noise neural units")
plt.show()


#%%############################################################################
# Tracking Analysis ###########################################################
###############################################################################

# load data ###################################################################

# video recording rate---------------------------------------------------------
fps = FPS

# excel file in behavioural tracking info
behaviour_data = pd.read_csv(BEHAVIOUR_CSV)

behaviour_timestamps = np.array(behaviour_data['time'])
behaviour_x = np.array(behaviour_data['x'])
behaviour_y = np.array(behaviour_data['y'])

#%%############################################################################
# remove zeros ################################################################

x2 = np.nonzero(behaviour_x)

x = behaviour_x[x2]
y = behaviour_y[x2]

behaviour_t = behaviour_timestamps[x2]/1000 #in seconds

#%%############################################################################
# moving average #############################################################

window_size = MOVING_AVG_WINDOW

x_av = np.convolve(x, np.ones(window_size), 'same') / window_size
y_av = np.convolve(y, np.ones(window_size), 'same') / window_size

#%%############################################################################
# Find centre point ###########################################################

max_x = max(x_av)
max_y = max(y_av)

min_x = min(x_av)
min_y = min(y_av)

cent_x = min_x + (max_x-min_x)/2
cent_y = min_y + (max_y-min_y)/2


# change x and y relative to centre coords ------------------------------------

x_fin = x_av - cent_x
y_fin = y_av - cent_y


#%%############################################################################
# Syncing spike data with tracking data #######################################
###############################################################################

#%%############################################################################
# plot tracked behaviour with corresponding spikes ############################
# Loop through all good units
#for u in good_units:
    # Load spikes
#    t = trn(dp, u)
#   print(f"Neuron {u} has {t.shape[0]} spikes.")

#    t = t / fs  # Convert from samples to seconds


# pick unit
u = good_units[UNIT_IDX_ANALYSIS]

# Load spikes
t = trn(dp, u)
print(f"Neuron {u} has {t.shape[0]} spikes.")
t = t / fs # convert from samples to seconds
# Plot
plt.scatter(t, t * 0, marker="|")
fig = mplp(xlim=RASTER_XLIM, ylabel=f"Spikes of neuron {u}\n between 0 and 1s", xlabel="Time (s)")

# Show each plot before moving to the next neuron
plt.show()

# Convert spike timestamps from samples to seconds
spike_timestamps = t  # Spike times (s)

# Behavior timestamps (assumed to be already in seconds)
# behaviour_t should be a sorted NumPy array
behaviour_t = np.array(behaviour_t)  # Ensure it's a NumPy array

# Initialize matched timestamps
sp_t = []

# Match each spike timestamp to the closest behavior timestamp
for spike in spike_timestamps:
    idx = np.searchsorted(behaviour_t, spike)  # Find closest index

    # Find the closest behavior timestamp
    if idx == 0:
        closest_t = behaviour_t[0]
    elif idx == len(behaviour_t):
        closest_t = behaviour_t[-1]
    else:
        before = behaviour_t[idx - 1]  # Previous behavior timestamp
        after = behaviour_t[idx]  # Next behavior timestamp

        # Pick the closer one
        closest_t = before if abs(spike - before) < abs(spike - after) else after

    # Check if the difference is within SPIKE_MATCH_TOLERANCE seconds
    if abs(spike - closest_t) <= SPIKE_MATCH_TOLERANCE:
        sp_t.append(closest_t)

# Convert to NumPy array
sp_t = np.array(sp_t)

# Plot results
plt.figure(figsize=(10, 4))
plt.scatter(spike_timestamps, np.zeros_like(spike_timestamps), marker="|", color="r", label="Original Spikes")
plt.scatter(sp_t, np.ones_like(sp_t), marker="|", color="b", label="Matched Behavior Timestamps")
plt.xlabel("Time (s)")
plt.ylabel("Spikes (Red) / Matched Behavior (Blue)")
plt.legend()
plt.title("Spike Timestamps Matched to Behavior Timestamps")
plt.show()


#%%############################################################################
# ONLY FOR BATCH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ##########################
###############################################################################

if RUN_BATCH_LOOP:
    # Loop through all good units
    for u in good_units:
        # Load spikes
        t = trn(dp, u)
        print(f"Neuron {u} has {t.shape[0]} spikes.")
        t = t / fs  # Convert from samples to seconds

        # Plot spike raster for the unit
        plt.scatter(t, t * 0, marker="|")
        fig = mplp(xlim=RASTER_XLIM, ylabel=f"Spikes of neuron {u}\n between 0 and 1s", xlabel="Time (s)")
        plt.show()

        # Behavior timestamps (assumed to be already in seconds)
        behaviour_t = np.array(behaviour_t)  # Ensure it's a NumPy array

        # Initialize matched timestamps
        sp_t = []

        # Match each spike timestamp to the closest behavior timestamp
        for spike in t:
            idx = np.searchsorted(behaviour_t, spike)  # Find closest index

            if idx == 0:
                closest_t = behaviour_t[0]
            elif idx == len(behaviour_t):
                closest_t = behaviour_t[-1]
            else:
                before = behaviour_t[idx - 1]
                after = behaviour_t[idx]
                closest_t = before if abs(spike - before) < abs(spike - after) else after

            # Keep if close enough
            if abs(spike - closest_t) <= SPIKE_MATCH_TOLERANCE:
                sp_t.append(closest_t)

        sp_t = np.array(sp_t)

        # Plot matched timestamps
        plt.figure(figsize=(10, 4))
        plt.scatter(t, np.zeros_like(t), marker="|", color="r", label="Original Spikes")
        plt.scatter(sp_t, np.ones_like(sp_t), marker="|", color="b", label="Matched Behavior Timestamps")
        plt.xlabel("Time (s)")
        plt.ylabel("Spikes (Red) / Matched Behavior (Blue)")
        plt.legend()
        plt.title(f"Neuron {u} - Spike Timestamps Matched to Behavior Timestamps")
        plt.show()

#%%----------------------------------------------------------------------------
all_units = get_units(dp)
good_units = get_units(dp, UNIT_QUALITY_SECOND)

ts_list = []  # List to store spike times for each unit

for i, u in enumerate(good_units):
    t = trn(dp, u) / fs  # Convert from samples to seconds
    ts_list.append((t, i * RASTER_Y_OFFSET))  # Store spike times with vertical offset
    print(f"Neuron {u} has {t.shape[0]} spikes.")

# Plot raster
plt.figure(figsize=(10, 6))
for t, y_offset in ts_list:
    plt.scatter(t, [y_offset] * len(t), marker="|", color="black")

plt.xlim(*RASTER_XLIM)
plt.xlabel("Time (s)")
plt.ylabel("Neuron index")
plt.title("Raster plot of noise neural units")
plt.show()

#%%----------------------------------------------------------------------------
# Convert behavior timestamps to NumPy array if not already
behaviour_t = np.array(behaviour_t)

# Convert (x, y) coordinates to NumPy arrays
x_fin = np.array(x_fin)
y_fin = np.array(y_fin)

# Find indices in behaviour_t that correspond to sp_t
spike_indices = np.searchsorted(behaviour_t, sp_t)

# Remove indices that are out of bounds
valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(x_fin))]

# Plot behavior tracking and spike locations
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_fin, y_fin, 'k', label='Behavioral tracking', alpha=0.5)  # Plot full path in black
ax.plot(x_fin[valid_indices], y_fin[valid_indices], 'r.', label='Neuronal spikes')  # Plot spikes in red

# Formatting
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
ax.set_title("Animal Trajectory with Spike Locations")

plt.show()


#%%############################################################################
# Filtering by speed ##########################################################

# Convert to NumPy arrays
behaviour_t = np.array(behaviour_t)
x_fin = np.array(x_fin)
y_fin = np.array(y_fin)

# Compute time differences
dt = np.diff(behaviour_t)  # Time intervals between frames

# Compute Euclidean distance between consecutive points
dx = np.diff(x_fin)
dy = np.diff(y_fin)
distances = np.sqrt(dx**2 + dy**2)

# Compute speed (cm/s)
speed = distances / dt

# Define speed threshold
speed_threshold = SPEED_THRESHOLD
valid_points = np.insert(speed < speed_threshold, 0, True)  # Keep first point

# Remove jumps by interpolation
x_filtered = np.interp(behaviour_t, behaviour_t[valid_points], x_fin[valid_points])
y_filtered = np.interp(behaviour_t, behaviour_t[valid_points], y_fin[valid_points])

# Find spike indices within valid range
spike_indices = np.searchsorted(behaviour_t, sp_t)
valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(x_filtered))]

# Plot the cleaned trajectory
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_filtered, y_filtered, 'k', alpha=0.5, label='Filtered Tracking')
ax.plot(x_filtered[valid_indices], y_filtered[valid_indices], 'r.', label='Spike Locations')

# Formatting
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
ax.set_title("Filtered Animal Trajectory with Spikes")

plt.show()


#%%############################################################################
# Filtering by centre proximity and jump distance #############################

# Convert to NumPy arrays
behaviour_t = np.array(behaviour_t)
x_fin = np.array(x_fin)
y_fin = np.array(y_fin)

# 1. Calculate distance from center to identify center jumps
center = np.array([np.mean(x_fin), np.mean(y_fin)])  # Estimate arena center
dist_from_center = np.sqrt((x_fin - center[0])**2 + (y_fin - center[1])**2)

# 2. Calculate median radius of trajectory (assuming mostly circular)
median_radius = np.median(dist_from_center)

# 3. Identify points that are too close to center (potential tracking errors)
center_threshold = median_radius * CENTER_THRESHOLD_FACTOR  # Adjust as needed
valid_points = dist_from_center > center_threshold

# 4. Filter based on both distance jumps AND center proximity
dx = np.diff(x_fin)
dy = np.diff(y_fin)
distances = np.sqrt(dx**2 + dy**2)
max_allowed_jump = median_radius * MAX_JUMP_FACTOR_CIRCULAR  # Adjust as needed

# Combine both criteria (first point is always valid)
distance_valid = np.insert(distances < max_allowed_jump, 0, True)
valid_points = valid_points & distance_valid

# 5. Interpolate missing points
x_fin = np.interp(behaviour_t, behaviour_t[valid_points], x_fin[valid_points])
y_fin = np.interp(behaviour_t, behaviour_t[valid_points], y_fin[valid_points])

# 6. Optionally: Project all points onto ideal circle if needed
# theta = np.arctan2(y_filtered - center[1], x_filtered - center[0])
# x_filtered = center[0] + median_radius * np.cos(theta)
# y_filtered = center[1] + median_radius * np.sin(theta)

# Plot results
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_fin, y_fin, 'k', alpha=0.5, label='Filtered Tracking')
ax.plot(x_fin[~valid_points], y_fin[~valid_points], 'rx', alpha=0.3, label='Removed Points')

# Add spike locations
spike_indices = np.searchsorted(behaviour_t, sp_t)
valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(x_filtered))]
ax.plot(x_fin[valid_indices], y_fin[valid_indices], 'r.', label='Spike Locations')

ax.set_aspect('equal')
ax.set_title("Filtered Animal Trajectory with Spikes")
ax.legend()
plt.show()


#%%############################################################################
# Filtering by pixel value ####################################################

# Convert to NumPy arrays
behaviour_t = np.array(behaviour_t)
x_fin = np.array(x_fin)
y_fin = np.array(y_fin)

# Compute the difference between consecutive x and y coordinates
dx = np.diff(x_fin)
dy = np.diff(y_fin)

# Compute Euclidean distance between consecutive points
distances = np.sqrt(dx**2 + dy**2)

# Calculate average and maximum distances
average_distance = np.mean(distances)
max_distance = np.max(distances)

print(f"Average distance between consecutive points: {average_distance} units")
print(f"Maximum distance between consecutive points: {max_distance} units")

# Plot the distances to visually inspect the jump sizes
plt.figure(figsize=(10, 5))
plt.plot(distances, label="Distance between consecutive points")
plt.axhline(y=average_distance, color='r', linestyle='--', label="Average Distance")
plt.axhline(y=max_distance, color='g', linestyle='--', label="Maximum Distance")
plt.xlabel("Index")
plt.ylabel("Distance (units)")
plt.title("Distances between Consecutive Points")
plt.legend()
plt.show()

# Define your threshold for maximum allowed jump
max_allowed_jump = MAX_ALLOWED_JUMP_PIXEL #max_distance * 0.25  # Set your own threshold here

# Filter out jumps that exceed the threshold
valid_points = np.insert(distances < max_allowed_jump, 0, True)  # Keep first point

# Apply the filter to x and y coordinates
x_filtered = np.interp(behaviour_t, behaviour_t[valid_points], x_fin[valid_points])
y_filtered = np.interp(behaviour_t, behaviour_t[valid_points], y_fin[valid_points])

# Find spike indices within valid range
spike_indices = np.searchsorted(behaviour_t, sp_t)
valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(x_filtered))]

# Plot the cleaned trajectory
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_filtered, y_filtered, 'k', alpha=0.5, label='Filtered Tracking')
ax.plot(x_filtered[valid_indices], y_filtered[valid_indices], 'r.', label='Spike Locations')

# Formatting
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
ax.set_title("Filtered Animal Trajectory with Spikes")

plt.show()


#%%############################################################################
# spatial matching of spikes ##################################################

# Define bin size in cm
bin_size = SPATIAL_BIN_SIZE

# Convert x and y coordinates to binned spatial indices
bin_beh_x = (x_fin / bin_size).astype(int)
bin_beh_y = (y_fin / bin_size).astype(int)

# Stack them into a single array
bin_beh = np.column_stack((bin_beh_x, bin_beh_y))

# Find indices in behaviour_t that correspond to sp_t
spike_indices = np.searchsorted(behaviour_t, sp_t)

# Filter out indices that are out of bounds
valid_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(bin_beh))]

# Extract binned spike positions
bin_sp = bin_beh[valid_indices]

# Convert to integer
bin_sp = bin_sp.astype(int)

# Output
print(f"Binned behavioral coordinates: {bin_beh.shape}")
print(f"Binned spike coordinates: {bin_sp.shape}")

#%%----------------------------------------------------------------------------
figure, ax = plt.subplots()
ax.plot(bin_beh.T[0], bin_beh.T[1],'k', label = 'Behavioural tracking binned')
ax.plot(bin_sp.T[0], bin_sp.T[1], 'r.', label = 'Neuronal spikes binned')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc = 'lower left', bbox_to_anchor = (0, 1), ncol = 2)
plt.show()


#%%############################################################################
# basic firing rate map #######################################################

# find visited bins -----------------------------------------------------------

visited_bins = []

for i in range(len(bin_beh)):

    if list(bin_beh[i]) not in visited_bins:
        visited_bins.append(list(bin_beh[i]))


# create basic rates -----------------------------------------------------------

bin_fr = np.zeros(len(visited_bins))

counter = 0
for i in visited_bins:

    spikes = np.where(bin_sp == i)[0]
    if len(spikes) == 0:
        bin_fr[counter] = 0.0
    else:
        beh_t = len(np.where(bin_beh == i)[0]) * (1/fps)

        bin_fr[counter] = len(spikes)/beh_t

    counter += 1


max_fr = max(bin_fr)
mean_fr = np.average(bin_fr)

#%%############################################################################
# plot basic firing rate map ##################################################

figure, ax = plt.subplots()
f = ax.scatter(np.transpose(visited_bins)[0], np.transpose(visited_bins)[1],
               s = 2,c=bin_fr, marker = 's', cmap='jet')

ax.set_xticks([])
ax.set_yticks([])
figure.colorbar(f, label = 'unsmoothed firing rate (Hz)')

ax.text(-20,-10,'max fr = ' + str(round(max_fr,2)))
ax.text(-20,10,'mean fr = ' + str(round(mean_fr,2)))
plt.show()


#%%############################################################################
# apply gaussian smoothing ####################################################

k = GAUSS_K # gaussian smoothing kernal
h = GAUSS_H # max distance to avvoid extrapolation error


# gaussian smooth behaviour ----------------------------------------------------

beh_g = np.zeros(len(visited_bins))

counter = 0
for i in visited_bins:
    print(counter/len(visited_bins))

    beh_gt = []
    for j in range(len(bin_beh)):

        xyt_dist = distance.euclidean(bin_beh[j], i)

        if xyt_dist < h:
            xt = bin_beh[j][0]- i[0]
            yt = bin_beh[j][1]- i[1]
            beh_gt.append(1/(2*m.pi*k**2) * m.exp(-0.5 * (xt**2 + yt**2) /(k**2))) #apply behavioural gaussian


    beh_g[counter] = np.trapz(beh_gt)

    counter +=1




# gaussian smooth spikes -----------------------------------------------------
fr_s = np.zeros(len(visited_bins))

counter = 0

for i in visited_bins:

    print(counter/len(visited_bins))

    sp_gi = [0] *len(bin_sp)

    for j in range(len(bin_sp)):

        xt = bin_sp[j][0]- i[0]
        yt = bin_sp[j][1]- i[1]
        sp_gi[j] = (1/(2*m.pi*k**2) * m.exp(-0.5 * (xt**2 + yt**2) /(k**2))) #apply behavioural gaussian

    sp_g = sum(sp_gi)

    if any(a ==0 for a in [beh_g[counter],sp_g]):
        fr_s[counter] = 0.0
    else:
        fr_s[counter] = sp_g/beh_g[counter]


    counter +=1



max_smooth_fr = max(fr_s)
mean_smooth_fr = np.average(fr_s)

#%%############################################################################
# calculate spatial information rate ------------------------------------------

recording_length = (behaviour_t[-1] - behaviour_t[0])

sir_i  = [0.0]*len(visited_bins)
for i in range(len(visited_bins)):

    pi = (len(np.where(bin_beh == visited_bins[i])[0]) * 1/fps) / recording_length

    if fr_s[i]==0:
        sir_i[i] = 0.0
    else:
        sir_i[i] = pi * fr_s[i] * np.log2(fr_s[i]/mean_smooth_fr)

sir = sum(sir_i)

#%%############################################################################
# plot smoothed firing rate map ###############################################

figure, ax = plt.subplots()
f = ax.scatter(np.transpose(visited_bins)[0], np.transpose(visited_bins)[1],
               s = 2,c=fr_s, marker = 's', cmap='jet')

ax.set_xticks([])
ax.set_yticks([])
figure.colorbar(f, label = 'smoothed firing rate (Hz)')

ax.text(-20,-15,'max fr = ' + str(round(max_smooth_fr,2)))
ax.text(-20,0,'mean fr = ' + str(round(mean_smooth_fr,2)))
ax.text(-20,15,'sir = ' + str(round(sir,2)))
plt.show()


#%%############################################################################
# Heading direction ###########################################################

# Compute movement vectors
dx = np.diff(x_fin)  # Change in x
dy = np.diff(y_fin)  # Change in y

# Compute heading direction (angle in radians)
heading_rad = np.arctan2(dy, dx)

# Convert to degrees
heading_deg = np.degrees(heading_rad)

# Ensure heading is in the range [0, 360] degrees
heading_deg = (heading_deg + 360) % 360

# Append NaN to keep same length as original x, y
#heading_deg = np.insert(heading_deg, 0, np.nan)

# Output example
print(f"Heading direction (first 10 values): {heading_deg[:10]}")

#%%----------------------------------------------------------------------------
# Skip the first value in timestamps and coordinates
behaviour_t = behaviour_t[1:]
x_fin = x_fin[1:]
y_fin = y_fin[1:]

# Compute movement vectors
dx = np.diff(x_fin)  # Change in x
dy = np.diff(y_fin)  # Change in y

# Compute heading direction (angle in radians)
heading_rad = np.arctan2(dy, dx)

# Convert to degrees
heading_deg = np.degrees(heading_rad)

# Ensure heading is in the range [0, 360] degrees
heading_deg = (heading_deg + 360) % 360

# Ensure timestamps match the heading direction length
behaviour_t = behaviour_t[:-1]  # Remove last timestamp to align

# Output example
print(f"Aligned Heading Direction (first 10 values): {heading_deg[:10]}")
print(f"Aligned Timestamps (first 10 values): {behaviour_t[:10]}")

#%%----------------------------------------------------------------------------
# Ensure spike times and behavior times are NumPy arrays
spike_timestamps = np.array(spike_timestamps)
behaviour_t = np.array(behaviour_t)
heading_deg = np.array(heading_deg)  # Heading directions in degrees

# Match each spike timestamp to the closest behavior timestamp
sp_t = []
sp_heading = []  # Store heading direction at spike times

for spike in spike_timestamps:
    idx = np.searchsorted(behaviour_t, spike)  # Find closest index

    # Find the closest behavior timestamp
    if idx == 0:
        closest_t = behaviour_t[0]
    elif idx == len(behaviour_t):
        closest_t = behaviour_t[-1]
    else:
        before = behaviour_t[idx - 1]  # Previous behavior timestamp
        after = behaviour_t[idx]  # Next behavior timestamp

        # Pick the closer one
        closest_t = before if abs(spike - before) < abs(spike - after) else after

    # Check if the difference is within SPIKE_MATCH_TOLERANCE seconds
    if abs(spike - closest_t) <= SPIKE_MATCH_TOLERANCE:
        sp_t.append(closest_t)
        sp_heading.append(heading_deg[idx])  # Get heading direction at spike time

# Convert to NumPy array
sp_heading = np.array(sp_heading)

# Convert degrees to radians for polar plot
sp_heading_rad = np.radians(sp_heading)

# Plot polar histogram
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(sp_heading_rad, bins=HEADING_HIST_BINS, color='r', alpha=0.75)  # Histogram of spike angles

ax.set_theta_zero_location('N')  # 0° at the top
ax.set_theta_direction(-1)  # Clockwise rotation

ax.set_title("Polar Histogram of Neuronal Spikes at Heading Directions")
plt.show()

#%%----------------------------------------------------------------------------
# Sampling rate
sampling_rate = HEADING_SAMPLING_RATE  # Hz

# Compute heading direction
posgx, posgy = x_fin[:-1], y_fin[:-1]  # Remove last point to align shift
posrx, posry = np.roll(posgx, -1), np.roll(posgy, -1)  # Next position

# Compute heading in degrees
heading_deg = np.mod(np.degrees(np.arctan2(-(posry - posgy), posrx - posgx)), 360)

# Ensure sp_t contains valid integer indices
sp_t = np.array(sp_t, dtype=int)

# Match spikes to heading direction
spike_headings = heading_deg[sp_t]

# Define bins for heading directions
bin_size = HEADING_BIN_SIZE  # Degrees
bins = np.arange(0, 370, bin_size)  # 0 to 360 degrees
bin_centers = bins[:-1] + bin_size / 2  # Center of bins

# Compute histogram for spike counts and occupancy
spike_counts, _ = np.histogram(spike_headings, bins=bins)
occupancy, _ = np.histogram(heading_deg, bins=bins)

# Convert to spike rate (spikes per second)
spike_rate = spike_counts / (occupancy / sampling_rate)
spike_rate = np.nan_to_num(spike_rate)  # Replace NaNs with zero
smoothed_rate = gaussian_filter1d(spike_rate, sigma=HEADING_SMOOTH_SIGMA)  # Smooth data

# Convert bin centers to radians
angles_rad = np.radians(bin_centers)

# Polar plot of spike rate vs. heading direction
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(angles_rad, smoothed_rate, color='r', linewidth=2, label="Spike Rate")
ax.fill_between(angles_rad, 0, smoothed_rate, color='r', alpha=0.3)

ax.set_theta_zero_location('N')  # North = 0 degrees
ax.set_theta_direction(-1)  # Clockwise rotation
ax.set_xticks(np.radians([0, 90, 180, 270]))  # Label directions
ax.set_xticklabels(['N', 'E', 'S', 'W'])
ax.set_title("Polar Plot of Spiking Activity vs. Heading Direction")
plt.legend()
plt.show()

# Plot behavioral tracking with spikes
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x_fin, y_fin, color='gray', alpha=0.5, linewidth=3, label='Behavior Tracking')
ax.scatter(x_fin[sp_t], y_fin[sp_t], color='red', s=20, label='Spike Locations')

ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left')
ax.set_title("Behavioral Path with Spike Locations")
plt.show()
