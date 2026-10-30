# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:57:13 2022

Loading ncs files and performing fft

@author: Georgina fenton
"""

#%% import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

from scipy.fft import fft, fftfreq, fftshift


#%% load ncs files


# Enter folder path
#lfp_folder = 'path/to/folder'
lfp_folder = r'C:\Runita\NMR\Pierre\Analysis_Codes\ThetaAnalysis\LFP theta speed relationship\theta_plot'

dir_list = os.listdir(lfp_folder)

lfp_files = []
for i in range(len(dir_list)):
    if dir_list[i].endswith(".ncs"):
        if '0001' not in dir_list[i]:
            lfp_files.append(dir_list[i])
        
        

# neuralynx .ncs file format
ncs_dtype = np.dtype([
     ('timestamp'  , '<u8'),
     ('sc_number'  , '<u4'),
     ('cell_number', '<u4'),
     ('params'     , '<u4'),
     ('samples'  , '<i2', (512,)),
 ])


# load numpy memory map and import lfp data for each tetrode
lfp_data = []
for i in range(len(lfp_files)):
    
    lfp_datai = np.memmap(lfp_folder + '\\' + lfp_files[i], dtype=ncs_dtype, mode='readwrite', offset =16*1024 ) 
    
    lfp_data.append(np.concatenate(lfp_datai['samples']))
    
    
#%%
    
# Assuming a sampling rate of 32,000 Hz (standard for Neuralynx systems)
sampling_rate = 32000  # Hz
downsample_factor = 1000

# Downsample the LFP data from the first tetrode
lfp_downsampled = lfp_data[0][::downsample_factor]

# Compute the actual sampling rate after downsampling
downsampled_sampling_rate = sampling_rate // downsample_factor

# Create time vector for the downsampled signal
total_duration = len(lfp_downsampled) / downsampled_sampling_rate  # in seconds
time_vector = np.linspace(0, total_duration, len(lfp_downsampled))

# Plot parameters
window_duration = 1  # seconds
samples_per_window = int(window_duration * downsampled_sampling_rate)

# Initial view window (first second)
start_idx = 0
end_idx = samples_per_window

# Initialize the figure and plot
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)  # Leave space for widgets

line, = ax.plot(time_vector[start_idx:end_idx], lfp_downsampled[start_idx:end_idx], lw=1)
ax.set_xlim(0, window_duration)
ax.set_ylim(np.min(lfp_downsampled), np.max(lfp_downsampled))
ax.set_title('LFP Data')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')

# # Slider for moving through the time series
# ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position: [left, bottom, width, height]
# slider = Slider(ax_slider, 'Time', 0, total_duration - window_duration, valinit=0, valstep=window_duration)

# # Update function for the slider
# def update(val):
#     global start_idx, end_idx
#     start_time = slider.val
#     start_idx = int(start_time * downsampled_sampling_rate)
#     end_idx = start_idx + samples_per_window
#     line.set_xdata(time_vector[start_idx:end_idx])
#     line.set_ydata(lfp_downsampled[start_idx:end_idx])
#     ax.set_xlim(start_time, start_time + window_duration)
#     fig.canvas.draw_idle()

# slider.on_changed(update)

# # Buttons for navigation
# ax_button_prev = plt.axes([0.05, 0.05, 0.1, 0.04])  # Position: [left, bottom, width, height]
# button_prev = Button(ax_button_prev, 'Previous')

# ax_button_next = plt.axes([0.85, 0.05, 0.1, 0.04])  # Position: [left, bottom, width, height]
# button_next = Button(ax_button_next, 'Next')

# # Functions for button clicks
# def go_previous(event):
#     current_time = slider.val
#     if current_time - window_duration >= 0:
#         slider.set_val(current_time - window_duration)

# def go_next(event):
#     current_time = slider.val
#     if current_time + window_duration <= total_duration - window_duration:
#         slider.set_val(current_time + window_duration)

# button_prev.on_clicked(go_previous)
# button_next.on_clicked(go_next)

plt.show()    



#%% Plot the first 1-minute of LFP data (downsampled)

# # Assuming a sampling rate of 32,000 Hz (standard for Neuralynx systems)
# sampling_rate = 32000  # Hz
# downsample_factor = 512

# # Compute number of samples for 1 minute
# one_minute_samples = sampling_rate * 1

# # Extract first minute of data from the first tetrode
# lfp_first_minute = lfp_data[0][:one_minute_samples]

# # Downsample the data
# lfp_downsampled = lfp_first_minute[::downsample_factor]

# # Create time vector for the downsampled signal
# time_vector = np.linspace(0, 1, len(lfp_downsampled))

# # Plot the downsampled LFP
# plt.figure(figsize=(12, 6))
# plt.plot(time_vector, lfp_downsampled, label='Downsampled LFP')
# plt.title('First Minute of Downsampled LFP Data')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid()
# plt.show()

#%% run fourier transform
    
# # number of signal points

# N = len(lfp_data[0])

# # sample spacing
# T = 0.000032

# x = np.linspace(0.0, N*T, N, endpoint=False)


# yf = fft(lfp_data[0])

# xf = fftfreq(N, T)

# xf = fftshift(xf)

# yplot = fftshift(yf)

# #%% plot fft

# fig, ax = plt.subplots()
# ax.plot(xf, 1.0/N * np.abs(yplot))
# ax.grid()





















