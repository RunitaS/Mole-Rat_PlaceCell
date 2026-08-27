# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:39:30 2025

@author: shirdhankar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.signal import butter, filtfilt, hilbert, sawtooth, detrend
import os

# Butterworth bandpass filter for theta range
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Phase estimation using Hilbert transform
def estimate_phase(signal):
    analytic_signal = hilbert(signal)
    return np.angle(analytic_signal)

# Apply bandpass filter (1-100 Hz) and detrend raw LFP data
def preprocess_raw_lfp(raw_data, fs):
    filtered_data = bandpass_filter(raw_data, 1, 100, fs)
    return detrend(filtered_data)

# Load and process LFP data
def load_lfp_files(folder_path, sampling_rate=32000, downsample_factor=128):
    dir_list = os.listdir(folder_path)
    lfp_files = [f for f in dir_list if f.endswith(".ncs") and '0001' not in f]
    
    ncs_dtype = np.dtype([
        ('timestamp', '<u8'),
        ('sc_number', '<u4'),
        ('cell_number', '<u4'),
        ('params', '<u4'),
        ('samples', '<i2', (512,))
    ])
    
    raw_lfp_data, lfp_data = {}, {}
    
    for file in lfp_files:
        lfp_raw = np.memmap(os.path.join(folder_path, file), dtype=ncs_dtype, mode='r', offset=16*1024)
        data = np.concatenate(lfp_raw['samples'])
        downsampled_data = data[::downsample_factor]
        filtered_data = bandpass_filter(downsampled_data, 3, 7, sampling_rate // downsample_factor)
        
        raw_lfp_data[file] = downsampled_data
        lfp_data[file] = filtered_data
    
    return raw_lfp_data, lfp_data, sampling_rate // downsample_factor

# Plot LFP traces with interactive sliders
def plot_lfp_traces_with_sliders(raw_lfp_data, lfp_data, sampling_rate, initial_window_duration=1):
    files = list(lfp_data.keys())
    n_files = len(files)
    fig, axes = plt.subplots(n_files * 2, 1, figsize=(12, 10 + n_files * 6), sharex=True)

    if n_files * 2 == 1:
        axes = [axes]

    sliders = []
    scatters = []  # Store scatter plot handles
    sawtooth_lines = []  # Store sawtooth line handles
    lfp_lines = []  # Store LFP line handles
    raw_lines = []  # Store raw LFP line handles

    # Add a text box for time window input
    ax_time_window = plt.axes([0.2, 0.95, 0.1, 0.03], facecolor='lightgoldenrodyellow')
    time_window_textbox = TextBox(ax_time_window, 'Time Window (s): ', initial=str(initial_window_duration))

    for i, file in enumerate(files):
        raw_data = preprocess_raw_lfp(raw_lfp_data[file], sampling_rate)
        data = lfp_data[file]
        phase = estimate_phase(data)

        total_duration = len(data) / sampling_rate
        time_vector = np.linspace(0, total_duration, len(data))
        samples_per_window = int(initial_window_duration * sampling_rate)
        start_idx, end_idx = 0, samples_per_window

        # --- Raw LFP with Theta Overlay ---
        raw_line, = axes[2 * i].plot(time_vector[start_idx:end_idx], raw_data[start_idx:end_idx], lw=0.8, color='gray', alpha=0.5)
        sc = axes[2 * i].scatter(time_vector[start_idx:end_idx], data[start_idx:end_idx], c=phase[start_idx:end_idx], cmap='hsv', s=10)
        scatters.append(sc)  # Store scatter handle
        raw_lines.append(raw_line)  # Store raw LFP line handle
        
        axes[2 * i].set_ylabel('Amplitude (µV)')
        axes[2 * i].set_xlim(0, initial_window_duration)
        axes[2 * i].set_ylim(-7000, 7000)
        axes[2 * i].grid(True)

        # --- Sawtooth Phase vs LFP ---
        ax1, ax2 = axes[2 * i + 1], axes[2 * i + 1].twinx()
        sawtooth_wave = sawtooth(phase) * np.pi
        line1, = ax1.plot(time_vector[start_idx:end_idx], sawtooth_wave[start_idx:end_idx], lw=2, color='black', label='Theta Phase')
        line2, = ax2.plot(time_vector[start_idx:end_idx], data[start_idx:end_idx], lw=1, color='red', label='Theta Wave')

        sawtooth_lines.append(line1)  # Store sawtooth handle
        lfp_lines.append(line2)  # Store LFP handle

        ax1.set_ylabel('Phase (rad)', color='black')
        ax2.set_ylabel('LFP Amplitude (µV)', color='red')
        ax1.set_xlim(0, initial_window_duration)
        ax1.set_ylim(-np.pi, np.pi)
        ax2.set_ylim(-4000, 4000)
        ax1.grid(True)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_xlabel('Time (s)')

        # --- Slider for Time Window ---
        ax_slider = plt.axes([0.2, 0.03 - i * 0.03, 0.65, 0.02], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, f'Time {i + 1}', 0, total_duration - initial_window_duration, valinit=0, valstep=0.01)
        sliders.append(slider)

    plt.subplots_adjust(hspace=0.6, bottom=0.15)

    # --- Slider Update Function ---
    def update(val):
        window_duration = float(time_window_textbox.text)  # Get the current time window value
        for i, slider in enumerate(sliders):
            start_time = slider.val
            start_idx = int(start_time * sampling_rate)
            end_idx = start_idx + int(window_duration * sampling_rate)

            # Ensure indices are within bounds
            if end_idx > len(lfp_data[files[i]]):
                end_idx = len(lfp_data[files[i]])
                start_idx = end_idx - int(window_duration * sampling_rate)
                if start_idx < 0:
                    start_idx = 0

            time_vector = np.linspace(start_time, start_time + window_duration, end_idx - start_idx)
            phase = estimate_phase(lfp_data[files[i]])
            sawtooth_wave = sawtooth(phase) * np.pi

            # Update raw LFP plot
            raw_lines[i].set_xdata(time_vector)
            raw_lines[i].set_ydata(raw_lfp_data[files[i]][start_idx:end_idx])

            # Update scatter plot
            scatters[i].set_offsets(np.column_stack((time_vector, lfp_data[files[i]][start_idx:end_idx])))
            scatters[i].set_array(phase[start_idx:end_idx])  # Update colors

            # Update sawtooth & LFP plots
            sawtooth_lines[i].set_xdata(time_vector)
            sawtooth_lines[i].set_ydata(sawtooth_wave[start_idx:end_idx])
            lfp_lines[i].set_xdata(time_vector)
            lfp_lines[i].set_ydata(lfp_data[files[i]][start_idx:end_idx])

            # Update x-axis limits
            axes[2 * i].set_xlim(start_time, start_time + window_duration)
            axes[2 * i + 1].set_xlim(start_time, start_time + window_duration)

        fig.canvas.draw_idle()

    # Attach update function to sliders and text box
    for slider in sliders:
        slider.on_changed(update)
    time_window_textbox.on_submit(update)

    plt.show()

# Main execution
lfp_folder = r'C:\Runita\NMR\Pierre\Analysis_Codes\ThetaAnalysis\LFP theta speed relationship\theta_plot'
raw_lfp_data, lfp_data, sampling_rate = load_lfp_files(lfp_folder)
plot_lfp_traces_with_sliders(raw_lfp_data, lfp_data, sampling_rate)