# %%
import numpy as np
from fBOSCpy.fBOSCpy_wrapper import fBOSCpy_wrapper
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns

font_path = r"../misc/Inter-4.1/extras/ttf/Inter-Light.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()
plt.rcParams["font.size"] = 11

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    style="ticks",
    palette="colorblind",
    font_scale=1.2,
    rc=custom_params,
    font=prop.get_name(),
)

# %%
dt = 1 / 1000  # seconds, 1KHz sampling freq
T = 8 * 60  # seconds

timestamps = np.arange(0, T, dt)

# create a sine wavelet with frequency of 4Hz, only present for 10 seconds duration every 60 seconds
freq = 4  # Hz
wavelet_duration = 10  # seconds
wavelet_interval = 60  # seconds
wavelet = np.sin(2 * np.pi * freq * np.arange(0, wavelet_duration, dt))
signal = np.zeros_like(timestamps)

for start in np.arange(0, T, wavelet_interval):
    end = start + wavelet_duration
    signal[(timestamps >= start) & (timestamps < end)] += wavelet[
        : len(signal[(timestamps >= start) & (timestamps < end)])
    ]

noise = np.random.normal(0, 1, len(timestamps))

signal += noise


# %%
# add pink background noise in the entire signal
def pink_noise(N):
    """Generate pink noise with N samples."""
    # Voss-McCartney algorithm
    n = int(np.ceil(np.log2(N)))
    array = np.random.randn(n, N)
    array = np.cumsum(array, axis=0)
    weights = 1 / (2 ** np.arange(n))
    return np.dot(weights, array)[:N]


noise = pink_noise(len(timestamps))
signal += 0.3 * noise
signal -= np.mean(signal)
signal /= np.std(signal)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(timestamps, signal)
plt.xlim(68, 72)
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.title("Signal with 4Hz Wavelet and Random Noise")
plt.show()

# %%

from scipy.signal import welch
from scipy.stats import gaussian_kde

freqs_welch, psd_welch = welch(signal, fs=1000, window="hamming")

plt.plot(freqs_welch, psd_welch)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (normalized)")
# plt.xlim(0.1, 10)  # Adjust x-axis limits to avoid log scale issues at 0
plt.xscale("log")  # Set x-axis to log scale
plt.grid(
    True, which="both", linestyle="--"
)  # Enable grid for both major and minor ticks
plt.show()

# %%

episodesTable, episodesTablePostProc = fBOSCpy_wrapper(
    signal,
    timestamps,
    F_array=np.arange(1, 40.5, 0.5),
    Fs=1000,
    postproc=True,
    plot_histogram=False,
)

# %%
episodesTable_df = pd.DataFrame.from_dict(episodesTable)
episodesTablePostProc_df = pd.DataFrame.from_dict(episodesTablePostProc)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1: Histogram of Frequency
axs[0].hist(
    episodesTable_df["FrequencyMean"], bins=50, label="All detected episodes", alpha=0.5
)
axs[0].hist(
    episodesTablePostProc_df["FrequencyMean"],
    bins=50,
    label="Post-processed episodes",
    alpha=0.5,
)
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Count")
axs[0].legend()
axs[0].set_title("Histogram of Frequency of Detected Oscillatory Episodes")

# Subplot 2: Histogram of Duration
axs[1].hist(
    episodesTable_df["DurationS"], bins=50, label="All detected episodes", alpha=0.5
)
axs[1].hist(
    episodesTablePostProc_df["DurationS"],
    bins=50,
    label="Post-processed episodes",
    alpha=0.5,
)
axs[1].set_xlabel("Duration (s)")
axs[1].set_ylabel("Count")
axs[1].legend()
axs[1].set_title("Histogram of Duration of Detected Oscillatory Episodes")

plt.tight_layout()
plt.show()

# %%
# plot the detected episodes on top of the signal
plt.figure(figsize=(20, 3))
plt.plot(timestamps, signal, label="Signal")
for _, row in episodesTablePostProc_df.iterrows():
    plt.axvspan(row["Onset"], row["Offset"], color="red", alpha=0.3)
plt.xlim(55, 75)
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.title("Detected oscillatory Episodes (post-processed) overlaid on signal")
plt.legend()
plt.show()

# %% ############### TS using actual LFP ###############
fileNameNCS = r"../data/Fa8477/17Aug22_1D/17Aug_Session2/CSC1ch1_0001.ncs"

import LoadProcessCodes.DataUtils as DataUtils
from LoadProcessCodes.FileManager import FileManager

file_manager = FileManager(base_data_dir="../data")

file_manager.index_database()

lfp_samples, lfp_timestamps, lfp_header, lfp_tsd = DataUtils.load_lfp(
    fileNameNCS,
    file_manager,
    overwrite=False,
    denoise_smoothing=False,
    downsample=True,
    downsampling_freq=1000,
)

lfp_timestamps_zeroed = (lfp_timestamps - lfp_timestamps[0]) / 1e6
# %% plot the lfp signal

plt.figure(figsize=(12, 4))
plt.plot(lfp_timestamps_zeroed, lfp_samples)
plt.xlim(83, 92.5)
plt.xlabel("Time (s)")
plt.ylabel("LFP Amplitude")
plt.title("LFP Signal from CSC1ch1")
plt.show()

# %% run fBOSCpy on the lfp signal
from fBOSCpy.fBOSCpy_wrapper_v2 import fBOSCpy_wrapper_v2

# episodesTableLFP, episodesTablePostProcLFP = fBOSCpy_wrapper_v2(lfp_samples, lfp_timestamps_zeroed, F_array=np.arange(0.1, 40.5, 1.0), Fs=1000, postproc=True, plot_histogram=False)

# animal = 'Fa8477'
# day = '17Aug22_1D'
# session = 'Session1'
# tetrode_num = 1
# channel_num = 1
# results_root = r'../results/fBOSCpy_validation'

# episodesTable, episodesTablePostProc = fBOSCpy_wrapper_v2(animal, day, session, tetrode_num, channel_num,lfp_samples, lfp_timestamps_zeroed, F_array=np.arange(0.1, 40.5, 1), Fs=1000, postproc=True,plot_histogram=True, results_root=results_root)

# episodesTableLFP_df = pd.DataFrame.from_dict(episodesTableLFP)
# episodesTablePostProcLFP_df = pd.DataFrame.from_dict(episodesTablePostProcLFP)

# %% reading in the saved pkl
episodesTableLFP_df = pd.read_pickle(
    "../results/Fa8477/17Aug22_1D/17Aug_Session2/1_1_episodesTable.pkl"
)
# convert episodeTableLFP to a dataframe
episodesTableLFP_df = pd.DataFrame.from_dict(episodesTableLFP_df)

# substract lfp_timestamps[0] from both Onset and Offset and then divide by 1e6 to get zeroed times in seconds
episodesTableLFP_df["Onset"] = (episodesTableLFP_df["Onset"] - lfp_timestamps[0]) / 1e6
episodesTableLFP_df["Offset"] = (
    episodesTableLFP_df["Offset"] - lfp_timestamps[0]
) / 1e6

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1: Histogram of Frequency
axs[0].hist(
    episodesTableLFP_df["FrequencyMean"],
    bins=50,
    label="Raw Detected Episodes",
    alpha=0.5,
)
# axs[0].hist(episodesTablePostProcLFP_df['FrequencyMean'], bins=50, label='Post-Processed Episodes', alpha=0.5)
axs[0].axvspan(3, 6, color="red", alpha=0.2, label="Theta Band (3-6 Hz)")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Count")
axs[0].legend()
axs[0].set_title("Histogram of Frequency of Detected Oscillatory Episodes")

# Subplot 2: Histogram of Duration
axs[1].hist(
    episodesTableLFP_df["DurationS"], bins=50, label="Raw Detected Episodes", alpha=0.5
)
# axs[1].hist(episodesTablePostProcLFP_df['DurationS'], bins=50, label='Post-Processed Episodes', alpha=0.5)
axs[1].set_xlabel("Duration (s)")
axs[1].set_ylabel("Count")
axs[1].legend()
axs[1].set_title("Histogram of Duration of Detected Oscillatory Episodes")

plt.tight_layout()
plt.show()

# %% plot the detected episodes on top of the lfp signal
plt.figure(figsize=(12, 4))
plt.plot(lfp_timestamps_zeroed, lfp_samples, label="LFP Signal")
for _, row in episodesTableLFP_df.iterrows():
    if (row["FrequencyMean"] < 7) & (row["FrequencyMean"] > 2):
        plt.axvspan(row["Onset"], row["Offset"], color="red", alpha=0.3)
plt.xlim(80, 95)
plt.xlabel("Time (s)")
plt.ylabel("LFP Amplitude")
plt.title(
    "Detected osc episodes (Post-Processed, in 2-7 Hz Range) overlaid on LFP Signal"
)
plt.legend()
plt.show()

# %%
plt.figure(figsize=(16, 4))

# Plot the base signal in black
plt.plot(lfp_timestamps_zeroed, lfp_samples, color="black", linewidth=0.8)

# Overlay the detected episodes in red directly on the line
for _, row in episodesTableLFP_df.iterrows():
    if (row["FrequencyMean"] < 7) & (row["FrequencyMean"] > 2):
        mask = (lfp_timestamps_zeroed >= row["Onset"]) & (
            lfp_timestamps_zeroed <= row["Offset"]
        )
        plt.plot(
            lfp_timestamps_zeroed[mask], lfp_samples[mask], color="red", linewidth=0.8
        )

plt.xlim(162, 173)
plt.ylim(-700, 450)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
# plt.savefig("../figures/2-LFP/LFP_with_Detected_Episodes.svg",bbox_inches='tight')
plt.show()


# %% percentage of time spent in theta

theta_episodes = episodesTablePostProcLFP_df[
    (episodesTablePostProcLFP_df["FrequencyMean"] < 7)
    & (episodesTablePostProcLFP_df["FrequencyMean"] > 2)
]
total_theta_duration = theta_episodes["DurationS"].sum()
total_signal_duration = lfp_timestamps_zeroed[-1] - lfp_timestamps_zeroed[0]
percentage_theta = (total_theta_duration / total_signal_duration) * 100

# histogram of durations of theta episodes
plt.figure(figsize=(6, 4))
plt.hist(theta_episodes["DurationS"], bins=50, color="blue", alpha=0.7)
plt.xlabel("Duration (s)")
plt.ylabel("Count")
plt.title(f"Histogram of Duration of Detected Theta Episodes")
plt.show()

# %% bandpass lfp for theta range 3-6 hz and plot it with original signal at 70 to 80 seconds

from scipy.signal import butter, filtfilt


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


lfp_theta = bandpass_filter(lfp_samples, 3, 6, 1000, order=4)
plt.figure(figsize=(16, 4))
plt.plot(lfp_timestamps_zeroed, lfp_samples, label="LFP Signal", alpha=0.5)
plt.plot(lfp_timestamps_zeroed, lfp_theta, label="Theta Band (3-6 Hz)", color="orange")
plt.xlim(70, 85)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("LFP Signal and Theta Band (3-6 Hz)")
plt.legend()

# overlay the detected theta episodes
for _, row in theta_episodes.iterrows():
    plt.axvspan(row["Onset"], row["Offset"], color="red", alpha=0.3)
plt.show()

# %%
