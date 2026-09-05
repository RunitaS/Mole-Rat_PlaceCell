#%% load dependencies
import LoadProcessCodes.DataUtils as DataUtils
import MetaCodes.MetaUtils as MetaUtils
import fBOSCpy.fBOSCUtils as fBOSCUtils

import pynapple as nap
import numpy as np
import pandas as pd
import os
import pickle
#from scipy import signal
#import math 

from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns

font_path = r'../misc/Inter-4.1/extras/ttf/Inter-Light.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = 11

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.2, rc=custom_params, font=prop.get_name())

from tqdm import tqdm

file_manager = DataUtils.FileManager(base_data_dir='../data')

# Check if the database index exists; if not, index the database
index_file = 'database.pkl'
try:
    file_manager.load_index(index_file)
    print("Loaded indexed database.")
except FileNotFoundError:
    print("Index file not found. Indexing database...")
    file_manager.index_database()
    file_manager.save_index(index_file)
    print("Database indexed and saved.")

# load lfp file(s)

file_manager.get_unique_session_ids(animal='Fa8477OF')
lfp_files = file_manager.get_files(
    category='ncs',
    animal='Fa8477OF',
    date=["21Aug22"]
)
lfp_files

# filter lfp files for those that contain 'CSC*ch1.ncs'
lfp_files = [f for f in lfp_files if 'CSC' in f and 'ch1.ncs' in f]
lfp_files

lfp_file = lfp_files[1]

#%%
lfp_samples, lfp_timestamps, lfp_header, lfp_tsd = DataUtils.load_lfp(lfp_file, file_manager, overwrite=True, denoise_smoothing=False, downsample=True, downsampling_freq=1000)

lfp_timestamps_zeroed = (lfp_timestamps - lfp_timestamps[0])/1e6

# get metadata for the lfp file
metadata = file_manager.get_metadata_for_file(lfp_file, fields=['animal', 'date', 'session'])
if not metadata:
    raise ValueError(f"Metadata not found for LFP file: {lfp_file}")
animal = metadata['animal']
date = metadata['date']
session = metadata['session']
basename = file_manager.get_basename(lfp_file)
print(f"Processing LFP file: {basename}")

#print length of the LFP signal in seconds
lfp_length_s = len(lfp_samples) / 1000  # since we downsampled to 1000 Hz
print(f"LFP signal length: {lfp_length_s:.2f} seconds, {lfp_length_s/60:.2f} minutes")

#%% ##### Step 0 #####: Set up fBOSC configs
cfg_fBOSC = {}
cfg_fBOSC['F'] = np.arange(1, 40.5, 1)  # frequency sampling
cfg_fBOSC['wavenumber'] = 6     # wavelet parameter (time-frequency tradeoff)
cfg_fBOSC['fsample'] = 1000  # current sampling frequency of EEG data

cfg_fBOSC['pad.tfr_s'] = 1 # padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
cfg_fBOSC['pad.detection_s'] = .5 # padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
cfg_fBOSC['pad.background_s'] = 1 # padding of segments for BG (only avoiding edge artifacts)

cfg_fBOSC['pad.tfr_sample'] = int(cfg_fBOSC['pad.tfr_s'] * cfg_fBOSC['fsample'])
cfg_fBOSC['pad.detection_sample'] = int(cfg_fBOSC['pad.detection_s'] * cfg_fBOSC['fsample'])
cfg_fBOSC['pad.total_s'] = cfg_fBOSC['pad.tfr_s'] + cfg_fBOSC['pad.detection_s']
cfg_fBOSC['pad.total_sample'] = int(cfg_fBOSC['pad.tfr_sample'] + cfg_fBOSC['pad.detection_sample'])
cfg_fBOSC['pad.background_sample'] = int(cfg_fBOSC['pad.tfr_sample'])

n_freq = len(cfg_fBOSC['F'])  # number of frequencies
n_time = len(lfp_samples)
n_time_total = len(lfp_samples)
n_trial = 1

cfg_fBOSC['time.time_total'] = lfp_timestamps_zeroed # total time vector

# get timing and info for post-TFR padding removal
tfr_time2extract = np.arange(cfg_fBOSC['pad.tfr_sample']+1, n_time_total-cfg_fBOSC['pad.tfr_sample']+1,1)
cfg_fBOSC['time.time_tfr'] = cfg_fBOSC['time.time_total'][tfr_time2extract]
n_time_tfr = len(cfg_fBOSC['time.time_tfr'])
# get timing and info for post-detected padding removal
det_time2extract = np.arange(cfg_fBOSC['pad.detection_sample']+1, n_time_tfr-cfg_fBOSC['pad.detection_sample']+1,1)
cfg_fBOSC['time.time_det'] = cfg_fBOSC['time.time_tfr'][det_time2extract]
n_time_det = len(cfg_fBOSC['time.time_det'])

cfg_fBOSC['threshold.duration'] = np.kron(np.ones((1,len(cfg_fBOSC['F']))),3) # vector of duration thresholds at each frequency (previously: ncyc)
cfg_fBOSC['threshold.percentile'] = .95    # percentile of background fit for power threshold

cfg_fBOSC['fooof'] = {}
cfg_fBOSC['fooof']['peak_width_limits'] = [0.5, 12]  # peak width limits in Hz
cfg_fBOSC['fooof']['max_n_peaks'] = float('inf')  # maximum number of peaks to fit
cfg_fBOSC['fooof']['min_peak_height'] = 0.0  # minimum peak height
cfg_fBOSC['fooof']['peak_threshold'] = 2.0  # peak threshold
cfg_fBOSC['fooof']['aperiodic_mode'] = 'knee'
cfg_fBOSC['fooof']['verbose'] = True  # verbosity of fooof output

#%% ##### Step 1 #####: time-frequency wavelet decomposition for whole signal to prepare background fit
Fsample = cfg_fBOSC['fsample']  # sampling frequency of the LFP data
wavenumber = cfg_fBOSC['wavenumber']  # wavelet parameter (time-frequency tradeoff)
F = cfg_fBOSC['F']  # frequency sampling
TFR = np.zeros((1, n_freq, n_time))

[TFR[0,:,:], tmp, tmp] = fBOSCUtils.BOSC_tf(lfp_samples, F, Fsample, wavenumber)
del tmp

plt.imshow(TFR[0,:,:], extent=[0,1,0,1])

#%% ##### Step 2 #####: background fit (fooof)
fBOSC = {}
fBOSC['static'] = {}
fBOSC['static']['bg_pow'] = pd.DataFrame(columns = cfg_fBOSC['F'])
fBOSC['static']['bg_log10_pow'] = pd.DataFrame(columns = cfg_fBOSC['F'])
fBOSC['static']['pv'] = pd.DataFrame(columns = ['slope', 'intercept'])
fBOSC['static']['mp'] = pd.DataFrame(columns = cfg_fBOSC['F'])    
fBOSC['static']['pt'] = pd.DataFrame(columns = cfg_fBOSC['F'])   

[fBOSC, pt, dt] = fBOSCUtils.fBOSC_getThresholds(cfg_fBOSC, TFR, fBOSC)

#%% ##### Step 3 #####: detect rhythms and calculate Pepisode
# The next section applies both the power and the duration threshold to detect individual rhythmic segments in the continuous signals.

# tfr padding is removed to avoid edge artifacts from the wavelet transform. Note that a padding fpr detection remains attached so that there is no problems with too few sample points at the edges to fulfill the duration criterion.         

time2extract = np.arange(cfg_fBOSC['pad.tfr_sample']+1, TFR.shape[2]-cfg_fBOSC['pad.tfr_sample']+1,1)
TFR_ = np.transpose(TFR[0,:,time2extract],[1,0])

detected = np.zeros((TFR_.shape))
for f in range(len(cfg_fBOSC['F'])):
    detected[f,:] = fBOSCUtils.BOSC_detect(TFR_[f,:],pt[f],dt[0][f],cfg_fBOSC['fsample'])

# remove padding for detection (matrix with padding required for refinement)
time2encode = np.arange(cfg_fBOSC['pad.detection_sample'], detected.shape[1]-cfg_fBOSC['pad.detection_sample'],1)

# Multiindex for channel x trial x frequency x time
arrays = np.array([1,1,cfg_fBOSC['F'], cfg_fBOSC['time.time_det']],dtype=object)
#tuples = list(zip(*arrays))
names=["channel", "trial", "frequency", "time"]
# Ensure all elements in `arrays` are list-like
arrays = [arr if isinstance(arr, (list, tuple, np.ndarray)) else [arr] for arr in arrays]
# Create the MultiIndex
index = pd.MultiIndex.from_product(arrays, names=names)
nullData=np.zeros(len(arrays[0]) * len(arrays[1]) * len(arrays[2]) * len(arrays[3]) )
fBOSC['detected'] = pd.DataFrame(data = nullData, index = index)
fBOSC['detected_ep'] = fBOSC['detected'].copy()
del nullData, index

fBOSC['detected'].loc[(1,1)] = np.reshape(detected[:,time2encode],[-1,1])

#%% #### Step 4 ####: create table of rhythmic episodes
cfg_fBOSC['tmp_trial'] = 1
cfg_fBOSC['tmp_channel'] = 1
cfg_fBOSC['tmp_channelID'] = 0 # zero-based indexing for channel in fBOSC['static']['pt']

episodesTable, detected_new = fBOSCUtils.fBOSC_episode_create_v2(cfg_fBOSC, TFR_, detected, fBOSC)

#%%
plt.hist(episodesTable['FrequencyMean'], bins=50)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')
plt.title('Histogram of Episode Frequencies; fBOSCpy; withoutpostproc (Upper bound is 40 Hz)')

plt.figure()
plt.hist(episodesTable['DurationS'])
#%% save to pickle objects

#save cfg_fBOSC, episodesTable, detected_new, fBOSC to pickle object
import pickle

pickle_dir = f"../results/Fa8477/fBOSC/pickle"
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)
pickle_file = os.path.join(pickle_dir, f"fBOSC-{animal}-{date}-{session}-{basename}-50Hz.pkl")
with open(pickle_file, 'wb') as f:
    pickle.dump({
        'cfg_fBOSC': cfg_fBOSC,
        'episodesTable': episodesTable,
        'detected_new': detected_new,
        'fBOSC': fBOSC
    }, f)
print(f"fBOSC results saved to {pickle_file}")

#%% load pickle object
pickle_dir = f"../results/Fa8477/fBOSC/pickle"
pickle_file = os.path.join(pickle_dir, f"fBOSC-{animal}-{date}-{session}-{basename}-50Hz.pkl")
with open(pickle_file, 'rb') as f:
    loaded_data = pickle.load(f)
cfg_fBOSC = loaded_data['cfg_fBOSC']
episodesTable = loaded_data['episodesTable']
detected_new = loaded_data['detected_new']
fBOSC = loaded_data['fBOSC']

#%% ##### Step 5 #####: post-processing of rhythmic episodes

# Exclude temporal amplitude "leakage" due to wavelet smearing

# temporarily pass on power threshold for easier access
cfg_fBOSC['tmp.pt'] = fBOSC['static']['pt']

cfg_fBOSC['postproc.use'] = 'yes'
cfg_fBOSC['postproc.method'] = 'FWHM' # 'MaxBias' or 'FWHM'
cfg_fBOSC['postproc.edgeOnly'] = 'yes'      # Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
cfg_fBOSC['postproc.effSignal'] = 'PT'   
cfg_fBOSC['tmp_trial'] = 1
cfg_fBOSC['tmp_channel'] = 1

# only do this if there are any episodes to fine-tune
if cfg_fBOSC['postproc.use'] == 'yes' and len(episodesTable['Trial']) > 0:
    if cfg_fBOSC['postproc.method'] == 'FWHM':
        [episodesTablePostProc, detected_new_PostProc] = fBOSCUtils.eBOSC_episode_postproc_fwhm(cfg_fBOSC, episodesTable, TFR_)
    elif cfg_fBOSC['postproc.method'] == 'MaxBias':
        [episodesTable, detected_new] = fBOSCUtils.eBOSC_episode_postproc_maxbias_memmap(cfg_fBOSC, episodesTable, TFR_)
    
# remove episodes and part of episodes that fall into 'shoulder'
if len(episodesTable['Trial']) > 0 and cfg_fBOSC['pad.detection_sample']>0:
    episodesTablePostProc = fBOSCUtils.eBOSC_episode_rm_shoulder(cfg_fBOSC,detected_new_PostProc,episodesTablePostProc)

#%%
plt.figure()
plt.hist(episodesTablePostProc['FrequencyMean'], bins=50, label='FWHM')
plt.hist(episodesTable['FrequencyMean'], bins=50, alpha=0.5, label='Original', color='orange')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')
plt.title('Histogram of Episode Frequencies; fBOSCPy')
# %%
