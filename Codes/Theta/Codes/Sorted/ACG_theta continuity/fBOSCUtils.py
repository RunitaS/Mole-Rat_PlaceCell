import numpy as np
import os
import scipy.io as sio
from pathlib import Path
import warnings
from matplotlib import pyplot as plt


def BOSC_tf(eegsignal, F, Fsample, wavenumber):

    st = 1.0 / (2 * np.pi * (F / wavenumber))
    A = 1.0 / np.sqrt(st * np.sqrt(np.pi))
    # initialize the time-frequency matrix
    B = np.zeros((len(F), len(eegsignal)))
    B[:] = np.nan
    # loop through sampled frequencies
    for f in range(len(F)):
        # print(f)
        t = np.arange(-3.6 * st[f], (3.6 * st[f]), 1 / Fsample)
        # define Morlet wavelet
        m = (
            A[f]
            * np.exp(-(t**2) / (2 * st[f] ** 2))
            * np.exp(1j * 2 * np.pi * F[f] * t)
        )
        y = np.convolve(eegsignal, m, "full")
        y = abs(y) ** 2
        B[f, :] = y[
            np.arange(
                int(np.ceil(len(m) / 2)) - 1, len(y) - int(np.floor(len(m) / 2)), 1
            )
        ]
        T = np.arange(1, len(eegsignal) + 1, 1) / Fsample
    return B, T, F


def BOSC_detect(b, powthresh, durthresh, Fsample):
    """
    detected=BOSC_detect(b,powthresh,durthresh,Fsample)
    This function detects oscillations based on a wavelet power
    timecourse, b, a power threshold (powthresh) and duration
    threshold (durthresh) returned from BOSC_thresholds.m.

    It now returns the detected vector which is already episode-detected.

    b - the power timecourse (at one frequency of interest)

    durthresh - duration threshold in  required to be deemed oscillatory
    powthresh - power threshold

    returns:
    detected - a binary vector containing the value 1 for times at
               which oscillations (at the frequency of interest) were
               detected and 0 where no oscillations were detected.

    note: Remember to account for edge effects by including
    "shoulder" data and accounting for it afterwards!f

    To calculate Pepisode:
    Pepisode=length(find(detected))/(length(detected));
    """

    # number of time points
    nT = len(b)
    # t=np.arange(1,nT+1,1)/Fsample

    # Step 1: power threshold
    x = b > powthresh
    # we have to turn the boolean to numeric
    x = np.array(list(map(int, x)))
    # show the +1 and -1 edges
    dx = np.diff(x)
    if np.size(np.where(dx == 1)) != 0:
        pos = np.where(dx == 1)[0] + 1
        # pos = pos[0]
    else:
        pos = []
    if np.size(np.where(dx == -1)) != 0:
        neg = np.where(dx == -1)[0] + 1
        # neg = neg[0]
    else:
        neg = []

    # now do all the special cases to handle the edges
    detected = np.zeros(b.shape)
    if not any(pos) and not any(neg):
        # either all time points are rhythmic or none
        if all(x == 1):
            H = np.array([[0], [nT]])
        elif all(x == 0):
            H = np.array([])
    elif not any(pos):
        # i.e., starts on an episode, then stops
        H = np.array([[0], neg])
        # np.concatenate(([1],neg), axis=0)
    elif not any(neg):
        # starts, then ends on an ep.
        H = np.array([pos, [nT]])
        # np.concatenate((pos,[nT]), axis=0)
    else:
        # special-case, create the H double-vector
        if pos[0] > neg[0]:
            # we start with an episode
            pos = np.append(0, pos)
        if neg[-1] < pos[-1]:
            # we end with an episode
            neg = np.append(neg, nT)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        H = np.array([pos, neg])
        # np.concatenate((pos,neg), axis=0)

    if H.shape[0] > 0:
        # more than one "hole"
        # find epochs lasting longer than minNcycles*period
        goodep = H[1,] - H[0,] >= durthresh
        if not any(goodep):
            H = []
        else:
            H = H[:, goodep.nonzero()][:, 0]
            # mark detected episode on the detected vector
            for h in range(H.shape[1]):
                detected[np.arange(H[0][h], H[1][h], 1)] = 1

    # ensure that outputs are integer
    detected = np.array(list(map(int, detected)))
    return detected


def fooof_check_settings(settings):
    # Set defaults for all settings
    defaults = {
        "peak_width_limits": [0.5, 12],
        "max_n_peaks": float("inf"),
        "min_peak_height": 0.0,
        "peak_threshold": 2.0,
        "aperiodic_mode": "fixed",
        "verbose": True,
    }

    # Overwrite any non-existent or None settings with defaults
    for key, value in defaults.items():
        if key not in settings or settings[key] is None:
            settings[key] = value

    return settings


def fooof_unpack_results(results_in):
    results_out = {}

    results_out["aperiodic_params"] = list(results_in.aperiodic_params)

    temp = list(results_in.peak_params.ravel())
    results_out["peak_params"] = [temp[i : i + 3] for i in range(0, len(temp), 3)]

    temp = list(results_in.gaussian_params.ravel())
    results_out["gaussian_params"] = [temp[i : i + 3] for i in range(0, len(temp), 3)]

    results_out["error"] = float(results_in.error)

    # Note: r_squared gets recalculated, so doesn't need type casting
    #   Just in case, the code for type casting is:
    # results_out['r_squared'] = float(results_in.r_squared)

    return results_out


def fooof_get_model(fm):
    """
    fooof_get_model() - Return the model fit values from a FOOOF object

    Usage:
        model_fit = fooof_get_model(fm)

    Inputs:
        fm              = FOOOF object

    Outputs:
        model_fit       = model results, in a dictionary, including:
            model_fit['freqs']
            model_fit['power_spectrum']
            model_fit['fooofed_spectrum']
            model_fit['ap_fit']

    Notes:
        This function is mostly an internal function.
        It can be called directly by the user if you are interacting with FOOOF objects directly.
    """
    model_fit = {}

    model_fit["freqs"] = list(fm.freqs)
    model_fit["power_spectrum"] = list(fm.power_spectrum)
    model_fit["fooofed_spectrum"] = list(fm.fooofed_spectrum_)
    model_fit["ap_fit"] = list(getattr(fm, "_ap_fit"))

    return model_fit


def run_fooof(freqs, power_spectrum, f_range, settings, return_model):
    settings = fooof_check_settings(settings)

    freqs = np.asarray(freqs)
    power_spectrum = np.asarray(power_spectrum)

    fm = fooof.FOOOF(
        settings["peak_width_limits"],
        settings["max_n_peaks"],
        settings["min_peak_height"],
        settings["peak_threshold"],
        settings["aperiodic_mode"],
        settings["verbose"],
    )

    fm.fit(freqs, power_spectrum, f_range)

    fooof_results = fm.get_results()
    fooof_results = fooof_unpack_results(fooof_results)

    # Re-calculating r-squared
    # r_squared doesn't seem to get computed properly (in NaN).
    # It is unclear why this happens, other than the error can be traced
    # back to the internal call to `np.cov`, and fails when this function
    # gets two arrays as input.
    # Therefore, we can simply recalculate r-squared
    coefs = np.corrcoef(np.array(fm.power_spectrum), np.array(fm.fooofed_spectrum_))
    fooof_results["r_squared"] = coefs[0, 1]

    if return_model:
        model_out = fooof_get_model(fm)
        for field in model_out:
            fooof_results[field] = model_out[field]

    return fooof_results


import fooof
from scipy.stats.distributions import chi2


def fBOSC_getThresholds(cfg_fBOSC, TFR, fBOSC):
    """
    This function estimates the static duration and power thresholds and
    saves information regarding the overall spectrum and background.
    Inputs:
               cfg_fBOSC | config structure
               TFR | time-frequency matrix
               fBOSC | main fBOSC output structure; will be updated

    Outputs:
               fBOSC   | updated w.r.t. background info (see below)
                       | bg_pow: overall power spectrum
                       | bg_log10_pow: overall power spectrum (log10)
                       | pv: intercept and slope of fit
                       | mp: linear background power
                       | pt: power threshold
               pt | empirical power threshold
               dt | duration threshold
    """

    # remove BGpad at beginning and end to avoid edge artifacts
    time2extract = np.arange(
        cfg_fBOSC["pad.background_sample"] + 1,
        TFR.shape[2] - cfg_fBOSC["pad.background_sample"] + 1,
        1,
    )

    # index both trial and time dimension simultaneously
    # TFR = TFR[np.ix_(trial2extract,range(TFR.shape[1]),time2extract)]
    TFR = TFR[:, :, time2extract]

    # concatenate trials in time dimension: permute dimensions, then reshape
    TFR_t = np.transpose(TFR, [1, 2, 0])
    BG = TFR_t.reshape(TFR_t.shape[0], TFR_t.shape[1] * TFR_t.shape[2])
    del TFR_t, time2extract
    # plt.imshow(BG[:,0:100], extent=[0, 1, 0, 1])

    freqs = cfg_fBOSC["F"]
    mean_pow = 10 ** np.mean(np.log10(BG), axis=1)
    settings = cfg_fBOSC["fooof"]
    f_range = (freqs[0], freqs[-1])
    fooof_results = run_fooof(freqs, mean_pow, f_range, settings, True)

    # compute fBOSC power and duration threshold
    mp = 10 ** np.array(fooof_results["ap_fit"])
    pt = chi2.ppf(cfg_fBOSC["threshold.percentile"], 2) * mp
    dt = cfg_fBOSC["threshold.duration"] * cfg_fBOSC["fsample"] / cfg_fBOSC["F"]

    # save results in fBOSC structure
    # Save multiple time-invariant estimates that could be of interest:
    # Overall wavelet power spectrum (NOT only background)
    fBOSC["static"]["bg_pow"] = np.mean(
        BG[:, cfg_fBOSC["pad.total_sample"] : -cfg_fBOSC["pad.total_sample"]], axis=1
    )

    # Log10-transformed wavelet power spectrum (NOT only background)
    fBOSC["static"]["bg_log10_pow"] = np.mean(
        np.log10(BG[:, cfg_fBOSC["pad.total_sample"] : -cfg_fBOSC["pad.total_sample"]]),
        axis=1,
    )

    # Aperiodic parameters from fooof
    if cfg_fBOSC["fooof"]["aperiodic_mode"] != "old":
        fBOSC["static"]["aperiodic_params"] = fooof_results["aperiodic_params"]
        fBOSC["static"]["ap_fit"] = fooof_results["ap_fit"]
        fBOSC["static"]["r_squared"] = fooof_results["r_squared"]
        fBOSC["static"]["error"] = fooof_results["error"]
        fBOSC["static"]["fooofed_spectrum"] = fooof_results["fooofed_spectrum"]
        # Peak params (was computed by run_fooof but previously discarded here) --
        # kept so callers with return_diagnostics=True can plot/inspect this
        # background fit's periodic peaks (e.g. theta) alongside the aperiodic fit.
        fBOSC["static"]["peak_params"] = fooof_results["peak_params"]

    # Linear background power at each estimated frequency
    fBOSC["static"]["mp"] = mp
    # fBOSC['static']['mp_old'] = mp_old

    # Statistical power threshold
    fBOSC["static"]["pt"] = pt
    # fBOSC['static']['pt_old'] = pt_old

    return fBOSC, pt, dt


def find_nearest_value(array, value):
    """Find nearest value and index of float in array
    Parameters:
    array : Array of values [1d array]
    value : Value of interest [float]
    Returns:
    array[idx] : Nearest value [1d float]
    idx : Nearest index [1d float]
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR):
    """Sparsen the detected matrix along the frequency dimension"""
    print("Creating sparse detected matrix ...")

    freqWidth = (2 / cfg_eBOSC["wavenumber"]) * cfg_eBOSC["F"]
    lowFreq = cfg_eBOSC["F"] - (freqWidth / 2)
    highFreq = cfg_eBOSC["F"] + (freqWidth / 2)
    #  define range for each frequency across which max. is detected
    fmat = np.zeros([cfg_eBOSC["F"].shape[0], 3])
    for [indF, valF] in enumerate(cfg_eBOSC["F"]):
        # print(indF)
        lastVal = np.where(cfg_eBOSC["F"] <= lowFreq[indF])[0]
        if len(lastVal) > 0:
            # first freq falling into range
            fmat[indF, 0] = lastVal[-1] + 1
        else:
            fmat[indF, 0] = 0
        firstVal = np.where(cfg_eBOSC["F"] >= highFreq[indF])[0]
        if len(firstVal) > 0:
            # last freq falling into range
            fmat[indF, 2] = firstVal[0] - 1
        else:
            fmat[indF, 2] = cfg_eBOSC["F"].shape[0] - 1
    fmat[:, 1] = np.arange(0, cfg_eBOSC["F"].shape[0], 1)
    del indF
    range_cur = np.diff(fmat, axis=1)
    range_cur = [int(np.max(range_cur[:, 0])), int(np.max(range_cur[:, 1]))]
    # perform the actual search
    # initialize variables
    # append frequency search space (i.e. range at both ends. first index refers to lower range
    c1 = np.zeros([int(range_cur[0]), TFR.shape[1]])
    c2 = TFR * detected
    c3 = np.zeros([int(range_cur[1]), TFR.shape[1]])
    tmp_B = np.concatenate([c1, c2, c3])
    del c1, c2, c3
    # preallocate matrix (incl. padding , which will be removed)
    detected = np.zeros(tmp_B.shape)
    # loop across frequencies. note that indexing respects the appended segments
    freqs_to_search = np.arange(
        int(range_cur[0]), int(tmp_B.shape[0] - range_cur[1]), 1
    )
    for f in freqs_to_search:
        # encode detected positions where power is higher than in LOWER and HIGHER ranges
        range1 = [f + np.arange(1, int(range_cur[1]) + 1)][0]
        range2 = [f - np.arange(1, int(range_cur[0]) + 1)][0]
        ranges = np.concatenate([range1, range2])
        detected[f, :] = np.logical_and(
            tmp_B[f, :] != 0, np.min(tmp_B[f, :] >= tmp_B[ranges, :], axis=0)
        )
    # only retain data without padded zeros
    detected = detected[freqs_to_search, :]
    return detected


def fBOSC_episode_create(cfg_fBOSC, TFR_, detected, fBOSC):
    """This function creates continuous rhythmic "episodes" and attempts to control for the impact of wavelet parameters.
      Time-frequency points that best represent neural rhythms are identified by
      heuristically removing temporal and frequency leakage.

     Frequency leakage: at each frequency x time point, power has to exceed neighboring frequencies.
      Then it is checked whether the detected time-frequency points belong to
      a continuous episode for which (1) the frequency maximally changes by
      +/- n steps (cfg.fBOSC.fstp) from on time point to the next and (2) that is at
      least as long as n number of cycles (cfg.fBOSC.threshold.duration) of the average freqency
      of that episode (a priori duration threshold).

     Temporal leakage: The impact of the amplitude at each time point within a rhythmic episode on previous
      and following time points is tested with the goal to exclude supra-threshold time
      points that are due to the wavelet extension in time.

    Input:
               cfg         | config structure with cfg.fBOSC field
               TFR         | time-frequency matrix (excl. WLpadding)
               detected    | detected oscillations in TFR (based on power and duration threshold)
               fBOSC       | main fBOSC output structure; necessary to read in
                               prior eBOSC.episodes if they exist in a loop

    Output:
               detected_new    | new detected matrix with frequency leakage removed
               episodesTable   | table with specific episode information:
                     Trial: trial index (corresponds to cfg.eBOSC.trial)
                     Channel: channel index
                     FrequencyMean: mean frequency of episode (Hz)
                     DurationS: episode duration (in sec)
                     DurationC: episode duration (in cycles, based on mean frequency)
                     PowerMean: mean amplitude of amplitude
                     Onset: episode onset in s
                     Offset: episode onset in s
                     Power: (cell) time-resolved wavelet-based amplitude estimates during episode
                     Frequency: (cell) time-resolved wavelet-based frequency
                     RowID: (cell) row index (frequency dimension): following eBOSC_episode_rm_shoulder relative to data excl. detection padding
                     ColID: (cell) column index (time dimension)
                     SNR: (cell) time-resolved signal-to-noise ratio: momentary amplitude/static background estimate at channel*frequency
                     SNRMean: mean signal-to-noise ratio
    """
    # initialize dictionary to save results in
    episodesTable = {}
    episodesTable["RowID"] = []
    episodesTable["ColID"] = []
    episodesTable["Frequency"] = []
    episodesTable["FrequencyMean"] = []
    episodesTable["Power"] = []
    episodesTable["PowerMean"] = []
    episodesTable["DurationS"] = []
    episodesTable["DurationC"] = []
    episodesTable["Trial"] = []
    episodesTable["Channel"] = []
    episodesTable["Onset"] = []
    episodesTable["Offset"] = []
    episodesTable["SNR"] = []
    episodesTable["SNRMean"] = []

    #  Accounting for the frequency spread of the wavelet

    # Here, we compute the bandpass response as given by the wavelet
    # formula and apply half of the BP repsonse on top of the center frequency.
    # Because of log-scaling, the widths are not the same on both sides.

    detected = eBOSC_episode_sparsefreq(cfg_fBOSC, detected, TFR_)

    # Create continuous rhythmic episodes

    # define step size in adjacency matrix
    cfg_fBOSC["fstp"] = 1

    # add zeros
    padding = np.zeros([cfg_fBOSC["fstp"], detected.shape[1]])
    detected_remaining = np.vstack([padding, detected, padding])
    detected_remaining[:, 0] = 0
    detected_remaining[:, -1] = 0
    # print size of detected_remaining
    print(f"Size of detected_remaining: {detected_remaining.shape}")
    # detected_remaining serves as a dummy matrix; unless all entries from detected_remaining are
    # removed, we will continue extracting episodes
    tmp_B1 = np.vstack([padding, TFR_ * detected, padding])
    tmp_B1[:, 0] = 0
    tmp_B1[:, -1] = 0
    detected_new = np.zeros(detected.shape)

    while sum(sum(detected_remaining)) > 0:
        # sampling point counter
        x = []
        y = []
        # find seed (remember that numpy uses row-first format!)
        # we need increasing x-axis sorting here
        [tmp_y, tmp_x] = np.where(np.matrix.transpose(detected_remaining) == 1)
        x.append(tmp_x[0])
        y.append(tmp_y[0])
        # check next sampling point
        chck = 0
        while chck == 0:
            # next sampling point
            next_point = y[-1] + 1
            next_freqs = np.arange(
                x[-1] - cfg_fBOSC["fstp"], x[-1] + cfg_fBOSC["fstp"] + 1
            )
            tmp = np.where(detected_remaining[next_freqs, next_point] == 1)[0]
            if tmp.size > 0:
                y.append(next_point)
                if tmp.size > 1:
                    # JQK 161017: It is possible that an episode is branching
                    # two ways, hence we follow the 'strongest' branch;
                    # Note that there is no correction for 1/f here, but
                    # practically, it leads to satisfying results
                    # (i.e. following the longer episodes).
                    tmp_data = tmp_B1[next_freqs, next_point]
                    tmp = np.where(tmp_data == max(tmp_data))[0]
                x.append(next_freqs[tmp[0]])
            else:
                chck = 1

        # check for passing the duration requirement
        # get average frequency
        avg_frq = np.mean(cfg_fBOSC["F"][np.array(x) - cfg_fBOSC["fstp"]])
        # match to closest frequency
        [tmp_a, indF] = find_nearest_value(cfg_fBOSC["F"], avg_frq)
        # check number of data points to fulfill number of cycles criterion
        num_pnt = np.floor(
            (cfg_fBOSC["fsample"] / avg_frq)
            * int(cfg_fBOSC["threshold.duration"].flatten()[indF])
        )
        if len(y) >= num_pnt:
            #  encode episode that crosses duration threshold
            episodesTable["RowID"].append(np.array(x) - cfg_fBOSC["fstp"])
            episodesTable["ColID"].append([np.single(y[0]), np.single(y[-1])])
            episodesTable["Frequency"].append(
                np.single(cfg_fBOSC["F"][episodesTable["RowID"][-1]])
            )
            episodesTable["FrequencyMean"].append(np.single(avg_frq))
            tmp_x = episodesTable["RowID"][-1]
            tmp_y = np.arange(
                int(episodesTable["ColID"][-1][0]),
                int(episodesTable["ColID"][-1][1]) + 1,
            )
            linIdx = np.ravel_multi_index(
                [np.reshape(tmp_x, [-1, 1]), np.reshape(tmp_y, [-1, 1])],
                dims=TFR_.shape,
                order="C",
            )
            episodesTable["Power"].append(np.single(TFR_.flatten("C")[linIdx]))
            episodesTable["PowerMean"].append(np.mean(episodesTable["Power"][-1]))
            episodesTable["DurationS"].append(np.single(len(y) / cfg_fBOSC["fsample"]))
            episodesTable["DurationC"].append(
                episodesTable["DurationS"][-1] * episodesTable["FrequencyMean"][-1]
            )
            episodesTable["Trial"].append(
                cfg_fBOSC["tmp_trial"]
            )  # Note that the trial is non-zero-based
            episodesTable["Channel"].append(cfg_fBOSC["tmp_channel"])
            # episode onset in absolute time
            episodesTable["Onset"].append(
                cfg_fBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][0])]
            )
            # episode offset in absolute time
            episodesTable["Offset"].append(
                cfg_fBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][-1])]
            )
            # extract (static) background power at frequencies
            episodesTable["SNR"].append(
                episodesTable["Power"][-1]
                / fBOSC["static"]["pt"][episodesTable["RowID"][-1]]
            )
            episodesTable["SNRMean"].append(np.mean(episodesTable["SNR"][-1]))

            # remove processed segment from detected matrix
            detected_remaining[x, y] = 0
            # set all detected points to one in binary detected matrix
            rows = episodesTable["RowID"][-1]
            cols = np.arange(
                int(episodesTable["ColID"][-1][0]),
                int(episodesTable["ColID"][-1][1]) + 1,
            )
            detected_new[rows, cols] = 1
        else:
            #  remove episode from consideration due to being lower than duration
            detected_remaining[x, y] = 0

    return episodesTable, detected_new


import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, find
from collections import deque
from joblib import Parallel, delayed


def fBOSC_episode_create_v2(cfg_fBOSC, TFR_, detected, fBOSC):
    """
    Create continuous rhythmic episodes with optimized performance.

    Parameters:
    cfg_fBOSC : dict
        Configuration structure with fBOSC parameters.
    TFR_ : numpy.ndarray
        Time-frequency matrix (excluding wavelet padding).
    detected : numpy.ndarray
        Binary matrix of detected oscillations in TFR.
    fBOSC : dict
        Main fBOSC output structure.

    Returns:
    episodesTable : dict
        Table with specific episode information.
    detected_new : numpy.ndarray
        Updated binary matrix with frequency leakage removed.
    """
    # Initialize dictionary to save results
    episodesTable = {
        "RowID": [],
        "ColID": [],
        "Frequency": [],
        "FrequencyMean": [],
        "Power": [],
        "PowerMean": [],
        "DurationS": [],
        "DurationC": [],
        "Trial": [],
        "Channel": [],
        "Onset": [],
        "Offset": [],
        "SNR": [],
        "SNRMean": [],
    }
    #  Accounting for the frequency spread of the wavelet

    # Here, we compute the bandpass response as given by the wavelet
    # formula and apply half of the BP repsonse on top of the center frequency.
    # Because of log-scaling, the widths are not the same on both sides.

    detected = eBOSC_episode_sparsefreq(cfg_fBOSC, detected, TFR_)

    # Create continuous rhythmic episodes

    # define step size in adjacency matrix
    cfg_fBOSC["fstp"] = 1

    # Convert detected matrix to sparse format for efficiency
    detected_sparse = csr_matrix(detected)
    detected_remaining = lil_matrix(detected_sparse)
    detected_new = lil_matrix(detected_sparse.shape)

    # Precompute padded matrices
    padding = np.zeros([cfg_fBOSC["fstp"], detected_sparse.shape[1]])
    tmp_B1 = np.vstack([padding, TFR_ * detected, padding])
    tmp_B1[:, 0] = 0
    tmp_B1[:, -1] = 0

    # Function to process a single episode
    def process_episode(seed_x, seed_y):
        x, y = deque([seed_x]), deque([seed_y])
        chck = 0

        while chck == 0:
            next_point = y[-1] + 1
            next_freqs = np.arange(
                x[-1] - cfg_fBOSC["fstp"], x[-1] + cfg_fBOSC["fstp"] + 1
            )
            tmp = np.where(detected_remaining[next_freqs, next_point].toarray() == 1)[0]

            if tmp.size > 0:
                y.append(next_point)
                if tmp.size > 1:
                    tmp_data = tmp_B1[next_freqs, next_point]
                    tmp = np.where(tmp_data == np.max(tmp_data))[0]
                x.append(next_freqs[tmp[0]])
            else:
                chck = 1

        # Check duration requirement
        avg_frq = np.mean(cfg_fBOSC["F"][np.array(x) - cfg_fBOSC["fstp"]])
        _, indF = find_nearest_value(cfg_fBOSC["F"], avg_frq)
        num_pnt = np.floor(
            (cfg_fBOSC["fsample"] / avg_frq)
            * int(cfg_fBOSC["threshold.duration"].flatten()[indF])
        )

        if len(y) >= num_pnt:
            # Encode episode
            row_ids = np.array(x) - cfg_fBOSC["fstp"]
            col_ids = [np.single(y[0]), np.single(y[-1])]
            freq = np.single(cfg_fBOSC["F"][row_ids])
            # Flatten the indices and calculate power
            try:
                power = np.single(
                    TFR_.flatten("C")[
                        np.ravel_multi_index(
                            [
                                np.reshape(row_ids, [-1, 1]),
                                np.reshape(col_ids, [-1, 1]),
                            ],
                            dims=TFR_.shape,
                            order="C",
                        )
                    ]
                )
            except ValueError as e:
                print(f"Error in ravel_multi_index: {e}")
                print(
                    f"row_ids: {row_ids}, col_ids: {col_ids}, TFR_.shape: {TFR_.shape}"
                )
                raise

            return {
                "RowID": row_ids,
                "ColID": col_ids,
                "Frequency": freq,
                "FrequencyMean": np.single(avg_frq),
                "Power": power,
                "PowerMean": np.mean(power),
                "DurationS": np.single(len(y) / cfg_fBOSC["fsample"]),
                "DurationC": np.single(len(y) / cfg_fBOSC["fsample"] * avg_frq),
                "Trial": cfg_fBOSC["tmp_trial"],
                "Channel": cfg_fBOSC["tmp_channel"],
                "Onset": cfg_fBOSC["time.time_tfr"][int(col_ids[0])],
                "Offset": cfg_fBOSC["time.time_tfr"][int(col_ids[1])],
                "SNR": power / fBOSC["static"]["pt"][row_ids],
                "SNRMean": np.mean(power / fBOSC["static"]["pt"][row_ids]),
            }
        else:
            # Remove episode from consideration
            detected_remaining[x, y] = 0
            return None

    # Process all episodes
    seeds = find(detected_remaining)
    results = Parallel(n_jobs=-1)(
        delayed(process_episode)(seed_x, seed_y)
        for seed_x, seed_y in zip(seeds[0], seeds[1])
    )

    # Filter out None results and update episodesTable
    for result in results:
        if result:
            for key in episodesTable:
                episodesTable[key].append(result[key])
            # Remove processed segment from detected matrix
            detected_remaining[result["RowID"], result["ColID"]] = 0
            detected_new[result["RowID"], result["ColID"]] = 1

    return episodesTable, detected_new.toarray()


# Helper function to find the nearest value
def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can trivially be explained by the FWHM of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs:
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs:
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    """

    print("Applying FWHM post-processing ...")

    # re-initialize detected_new (for post-proc results)
    detected_new = np.zeros(TFR.shape)
    # initialize new dictionary to save results in
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []

    for e in range(len(episodes["Trial"])):
        # get temporary frequency vector
        f_ = episodes["Frequency"][e]
        f_unique = np.unique(f_)
        # find index within minor tolerance (float arrays)
        f_ind_unique = np.where(np.abs(cfg_eBOSC["F"][:, None] - f_unique) < 1e-5)
        f_ind_unique = f_ind_unique[0]
        # get temporary amplitude vector
        a_ = episodes["Power"][e]
        # location in time with reference to matrix TFR
        t_ind = np.int_(
            np.arange(episodes["ColID"][e][0], episodes["ColID"][e][-1] + 1)
        )
        # initiate bias matrix (only requires to encode frequencies occuring within episode)
        biasMat = np.zeros([len(f_unique), len(a_)])

        for tp in range(len(a_)):
            # The FWHM correction is done independently at each
            # frequency. To accomplish this, we actually reference
            # to the original data in the TF matrix.
            # search within frequencies that occur within the episode
            for f in range(len(f_unique)):
                # create wavelet with center frequency and amplitude at time point
                st = 1 / (2 * np.pi * (f_unique / cfg_eBOSC["wavenumber"]))
                step_size = 1 / cfg_eBOSC["fsample"]
                t = np.arange(-3.6 * st[f], 3.6 * st[f] + step_size, step_size)
                wave = np.exp(-(t**2) / (2 * st[f] ** 2)) * np.exp(
                    1j * 2 * np.pi * f_unique[f] * t
                )
                if cfg_eBOSC["postproc.effSignal"] == "all":
                    # Morlet wavelet with amplitude-power threshold modulation
                    m = TFR[f_ind_unique[f], int(t_ind[tp])] * wave
                elif cfg_eBOSC["postproc.effSignal"] == "PT":
                    m = (
                        TFR[f_ind_unique[f], int(t_ind[tp])]
                        - cfg_eBOSC["tmp.pt"][f_ind_unique[f]]
                    ) * wave
                # amplitude of wavelet
                wl_a = abs(m)
                maxval = max(wl_a)
                maxloc = np.where(np.abs(wl_a[:, None] - maxval) < 1e-5)[0][0]
                index_fwhm = np.where(wl_a >= maxval / 2)[0][0]
                # amplitude at fwhm, freq
                fwhm_a = wl_a[index_fwhm]
                if cfg_eBOSC["postproc.effSignal"] == "PT":
                    # re-add power threshold
                    fwhm_a = fwhm_a + cfg_eBOSC["tmp.pt"][f_ind_unique[f]]
                correctionDist = maxloc - index_fwhm
                # extract FWHM amplitude of frequency- and amplitude-specific wavelet
                # check that lower fwhm is part of signal
                if tp - correctionDist >= 0:
                    # and that existing value is lower than update
                    if biasMat[f, tp - correctionDist] < fwhm_a:
                        biasMat[f, tp - correctionDist] = fwhm_a
                # check that upper fwhm is part of signal
                if tp + correctionDist + 1 <= biasMat.shape[1]:
                    # and that existing value is lower than update
                    if biasMat[f, tp + correctionDist] < fwhm_a:
                        biasMat[f, tp + correctionDist] = fwhm_a

        # plt.imshow(biasMat, extent=[0, 1, 0, 1])

        # retain only those points that are larger than the FWHM
        aMat_retain = np.zeros(biasMat.shape)
        indFreqs = np.where(np.abs(f_[:, None] - f_unique) < 1e-5)
        indFreqs = indFreqs[1]
        for indF in range(len(f_unique)):
            aMat_retain[indF, np.where(indFreqs == indF)[0]] = np.transpose(
                a_[indFreqs == indF]
            )
        # anything that is lower than the convolved wavelet is removed
        aMat_retain[aMat_retain <= biasMat] = 0

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = aMat_retain.mean(0) > 0
        keep = keep > 0
        if cfg_eBOSC["postproc.edgeOnly"] == "yes":
            keepEdgeRemovalOnly = np.zeros([len(keep)], dtype=bool)
            keepEdgeRemovalOnly[
                np.arange(np.where(keep == 1)[0][0], np.where(keep == 1)[0][-1] + 1)
            ] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly

        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))

        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0] - 1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i], ind_epsd_end[i] + 1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC["F"], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    num_pnt = np.floor(
                        (cfg_eBOSC["fsample"] / avg_frq)
                        * int(
                            np.reshape(cfg_eBOSC["threshold.duration"], [-1])[indF]
                        )
                    )
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable["RowID"].append(episodes["RowID"][e][tmp_col])
                    episodesTable["ColID"].append(
                        [t_ind[tmp_col[0]], t_ind[tmp_col[-1]]]
                    )
                    episodesTable["Frequency"].append(f_[tmp_col])
                    episodesTable["FrequencyMean"].append(
                        np.mean(episodesTable["Frequency"][-1])
                    )
                    episodesTable["Power"].append(a_[tmp_col])
                    episodesTable["PowerMean"].append(
                        np.mean(episodesTable["Power"][-1])
                    )
                    episodesTable["DurationS"].append(
                        np.diff(episodesTable["ColID"][-1])[0] / cfg_eBOSC["fsample"]
                    )
                    episodesTable["DurationC"].append(
                        episodesTable["DurationS"][-1]
                        * episodesTable["FrequencyMean"][-1]
                    )
                    episodesTable["Trial"].append(cfg_eBOSC["tmp_trial"])
                    episodesTable["Channel"].append(cfg_eBOSC["tmp_channel"])
                    episodesTable["Onset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][0])]
                    )
                    episodesTable["Offset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][-1])]
                    )
                    episodesTable["SNR"].append(episodes["SNR"][e][tmp_col])
                    episodesTable["SNRMean"].append(np.mean(episodesTable["SNR"][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable["RowID"][-1], t_ind[tmp_col]] = 1

    # plt.imshow(detected_new, extent=[0, 1, 0, 1])
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new


def eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can be explained by the simulated extension of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs:
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs:
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix

    % This method works as follows: we estimate the bias introduced by
    % wavelet convolution. The bias is represented by the amplitudes
    % estimated for the zero-shouldered signal (i.e. for which no real
    % data was initially available). The influence of episodic
    % amplitudes on neighboring time points is assessed by scaling each
    % time point's amplitude with the last 'rhythmic simulated time
    % point', i.e. the first time wavelet amplitude in the simulated
    % rhythmic time points. At this time point the 'bias' is maximal,
    % although more precisely, this amplitude does not represent a
    % bias per se.
    """

    print("Applying maxbias post-processing ...")

    # re-initialize detected_new (for post-proc results)
    N_freq = TFR.shape[0]
    N_tp = TFR.shape[1]
    detected_new = np.zeros([N_freq, N_tp])
    # initialize new dictionary to save results in
    # this is required as episodes may split, thus needing novel entries
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []

    # generate "bias" matrix
    # the logic here is as follows: we take a sinusoid, zero-pad it, and get the TFR
    # the bias is the tfr power produced for the padding (where power should be zero)
    B_bias = np.zeros([len(cfg_eBOSC["F"]), len(cfg_eBOSC["F"]), 2 * N_tp + 1])
    amp_max = np.zeros([len(cfg_eBOSC["F"]), len(cfg_eBOSC["F"])])
    for f in range(len(cfg_eBOSC["F"])):
        # temporary time vector and signal
        step_size = 1 / cfg_eBOSC["fsample"]
        time = np.arange(step_size, 1 / cfg_eBOSC["F"][f] + step_size, step_size)
        tmp_sig = np.cos(time * 2 * np.pi * cfg_eBOSC["F"][f]) * -1 + 1
        # signal for time-frequency analysis
        signal = np.concatenate((np.zeros([N_tp]), tmp_sig, np.zeros([N_tp])))
        [tmp_bias_mat, tmp_time, tmp_freq] = BOSC_tf(
            signal, cfg_eBOSC["F"], cfg_eBOSC["fsample"], cfg_eBOSC["wavenumber"]
        )
        # bias matrix
        points_begin = np.arange(0, N_tp + 1)
        points_end = np.arange(N_tp, B_bias.shape[2] + 1)
        # for some reason, we have to transpose the matrix here, as the submatrix dimension order changes???
        B_bias[f, :, points_begin] = np.transpose(tmp_bias_mat[:, points_begin])
        B_bias[f, :, points_end] = np.transpose(
            np.fliplr(tmp_bias_mat[:, points_begin])
        )
        # maximum amplitude
        amp_max[f, :] = B_bias[f, :, :].max(1)
        # plt.imshow(amp_max, extent=[0, 1, 0, 1])

    # midpoint index
    ind_mid = N_tp + 1
    # loop episodes
    for e in range(len(episodes["Trial"])):
        # get temporary frequency vector
        f_ = episodes["Frequency"][e]
        # get temporary amplitude vector
        a_ = episodes["Power"][e]
        m_ = np.zeros([len(a_), len(a_)])
        # location in time with reference to matrix TFR
        t_ind = np.arange(
            int(episodes["ColID"][e][0]), int(episodes["ColID"][e][-1] + 1)
        )
        # indices of time points' frequencies within "bias" matrix
        f_vec = episodes["RowID"][e]
        # figure; hold on;
        for tp in range(len(a_)):
            # index of current point's frequency within "bias" matrix
            ind_f = f_vec[tp]
            # get bias vector that varies with frequency of the
            # timepoints in the episode
            temporalBiasIndices = np.arange(
                ind_mid + 1 - tp, ind_mid + len(a_) - tp + 1
            )
            ind1 = np.matlib.repmat(ind_f, len(f_vec), 1)
            ind2 = np.reshape(f_vec, [-1, 1])
            ind3 = np.reshape(temporalBiasIndices, [-1, 1])
            indices = np.ravel_multi_index(
                [ind1, ind2, ind3], dims=B_bias.shape, order="C"
            )
            tmp_biasVec = B_bias.flatten("C")[indices]
            # temporary "bias" vector (frequency-varying)
            if cfg_eBOSC["postproc.effSignal"] == "all":
                tmp_bias = (
                    tmp_biasVec / np.reshape(amp_max[ind_f, f_vec], [-1, 1])
                ) * a_[tp]
            elif cfg_eBOSC["postproc.effSignal"] == "PT":
                tmp_bias = (
                    (tmp_biasVec / np.reshape(amp_max[ind_f, f_vec], [-1, 1]))
                    * (a_[tp] - cfg_eBOSC["tmp.pt"][ind_f])
                ) + cfg_eBOSC["tmp.pt"][ind_f]
            # compare to data
            m_[tp, :] = np.transpose(a_ >= tmp_bias)
            # plot(a_', 'k'); hold on; plot(tmp_bias, 'r');

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = m_.sum(0) == len(a_)
        if cfg_eBOSC["postproc.edgeOnly"] == "yes":
            # keep everything that would be kept within the vector,
            # no removal within episode except for edges possible
            keepEdgeRemovalOnly = np.zeros([len(keep)], dtype=bool)
            keepEdgeRemovalOnly[
                np.arange(np.where(keep == 1)[0][0], np.where(keep == 1)[0][-1] + 1)
            ] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly

        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))

        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0] - 1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i], ind_epsd_end[i] + 1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC["F"], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor(
                    (cfg_eBOSC["fsample"] / avg_frq)
                    * int(np.reshape(cfg_eBOSC["threshold.duration"], [-1])[indF])
                )
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable["RowID"].append(episodes["RowID"][e][tmp_col])
                    episodesTable["ColID"].append(
                        [t_ind[tmp_col[0]], t_ind[tmp_col[-1]]]
                    )
                    episodesTable["Frequency"].append(f_[tmp_col])
                    episodesTable["FrequencyMean"].append(
                        np.mean(episodesTable["Frequency"][-1])
                    )
                    episodesTable["Power"].append(a_[tmp_col])
                    episodesTable["PowerMean"].append(
                        np.mean(episodesTable["Power"][-1])
                    )
                    episodesTable["DurationS"].append(
                        np.diff(episodesTable["ColID"][-1])[0] / cfg_eBOSC["fsample"]
                    )
                    episodesTable["DurationC"].append(
                        episodesTable["DurationS"][-1]
                        * episodesTable["FrequencyMean"][-1]
                    )
                    episodesTable["Trial"].append(cfg_eBOSC["tmp_trial"])
                    episodesTable["Channel"].append(cfg_eBOSC["tmp_channel"])
                    episodesTable["Onset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][0])]
                    )
                    episodesTable["Offset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][-1])]
                    )
                    episodesTable["SNR"].append(episodes["SNR"][e][tmp_col])
                    episodesTable["SNRMean"].append(np.mean(episodesTable["SNR"][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable["RowID"][-1], t_ind[tmp_col]] = 1
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new


def eBOSC_episode_postproc_maxbias_memmap(cfg_eBOSC, episodes, TFR):
    """
    This function performs post-processing of input episodes by checking
    whether 'detected' time points can be explained by the simulated extension of
    the wavelet used in the time-frequency transform.
    """

    print("Applying maxbias post-processing ...")

    # Re-initialize detected_new (for post-proc results)
    N_freq = TFR.shape[0]
    N_tp = TFR.shape[1]
    detected_new = np.zeros([N_freq, N_tp])

    # Initialize new dictionary to save results in
    episodesTable = {entry: [] for entry in episodes}

    # Create memory-mapped arrays for B_bias and amp_max
    mmap_dir = "./mmap_storage"
    if not os.path.exists(mmap_dir):
        os.makedirs(mmap_dir)

    B_bias_filename = os.path.join(mmap_dir, "B_bias.dat")
    amp_max_filename = os.path.join(mmap_dir, "amp_max.dat")

    B_bias = np.memmap(
        B_bias_filename,
        dtype="float32",
        mode="w+",
        shape=(len(cfg_eBOSC["F"]), len(cfg_eBOSC["F"]), 2 * N_tp + 1),
    )
    amp_max = np.memmap(
        amp_max_filename,
        dtype="float32",
        mode="w+",
        shape=(len(cfg_eBOSC["F"]), len(cfg_eBOSC["F"])),
    )

    # Generate "bias" matrix
    for f in range(len(cfg_eBOSC["F"])):
        # Temporary time vector and signal
        step_size = 1 / cfg_eBOSC["fsample"]
        time = np.arange(step_size, 1 / cfg_eBOSC["F"][f] + step_size, step_size)
        tmp_sig = np.cos(time * 2 * np.pi * cfg_eBOSC["F"][f]) * -1 + 1
        # Signal for time-frequency analysis
        signal = np.concatenate((np.zeros([N_tp]), tmp_sig, np.zeros([N_tp])))
        [tmp_bias_mat, tmp_time, tmp_freq] = BOSC_tf(
            signal, cfg_eBOSC["F"], cfg_eBOSC["fsample"], cfg_eBOSC["wavenumber"]
        )

        # Bias matrix
        points_begin = np.arange(0, N_tp)
        points_end = np.arange(N_tp, B_bias.shape[2] - 1)
        print(B_bias.shape, tmp_bias_mat.shape)
        print(N_tp, points_begin, points_end)

        # Transpose and assign values to memory-mapped array
        B_bias[f, :, points_begin] = np.transpose(tmp_bias_mat[:, points_begin])
        B_bias[f, :, points_end] = np.transpose(
            np.fliplr(tmp_bias_mat[:, points_begin])
        )
        # Maximum amplitude
        amp_max[f, :] = B_bias[f, :, :].max(1)

    # Midpoint index
    ind_mid = N_tp + 1

    # Loop through episodes
    for e in range(len(episodes["Trial"])):
        # Get temporary frequency vector
        f_ = episodes["Frequency"][e]
        # Get temporary amplitude vector
        a_ = episodes["Power"][e]
        m_ = np.zeros([len(a_), len(a_)])
        # Location in time with reference to matrix TFR
        t_ind = np.arange(
            int(episodes["ColID"][e][0]), int(episodes["ColID"][e][-1] + 1)
        )
        # Indices of time points' frequencies within "bias" matrix
        f_vec = episodes["RowID"][e]

        for tp in range(len(a_)):
            # Index of current point's frequency within "bias" matrix
            ind_f = f_vec[tp]
            # Get bias vector that varies with frequency of the timepoints in the episode
            temporalBiasIndices = np.arange(
                ind_mid + 1 - tp, ind_mid + len(a_) - tp + 1
            )
            ind1 = np.matlib.repmat(ind_f, len(f_vec), 1)
            ind2 = np.reshape(f_vec, [-1, 1])
            ind3 = np.reshape(temporalBiasIndices, [-1, 1])
            indices = np.ravel_multi_index(
                [ind1, ind2, ind3], dims=B_bias.shape, order="C"
            )
            tmp_biasVec = B_bias.flatten("C")[indices]

            # Temporary "bias" vector (frequency-varying)
            if cfg_eBOSC["postproc.effSignal"] == "all":
                tmp_bias = (
                    tmp_biasVec / np.reshape(amp_max[ind_f, f_vec], [-1, 1])
                ) * a_[tp]
            elif cfg_eBOSC["postproc.effSignal"] == "PT":
                tmp_bias = (
                    (tmp_biasVec / np.reshape(amp_max[ind_f, f_vec], [-1, 1]))
                    * (a_[tp] - cfg_eBOSC["tmp.pt"][ind_f])
                ) + cfg_eBOSC["tmp.pt"][ind_f]

            # Compare to data
            m_[tp, :] = np.transpose(a_ >= tmp_bias)

        # Identify which time points to retain and discard
        keep = m_.sum(0) == len(a_)
        if cfg_eBOSC["postproc.edgeOnly"] == "yes":
            keepEdgeRemovalOnly = np.zeros([len(keep)], dtype=bool)
            keepEdgeRemovalOnly[
                np.arange(np.where(keep == 1)[0][0], np.where(keep == 1)[0][-1] + 1)
            ] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly

        # Get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))

        if max(d_keep) == 1 and min(d_keep) == -1:
            # Start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0] - 1
            for i in range(len(ind_epsd_begin)):
                # Check for passing the duration requirement
                tmp_col = np.arange(ind_epsd_begin[i], ind_epsd_end[i] + 1)
                avg_frq = np.mean(f_[tmp_col])
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC["F"], avg_frq)
                num_pnt = np.floor(
                    (cfg_eBOSC["fsample"] / avg_frq)
                    * int(np.reshape(cfg_eBOSC["threshold.duration"], [-1])[indF])
                )

                if len(tmp_col) >= num_pnt:
                    # Update all data in table with new episode limits
                    episodesTable["RowID"].append(episodes["RowID"][e][tmp_col])
                    episodesTable["ColID"].append(
                        [t_ind[tmp_col[0]], t_ind[tmp_col[-1]]]
                    )
                    episodesTable["Frequency"].append(f_[tmp_col])
                    episodesTable["FrequencyMean"].append(
                        np.mean(episodesTable["Frequency"][-1])
                    )
                    episodesTable["Power"].append(a_[tmp_col])
                    episodesTable["PowerMean"].append(
                        np.mean(episodesTable["Power"][-1])
                    )
                    episodesTable["DurationS"].append(
                        np.diff(episodesTable["ColID"][-1])[0] / cfg_eBOSC["fsample"]
                    )
                    episodesTable["DurationC"].append(
                        episodesTable["DurationS"][-1]
                        * episodesTable["FrequencyMean"][-1]
                    )
                    episodesTable["Trial"].append(cfg_eBOSC["tmp_trial"])
                    episodesTable["Channel"].append(cfg_eBOSC["tmp_channel"])
                    episodesTable["Onset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][0])]
                    )
                    episodesTable["Offset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][-1])]
                    )
                    episodesTable["SNR"].append(episodes["SNR"][e][tmp_col])
                    episodesTable["SNRMean"].append(np.mean(episodesTable["SNR"][-1]))
                    detected_new[episodesTable["RowID"][-1], t_ind[tmp_col]] = 1

    # Return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new


def eBOSC_episode_rm_shoulder(cfg_eBOSC, detected1, episodes):
    """Remove parts of the episode that fall into the 'shoulder' of individual
    trials. There is no check for adherence to a given duration criterion necessary,
    as the point of the padding of the detected matrix is exactly to account
    for allowing the presence of a few cycles.
    """

    print("Removing padding from detected episodes")

    ind1 = cfg_eBOSC["pad.detection_sample"]
    ind2 = detected1.shape[1] - cfg_eBOSC["pad.detection_sample"]
    rmv = []
    for j in range(len(episodes["Trial"])):
        # get time points of current episode
        tmp_col = np.arange(episodes["ColID"][j][0], episodes["ColID"][j][1] + 1)
        # find time points that fall inside the padding (i.e. on- and offset)
        ex = np.where(np.logical_or(tmp_col < ind1, tmp_col >= ind2))[0]
        # remove padded time points from episodes
        tmp_col = np.delete(tmp_col, ex)
        episodes["RowID"][j] = np.delete(episodes["RowID"][j], ex)
        episodes["Power"][j] = np.delete(episodes["Power"][j], ex)
        episodes["Frequency"][j] = np.delete(episodes["Frequency"][j], ex)
        episodes["SNR"][j] = np.delete(episodes["SNR"][j], ex)
        # if nothing remains of episode: retain for later deletion
        if len(tmp_col) == 0:
            rmv.append(j)
        else:
            # shift onset according to padding
            # Important: new col index is indexing w.r.t. to matrix AFTER
            # detected padding is removed!
            tmp_col = tmp_col - ind1
            episodes["ColID"][j] = [tmp_col[0], tmp_col[-1]]
            # re-compute mean frequency
            episodes["FrequencyMean"][j] = np.mean(episodes["Frequency"][j])
            # re-compute mean amplitude
            episodes["PowerMean"][j] = np.mean(episodes["Power"][j])
            # re-compute mean SNR
            episodes["SNRMean"][j] = np.mean(episodes["SNR"][j])
            # re-compute duration
            episodes["DurationS"][j] = (
                np.diff(episodes["ColID"][j])[0] / cfg_eBOSC["fsample"]
            )
            episodes["DurationC"][j] = (
                episodes["DurationS"][j] * episodes["FrequencyMean"][j]
            )
            # update absolute on-/offsets (should remain the same)
            episodes["Onset"][j] = cfg_eBOSC["time.time_det"][
                int(episodes["ColID"][j][0])
            ]
            episodes["Offset"][j] = cfg_eBOSC["time.time_det"][
                int(episodes["ColID"][j][-1])
            ]
    # remove now empty episodes from table
    for entry in episodes:
        # https://stackoverflow.com/questions/21032034/deleting-multiple-indexes-from-a-list-at-once-python
        episodes[entry] = [v for i, v in enumerate(episodes[entry]) if i not in rmv]
    return episodes


########################### v2 code for fBOSC ###########################
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, find
from collections import deque
import numpy as np
from scipy.sparse import find as sparse_find
from collections import deque

# ── helpers ──────────────────────────────────────────────────────────────────


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def _process_episode_serial_fast(
    seed_x, seed_y, detected_remaining, TFR_, tmp_B1, cfg_fBOSC, fBOSC, time_offset=0
):
    """
    Same logic as before but:
    - no argwhere inside the loop
    - tmp_B1 row lookup uses direct indexing (no clip in inner loop)
    - early exit if seed already consumed by a prior episode
    """
    if detected_remaining[seed_x, seed_y] == 0:
        return None  # already consumed by a previous episode

    fstp = cfg_fBOSC["fstp"]
    nfreq, ntime = TFR_.shape

    x, y = deque([seed_x]), deque([seed_y])
    detected_remaining[seed_x, seed_y] = 0  # consume seed immediately

    while True:
        next_point = int(y[-1]) + 1
        if next_point >= ntime:
            break

        cur_x = int(x[-1])
        f0 = max(0, cur_x - fstp)
        f1 = min(nfreq, cur_x + fstp + 1)
        next_freqs = np.arange(f0, f1, dtype=np.int32)

        hits = next_freqs[detected_remaining[next_freqs, next_point] == 1]

        if hits.size == 0:
            break

        y.append(next_point)
        if hits.size > 1:
            vals = tmp_B1[hits + fstp, next_point]
            best = hits[np.argmax(vals)]
        else:
            best = hits[0]

        x.append(best)
        detected_remaining[best, next_point] = 0  # consume as we grow

    # --- duration check ---
    x_arr = np.asarray(x, dtype=np.int32) - fstp
    x_arr = np.clip(x_arr, 0, nfreq - 1)
    avg_frq = float(np.mean(cfg_fBOSC["F"][x_arr]))

    F_arr = np.asarray(cfg_fBOSC["F"])
    indF = int(np.argmin(np.abs(F_arr - avg_frq)))
    num_pnt = int(
        np.floor(
            (cfg_fBOSC["fsample"] / avg_frq)
            * cfg_fBOSC["threshold.duration"].flatten()[indF]
        )
    )

    n_cols = len(y)
    if n_cols < num_pnt:
        return None  # too short — cells already consumed, just skip

    y_arr = np.arange(int(y[0]), int(y[-1]) + 1, dtype=np.int32)
    y_arr = np.clip(y_arr, 0, ntime - 1)
    power = np.single(TFR_[x_arr, y_arr])

    g0 = int(y[0]) + time_offset
    g1 = int(y[-1]) + time_offset
    pt = fBOSC["static"]["pt"]

    return {
        "RowID": x_arr,
        "ColID": [g0, g1],
        "Frequency": F_arr[x_arr],
        "FrequencyMean": avg_frq,
        "Power": power,
        "PowerMean": float(np.mean(power)),
        "DurationS": n_cols / cfg_fBOSC["fsample"],
        "DurationC": n_cols / cfg_fBOSC["fsample"] * avg_frq,
        "Trial": cfg_fBOSC["tmp_trial"],
        "Channel": cfg_fBOSC["tmp_channel"],
        "Onset": cfg_fBOSC["time.time_tfr"][g0],
        "Offset": cfg_fBOSC["time.time_tfr"][g1],
        "SNR": power / pt[x_arr],
        "SNRMean": float(np.mean(power / pt[x_arr])),
    }


def fBOSC_episode_create_segment(cfg_fBOSC, TFR_, detected, fBOSC, time_offset=0):
    detected = eBOSC_episode_sparsefreq(cfg_fBOSC, detected, TFR_)

    cfg_fBOSC = dict(cfg_fBOSC)
    cfg_fBOSC["fstp"] = 1

    nfreq, ntime = TFR_.shape
    fstp = cfg_fBOSC["fstp"]

    padding = np.zeros((fstp, ntime), dtype=TFR_.dtype)
    tmp_B1 = np.vstack([padding, TFR_ * detected, padding])
    tmp_B1[:, 0] = 0
    tmp_B1[:, -1] = 0

    # ── key fix: precompute ALL seeds once, column-major order ──
    # Column-major means we visit time left-to-right, which matches
    # the episode-growing direction and maximises cache locality.
    detected_remaining = np.asfortranarray((detected > 0).astype(np.uint8))
    seeds_row, seeds_col = np.where(detected_remaining)  # one call, O(n)

    # Sort by time (col) first, then frequency (row) — natural episode order
    order = np.lexsort((seeds_row, seeds_col))
    seeds_row = seeds_row[order]
    seeds_col = seeds_col[order]

    episodes = []
    detected_new = np.zeros((nfreq, ntime), dtype=np.uint8)

    for sx, sy in zip(seeds_row, seeds_col):
        # Skip if already consumed by a prior episode
        if detected_remaining[sx, sy] == 0:
            continue

        result = _process_episode_serial_fast(
            sx, sy, detected_remaining, TFR_, tmp_B1, cfg_fBOSC, fBOSC, time_offset
        )

        if result is not None:
            episodes.append(result)
            r = result["RowID"]
            c0 = result["ColID"][0] - time_offset
            c1 = result["ColID"][1] - time_offset
            detected_new[r, np.arange(c0, c1 + 1)] = 1

    return episodes, detected_new


# ── merge / stitch episodes across segment boundaries ────────────────────────


def _episodes_overlap(ep_a, ep_b, freq_tol=1):
    """
    Return True when ep_a and ep_b could be the same oscillatory episode
    split across a segment boundary.

    Criteria
    --------
    1. Temporal adjacency  : ep_b starts within 1 sample of ep_a ending.
    2. Frequency proximity : mean frequencies agree within `freq_tol` Hz.
    """
    temporal_adj = ep_b["ColID"][0] <= ep_a["ColID"][1] + 1
    freq_close = abs(ep_a["FrequencyMean"] - ep_b["FrequencyMean"]) <= freq_tol
    return temporal_adj and freq_close


def _merge_two(ep_a, ep_b, cfg_fBOSC, fBOSC):
    """Fuse two episode dicts into one, properly handling temporal overlap."""

    # Calculate how many columns actually overlap
    overlap_count = max(0, ep_a["ColID"][1] - ep_b["ColID"][0] + 1)

    # Slice the arrays from ep_b to remove duplicated data
    b_row_ids = ep_b["RowID"][overlap_count:]
    b_power = ep_b["Power"][overlap_count:]
    b_freq = ep_b["Frequency"][overlap_count:]

    # Safely concatenate without duplicates
    row_ids = np.concatenate([ep_a["RowID"], b_row_ids])
    power = np.concatenate([ep_a["Power"], b_power])
    freqs = np.concatenate([ep_a["Frequency"], b_freq])

    avg_frq = float(np.mean(freqs))

    # Recalculate duration using the true length
    total_len = len(power)
    duration_s = total_len / cfg_fBOSC["fsample"]

    return {
        "RowID": row_ids,
        "ColID": [ep_a["ColID"][0], max(ep_a["ColID"][1], ep_b["ColID"][1])],
        "Frequency": freqs,
        "FrequencyMean": avg_frq,
        "Power": power,
        "PowerMean": float(np.mean(power)),
        "DurationS": duration_s,
        "DurationC": duration_s * avg_frq,
        "Trial": ep_a["Trial"],
        "Channel": ep_a["Channel"],
        "Onset": ep_a["Onset"],
        "Offset": max(ep_a["Offset"], ep_b["Offset"]),
        "SNR": power / fBOSC["static"]["pt"][row_ids],
        "SNRMean": float(np.mean(power / fBOSC["static"]["pt"][row_ids])),
    }


def merge_segment_episodes(all_episodes_per_seg, cfg_fBOSC, fBOSC, freq_tol=1):
    """
    Given a list-of-lists (one list per segment, each element a dict),
    stitch episodes that were split across boundaries and return a flat list.

    Algorithm
    ---------
    Process segments left-to-right. For every episode that ends at or near the
    right edge of the current segment, check whether any episode in the next
    segment starts at (or just after) that edge AND has a compatible frequency.
    If so, merge them. Duplicate episodes that fall entirely inside the overlap
    zone are deduplicated by keeping the one with higher mean power.
    """
    if not all_episodes_per_seg:
        return []

    merged = list(all_episodes_per_seg[0])  # start with segment 0

    for seg_episodes in all_episodes_per_seg[1:]:
        next_eps = list(seg_episodes)
        final_eps = []

        for ep_next in next_eps:
            stitched = False
            for i, ep_prev in enumerate(merged):
                if _episodes_overlap(ep_prev, ep_next, freq_tol):
                    merged[i] = _merge_two(ep_prev, ep_next, cfg_fBOSC, fBOSC)
                    stitched = True
                    break
            if not stitched:
                final_eps.append(ep_next)

        merged.extend(final_eps)

    # Deduplicate using integer boundaries and a slightly looser frequency match
    seen = set()
    deduped = []

    for ep in merged:
        # Integers are perfectly safe for set hashing
        start_idx = ep["ColID"][0]
        end_idx = ep["ColID"][1]

        # Rounding to 1 decimal gives a bit more breathing room for minor float variations
        freq_key = round(ep["FrequencyMean"], 1)

        key = (start_idx, end_idx, freq_key)

        if key not in seen:
            seen.add(key)
            deduped.append(ep)

    return deduped


# ── public entry point ────────────────────────────────────────────────────────


def fBOSC_episode_create_v3(
    cfg_fBOSC, TFR_, detected, fBOSC, segment_len_s=60, overlap_s=5
):
    """
    Memory-safe, crash-proof replacement for fBOSC_episode_create_v2.

    Key changes vs v2
    -----------------
    * Serial episode processing (eliminates joblib shared-state race condition).
    * Dense numpy arrays instead of lil_matrix (no sparse overhead or .toarray() blowup).
    * Power extracted per-element instead of via huge index broadcasting.
    * Long signals are split into overlapping segments; boundary episodes are
      stitched back together automatically.

    Parameters
    ----------
    cfg_fBOSC : dict
    TFR_      : ndarray (nfreq x ntime)  - wavelet-padding already removed
    detected  : ndarray (nfreq x ntime)  - binary
    fBOSC     : dict
    segment_len_s : float  - target segment length in seconds (default 60 s)
    overlap_s     : float  - overlap on each side in seconds (default 5 s)
                    Rule of thumb: set to the longest expected oscillatory
                    episode duration you care about.

    Returns
    -------
    episodesTable : dict   - flat table with one entry per episode
    detected_new  : ndarray (nfreq x ntime)
    """
    fs = cfg_fBOSC["fsample"]
    nfreq, ntime = TFR_.shape

    seg_len = int(segment_len_s * fs)
    overlap = int(overlap_s * fs)

    # Build segment boundaries (start, end) in sample indices
    starts = list(range(0, ntime, seg_len))
    segments = []
    for s in starts:
        t0 = max(0, s - overlap)
        t1 = min(ntime, s + seg_len + overlap)
        segments.append((t0, t1))

    print(
        f"Processing {len(segments)} segments "
        f"(seg_len={segment_len_s}s, overlap={overlap_s}s) …"
    )

    all_episodes_per_seg = []
    detected_new_global = np.zeros((nfreq, ntime), dtype=np.uint8)

    for seg_idx, (t0, t1) in enumerate(segments):
        print(
            f"  Segment {seg_idx+1}/{len(segments)}: " f"samples [{t0}, {t1}) …",
            end=" ",
            flush=True,
        )

        TFR_seg = TFR_[:, t0:t1]
        detected_seg = detected[:, t0:t1].copy()

        eps_seg, det_new_seg = fBOSC_episode_create_segment(
            cfg_fBOSC, TFR_seg, detected_seg, fBOSC, time_offset=t0
        )

        print(f"{len(eps_seg)} episodes found.")

        all_episodes_per_seg.append(eps_seg)

        # Accumulate detected_new in global coords
        detected_new_global[:, t0:t1] = np.maximum(
            detected_new_global[:, t0:t1], det_new_seg
        )

    # Stitch boundary episodes
    print("Stitching episodes across segment boundaries …")
    all_episodes = merge_segment_episodes(all_episodes_per_seg, cfg_fBOSC, fBOSC)
    print(f"Total episodes after merge: {len(all_episodes)}")

    # Pack into flat episodesTable dict
    episodesTable = {
        k: []
        for k in [
            "RowID",
            "ColID",
            "Frequency",
            "FrequencyMean",
            "Power",
            "PowerMean",
            "DurationS",
            "DurationC",
            "Trial",
            "Channel",
            "Onset",
            "Offset",
            "SNR",
            "SNRMean",
        ]
    }

    for ep in all_episodes:
        for k in episodesTable:
            episodesTable[k].append(ep[k])

    return episodesTable, detected_new_global


def eBOSC_episode_postproc_fwhm_v2(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can trivially be explained by the FWHM of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs:
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs:
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    """

    print("Applying FWHM post-processing ...")

    # re-initialize detected_new (for post-proc results)
    detected_new = np.zeros(TFR.shape)
    # initialize new dictionary to save results in
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []

    for e in range(len(episodes["Trial"])):
        # get temporary frequency vector
        f_ = episodes["Frequency"][e]
        f_unique = np.unique(f_)
        # find index within minor tolerance (float arrays)
        # f_ind_unique = np.where(np.abs(cfg_eBOSC['F'][:,None] - f_unique) < 1e-5)
        # f_ind_unique = f_ind_unique[0]
        f_ind_unique = np.array(
            [int(np.argmin(np.abs(cfg_eBOSC["F"] - fv))) for fv in f_unique], dtype=int
        )
        # get temporary amplitude vector
        a_ = episodes["Power"][e]
        # location in time with reference to matrix TFR
        t_ind = np.int_(
            np.arange(episodes["ColID"][e][0], episodes["ColID"][e][-1] + 1)
        )
        # initiate bias matrix (only requires to encode frequencies occuring within episode)
        biasMat = np.zeros([len(f_unique), len(a_)])

        for tp in range(len(a_)):
            # The FWHM correction is done independently at each
            # frequency. To accomplish this, we actually reference
            # to the original data in the TF matrix.
            # search within frequencies that occur within the episode
            for f in range(len(f_unique)):
                # create wavelet with center frequency and amplitude at time point
                st = 1 / (2 * np.pi * (f_unique / cfg_eBOSC["wavenumber"]))
                step_size = 1 / cfg_eBOSC["fsample"]
                t = np.arange(-3.6 * st[f], 3.6 * st[f] + step_size, step_size)
                wave = np.exp(-(t**2) / (2 * st[f] ** 2)) * np.exp(
                    1j * 2 * np.pi * f_unique[f] * t
                )
                freq_idx = int(f_ind_unique[f])
                col_idx = int(t_ind[tp])
                if (
                    freq_idx < 0
                    or freq_idx >= TFR.shape[0]
                    or col_idx < 0
                    or col_idx >= TFR.shape[1]
                ):
                    continue
                if cfg_eBOSC["postproc.effSignal"] == "all":
                    # Morlet wavelet with amplitude-power threshold modulation
                    m = TFR[freq_idx, col_idx] * wave
                elif cfg_eBOSC["postproc.effSignal"] == "PT":
                    m = (TFR[freq_idx, col_idx] - cfg_eBOSC["tmp.pt"][freq_idx]) * wave

                # amplitude of wavelet
                wl_a = abs(m)
                maxval = max(wl_a)
                maxloc = np.where(np.abs(wl_a[:, None] - maxval) < 1e-5)[0][0]
                index_fwhm = np.where(wl_a >= maxval / 2)[0][0]

                # amplitude at fwhm, freq
                fwhm_a = wl_a[index_fwhm]
                if cfg_eBOSC["postproc.effSignal"] == "PT":
                    # re-add power threshold
                    fwhm_a = fwhm_a + cfg_eBOSC["tmp.pt"][f_ind_unique[f]]
                correctionDist = maxloc - index_fwhm
                # extract FWHM amplitude of frequency- and amplitude-specific wavelet
                # check that lower fwhm is part of signal
                if tp - correctionDist >= 0:
                    # and that existing value is lower than update
                    if biasMat[f, tp - correctionDist] < fwhm_a:
                        biasMat[f, tp - correctionDist] = fwhm_a
                # check that upper fwhm is part of signal
                if tp + correctionDist + 1 <= biasMat.shape[1]:
                    # and that existing value is lower than update
                    if biasMat[f, tp + correctionDist] < fwhm_a:
                        biasMat[f, tp + correctionDist] = fwhm_a

        # plt.imshow(biasMat, extent=[0, 1, 0, 1])

        # retain only those points that are larger than the FWHM
        aMat_retain = np.zeros(biasMat.shape)
        indFreqs = np.where(np.abs(f_[:, None] - f_unique) < 1e-5)
        indFreqs = indFreqs[1]
        for indF in range(len(f_unique)):
            aMat_retain[indF, np.where(indFreqs == indF)[0]] = np.transpose(
                a_[indFreqs == indF]
            )
        # anything that is lower than the convolved wavelet is removed
        aMat_retain[aMat_retain <= biasMat] = 0

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = aMat_retain.mean(0) > 0
        # keep = keep>0

        # print(
        # f"Episode {e}: {np.sum(keep)} / {len(keep)} time points retained #after FWHM correction."
        # )

        # if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
        #     keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
        #     keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
        #     keep = keepEdgeRemovalOnly
        #     del keepEdgeRemovalOnly

        if cfg_eBOSC["postproc.edgeOnly"] == "yes":
            keep_inds = np.where(keep == 1)[0]
            if len(keep_inds) == 0:  # ← fix 1: nothing survived FWHM
                continue
            keepEdgeRemovalOnly = np.zeros(len(keep), dtype=bool)
            keepEdgeRemovalOnly[np.arange(keep_inds[0], keep_inds[-1] + 1)] = True
            keep = keepEdgeRemovalOnly

        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))

        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0] - 1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i], ind_epsd_end[i] + 1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC["F"], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    num_pnt = np.floor(
                        (cfg_eBOSC["fsample"] / avg_frq)
                        * int(
                            np.reshape(cfg_eBOSC["threshold.duration"], [-1])[indF]
                        )
                    )
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable["RowID"].append(episodes["RowID"][e][tmp_col])
                    episodesTable["ColID"].append(
                        [t_ind[tmp_col[0]], t_ind[tmp_col[-1]]]
                    )
                    episodesTable["Frequency"].append(f_[tmp_col])
                    episodesTable["FrequencyMean"].append(
                        np.mean(episodesTable["Frequency"][-1])
                    )
                    episodesTable["Power"].append(a_[tmp_col])
                    episodesTable["PowerMean"].append(
                        np.mean(episodesTable["Power"][-1])
                    )
                    episodesTable["DurationS"].append(
                        np.diff(episodesTable["ColID"][-1])[0] / cfg_eBOSC["fsample"]
                    )
                    episodesTable["DurationC"].append(
                        episodesTable["DurationS"][-1]
                        * episodesTable["FrequencyMean"][-1]
                    )
                    episodesTable["Trial"].append(cfg_eBOSC["tmp_trial"])
                    episodesTable["Channel"].append(cfg_eBOSC["tmp_channel"])
                    episodesTable["Onset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][0])]
                    )
                    episodesTable["Offset"].append(
                        cfg_eBOSC["time.time_tfr"][int(episodesTable["ColID"][-1][-1])]
                    )
                    episodesTable["SNR"].append(episodes["SNR"][e][tmp_col])
                    episodesTable["SNRMean"].append(np.mean(episodesTable["SNR"][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable["RowID"][-1], t_ind[tmp_col]] = 1

    # plt.imshow(detected_new, extent=[0, 1, 0, 1])
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new


def plot_episodetable_stats(
    animal,
    day,
    session,
    tetrode_num,
    channel_num,
    episodesTable,
    post_proc=True,
    results_root=None,
    save=False,
):
    if post_proc == True:
        postproc = "postproc"
    else:
        postproc = "raw"

    freq_mean = np.array(episodesTable["FrequencyMean"])
    duration = np.array(episodesTable["DurationS"])

    # Duration mask for 3–6 Hz episodes
    theta_mask = (freq_mean >= 3) & (freq_mean <= 6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: frequency mean ──────────────────────────────────────────────────────
    axes[0].hist(
        freq_mean, bins=50, color="steelblue", edgecolor="white", linewidth=0.5
    )
    axes[0].set_xlabel("Frequency mean (Hz)")
    axes[0].set_ylabel("Episode count")
    axes[0].set_title("Episode frequency distribution")

    # ── Right: duration (all vs 3–6 Hz) ──────────────────────────────────────────
    bins = np.linspace(0, np.percentile(duration, 99), 60)  # clip extreme tail

    axes[1].hist(
        duration,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.6,
        label="All episodes",
    )
    axes[1].hist(
        duration[theta_mask],
        bins=bins,
        color="tomato",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        label="3-6 Hz episodes",
    )
    axes[1].set_xlabel("Duration (s)")
    axes[1].set_ylabel("Episode count")
    axes[1].set_title("Episode duration distribution")
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"Animal {animal}, {day}, Session {session} \n Tetrode {tetrode_num}, Channel {channel_num} - {postproc} Episode Statistics",
        fontsize=16,
    )

    fig.tight_layout()
    if save:
        os.makedirs(results_root, exist_ok=True)
        save_path_png = os.path.join(
            results_root, f"{tetrode_num}_{channel_num}_{postproc}_fBOSC_summary.png"
        )
        save_path_svg = os.path.join(
            results_root, f"{tetrode_num}_{channel_num}_{postproc}_fBOSC_summary.svg"
        )
        fig.savefig(save_path_png, bbox_inches="tight", dpi=300)
        fig.savefig(save_path_svg, bbox_inches="tight", dpi=300)
    else:
        plt.show()

    plt.close(fig)
