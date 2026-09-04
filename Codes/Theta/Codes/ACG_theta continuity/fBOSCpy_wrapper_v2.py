import numpy as np
import pandas as pd
import os

try:
    from fBOSCpy import fBOSCUtils
except ModuleNotFoundError:
    # Fallback for running scripts from inside the fBOSCpy/ folder itself
    # (rather than importing fBOSCpy as a package from its parent directory).
    import fBOSCUtils


def fBOSCpy_wrapper_v2(
    animal,
    day,
    session,
    tetrode_num,
    channel_num,
    lfp_samples,
    lfp_timestamps,
    F_array,
    Fs,
    postproc=True,
    min_ncycles=3,
    min_duration_s=0.15,
    plot_histogram=True,
    results_root=None,
    return_diagnostics=False,
):
    """
    Wrapper function to run fBOSCpy on LFP signals.

    min_ncycles, min_duration_s: together set the per-frequency episode
        duration threshold as dt_cycles(F) = max(min_ncycles, min_duration_s * F)
        -- i.e. an episode must last at least min_ncycles cycles AND at least
        min_duration_s seconds, whichever is stricter at that frequency. Below
        the crossover frequency (min_ncycles / min_duration_s) this reduces to
        the classic fixed-cycle-count rule; above it, progressively more cycles
        are required so the absolute duration floor still holds. Without this,
        a fixed cycle count becomes an ever-shorter absolute-time requirement as
        F grows (e.g. 3 cycles at 100 Hz is only 30 ms), letting brief noise or
        artifact transients satisfy the duration criterion purely because F is
        high -- inflating detected episodes at the top of F_array regardless of
        whether a genuine oscillation is present there.

    If return_diagnostics=True, also returns a dict with the intermediate
    objects needed for per-file QC plots (TFR heatmap, background/FOOOF fit):
        "TFR"        | time-frequency matrix, padding trimmed [freq x time]
        "time_tfr"   | time vector (s) matching TFR's time axis
        "fBOSC"      | fBOSC["static"] holds bg_pow/bg_log10_pow/mp/pt/ap_fit/etc.
        "cfg_fBOSC"  | full config dict used for the run
    """
    ##### Step 0 #####: Set up fBOSC configs

    # general setup
    cfg_fBOSC = {}
    cfg_fBOSC["F"] = F_array  # np.arange(1, 40.5, 1)  # frequency sampling
    cfg_fBOSC["wavenumber"] = 8  # wavelet parameter (time-frequency tradeoff)
    cfg_fBOSC["fsample"] = Fs  # 1000  # current sampling frequency of EEG data

    # padding
    cfg_fBOSC["pad.tfr_s"] = (
        1  # padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
    )
    cfg_fBOSC["pad.detection_s"] = (
        0.5  # padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
    )
    cfg_fBOSC["pad.background_s"] = (
        1  # padding of segments for BG (only avoiding edge artifacts)
    )

    # fooof parameters
    cfg_fBOSC["fooof"] = {}
    cfg_fBOSC["fooof"]["peak_width_limits"] = [1.0, 8.0]  # peak width limits in Hz
    cfg_fBOSC["fooof"]["max_n_peaks"] = 6  # maximum number of peaks to fit
    cfg_fBOSC["fooof"]["min_peak_height"] = 0.1  # minimum peak height
    cfg_fBOSC["fooof"]["peak_threshold"] = 2.0  # peak threshold
    cfg_fBOSC["fooof"]["aperiodic_mode"] = "knee"
    cfg_fBOSC["fooof"]["verbose"] = True  # verbosity of fooof output

    # threshold settings
    # Duration threshold, in cycles, with an absolute-time floor:
    #   dt_cycles(F) = max(min_ncycles, min_duration_s * F)
    # See the min_ncycles / min_duration_s docstring entry above for rationale.
    # np.ceil rounds up to a whole cycle count -- several downstream functions
    # in fBOSCUtils.py do int(threshold.duration[f]) when re-checking duration
    # (e.g. during FWHM post-processing), which would silently truncate a
    # fractional value like 3.75 down to 3 and quietly weaken the floor;
    # rounding up here keeps it a no-op everywhere it gets cast to int.
    cfg_fBOSC["threshold.duration"] = np.ceil(
        np.maximum(min_ncycles, min_duration_s * cfg_fBOSC["F"])
    ).reshape(1, -1)  # vector of duration thresholds (in cycles) at each frequency
    cfg_fBOSC["threshold.percentile"] = (
        0.95  # percentile of background fit for power threshold
    )

    # episode post processing
    if postproc:
        cfg_fBOSC["postproc.use"] = "yes"
    else:
        cfg_fBOSC["postproc.use"] = "no"
    cfg_fBOSC["postproc.method"] = "FWHM"  # 'MaxBias' or 'FWHM'
    cfg_fBOSC["postproc.edgeOnly"] = (
        "yes"  # Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
    )
    cfg_fBOSC["postproc.effSignal"] = "PT"

    # calculate the sample points for padding
    cfg_fBOSC["pad.tfr_sample"] = int(
        cfg_fBOSC["pad.tfr_s"] * cfg_fBOSC["fsample"]
    )  # automatic sample point calculation
    cfg_fBOSC["pad.detection_sample"] = int(
        cfg_fBOSC["pad.detection_s"] * cfg_fBOSC["fsample"]
    )
    cfg_fBOSC["pad.total_s"] = (
        cfg_fBOSC["pad.tfr_s"] + cfg_fBOSC["pad.detection_s"]
    )  # complete padding (WL + shoulder)
    cfg_fBOSC["pad.total_sample"] = int(
        cfg_fBOSC["pad.tfr_sample"] + cfg_fBOSC["pad.detection_sample"]
    )
    cfg_fBOSC["pad.background_sample"] = int(cfg_fBOSC["pad.tfr_sample"])

    n_freq = len(cfg_fBOSC["F"])  # number of frequencies
    n_time = len(lfp_samples)
    n_time_total = len(lfp_samples)
    n_trial = 1

    cfg_fBOSC["time.time_total"] = lfp_timestamps  # total time vector

    ##### Step 1 #####: time-frequency wavelet decomposition for whole signal to prepare background fit
    TFR = np.zeros((1, n_freq, n_time))

    [TFR[0, :, :], tmp, tmp] = fBOSCUtils.BOSC_tf(
        lfp_samples, cfg_fBOSC["F"], cfg_fBOSC["fsample"], cfg_fBOSC["wavenumber"]
    )
    del tmp
    # plt.imshow(TFR[0,:,:], extent=[0,1,0,1])

    print("Step 1: Time-frequency wavelet decomposition completed successfully.")

    ##### Step 2 #####: background fit for power threshold using fooof
    fBOSC = {}
    fBOSC["static"] = {}
    fBOSC["static"]["bg_pow"] = pd.DataFrame(columns=cfg_fBOSC["F"])
    fBOSC["static"]["bg_log10_pow"] = pd.DataFrame(columns=cfg_fBOSC["F"])
    fBOSC["static"]["pv"] = pd.DataFrame(columns=["slope", "intercept"])
    fBOSC["static"]["mp"] = pd.DataFrame(columns=cfg_fBOSC["F"])
    fBOSC["static"]["pt"] = pd.DataFrame(columns=cfg_fBOSC["F"])

    [fBOSC, pt, dt] = fBOSCUtils.fBOSC_getThresholds(cfg_fBOSC, TFR, fBOSC)

    print("Step 2: Background power fit completed successfully.")

    ##### Step 3 #####: detect rhythms and calculate Pepisode

    # get timing and info for post-TFR padding removal
    tfr_time2extract = np.arange(
        cfg_fBOSC["pad.tfr_sample"] + 1,
        n_time_total - cfg_fBOSC["pad.tfr_sample"] + 1,
        1,
    )
    cfg_fBOSC["time.time_tfr"] = cfg_fBOSC["time.time_total"][tfr_time2extract]
    n_time_tfr = len(cfg_fBOSC["time.time_tfr"])
    # get timing and info for post-detected padding removal
    det_time2extract = np.arange(
        cfg_fBOSC["pad.detection_sample"] + 1,
        n_time_tfr - cfg_fBOSC["pad.detection_sample"] + 1,
        1,
    )
    cfg_fBOSC["time.time_det"] = cfg_fBOSC["time.time_tfr"][det_time2extract]
    n_time_det = len(cfg_fBOSC["time.time_det"])

    time2extract = np.arange(
        cfg_fBOSC["pad.tfr_sample"] + 1,
        TFR.shape[2] - cfg_fBOSC["pad.tfr_sample"] + 1,
        1,
    )
    TFR_ = np.transpose(TFR[0, :, time2extract], [1, 0])

    detected = np.zeros((TFR_.shape))
    for f in range(len(cfg_fBOSC["F"])):
        detected[f, :] = fBOSCUtils.BOSC_detect(
            TFR_[f, :], pt[f], dt[0][f], cfg_fBOSC["fsample"]
        )

    # remove padding for detection (matrix with padding required for refinement)
    time2encode = np.arange(
        cfg_fBOSC["pad.detection_sample"],
        detected.shape[1] - cfg_fBOSC["pad.detection_sample"],
        1,
    )

    # Multiindex for channel x trial x frequency x time
    arrays = np.array([1, 1, cfg_fBOSC["F"], cfg_fBOSC["time.time_det"]], dtype=object)
    # tuples = list(zip(*arrays))
    names = ["channel", "trial", "frequency", "time"]
    # Ensure all elements in `arrays` are list-like
    arrays = [
        arr if isinstance(arr, (list, tuple, np.ndarray)) else [arr] for arr in arrays
    ]
    # Create the MultiIndex
    index = pd.MultiIndex.from_product(arrays, names=names)
    nullData = np.zeros(
        len(arrays[0]) * len(arrays[1]) * len(arrays[2]) * len(arrays[3])
    )
    fBOSC["detected"] = pd.DataFrame(data=nullData, index=index)
    fBOSC["detected_ep"] = fBOSC["detected"].copy()
    del nullData, index

    fBOSC["detected"].loc[(1, 1)] = np.reshape(detected[:, time2encode], [-1, 1])

    print("Step 3: Rhythm detection completed successfully.")

    ##### Step 4 #####: create a table of detected continuous episodes

    cfg_fBOSC["tmp_trial"] = 1
    cfg_fBOSC["tmp_channel"] = 1
    cfg_fBOSC["tmp_channelID"] = (
        0  # zero-based indexing for channel in fBOSC['static']['pt']
    )

    # episodesTable, detected_new = fBOSCUtils.fBOSC_episode_create(
    #    cfg_fBOSC, TFR_, detected, fBOSC
    # )

    episodesTable, detected_new = fBOSCUtils.fBOSC_episode_create_v3(
        cfg_fBOSC,
        TFR_,
        detected,
        fBOSC,
        segment_len_s=60,  # process in 60-second chunks
        overlap_s=5,  # 5-second overlap on each side of each chunk
    )

    print("Step 4: Episode table created successfully.")

    ##### Step 5 #####: post-processing of episodes

    # temporarily pass on power threshold for easier access
    cfg_fBOSC["tmp.pt"] = fBOSC["static"]["pt"]

    # Default to the raw (pre-postproc) results so that channels with zero
    # detected episodes return empty tables instead of raising
    # UnboundLocalError below.
    episodesTablePostProc = episodesTable
    detected_new_PostProc = detected_new

    # only do this if there are any episodes to fine-tune
    if cfg_fBOSC["postproc.use"] == "yes" and len(episodesTable["Trial"]) > 0:
        if cfg_fBOSC["postproc.method"] == "FWHM":
            [episodesTablePostProc, detected_new_PostProc] = (
                fBOSCUtils.eBOSC_episode_postproc_fwhm_v2(
                    cfg_fBOSC, episodesTable, TFR_
                )
            )
        elif cfg_fBOSC["postproc.method"] == "MaxBias":
            [episodesTable, detected_new] = (
                fBOSCUtils.eBOSC_episode_postproc_maxbias_memmap(
                    cfg_fBOSC, episodesTable, TFR_
                )
            )

    # remove episodes and part of episodes that fall into 'shoulder'
    if len(episodesTable["Trial"]) > 0 and cfg_fBOSC["pad.detection_sample"] > 0:
        episodesTablePostProc = fBOSCUtils.eBOSC_episode_rm_shoulder(
            cfg_fBOSC, detected_new_PostProc, episodesTablePostProc
        )

    print("Step 5: Episode post-processing completed successfully.")

    if plot_histogram:
        fBOSCUtils.plot_episodetable_stats(
            animal,
            day,
            session,
            tetrode_num,
            channel_num,
            episodesTable,
            post_proc=False,
            results_root=results_root,
            save=True,
        )

        if cfg_fBOSC["postproc.use"] == "yes":
            fBOSCUtils.plot_episodetable_stats(
                animal,
                day,
                session,
                tetrode_num,
                channel_num,
                episodesTablePostProc,
                postproc,
                results_root=results_root,
                save=True,
            )

    if return_diagnostics:
        diagnostics = {
            "TFR": TFR_,
            "time_tfr": cfg_fBOSC["time.time_tfr"],
            "fBOSC": fBOSC,
            "cfg_fBOSC": cfg_fBOSC,
        }
        return episodesTable, episodesTablePostProc, diagnostics

    return episodesTable, episodesTablePostProc
