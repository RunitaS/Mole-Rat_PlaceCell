"""
Theta frequency / power vs. running speed, on the epochs of the autocorrelogram
(ACG) "theta continuity" analysis.

All of the ACG analysis -- .ncs loading and cleaning, epoching, common-noise /
delta-theta / MAD epoch rejection, moving/immobile labelling and the Dunn et
al. 2022 peak-range quantification -- is done by ACG_Utils.py, which must sit
in ACG_UTILS_DIR (default: this script's folder). ACG_Utils.py is the latest
version of the main ACG script (e.g. Figc_ACG_ScalingCorrected_v5_CommonNoise.py)
saved under that name; this script never re-implements any of it, so every
change to the ACG procedure is picked up from there. ACG-specific settings
(bands, thresholds, ED_min gate, reference bank ...) are read from ACG_Utils.py;
only its output paths are redirected to this script's OUTPUT_DIR.

This script adds, per ACG epoch, the power columns the speed models need
(add_epoch_power: broadband, theta-band, non-theta and relative theta power,
computed on the exact same cleaned epochs), then fits the theta vs. speed
random-intercept mixed models (one value per recording x speed bin, grouped by
animal) for matched-sinusoid frequency and each power measure. The ACG summary
figures (freq_est histograms, Fig. 4-style panels, ...) are produced by
ACG_Utils.run_pipeline when RUN_ACG_FIGURES is True.
"""

import os
import sys
import shutil
import socket
import getpass
import hashlib
import inspect
import platform
import importlib
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_script_path():
    """Absolute path of this script -- `__file__` when run as a script,
    `__vsc_ipynb_file__` when run cell-by-cell in the VS Code Interactive
    Window. None if neither is available."""
    for cand in (globals().get('__file__'), globals().get('__vsc_ipynb_file__')):
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


# %% ==================== ACG analysis (ACG_Utils.py) ===============================

# Folder holding ACG_Utils.py (the latest main ACG analysis script).
ACG_UTILS_DIR = os.path.dirname(get_script_path() or os.path.join(os.getcwd(), '_'))
if ACG_UTILS_DIR not in sys.path:
    sys.path.insert(0, ACG_UTILS_DIR)

import ACG_Utils as acg  # noqa: E402
# re-read ACG_Utils.py when this cell is re-run in an interactive session, so
# an updated ACG procedure is never silently replaced by a cached one
acg = importlib.reload(acg)


# %% ==================== Configuration ===========================================

# Root directory to search recursively for .ncs files. Every .ncs found
# anywhere under this tree (in any subfolder) is processed.
ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Where results/figures are written (default: alongside ROOT_DIR).
OUTPUT_DIR = os.path.join(ROOT_DIR, 'ACG_ThetaVsSpeed_ACGUtils')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
# Run metadata (all settings, data/script paths) + a copy of this script
# are written here -- see save_run_metadata. ACG_Utils' own run_metadata.csv
# and a copy of ACG_Utils.py go to the ACG_META_DIR subfolder.
META_DIR = os.path.join(OUTPUT_DIR, 'meta')
ACG_META_DIR = os.path.join(META_DIR, 'ACG_Utils')
# Per-session common-noise diagnostic figures written by ACG_Utils.
COMMON_NOISE_FIG_DIR = os.path.join(FIGURE_DIR, 'CommonNoise')

# Also write the ACG summary tables/figures (ACG_Utils.run_pipeline) for
# this ROOT_DIR into OUTPUT_DIR / FIGURE_DIR.
RUN_ACG_FIGURES = True

# ---- Theta vs. speed (mixed models) ----
# Before the freq/power ~ speed mixed models are fit, the theta-positive
# (non-skipped) epochs of each recording (.ncs file) are grouped into
# SPEED_BIN_WIDTH_CMS-wide bins of median epoch speed ([0,2), [2,4), ... cm/s)
# and reduced to one BIN_AGG_METHOD value per (recording, speed bin) -- see
# aggregate_by_speed_bin. The model's predictor is the bin center (1, 3, 5,
# ... cm/s), so it is fit on one value per speed bin visited per recording
# rather than on thousands of individual epochs. Same binning as
# ThetaVsSpeed_v6_Ferret_Final.py.
SPEED_BIN_WIDTH_CMS = 2.0
BIN_AGG_METHOD = 'median'   # 'median' or 'mean'

# ---- Relative theta power (method of ThetaVsSpeed_v6_Ferret_Final.py) ----
# epoch_theta_power_rel = the epoch's theta power (mean square of the
# ACG_Utils THETA_BAND band-passed LFP) as % of the recording's total
# REL_POWER_BAND power (mean square of the REL_POWER_BAND band-passed LFP) --
# the time-domain (Parseval) equivalent of the PSD normalization in
# Fig4a_PSD_v2_Simul.py. Dividing by each recording's own total power removes
# per-recording gain differences (electrode impedance/placement), so animals
# are compared on the share of power in theta rather than on absolute uV^2.
REL_POWER_BAND = (1.0, 100.0)   # Hz -- matches NORM_BAND in Fig4a_PSD_v2_Simul.py
REL_POWER_FILTER_ORDER = 4
THETA_POWER_FILTER_ORDER = 5    # Butterworth order of the THETA_BAND band-pass (v6's FILTER_ORDER)

# ACG_Utils' path settings point at its own ROOT_DIR; send everything it
# writes during this run (common-noise figures, its run metadata, the
# run_pipeline outputs) to this script's folders instead.
acg.ROOT_DIR = ROOT_DIR
acg.OUTPUT_DIR = OUTPUT_DIR
acg.FIGURE_DIR = FIGURE_DIR
acg.META_DIR = ACG_META_DIR
acg.COMMON_NOISE_FIG_DIR = COMMON_NOISE_FIG_DIR
acg.ARENA_FIGURE_DIR = os.path.join(FIGURE_DIR, 'ArenaType')
acg.SESSION_FIGURE_DIR = os.path.join(FIGURE_DIR, 'SessionType')


def legend_and_stats_below(ax, stats_text=None, handles=None, labels=None, ncol=None,
                           fontsize=7, stats_fontsize=8, linespacing=None, pad_in=0.08,
                           **legend_kw):
    """Draw `ax`'s legend and an optional stats text block BELOW the axes --
    legend first, stats under it, both clear of the x tick labels / x-label --
    so neither covers the data points.

    The offset from the axes is fixed in inches (not axes fraction), so it
    stays correct after tight_layout() or bbox_inches='tight' resize the
    axes. `handles`/`labels` default to ax.get_legend_handles_labels(); any
    legend already on `ax` is replaced. Call before fig.tight_layout().
    """
    fig = ax.figure
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    canvas = fig.canvas
    renderer = canvas.get_renderer() if hasattr(canvas, 'get_renderer') else fig._get_renderer()
    # distance (in) from the axes' bottom edge to the bottom of its tick labels + x-label
    below_in = (ax.get_window_extent(renderer).y0 - ax.get_tightbbox(renderer).y0) / fig.dpi
    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, y=-(below_in + pad_in), units='inches')

    anchor = trans
    if handles:
        anchor = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0),
                           bbox_transform=trans, ncol=ncol or min(len(handles), 3),
                           fontsize=fontsize, frameon=False, **legend_kw)
    if stats_text:
        ax.annotate(stats_text, xy=(0.5, 0), xycoords=anchor, xytext=(0, -4),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=stats_fontsize, linespacing=linespacing, color='0.1',
                    annotation_clip=False)


# %% ==================== ACG epochs + epoch power ==================================

def add_epoch_power(result, clean_lfp, common_noise=None):
    """Add the per-epoch power columns of the theta vs. speed models to one
    .ncs file's ACG_Utils.process_ncs_for_acg result (one row per epoch).

    `clean_lfp` is the file's ACG_Utils.load_clean_lfp trace (before the ACG
    low-pass). It gets the same low-pass and epoching as in
    process_ncs_for_acg, so result['epoch_index'] picks out exactly the
    epochs the ACG was computed on. Power is in uV^2: load_ncs scales the ADC
    counts by the header's ADBitVolts and every cleaning step is
    amplitude-preserving.

    Columns added:
      epoch_power_uv2          mean square of the epoch's cleaned LFP
                               (<= ACG_Utils LOWPASS_CUTOFF_HZ)
      epoch_theta_power_uv2    mean square of the THETA_BAND band-passed epoch
      epoch_nontheta_power_uv2 epoch_power_uv2 - epoch_theta_power_uv2: by
                               Parseval the broadband mean square is the sum of
                               its band powers, so this is the power outside
                               THETA_BAND (delta, beta, gamma ...)
      session_total_power_uv2  the recording's REL_POWER_BAND power (below)
      epoch_theta_power_rel    theta power as % of session_total_power_uv2

    The recording's total power is the mean square of the REL_POWER_BAND
    trace over every epoch that is neither common noise nor a MAD
    peak-to-peak outlier -- any movement state, regardless of the delta/theta
    filter -- i.e. the whole session as in ThetaVsSpeed_v6_Ferret_Final.py,
    minus the large transients that would otherwise inflate one recording's
    denominator. The MAD reference statistics come from the same epochs as in
    ACG_Utils.process_ncs_for_acg (common-noise- and delta/theta-kept).
    """
    fs = acg.FS_ACG
    nperseg = int(round(fs * acg.EPOCH_SEC))
    lfp = acg.lowpass_filter(clean_lfp, fs, acg.LOWPASS_CUTOFF_HZ, acg.LOWPASS_ORDER)
    epochs, n_total = acg.build_epochs(lfp, nperseg)
    idx = result['epoch_index'].to_numpy(dtype=int)

    keep_common = np.ones(n_total, dtype=bool)
    if common_noise is not None:
        n = min(n_total, len(common_noise))
        keep_common[:n] = ~np.asarray(common_noise[:n], dtype=bool)
    mad_ref = keep_common & acg.reject_deltatheta_epochs(epochs, fs)
    artifact_free = keep_common & acg.reject_artifact_epochs(epochs, ref_mask=mad_ref)

    theta_epochs, _ = acg.build_epochs(
        acg.bandpass_filter(lfp, fs, acg.THETA_BAND, THETA_POWER_FILTER_ORDER), nperseg)
    broadband_epochs, _ = acg.build_epochs(
        acg.bandpass_filter(lfp, fs, REL_POWER_BAND, REL_POWER_FILTER_ORDER), nperseg)
    total_power = np.mean(broadband_epochs[artifact_free] ** 2)

    result['epoch_power_uv2'] = np.mean(epochs[idx] ** 2, axis=1)
    result['epoch_theta_power_uv2'] = np.mean(theta_epochs[idx] ** 2, axis=1)
    result['epoch_nontheta_power_uv2'] = result['epoch_power_uv2'] - result['epoch_theta_power_uv2']
    result['session_total_power_uv2'] = total_power
    result['epoch_theta_power_rel'] = 100 * result['epoch_theta_power_uv2'] / total_power
    return result


def run_acg_analysis(root_dir):
    """ACG_Utils.process_root_directory on `root_dir`, with the epoch power
    columns (add_epoch_power) added to every file's result.

    Each .ncs file's process_ncs_for_acg call is wrapped for the duration of
    the run, so the power is computed from the cleaned trace ACG_Utils
    already loaded (no second read of the .ncs) and on the same common-noise
    mask; the common-noise figures are sent to COMMON_NOISE_FIG_DIR. Both
    ACG_Utils functions are restored afterwards.

    Returns (results DataFrame, common-noise per-epoch DataFrame,
    {abs .ncs path: error message} for files ACG_Utils skipped).
    """
    orig_process = acg.process_ncs_for_acg
    orig_common_noise = acg.session_common_noise
    file_errors = {}

    def _process(fpath, *args, clean=None, common_noise=None, **kwargs):
        try:
            if clean is None:
                clean = acg.load_clean_lfp(fpath)
            res = orig_process(fpath, *args, clean=clean, common_noise=common_noise, **kwargs)
            return add_epoch_power(res, clean[0], common_noise)
        except Exception as e:
            file_errors[os.path.abspath(fpath)] = f'{type(e).__name__}: {e}'
            raise

    def _common_noise(folder_rel, cleaned, **kwargs):
        kwargs.setdefault('fig_dir', COMMON_NOISE_FIG_DIR)
        return orig_common_noise(folder_rel, cleaned, **kwargs)

    acg.process_ncs_for_acg, acg.session_common_noise = _process, _common_noise
    try:
        results_df, common_noise_df = acg.process_root_directory(root_dir)
    finally:
        acg.process_ncs_for_acg, acg.session_common_noise = orig_process, orig_common_noise
    return results_df, common_noise_df, file_errors


def build_file_log(root_dir, results_df, file_errors):
    """One dict per .ncs file under `root_dir` (path, tracking .csv, OK/SKIP
    status, epoch counts) -- used by save_run_metadata_excel to record
    exactly which data went into a run."""
    file_log = []
    for root, _dirs, files in os.walk(root_dir):
        for fname in sorted(files):
            if not fname.endswith('.ncs'):
                continue
            fpath = os.path.abspath(os.path.join(root, fname))
            rel = os.path.relpath(fpath, root_dir)
            row = {'relative_path': rel,
                   'ncs_file': fname,
                   'session_dir': os.path.dirname(fpath),
                   'abs_path': fpath,
                   **acg.parse_metadata_from_path(rel)}
            try:
                row['tracking_csv'] = acg.find_position_file(fpath)
                row['ncs_size_bytes'] = os.path.getsize(fpath)
                row['ncs_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(fpath)).isoformat(sep=' ', timespec='seconds')
            except OSError as e:
                row['tracking_csv'] = f'ERROR: {e}'

            res = results_df[results_df['file'] == fpath]
            if len(res):
                n_edmin = int(res['skipped_edmin'].sum())
                row.update({'status': 'OK', 'error': '',
                            'arena_type': res['arena_type'].iloc[0],
                            'session_type': res['session_type'].iloc[0],
                            'n_epochs_kept': len(res),
                            'n_moving': int((res['movement'] == 'moving').sum()),
                            'n_immobile': int((res['movement'] == 'immobile').sum()),
                            'n_edmin_rejected': n_edmin,
                            'pct_edmin_rejected': round(100 * n_edmin / len(res), 2),
                            'n_outofband_rejected': int(res['skipped_outofband'].sum()),
                            'n_accepted': int((~res['skipped']).sum())})
            else:
                row.update({'status': 'SKIP',
                            'error': file_errors.get(
                                fpath, 'not analysed -- .ncs could not be loaded (see console)')})
            file_log.append(row)
    return file_log


# %% ==================== Theta vs. speed (mixed models) ============================

def fit_freq_speed_mixed_model(df, response='freq', predictor='median_speed_cms', group='animal'):
    """Random-intercept linear mixed model: response ~ predictor, grouped by
    `group` (default: animal), so repeated epochs from the same animal don't
    count as independent observations. Same mixed-model approach as
    ThetaVsSpeed_v2.py's fit_mixed_model, ported here for the matched-sinusoid
    freq_est vs. running-speed comparison."""
    model = smf.mixedlm(f"{response} ~ {predictor}", df, groups=df[group])
    return model.fit()


def save_mixedlm_stats_excel(result, out_path, model_desc=''):
    """Write a mixedlm result's fixed-effect coefficient table (coef, std
    err, z, p-value, 95% CI) to an .xlsx file, plus a second sheet with basic
    model info (formula, n obs, n groups, convergence)."""
    coef_table = pd.DataFrame({
        'coef': result.params,
        'std_err': result.bse,
        'z': result.tvalues,
        'p_value': result.pvalues,
    })
    ci = result.conf_int()
    coef_table['CI_2.5%'] = ci[0]
    coef_table['CI_97.5%'] = ci[1]
    coef_table.index.name = 'term'

    info = pd.DataFrame({
        'field': ['model', 'n_obs', 'n_groups', 'converged'],
        'value': [model_desc, result.nobs, result.model.n_groups, result.converged],
    })

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        coef_table.to_excel(writer, sheet_name='Fixed effects')
        info.to_excel(writer, sheet_name='Model info', index=False)


def aggregate_by_speed_bin(df, response, speed_col='median_speed_cms', recording='file',
                           group='animal', bin_width=SPEED_BIN_WIDTH_CMS,
                           agg_method=BIN_AGG_METHOD):
    """One `agg_method` value of `response` per (recording, speed bin).

    Epochs are grouped by `bin_width`-wide bins of their median speed ([0,w),
    [w,2w), ... cm/s) within each recording, so the mixed model is fit on a
    value that summarises a distinct speed rather than on many individual
    epochs. `speed_within_bin` is the agg_method'd speed of the bin's epochs;
    `speed_bin_center` ((k + 0.5) * w for bin [k*w, (k+1)*w)) is the model's
    predictor; `n_epochs` is how many epochs went into the bin.
    """
    d = df.dropna(subset=[response, speed_col, group]).copy()
    d['speed_bin'] = np.floor(d[speed_col] / bin_width)
    agg = (d.groupby([group, recording, 'speed_bin'])
            .agg(**{response: (response, agg_method)},
                 speed_within_bin=(speed_col, agg_method),
                 n_epochs=(response, 'size'))
            .reset_index())
    agg['speed_bin_center'] = (agg['speed_bin'] + 0.5) * bin_width
    return agg.drop(columns='speed_bin')


def _plot_vs_speed_mixedmodel(df, response, ylabel, title, unit, animal=None, ax=None,
                              figsize=(7, 5), color='#4C72B0'):
    """Shared body of the *_vs_speed_mixedmodel plots (same style as
    ThetaVsSpeed_v6_Ferret_Final.py's plot_mixed_model_fit): theta-positive
    (non-skipped) epochs reduced to one value per (recording,
    SPEED_BIN_WIDTH_CMS speed bin) (aggregate_by_speed_bin) and scattered per
    animal against the bin center, the random-intercept mixed model's
    fixed-effect line + 95% CI band, and the legend plus full model
    statistics written below the plot.

    The model is always fit across ALL animals; `animal`, if given, only
    restricts which points are drawn on top of that shared fit.
    """
    predictor, group = 'speed_bin_center', 'animal'
    binned = aggregate_by_speed_bin(df[~df['skipped']], response, group=group)
    if len(binned) < 2 or binned[group].nunique() < 2:
        raise ValueError('Not enough theta-positive speed bins / animals to fit a mixed model '
                         f'(need >=2 animals with valid {response}/median_speed_cms).')

    result = fit_freq_speed_mixed_model(binned, response=response, predictor=predictor, group=group)
    sub = binned
    scope = sub if animal is None else sub[sub[group] == animal]

    intercept = result.params['Intercept']
    slope = result.params[predictor]
    pvalue = result.pvalues[predictor]
    slope_se = result.bse[predictor]
    slope_z = result.tvalues[predictor]
    slope_ci_low, slope_ci_high = result.conf_int().loc[predictor]
    cov = result.cov_params()

    x_range = np.linspace(sub[predictor].min(), sub[predictor].max(), 100)
    y_pred = intercept + slope * x_range
    se = np.sqrt(
        cov.loc['Intercept', 'Intercept']
        + x_range ** 2 * cov.loc[predictor, predictor]
        + 2 * x_range * cov.loc['Intercept', predictor]
    )
    z = norm.ppf(0.975)
    y_ci_low = y_pred - z * se
    y_ci_high = y_pred + z * se

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    handles = []
    for animal_label in acg.ordered_animals(scope[group]):
        pts = scope[scope[group] == animal_label]
        handles.append(ax.scatter(pts[predictor], pts[response],
                                  color=acg.animal_color(animal_label, default=color),
                                  edgecolor='k', linewidth=0.5, alpha=0.8, s=25,
                                  label=animal_label))
    handles += ax.plot(x_range, y_pred, color='black', linewidth=2, label='Mixed Linear Fit')
    handles.append(ax.fill_between(x_range, y_ci_low, y_ci_high, color='grey', alpha=0.5,
                                   label='95% CI'))
    ax.set_xlabel('Speed bin center (cm/s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{animal}: {title}' if animal else title, fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    details = '\n'.join([
        f'Model: {response} ~ {predictor} + (1 | {group})   [linear mixed model, REML]',
        f'Slope β = {slope:.4g} ± {slope_se:.3g} SE {unit} per cm/s,  '
        f'95% CI [{slope_ci_low:.4g}, {slope_ci_high:.4g}],  z = {slope_z:.2f},  '
        + (f'p = {pvalue:.3g}' if pvalue > 0 else 'p < 1e-300'),
        f'Intercept = {intercept:.4g} {unit};  random-intercept var = {result.cov_re.iloc[0, 0]:.3g};  '
        f'residual var = {result.scale:.3g};  converged = {result.converged}',
        f'n = {len(sub)} speed bins from {sub["file"].nunique()} recordings, '
        f'{sub[group].nunique()} animals ({int(sub["n_epochs"].sum())} theta-positive epochs)'
        + (f';  points shown: {animal} only' if animal else ''),
        f'bin width = {SPEED_BIN_WIDTH_CMS:g} cm/s,  value per (recording, bin) = {BIN_AGG_METHOD}',
    ])
    legend_and_stats_below(ax, details, handles=handles, labels=[h.get_label() for h in handles],
                           ncol=3, fontsize=9, stats_fontsize=9, linespacing=1.4)

    fig.tight_layout()
    return fig, ax, result


def plot_freq_vs_speed_mixedmodel(df, animal=None, ax=None, figsize=(7, 5), color='#4C72B0'):
    """Theta-positive epochs only (non-skipped, i.e. passed every ACG_Utils
    epoch rejection, had an acceptable sinusoid fit and an in-band freq_est):
    scatter of matched-sinusoid frequency ('freq') vs. running speed, one
    value per (recording, SPEED_BIN_WIDTH_CMS speed bin of
    'median_speed_cms'), with a random-intercept mixed-model fit (grouped by
    animal) overlaid as the fixed-effect line and its 95% CI band -- the same
    fit_mixed_model/plot_mixed_model_fit stats used in ThetaVsSpeed_v2.py,
    applied here to the ACG epochs' own freq_est/speed columns instead of the
    Generalized-Phase instantaneous frequency/speed bins.

    The mixed model is always fit across ALL animals (so the random intercept
    has more than one group to draw on); `animal`, if given, only restricts
    which points are drawn on top of that shared fit.

    Returns (fig, ax, mixedlm_result).
    """
    return _plot_vs_speed_mixedmodel(
        df, 'freq', ylabel='Theta Frequency, freq_est [Hz]',
        title='Theta Frequency vs. Speed (Mixed Model)', unit='Hz',
        animal=animal, ax=ax, figsize=figsize, color=color)


def plot_power_vs_speed_mixedmodel(df, animal=None, ax=None, figsize=(7, 5), color='#55A868'):
    """Theta-positive epochs only (non-skipped): scatter of per-epoch
    broadband LFP power ('epoch_power_uv2', mean squared amplitude of the
    cleaned, ADBitVolts-scaled uV waveform -- see add_epoch_power) vs.
    running speed, binned like plot_freq_vs_speed_mixedmodel, with a
    random-intercept mixed-model fit (grouped by animal) overlaid as the
    fixed-effect line and its 95% CI band.

    The mixed model is always fit across ALL animals (so the random
    intercept has more than one group to draw on); `animal`, if given, only
    restricts which points are drawn on top of that shared fit.

    Returns (fig, ax, mixedlm_result).
    """
    return _plot_vs_speed_mixedmodel(
        df, 'epoch_power_uv2', ylabel='Epoch LFP Power [µV²]',
        title='Epoch LFP Power vs. Speed (Mixed Model)', unit='µV²',
        animal=animal, ax=ax, figsize=figsize, color=color)


def plot_theta_power_vs_speed_mixedmodel(df, animal=None, ax=None, figsize=(7, 5),
                                         color='#4C72B0'):
    """Theta-band power ('epoch_theta_power_uv2', mean square of the
    THETA_BAND band-passed epoch, uV^2 -- no normalization) vs. speed, same
    binning / mixed model as plot_freq_vs_speed_mixedmodel. The un-normalized
    counterpart of plot_relpower_vs_speed_mixedmodel.

    Returns (fig, ax, mixedlm_result).
    """
    return _plot_vs_speed_mixedmodel(
        df, 'epoch_theta_power_uv2', ylabel='Theta Power [µV²]',
        title='Theta Power vs. Speed (no normalization, Mixed Model)', unit='µV²',
        animal=animal, ax=ax, figsize=figsize, color=color)


def plot_relpower_vs_speed_mixedmodel(df, animal=None, ax=None, figsize=(7, 5),
                                      color='#4C72B0'):
    """Relative theta power ('epoch_theta_power_rel', % of the recording's
    REL_POWER_BAND power -- see REL_POWER_BAND) vs. speed, same binning /
    mixed model as plot_freq_vs_speed_mixedmodel.

    Returns (fig, ax, mixedlm_result).
    """
    return _plot_vs_speed_mixedmodel(
        df, 'epoch_theta_power_rel',
        ylabel=f'Relative Theta Power [% of {REL_POWER_BAND[0]:g}–{REL_POWER_BAND[1]:g} Hz power]',
        title='Theta Power vs. Speed (relative power, Mixed Model)', unit='%',
        animal=animal, ax=ax, figsize=figsize, color=color)


def plot_nontheta_power_vs_speed_mixedmodel(df, animal=None, ax=None, figsize=(7, 5),
                                            color='#4C72B0'):
    """Power outside THETA_BAND ('epoch_nontheta_power_uv2' = epoch_power_uv2 -
    epoch_theta_power_uv2, uV^2) vs. speed, same binning / mixed model as
    plot_freq_vs_speed_mixedmodel -- the part of the un-filtered power
    (delta, beta, gamma ...) that is not theta.

    Returns (fig, ax, mixedlm_result).
    """
    return _plot_vs_speed_mixedmodel(
        df, 'epoch_nontheta_power_uv2', ylabel='Non-theta Power [µV²]',
        title='Non-theta Power vs. Speed (Mixed Model)', unit='µV²',
        animal=animal, ax=ax, figsize=figsize, color=color)


def plot_power_band_comparison_vs_speed(df, animal=None, figsize=(22, 6)):
    """Side-by-side check of how much of the un-normalized power ~ speed
    relationship is theta: the same theta-positive epochs' power (a) without
    band-pass (<= ACG_Utils LOWPASS_CUTOFF_HZ), (b) after the THETA_BAND
    band-pass, and (c) outside THETA_BAND (a minus b) -- each vs. speed with
    its own binned random-intercept mixed model (see
    _plot_vs_speed_mixedmodel). If (a)'s slope is driven by faster rhythms
    (e.g. gamma rising with speed), it shows up in (c) rather than (b).

    Returns (fig, axes, {response: mixedlm_result}).
    """
    theta_lbl = f'{acg.THETA_BAND[0]:g}–{acg.THETA_BAND[1]:g} Hz'
    lowpass = acg.LOWPASS_CUTOFF_HZ
    panels = [
        ('epoch_power_uv2', f'LFP Power ≤{lowpass:g} Hz [µV²]',
         f'(a) ≤{lowpass:g} Hz, no band-pass'),
        ('epoch_theta_power_uv2', f'Theta Power {theta_lbl} [µV²]',
         f'(b) {theta_lbl} band-pass'),
        ('epoch_nontheta_power_uv2', 'Non-theta Power [µV²]',
         f'(c) outside {theta_lbl} (a − b)'),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    results = {}
    for ax, (col, ylabel, title) in zip(axes, panels):
        _fig, _ax, results[col] = _plot_vs_speed_mixedmodel(
            df, col, ylabel=ylabel, title=title, unit='µV²', animal=animal, ax=ax)
    title = 'Epoch power vs. speed: un-filtered vs. theta band-passed'
    fig.suptitle(f'{animal}: {title}' if animal else title, fontsize=15, weight='bold')
    fig.tight_layout()
    return fig, axes, results


# (response column, plot function, output file stem, slope unit) of every
# theta vs. speed mixed model the driver fits and plots
SPEED_MODELS = [
    ('freq',                     plot_freq_vs_speed_mixedmodel,           'freq_est_vs_speed_mixedmodel',             'Hz'),
    ('epoch_power_uv2',          plot_power_vs_speed_mixedmodel,          'power_vs_speed_mixedmodel',                'uV^2'),
    ('epoch_theta_power_uv2',    plot_theta_power_vs_speed_mixedmodel,    'theta_power_vs_speed_mixedmodel',          'uV^2'),
    ('epoch_nontheta_power_uv2', plot_nontheta_power_vs_speed_mixedmodel, 'nontheta_power_vs_speed_mixedmodel',       'uV^2'),
    ('epoch_theta_power_rel',    plot_relpower_vs_speed_mixedmodel,       'theta_power_relative_vs_speed_mixedmodel', '%'),
]


def save_speed_figure(fig, path):
    """Save a theta vs. speed figure (padded so the stats below the axes
    aren't clipped) and close it."""
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f'Saved figure -> {path}')


# %% ==================== Run metadata (provenance) =================================
#
# Every run writes RUN_METADATA_FILENAME into FIGURE_DIR, recording which code
# versions produced the figures (this script AND ACG_Utils.py: path, SHA-256 of
# the source, git state, and a full copy of the source), every configuration
# setting of both, every .ncs file / session directory processed (with
# OK/SKIP status), and every output file written.

RUN_METADATA_FILENAME = 'run_metadata_ACG_ThetaVsSpeed.xlsx'

# (section, constant name, description) of this script's settings. Values are
# read from the module globals at write time, so they always reflect what was
# actually used. Any other UPPERCASE module-level constant of this script is
# still recorded (section 'Other'), and every UPPERCASE constant of
# ACG_Utils.py is recorded under section 'ACG_Utils'.
CONFIG_SETTINGS = [
    ('Paths',           'ROOT_DIR',                 'Root directory searched recursively for .ncs files'),
    ('Paths',           'OUTPUT_DIR',               'Results tables output directory'),
    ('Paths',           'FIGURE_DIR',               'Figure output directory'),
    ('Paths',           'META_DIR',                 'run_metadata.csv + script copy output directory'),
    ('Paths',           'ACG_META_DIR',             "ACG_Utils' run_metadata.csv + ACG_Utils.py copy"),
    ('Paths',           'COMMON_NOISE_FIG_DIR',     'ACG_Utils per-session common-noise figures'),
    ('Paths',           'ACG_UTILS_DIR',            'Folder ACG_Utils.py is imported from'),
    ('ACG',             'RUN_ACG_FIGURES',          'Also run ACG_Utils.run_pipeline (ACG summary tables/figures)'),
    ('Theta vs. speed', 'SPEED_BIN_WIDTH_CMS',      'Width (cm/s) of the speed bins epochs are grouped into per recording before the mixed models'),
    ('Theta vs. speed', 'BIN_AGG_METHOD',           "Value per (recording, speed bin): 'median' or 'mean'"),
    ('Relative power',  'REL_POWER_BAND',           'Band (Hz) whose total power (recording mean square) normalizes theta power'),
    ('Relative power',  'REL_POWER_FILTER_ORDER',   'Butterworth order of the REL_POWER_BAND band-pass (sosfiltfilt)'),
    ('Relative power',  'THETA_POWER_FILTER_ORDER', 'Butterworth order of the THETA_BAND band-pass for epoch theta power (sosfiltfilt)'),
]

# Processing choices hard-coded in the function bodies (not module constants).
PIPELINE_NOTES = [
    ('ACG analysis', 'ACG_Utils.process_root_directory(ROOT_DIR): all LFP cleaning, epoching, '
                     'common-noise / delta-theta / MAD epoch rejection, moving/immobile labelling '
                     'and ACG peak-range quantification -- settings under section ACG_Utils and in '
                     'ACG_META_DIR\\run_metadata.csv; common-noise figures redirected to '
                     'COMMON_NOISE_FIG_DIR'),
    ('ACG figures', 'ACG_Utils.run_pipeline(results, OUTPUT_DIR, FIGURE_DIR) when RUN_ACG_FIGURES'),
    ('Epoch power', 'add_epoch_power on the cleaned trace ACG_Utils loaded: ACG_Utils low-pass, '
                    'EPOCH_SEC epochs, rows = ACG epoch_index'),
    ('Theta power', 'epoch_theta_power_uv2 = per-epoch mean square of the ACG_Utils THETA_BAND '
                    'band-passed LFP'),
    ('Non-theta power', 'epoch_nontheta_power_uv2 = epoch_power_uv2 - epoch_theta_power_uv2'),
    ('Relative theta power', 'epoch_theta_power_rel = 100 * epoch_theta_power_uv2 / mean square of the '
                             'REL_POWER_BAND band-passed LFP over all epochs of the recording that are '
                             'not common noise and passed MAD peak-to-peak rejection'),
    ('Speed binning', 'non-skipped epochs grouped per (animal, file, floor(median_speed_cms / '
                      'SPEED_BIN_WIDTH_CMS)); BIN_AGG_METHOD of the response per group; '
                      'predictor = bin center'),
    ('Mixed models', 'smf.mixedlm: <response> ~ speed_bin_center, groups = animal; responses = freq, '
                     'epoch_power_uv2, epoch_theta_power_uv2, epoch_nontheta_power_uv2, '
                     'epoch_theta_power_rel'),
    ('Figure dpi', '200'),
]

# Functions whose keyword defaults are recorded (sheet 'Function defaults'),
# so plot/analysis parameters not exposed as module constants are captured
# too -- this script's, then ACG_Utils' (skipped if renamed there).
METADATA_FUNCTIONS = [
    'fit_freq_speed_mixed_model', 'aggregate_by_speed_bin',
    'plot_freq_vs_speed_mixedmodel', 'plot_power_vs_speed_mixedmodel',
    'plot_theta_power_vs_speed_mixedmodel', 'plot_relpower_vs_speed_mixedmodel',
    'plot_nontheta_power_vs_speed_mixedmodel', 'plot_power_band_comparison_vs_speed',
]
ACG_METADATA_FUNCTIONS = [
    'load_clean_lfp', 'process_ncs_for_acg', 'quantify_xcorr_epochs', 'create_sine_ref_xcorrs',
    'reject_deltatheta_epochs', 'reject_artifact_epochs', 'detect_common_noise_epochs',
    'session_common_noise', 'classify_epoch_movement', '_smooth_tracking_position',
]


def snapshot_script(script_path):
    """Read a script's source at the START of a run (so later edits made
    while a long run is still going don't get recorded by mistake).
    Returns (script_path, source_text, sha256_hex, modified_time_str)."""
    if script_path is None:
        return None, '', 'UNKNOWN (script path not found)', ''
    with open(script_path, 'rb') as f:
        raw = f.read()
    modified = datetime.fromtimestamp(os.path.getmtime(script_path)).isoformat(
        sep=' ', timespec='seconds')
    return script_path, raw.decode('utf-8', errors='replace'), hashlib.sha256(raw).hexdigest(), modified


def save_run_metadata(meta_dir=META_DIR):
    """Write this script's settings, the data directory it was run on, and
    the script's name/directory and run time to a 2-column (parameter, value)
    CSV in `meta_dir`, and save a copy of the script itself there. ACG_Utils'
    own save_run_metadata writes the ACG settings and a copy of ACG_Utils.py
    to ACG_META_DIR, so the results folder records exactly how it was made."""
    script_path = get_script_path() or os.path.join(
        os.getcwd(), 'ThetaVsSpeed_ACGMethod_ScalingCorrected_v3.py')
    meta = {
        # ---- provenance ----
        'script_name':              os.path.basename(script_path),
        'script_dir':               os.path.dirname(script_path),
        'run_datetime':             datetime.now().isoformat(timespec='seconds'),
        'data_dir (ROOT_DIR)':      ROOT_DIR,
        'OUTPUT_DIR':               OUTPUT_DIR,
        'FIGURE_DIR':               FIGURE_DIR,
        # ---- ACG analysis ----
        'acg_utils_path':           os.path.abspath(acg.__file__),
        'acg_utils_settings':       os.path.join(ACG_META_DIR, 'run_metadata.csv'),
        'RUN_ACG_FIGURES':          RUN_ACG_FIGURES,
        # ---- theta vs. speed (mixed model, grouped by animal) ----
        'SPEED_BIN_WIDTH_CMS':      SPEED_BIN_WIDTH_CMS,
        'BIN_AGG_METHOD':           BIN_AGG_METHOD,
        'speed_models':             'smf.mixedlm: <response> ~ speed_bin_center, groups=animal; one '
                                    'BIN_AGG_METHOD value per (recording, speed bin) of non-skipped epochs',
        'speed_model_responses':    [m[0] for m in SPEED_MODELS],
        # ---- theta power / relative-power normalization ----
        'theta_band (ACG_Utils)':   acg.THETA_BAND,
        'THETA_POWER_FILTER_ORDER': THETA_POWER_FILTER_ORDER,
        'REL_POWER_BAND':           REL_POWER_BAND,
        'REL_POWER_FILTER_ORDER':   REL_POWER_FILTER_ORDER,
        'rel_power_denominator':    'mean square of REL_POWER_BAND trace over the recording\'s '
                                    'common-noise-free, MAD-artifact-free epochs',
        'figure_dpi':               200,
    }
    os.makedirs(meta_dir, exist_ok=True)
    out_path = os.path.join(meta_dir, 'run_metadata.csv')
    pd.DataFrame({'parameter': list(meta.keys()),
                  'value':     [str(v) for v in meta.values()]}).to_csv(out_path, index=False)
    print(f'Saved run metadata -> {out_path}')

    if os.path.isfile(script_path):
        shutil.copy2(script_path, meta_dir)
        print(f'Saved script copy -> {os.path.join(meta_dir, os.path.basename(script_path))}')
    else:
        print(f'  Script not found at {script_path} -- no copy saved to {meta_dir}')

    acg.save_run_metadata(ACG_META_DIR)


def _git(args, cwd):
    """stdout of `git -C cwd <args>`, or None if git / the repo isn't available."""
    try:
        out = subprocess.run(['git', '-C', cwd, *args], capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def collect_code_info(code_snapshots):
    """Code identity (name, directory, hash, git state) of every script in
    `code_snapshots` ({section: snapshot_script(...)}) + environment rows."""
    from importlib.metadata import version, PackageNotFoundError

    rows = []
    repo_cwd = None
    for section, (script_path, _source, script_sha256, script_modified) in code_snapshots.items():
        rows += [
            (section, 'script_name', os.path.basename(script_path) if script_path else 'UNKNOWN'),
            (section, 'script_dir', os.path.dirname(script_path) if script_path else 'UNKNOWN'),
            (section, 'script_path', script_path or 'UNKNOWN'),
            (section, 'script_modified', script_modified),
            (section, 'script_sha256', script_sha256),
        ]
        if not script_path:
            continue
        cwd = os.path.dirname(script_path)
        if not _git(['rev-parse', '--show-toplevel'], cwd):
            continue
        repo_cwd = repo_cwd or cwd
        status = _git(['status', '--porcelain', '--', script_path], cwd)
        if status == '':
            script_state = 'committed, no local edits (code = HEAD_commit)'
        elif status and status.startswith('??'):
            script_state = ('UNTRACKED (script never committed) -- '
                            f'identify version by script_sha256 / {section} snapshot sheet')
        else:
            script_state = ('MODIFIED since last commit -- '
                            f'identify version by script_sha256 / {section} snapshot sheet')
        rows += [
            (section, 'git_state', script_state),
            (section, 'last_commit_touching_script',
             _git(['log', '-1', '--format=%H  %ci  %s', '--', script_path], cwd)
             or 'none (script not committed)'),
        ]

    if repo_cwd:
        rows += [
            ('Git', 'repo_root', _git(['rev-parse', '--show-toplevel'], repo_cwd)),
            ('Git', 'branch', _git(['rev-parse', '--abbrev-ref', 'HEAD'], repo_cwd)),
            ('Git', 'HEAD_commit', _git(['rev-parse', 'HEAD'], repo_cwd)),
            ('Git', 'HEAD_commit_date', _git(['log', '-1', '--format=%ci'], repo_cwd)),
        ]
    else:
        rows.append(('Git', 'repo_root', 'not a git repository (or git unavailable)'))

    rows += [
        ('Environment', 'user', getpass.getuser()),
        ('Environment', 'hostname', socket.gethostname()),
        ('Environment', 'platform', platform.platform()),
        ('Environment', 'python_executable', sys.executable),
        ('Environment', 'python_version', sys.version.split()[0]),
    ]
    for pkg in ('numpy', 'scipy', 'pandas', 'statsmodels', 'matplotlib', 'seaborn', 'openpyxl'):
        try:
            rows.append(('Environment', f'{pkg}_version', version(pkg)))
        except PackageNotFoundError:
            rows.append(('Environment', f'{pkg}_version', 'not installed'))
    return rows


def _setting_value(v):
    """Excel-friendly representation of a setting value."""
    if v is None or isinstance(v, (bool, int, float, str, np.integer, np.floating)):
        return 'None' if v is None else v
    return repr(v)


def _module_constants(namespace, skip=()):
    """(name, value) of every UPPERCASE, non-callable, non-module global in `namespace`."""
    return [(name, namespace[name]) for name in sorted(namespace)
            if name.isupper() and name not in skip and not callable(namespace[name])
            and not inspect.ismodule(namespace[name])]


def collect_settings():
    """Configuration actually in effect (this script's + ACG_Utils'), plus
    hard-coded pipeline choices."""
    g = globals()
    rows = [(sec, name, _setting_value(g.get(name, 'NOT DEFINED')), desc)
            for sec, name, desc in CONFIG_SETTINGS]
    skip = {name for _, name, _ in CONFIG_SETTINGS} | {
        'RUN_METADATA_FILENAME', 'CONFIG_SETTINGS', 'PIPELINE_NOTES',
        'METADATA_FUNCTIONS', 'ACG_METADATA_FUNCTIONS', 'SPEED_MODELS'}
    rows += [('Other', name, _setting_value(v), '(not documented in CONFIG_SETTINGS)')
             for name, v in _module_constants(g, skip)]
    rows += [('ACG_Utils', name, _setting_value(v), 'ACG_Utils.py')
             for name, v in _module_constants(vars(acg))]
    rows += [('Pipeline (hard-coded)', name, value, '') for name, value in PIPELINE_NOTES]
    return pd.DataFrame(rows, columns=['section', 'setting', 'value', 'description'])


def collect_function_defaults():
    """Keyword defaults of the analysis/plotting functions as actually defined."""
    rows = []
    for module, namespace, names in (('ThetaVsSpeed', globals(), METADATA_FUNCTIONS),
                                     ('ACG_Utils', vars(acg), ACG_METADATA_FUNCTIONS)):
        for fname in names:
            func = namespace.get(fname)
            if func is None:
                continue
            for pname, p in inspect.signature(func).parameters.items():
                if p.default is not inspect.Parameter.empty:
                    rows.append((module, fname, pname, _setting_value(p.default)))
    return pd.DataFrame(rows, columns=['module', 'function', 'parameter', 'default_value'])


def collect_output_files(output_dir, run_start, tolerance_s=60):
    """Every file under `output_dir`, flagged by whether it was (re)written
    during this run -- older leftovers from previous runs are flagged NO."""
    rows = []
    t0 = run_start.timestamp() - tolerance_s
    for root, _dirs, files in os.walk(output_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname == RUN_METADATA_FILENAME or fname.startswith('~$'):
                continue
            mtime = os.path.getmtime(fpath)
            rows.append({
                'file': fname,
                'relative_path': os.path.relpath(fpath, output_dir),
                'folder': root,
                'written_this_run': 'YES' if mtime >= t0 else 'NO (older file)',
                'modified': datetime.fromtimestamp(mtime).isoformat(sep=' ', timespec='seconds'),
                'size_bytes': os.path.getsize(fpath),
            })
    return pd.DataFrame(rows)


def _excel_safe_df(df):
    """Strip characters openpyxl rejects and stop strings being read as formulas."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    def _clean(v):
        if isinstance(v, str):
            v = ILLEGAL_CHARACTERS_RE.sub('', v)
            if v.startswith('='):
                v = ' ' + v
        return v
    return df.apply(lambda col: col.map(_clean))


def save_run_metadata_excel(out_path, run_start, run_status, code_snapshots, file_log,
                            results_df=None):
    """Write the run-provenance workbook (see section header above).

    `code_snapshots` is {section: snapshot_script(...)} for this script
    ('Code') and ACG_Utils.py ('ACG_Utils').

    Sheets: 'Run info' (code identity, git, environment, run timing/status,
    headline counts), 'Settings', 'Function defaults', 'Input directories',
    'Input files', 'Output files', and one '<section> snapshot' sheet per
    script (full source as run). If `out_path` is locked (e.g. open in
    Excel), a timestamped copy is written next to it instead. Returns the
    path actually written.
    """
    now = datetime.now()
    run_rows = [
        ('Run', 'status', run_status),
        ('Run', 'run_start', run_start.isoformat(sep=' ', timespec='seconds')),
        ('Run', 'metadata_written', now.isoformat(sep=' ', timespec='seconds')),
        ('Run', 'elapsed_min', round((now - run_start).total_seconds() / 60, 2)),
        ('Run', 'root_dir', ROOT_DIR),
        ('Run', 'output_dir', OUTPUT_DIR),
        ('Run', 'figure_dir', FIGURE_DIR),
    ]
    run_rows += collect_code_info(code_snapshots)

    files_df = pd.DataFrame(file_log)
    n_ok = int((files_df['status'] == 'OK').sum()) if len(files_df) else 0
    run_rows += [
        ('Data', 'n_ncs_found', len(files_df)),
        ('Data', 'n_ncs_processed_ok', n_ok),
        ('Data', 'n_ncs_skipped', len(files_df) - n_ok),
        ('Data', 'n_session_dirs',
         files_df['session_dir'].nunique() if len(files_df) else 0),
    ]
    if results_df is not None and len(results_df):
        run_rows += [
            ('Data', 'animals', ', '.join(map(str, sorted(results_df['animal'].unique())))),
            ('Data', 'n_epochs_kept', len(results_df)),
            ('Data', 'n_epochs_accepted (non-skipped)', int((~results_df['skipped']).sum())),
            ('Data', 'n_epochs_edmin_rejected', int(results_df['skipped_edmin'].sum())),
            ('Data', 'n_epochs_outofband_rejected', int(results_df['skipped_outofband'].sum())),
        ]
    run_df = pd.DataFrame(run_rows, columns=['section', 'field', 'value'])

    if len(files_df):
        dirs_df = (files_df.groupby('session_dir')
                   .agg(animal=('animal', 'first'), date=('date', 'first'),
                        session=('session', 'first'),
                        n_ncs=('ncs_file', 'count'),
                        n_ok=('status', lambda s: int((s == 'OK').sum())),
                        tracking_csv=('tracking_csv', 'first'))
                   .reset_index())
    else:
        dirs_df = pd.DataFrame(columns=['session_dir'])

    sheets = {
        'Run info': run_df,
        'Settings': collect_settings(),
        'Function defaults': collect_function_defaults(),
        'Input directories': dirs_df,
        'Input files': files_df,
        'Output files': collect_output_files(OUTPUT_DIR, run_start),
    }
    for section, (_path, source, _sha, _modified) in code_snapshots.items():
        lines = source.splitlines()
        sheets[f'{section} snapshot'] = pd.DataFrame({'line': np.arange(1, len(lines) + 1),
                                                      'code': lines})

    def _write(path):
        from openpyxl.utils import get_column_letter
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for name, df in sheets.items():
                _excel_safe_df(df).to_excel(writer, sheet_name=name, index=False)
                ws = writer.sheets[name]
                ws.freeze_panes = 'A2'
                for i, col in enumerate(df.columns, start=1):
                    sample = [str(col)] + [str(v) for v in df[col].head(500)]
                    width = min(max(len(s) for s in sample) + 2, 100)
                    ws.column_dimensions[get_column_letter(i)].width = width

    try:
        _write(out_path)
    except PermissionError:
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}_{now.strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f'  Metadata file locked (open in Excel?) -- writing {out_path} instead')
        _write(out_path)
    return out_path


# %% ==================== Driver =====================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    save_run_metadata(META_DIR)

    # Snapshot both scripts before the (long) run starts, so edits made while
    # it runs aren't mistaken for the versions that produced these results.
    run_start = datetime.now()
    code_snapshots = {'Code': snapshot_script(get_script_path()),
                      'ACG_Utils': snapshot_script(os.path.abspath(acg.__file__))}
    metadata_path = os.path.join(FIGURE_DIR, RUN_METADATA_FILENAME)

    print(f'ACG analysis: {os.path.abspath(acg.__file__)}')
    results_df, common_noise_df, file_errors = run_acg_analysis(ROOT_DIR)
    file_log = build_file_log(ROOT_DIR, results_df, file_errors)

    # Written once now (so a crash during plotting still leaves a record of
    # code/settings/inputs), then overwritten with the final version at the end.
    metadata_path = save_run_metadata_excel(
        metadata_path, run_start, 'IN PROGRESS (or crashed during plotting)',
        code_snapshots, file_log, results_df)

    if not common_noise_df.empty:
        common_path = os.path.join(OUTPUT_DIR, 'common_noise_epochs.csv')
        common_noise_df.to_csv(common_path, index=False)
        n_common = int(common_noise_df['common_noise'].sum())
        print(f'Common noise rejected {n_common}/{len(common_noise_df)} session epochs '
              f'({100 * n_common / len(common_noise_df):.1f}%) in '
              f'{common_noise_df["session_folder"].nunique()} tested folders '
              f'-> {common_path}')

    n_edmin_rejected = int(results_df['skipped_edmin'].sum())
    pct_edmin_rejected = 100 * n_edmin_rejected / len(results_df) if len(results_df) else 0.0
    print(f'ED_min > {acg.ACG_ED_MIN_THRESH} rejected {n_edmin_rejected}/{len(results_df)} '
          f'epochs overall ({pct_edmin_rejected:.1f}%)')
    n_outofband = int(results_df['skipped_outofband'].sum())
    pct_outofband = 100 * n_outofband / len(results_df) if len(results_df) else 0.0
    print(f'freq_est outside {acg.ACG_FREQ_RANGE} rejected {n_outofband}/{len(results_df)} '
          f'epochs overall ({pct_outofband:.1f}%)')

    results_path = os.path.join(OUTPUT_DIR, 'acg_theta_continuity_epochs.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Saved {len(results_df)} epoch-level rows -> {results_path}')

    # ACG summary tables/figures (freq_est histograms, ED_min plots, Fig.
    # 4-style panels, centroids ...), all made by ACG_Utils
    if RUN_ACG_FIGURES:
        acg.run_pipeline(results_df, OUTPUT_DIR, FIGURE_DIR)

    # Theta-positive (non-skipped) epochs vs. running speed, one value per
    # (recording, SPEED_BIN_WIDTH_CMS speed bin), mixed-model fit (grouped by
    # animal): matched-sinusoid freq_est, broadband epoch power (uV^2), and
    # theta power both un-normalized (uV^2) and relative (% of REL_POWER_BAND
    # power) -- see SPEED_MODELS
    speed_binned = {}
    for response, plot_func, stem, unit in SPEED_MODELS:
        fig_speed, _ax_speed, speed_result = plot_func(results_df)
        save_speed_figure(fig_speed, os.path.join(FIGURE_DIR, f'{stem}.png'))
        speed_stats_path = os.path.join(OUTPUT_DIR, f'{stem}_stats.xlsx')
        save_mixedlm_stats_excel(
            speed_result, speed_stats_path,
            model_desc=f'{response} ~ speed_bin_center, groups=animal ({SPEED_BIN_WIDTH_CMS:g} cm/s '
                       f'speed bins, {BIN_AGG_METHOD} per recording x bin, theta-positive epochs only)')
        print(f'Saved mixed-model stats -> {speed_stats_path}')
        print(f"  {response} ~ speed slope = {speed_result.params['speed_bin_center']:.4g} {unit} per cm/s "
              f"(p = {speed_result.pvalues['speed_bin_center']:.2e})")
        speed_binned[response] = aggregate_by_speed_bin(results_df[~results_df['skipped']], response)

    speed_binned_path = os.path.join(OUTPUT_DIR, 'theta_vs_speed_binned.xlsx')
    with pd.ExcelWriter(speed_binned_path, engine='openpyxl') as writer:
        for response, binned in speed_binned.items():
            binned.to_excel(writer, sheet_name=response, index=False)
    print(f'Saved speed-binned values (what the mixed models are fit on) -> {speed_binned_path}')

    # Un-filtered (<= LOWPASS_CUTOFF_HZ) vs. THETA_BAND band-passed vs.
    # non-theta epoch power, side by side -- is the un-normalized power ~
    # speed slope theta, or faster rhythms (e.g. gamma)?
    fig_bands, _axes_bands, _ = plot_power_band_comparison_vs_speed(results_df)
    save_speed_figure(fig_bands, os.path.join(
        FIGURE_DIR, 'power_band_comparison_vs_speed_mixedmodel.png'))

    # freq_est / power / theta power / relative theta power vs. speed, each
    # animal's speed bins plotted against the shared (all-animals) mixed-model fit
    for animal_label in acg.ordered_animals(results_df['animal']):
        for _response, plot_func, stem, _unit in SPEED_MODELS:
            try:
                fig_speed_a, _ax_speed_a, _ = plot_func(results_df, animal=animal_label)
                save_speed_figure(fig_speed_a, os.path.join(FIGURE_DIR, f'{animal_label}_{stem}.png'))
            except ValueError as e:
                print(f'  SKIP {stem} plot for {animal_label}: {e}')
        try:
            fig_bands_a, _axes_bands_a, _ = plot_power_band_comparison_vs_speed(
                results_df, animal=animal_label)
            save_speed_figure(fig_bands_a, os.path.join(
                FIGURE_DIR, f'{animal_label}_power_band_comparison_vs_speed_mixedmodel.png'))
        except ValueError as e:
            print(f'  SKIP power band comparison plot for {animal_label}: {e}')

    metadata_path = save_run_metadata_excel(
        metadata_path, run_start, 'COMPLETED', code_snapshots, file_log, results_df)
    print(f'Saved run metadata (code versions, settings, inputs, outputs) -> {metadata_path}')
