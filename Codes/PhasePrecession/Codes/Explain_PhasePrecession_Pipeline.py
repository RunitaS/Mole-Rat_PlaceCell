# -*- coding: utf-8 -*-
"""
Pictorial walk-through of ThetaMod_PhasePrec_v19_ArenaSesCompare.py, from the
spikes that survive the ACG theta-epoch selection up to the final
PrecessionClass (phase_precessing / phase_succeeding / phase_locked /
no_phase_relation), with every quantified parameter.

A small session is SIMULATED (a rodent running laps on a wide linear track,
pausing at the ends now and then; theta LFP while running, delta while
paused; five model cells with known, designed phase behaviour). All analysis
is then done by the pipeline's OWN functions, imported unchanged from
ThetaMod_PhasePrec_v19_ArenaSesCompare.py, so every number in the figures is
what the real pipeline would report for that data.

    Cell A  designed to PRECESS    (-300 deg/pass)
    Cell B  designed to SUCCEED    (+120 deg/pass)
    Cell C  designed with a shallow, consistent drift (-10 deg/pass)
    Cell D  designed perfectly phase-locked (flat, 200 deg)
    Cell E  designed NOT theta-modulated

Output (OUT_DIR):
    figures/fig00 ... fig12 *.png    one figure per pipeline step
    demo_cell_results.csv            all parameters of the five model cells

The companion explanation is PhasePrecession_Pipeline_Explained.md in OUT_DIR.

Run:  python Explain_PhasePrecession_Pipeline.py   (from this folder)
"""

from pathlib import Path
from types import SimpleNamespace
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from scipy import signal, stats
from scipy.special import i0

import ThetaMod_PhasePrec_v19_ArenaSesCompare as pp

# ============================================================================
# Configuration
# ============================================================================
OUT_DIR = Path(__file__).resolve().parent / 'PhasePrecession_Explained'
FIG_DIR = OUT_DIR / 'figures'
SEED = 7

# --- simulated session ---
FS = 1000.0                 # Hz, LFP sampling rate
DURATION_S = 600.0
T0 = 100.0                  # s, first LFP timestamp (tracking shares this clock)
TRACK_LEN = 100.0           # cm (x)
TRACK_WIDTH = 30.0          # cm (y); a wide runway keeps the auto filter band sensible
RUN_S = 3.0                 # s for one end-to-end run
REST_S = 5.0                # s pause at a track end ...
REST_EVERY_RUNS = 7         # ... after every 7th run (no theta while paused)
TRACK_STEP = 33             # LFP samples per tracking sample (~30 Hz)
THETA_HZ = 5.5
FIELD_CENTER = 50.0         # cm
FIELD_SIGMA = 20.0          # cm
PEAK_RATE = 10.0            # Hz
BASE_RATE = 0.3             # Hz

# Designed preferred phase psi (deg) as a function of the pipeline's pass index p.
# p spans 2 units per pass, so psi = c + (S/2) * p gives S deg/pass.
CELLS = [
    dict(key='A', design='designed: precessing, -300 deg/pass', psi=lambda p: 180 - 150 * p, kappa=2.0),
    dict(key='B', design='designed: succeeding, +120 deg/pass', psi=lambda p: 180 + 60 * p, kappa=2.0),
    dict(key='C', design='designed: shallow drift, -14 deg/pass', psi=lambda p: 200 - 7 * p, kappa=4.0, peak=30.0),
    dict(key='D', design='designed: flat, locked at 200 deg', psi=lambda p: 200 + 0 * p, kappa=3.0),
    dict(key='E', design='designed: no theta modulation', psi=lambda p: 0 * p, kappa=0.0),
]
WALKTHROUGH_CELL = 'A'

# --- colours (validated reference palette) ---
INK, INK2, MUTED, LIGHT = '#0b0b0b', '#52514e', '#898781', '#c3c2b7'
GRID, PANEL = '#e1e0d9', '#f4f3ef'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#4a3aa7', '#e34948'
PREC_RANGE_C, PROC_RANGE_C = '#e4d3ec', '#fbf3c9'     # light purple / light yellow bands
SEQ_CMAP = LinearSegmentedColormap.from_list(
    'seq_blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
CLASS_STYLE = {
    'phase_precessing': ('PHASE PRECESSING', BLUE),
    'phase_succeeding': ('PHASE SUCCEEDING', ORANGE),
    'phase_locked': ('PHASE LOCKED', AQUA),
    'no_phase_relation': ('NO PHASE RELATION', MUTED),
    'non_precessing': ('TOO STEEP (MultiLinesFit)', VIOLET),
    'not_tested': ('NOT TESTED (failed TMI)', LIGHT),
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'font.size': 9, 'axes.edgecolor': LIGHT, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 160,
    'mathtext.fontset': 'dejavusans',
})


# ============================================================================
# Small helpers
# ============================================================================

def fmt_p(p):
    """p of exactly 0 only means 1 - erf(|z|/sqrt 2) underflowed in float64."""
    return '< 1e-15' if p < 1e-15 else f'{p:.2g}'


def wrap_pm_pi(u):
    return np.mod(u + np.pi, 2 * np.pi) - np.pi


def break_wraps(y, jump):
    """Insert NaN where a wrapped trace jumps, so lines aren't drawn across."""
    y = np.asarray(y, dtype=float).copy()
    y[np.flatnonzero(np.abs(np.diff(y)) > jump) + 1] = np.nan
    return y


def _wrap_math_safe(par, width):
    """textwrap.fill that never breaks inside a $...$ mathtext span."""
    parts = par.split('$')
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].replace(' ', '\x00')
    out = textwrap.fill('$'.join(parts), width, break_long_words=False, break_on_hyphens=False)
    return out.replace('\x00', ' ')


def note(ax, text, title=None, width=60, fontsize=8.6):
    """Axis turned into a shaded explanation card. Each '\\n'-separated paragraph is
    drawn separately and stacked by its measured height; paragraphs starting with '$'
    are formulas, drawn larger and unwrapped."""
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle='round,pad=0.0,rounding_size=0.03',
                                transform=ax.transAxes, fc=PANEL, ec='none', zorder=0))
    renderer = ax.figure.canvas.get_renderer()
    ax_h = ax.get_window_extent(renderer).height
    y = 0.95
    if title:
        t = ax.text(0.04, y, title, transform=ax.transAxes, fontsize=fontsize + 1, fontweight='bold',
                    color=INK, va='top')
        y -= t.get_window_extent(renderer).height / ax_h + 0.04
    for par in text.split('\n'):
        if not par.strip():
            y -= 0.03
            continue
        is_formula = par.startswith('$')
        if is_formula and r'\frac' in par:
            y -= 0.035          # tall fractions extend above the measured box
        t = ax.text(0.04, y, par if is_formula else _wrap_math_safe(par, width),
                    transform=ax.transAxes, fontsize=fontsize + 2.5 if is_formula else fontsize,
                    color=INK if is_formula else INK2, va='top', linespacing=1.4)
        y -= t.get_window_extent(renderer).height / ax_h + (0.035 if is_formula else 0.022)


def box(ax, cx, cy, w, h, text, fc=PANEL, ec=LIGHT, fontsize=8.5, color=INK, bold=False, lw=1.0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=lw))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, color=color,
            fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, color=INK2, lx=0.08, ly=0.0, ha='left'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.1, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=8, color=color,
                fontweight='bold', ha=ha, va='center')


def draw_fit(ax, s, b, color=RED, lw=2.0, x0=-1, x1=1, label=None):
    """Fitted phase line 2*pi*s*x + b, wrapped to [0, 360) and drawn twice (0-720)."""
    xg = np.linspace(x0, x1, 800)
    phi = np.rad2deg(np.mod(2 * np.pi * s * xg + b, 2 * np.pi))
    phi = break_wraps(phi, 180)
    ax.plot(xg, phi, color=color, lw=lw, label=label, zorder=4)
    ax.plot(xg, phi + 360, color=color, lw=lw, zorder=4)


def phase_axes(ax, ylabel='Theta phase of spike (deg)'):
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 720)
    ax.set_yticks(np.arange(0, 721, 90))
    for yy in (0, 360, 720):
        ax.axhline(yy, color=GRID, lw=0.8, zorder=0)
    for yy in (180, 540):
        ax.axhline(yy, color=GRID, lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('Pass index  (-1 = entering ... 0 = field centre ... +1 = leaving)')
    ax.set_ylabel(ylabel)


def double(phase_rad):
    d = np.rad2deg(np.mod(phase_rad, 2 * np.pi))
    return np.concatenate([d, d + 360])


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {path.name}')


# ============================================================================
# Simulation
# ============================================================================

def simulate_behaviour():
    n = int(round(DURATION_S * FS))
    t = T0 + np.arange(n) / FS
    x, y = np.zeros(n), np.zeros(n)
    running = np.zeros(n, dtype=bool)
    run_n, rest_n = int(RUN_S * FS), int(REST_S * FS)
    i, pos, k, y_now = 0, 0.0, 0, TRACK_WIDTH / 2
    while i < n:
        target = TRACK_LEN - pos
        j = min(i + run_n, n)
        tau = np.arange(j - i) / run_n
        xs = pos + (target - pos) * (1 - np.cos(np.pi * tau)) / 2
        ys = TRACK_WIDTH / 2 + 12 * np.sin(2 * np.pi * xs / 80 + 1.7 * k)
        x[i:j], y[i:j], running[i:j] = xs, ys, True
        y_now = ys[-1]
        i, pos, k = j, target, k + 1
        if k % REST_EVERY_RUNS == 0 and i < n:
            j = min(i + rest_n, n)
            x[i:j], y[i:j] = pos, y_now
            i = j
    win = signal.windows.hann(401)
    env = np.clip(np.convolve(running.astype(float), win / win.sum(), mode='same'), 0, 1)
    return t, x, y, running, env


def simulate_lfp(rng, t, env):
    n = len(t)
    lp = signal.butter(2, 0.2, fs=FS, output='sos')
    slow = signal.sosfiltfilt(lp, rng.standard_normal(n))
    slow /= slow.std()
    slow2 = signal.sosfiltfilt(lp, rng.standard_normal(n))
    slow2 /= slow2.std()
    theta_phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(THETA_HZ + 0.35 * slow) / FS
    amp = (0.12 + 0.88 * env) * (1 + 0.15 * slow2)
    delta = signal.sosfiltfilt(signal.butter(2, [1.0, 3.0], btype='band', fs=FS, output='sos'),
                               rng.standard_normal(n))
    delta /= delta.std()
    bg = signal.sosfiltfilt(signal.butter(2, 40.0, fs=FS, output='sos'), rng.standard_normal(n))
    bg /= bg.std()
    lfp = 150.0 * (amp * np.cos(theta_phase) + (0.2 + 1.2 * (1 - env)) * delta + 0.2 * bg)
    # a few movement/chewing-like artefacts during running
    for c in rng.choice(np.flatnonzero(env > 0.99), 4, replace=False):
        lfp += 3000.0 * np.exp(-0.5 * ((t - t[c]) / 0.008) ** 2)
    return lfp, theta_phase


def pass_index_steps(pos_ts, pos_xy, spk_ts_all):
    """Step 3a/3b intermediates, computed exactly as in pp.compute_pass_index."""
    n_dims = pos_xy.shape[1]
    binside = 2.0 * n_dims if pp.BINSIDE == 'auto' else pp.BINSIDE
    smth = 3.0 * binside if pp.SMTH_WIDTH == 'auto' else pp.SMTH_WIDTH
    spk_xy, _ = pp.spk_pos(pos_ts, pos_xy, spk_ts_all)
    rmap, occ, xe, ye = pp.rate_map(pos_ts, pos_xy, spk_xy, binside, smth)
    fi_map = pp.field_index_map(rmap, occ, pp.METHOD)
    fi = pp.field_index_per_position(pos_xy, fi_map, xe, ye)
    cc, ts2, resampled = pp.sample_along_arc(pos_ts, pos_xy, fi)
    band = (pp.auto_filter_band(pp.METHOD, rmap, occ, binside, n_dims)
            if pp.FILTER_BAND == 'auto' else pp.FILTER_BAND)
    fs_arc = 1.0 / np.mean(np.diff(cc))
    ffi = pp.bandpass_filter(resampled, band[0], band[1], fs_arc)
    peak = np.nanmax(rmap)
    area = float(np.sum(rmap > 0.2 * peak)) * binside ** n_dims
    return SimpleNamespace(binside=binside, smth=smth, rmap=rmap, occ=occ, xe=xe, ye=ye,
                           fi_map=fi_map, cc=cc, ts2=ts2, resampled=resampled, band=band,
                           fs_arc=fs_arc, ffi=ffi, area=area, radius=np.sqrt(area / np.pi),
                           spk_xy=spk_xy)


def pass_index_at(steps, times):
    return wrap_pm_pi(pp.peak_interp_phase(steps.ffi, steps.ts2, times)) / np.pi


def simulate_spikes(rng, t, x, env, theta_phase, p_at_t, psi_fn, kappa, peak=PEAK_RATE):
    g = np.exp(-(x - FIELD_CENTER) ** 2 / (2 * FIELD_SIGMA ** 2))
    if kappa > 0:
        psi = np.deg2rad(psi_fn(p_at_t))
        m = np.exp(kappa * np.cos(theta_phase - psi)) / i0(kappa)
        m[~np.isfinite(m)] = 1.0
    else:
        m = 1.0
    lam = BASE_RATE + peak * g * env * m
    fire = rng.random(len(t)) < lam / FS
    return np.sort(t[fire] + rng.uniform(0, 0.4 / FS, fire.sum()))


def build_session():
    rng = np.random.default_rng(SEED)
    t, x, y, running, env = simulate_behaviour()
    lfp, theta_phase = simulate_lfp(rng, t, env)
    pos_ts = t[::TRACK_STEP]
    pos_xy = np.column_stack([x, y])[::TRACK_STEP]

    # pass-index map of the field (from a place-only cell) used to DESIGN phase behaviour
    proto = simulate_spikes(rng, t, x, env, theta_phase, None, None, 0.0)
    p_at_t = pass_index_at(pass_index_steps(pos_ts, pos_xy, proto), t)

    for c in CELLS:
        c['spk_all'] = simulate_spikes(rng, t, x, env, theta_phase, p_at_t, c['psi'], c['kappa'],
                                       c.get('peak', PEAK_RATE))

    filtered = pp.bandpass_filter(lfp, pp.LFP_FILTER_BAND[0], pp.LFP_FILTER_BAND[1], FS)
    acg_ok, ed_min = pp.acg_theta_epoch_mask(lfp, FS)
    theta_mask = pp.times_in_ok_epochs(t, t[0], acg_ok)
    return SimpleNamespace(t=t, x=x, y=y, running=running, env=env, lfp=lfp, filtered=filtered,
                           theta_phase=theta_phase, pos_ts=pos_ts, pos_xy=pos_xy,
                           acg_ok=acg_ok, ed_min=ed_min, theta_mask=theta_mask)


# ============================================================================
# Run the pipeline (same calls / order as pp.process_session)
# ============================================================================

def run_cell(c, S, idx):
    spk_all = c['spk_all']
    spk = spk_all[pp.times_in_ok_epochs(spk_all, S.t[0], S.acg_ok)]       # ACG theta-positive spikes
    metrics, phase_deg = pp.compute_theta_modulation(spk, S.t, S.filtered, FS,
                                                     np.random.default_rng(SEED + idx))
    ph = pp.assign_spike_phase(spk, S.t, S.filtered, FS)
    spk_ph = spk[np.isfinite(ph)]
    # identical RNG stream -> identical shuffle distribution to the one inside compute_theta_modulation
    _p, shuffle_tmis = pp.shuffle_tmi_significance(spk_ph, phase_deg, metrics['TMI'],
                                                   np.random.default_rng(SEED + idx))

    t_start, t_stop = max(S.pos_ts.min(), S.t.min()), min(S.pos_ts.max(), S.t.max())
    spk_ov = spk[(spk >= t_start) & (spk <= t_stop)]
    spk_all_ov = spk_all[(spk_all >= t_start) & (spk_all <= t_stop)]
    tested = bool(metrics['TMI_Significant']) and len(spk_ov) >= pp.MIN_SPIKES_FOR_FIT
    # Step 3 is computed for every cell so it can be drawn, but only TMI-significant
    # cells are 'tested' (classified) -- exactly as the pipeline gates it.
    res = pp.compute_pass_index(S.pos_ts, S.pos_xy, spk_all_ov, S.t, S.lfp, FS,
                                np.random.default_rng(SEED + idx),
                                method=pp.METHOD, binside=pp.BINSIDE, smth_width=pp.SMTH_WIDTH,
                                filter_band=pp.FILTER_BAND, lfp_filter_band=pp.LFP_FILTER_BAND,
                                slope_bnds=pp.SLOPE_BNDS, phase_spk_ts=spk_ov,
                                lfp_theta_mask=S.theta_mask)
    steps = pass_index_steps(S.pos_ts, S.pos_xy, spk_all_ov)
    c.update(spk=spk, spk_ph=spk_ph, phase_deg=phase_deg, metrics=metrics,
             shuffle_tmis=shuffle_tmis, res=res, steps=steps, tested=tested,
             cls=res['precession_class'] if tested else 'not_tested', spk_ov=spk_ov)


# ============================================================================
# Figures
# ============================================================================

def fig00_overview():
    fig, ax = plt.subplots(figsize=(11, 15.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 31)
    ax.axis('off')
    cx, w, h = 4.3, 7.4, 1.45
    gate_fc, stop_fc = '#fdf0cc', '#efeeea'
    ax.text(0.2, 30.6, 'Phase-precession pipeline after ACG theta-epoch extraction (one unit)',
            fontsize=13, fontweight='bold', color=INK)
    rows = [
        (29.0, 'INPUT for one unit\nonly spikes inside theta-positive 1-s epochs (ACG ED_min < 1)\n'
               '+ LFP band-passed 3-7 Hz  + tracking (x, y in cm)', PANEL),
        (27.0, 'STEP 1   Theta phase of every spike\npeak-to-peak linear interpolation: '
               '0 deg at an LFP peak -> 360 deg at the next peak', PANEL),
        (25.0, 'STEP 1b  Phase-locking statistics (reported, not a gate)\n'
               'mean resultant length (MRL), preferred phase, Rayleigh p, polar plot', PANEL),
        (23.0, 'STEP 2   Theta Modulation Index (TMI)\nTMI = 1 - trough of the smoothed, peak-normalised '
               'phase histogram\ncompared with 1000 burst-preserving random-phase shuffles', PANEL),
        (21.0, 'GATE 1   Is TMI shuffle p < 0.05 ?', gate_fc),
        (19.2, 'GATE 2   >= 50 theta spikes inside the tracking & LFP overlap ?', gate_fc),
        (17.4, 'STEP 3a  Rate map (ALL spikes) -> Field-index map  (0 = silent, 1 = peak)', PANEL),
        (15.4, 'STEP 3b  Field index along the path -> band-pass filter -> peak-to-peak phase\n'
               '-> PASS INDEX per spike  (-1 entering, 0 field centre, +1 leaving)', PANEL),
        (13.4, 'STEP 3c  Circular-linear fit (Kempter et al. 2012)\n'
               'phase = 2*pi*s*PI + b ;  s, b chosen to maximise R(s)  ->  slope s, intercept b, fit_R', PANEL),
        (11.4, 'STEP 3d  Circular-linear correlation\nposition -> angle phi = 2*pi*|s|*PI ;  '
               'rho(phi, theta)  ->  z  ->  p', PANEL),
        (9.4, 'GATE 3   p < 0.05  AND  wrapped fit line has <= 3 segments ?', gate_fc),
        (7.6, 'STEP 3e  slope in deg/pass = 720 * s', PANEL),
    ]
    hts = {y0: (h if '\n' in text else 1.0) for y0, text, _fc in rows}
    for y0, text, fc in rows:
        box(ax, cx, y0, w, hts[y0], text, fc=fc, bold=fc == gate_fc)
    for (y1, *_), (y2, *_) in zip(rows[:-1], rows[1:]):
        arrow(ax, cx, y1 - hts[y1] / 2, cx, y2 + hts[y2] / 2,
              'yes' if y1 in (21.0, 19.2, 9.4) else None)
    # stops
    sx, sw = 9.55, 2.6
    box(ax, sx, 21.0, sw, 1.3, 'STOP\nnot theta-modulated\nPrecessionTested = False', fc=stop_fc, fontsize=8)
    arrow(ax, cx + w / 2, 21.0, sx - sw / 2, 21.0, 'no', lx=-0.15, ly=0.25)
    box(ax, sx, 19.2, sw, 1.0, 'STOP\ntoo few spikes', fc=stop_fc, fontsize=8)
    arrow(ax, cx + w / 2, 19.2, sx - sw / 2, 19.2, 'no', lx=-0.15, ly=0.25)
    box(ax, sx, 9.4, sw, 2.3, 'p >= 0.05:\nNO PHASE RELATION\n\n> 3 segments:\nnon_precessing\n(MultiLinesFit)',
        fc=stop_fc, fontsize=8, color=INK2)
    arrow(ax, cx + w / 2, 9.4, sx - sw / 2, 9.4, 'no', lx=-0.15, ly=0.25)
    # classes
    cls_y = 4.9
    for xx, (k, rule) in zip((1.6, 4.3, 7.0), (('phase_precessing', 'slope < -15 deg/pass'),
                                                ('phase_locked', '-15 <= slope <= +15'),
                                                ('phase_succeeding', 'slope > +15 deg/pass'))):
        label, col = CLASS_STYLE[k]
        box(ax, xx, cls_y, 2.5, 1.3, f'{rule}\n\n{label}', fc='white', ec=col, lw=2.2, fontsize=8.5)
        arrow(ax, cx, 7.1, xx, cls_y + 0.67)
    box(ax, cx, 2.3, w + 2.5, 1.7,
        'Also quantified for every tested cell:\nr^2 = rho^2  |  fit_R (goodness of fit)  |  '
        'phase range = |slope| x (PI extent)\nphase at PI -1 / 0 / +1 (field entry / centre / exit) '
        '+ precession (70-250 deg) or procession range\nn_fit_lines  |  spike-density map '
        '(pass index x theta phase)', fc=PANEL, fontsize=8.5)
    ax.text(0.2, 0.4, 'Numbers in grey boxes are the current config of ThetaMod_PhasePrec_v19 '
            '(ALPHA = 0.05, SLOPE_THRESH_DEG_PER_PASS = 15, MAX_FIT_LINES = 3, MIN_SPIKES = 50).',
            fontsize=8, color=MUTED)
    save(fig, 'fig00_pipeline_overview.png')


def fig01_acg_recap(S, c):
    t_rest = S.t[np.flatnonzero((~S.running) & (S.t > T0 + 40))[0]]
    w0, w1 = t_rest - 9, t_rest + 11
    sel = (S.t >= w0) & (S.t <= w1)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.2, 0.8, 1.2], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.plot(S.t[sel] - w0, S.lfp[sel], color=INK2, lw=0.5, label='raw LFP')
    ax.plot(S.t[sel] - w0, S.filtered[sel], color=BLUE, lw=1.0, label='3-7 Hz band-pass')
    e0 = int(np.floor(w0 - S.t[0]))
    for e in range(e0, e0 + 22):
        es = S.t[0] + e - w0
        if e >= len(S.acg_ok) or es > w1 - w0:
            break
        col = AQUA if S.acg_ok[e] else LIGHT
        ax.axvspan(max(es, 0), es + 1, color=col, alpha=0.18 if S.acg_ok[e] else 0.35, lw=0)
    ax.set_xlim(0, w1 - w0)
    ax.set_ylabel('LFP (uV)')
    ax.set_title('Recap - which 1-s epochs count as theta', pad=22)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1, label='raw LFP'),
                       Line2D([], [], color=BLUE, lw=1.4, label='3-7 Hz band-pass'),
                       Patch(color=AQUA, alpha=0.3, label='theta-positive epoch (kept)'),
                       Patch(color=LIGHT, alpha=0.6, label='rejected epoch')],
              loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=8)
    ax.text(t_rest - w0 + 0.2, ax.get_ylim()[0] * 0.92, 'animal pauses here -> delta, no theta',
            fontsize=8, color=INK2, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

    ax = fig.add_subplot(gs[1])
    idx = np.arange(e0, min(e0 + 21, len(S.ed_min)))
    centers = S.t[0] + idx + 0.5 - w0
    ed = S.ed_min[idx]
    fin = np.isfinite(ed)
    ax.scatter(centers[fin], ed[fin], s=40, c=np.where(ed[fin] < pp.ACG_ED_MIN_THRESH, AQUA, MUTED),
               edgecolors='white', lw=1, zorder=3)
    ax.scatter(centers[~fin], np.full((~fin).sum(), 1.35), marker='x', s=40, color=MUTED, zorder=3)
    ax.axhline(pp.ACG_ED_MIN_THRESH, color=INK2, ls='--', lw=1)
    ax.text(0.1, pp.ACG_ED_MIN_THRESH + 0.04, 'threshold ED_min = 1', fontsize=8, color=INK2)
    ax.text(0.1, 1.42, 'x = screened out before the ACG test (delta > theta power, or amplitude artefact)',
            fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.6)
    ax.set_xlim(0, w1 - w0)
    ax.set_ylabel('ED_min')
    ax.set_title('Distance between each epoch\'s autocorrelogram and the closest 3-7 Hz sinusoid', fontsize=9.5)

    ax = fig.add_subplot(gs[2])
    s_all = c['spk_all'][(c['spk_all'] >= w0) & (c['spk_all'] <= w1)]
    kept = pp.times_in_ok_epochs(s_all, S.t[0], S.acg_ok)
    ax.vlines(s_all[kept] - w0, 0.55, 1.0, color=BLUE, lw=1.2)
    ax.vlines(s_all[~kept] - w0, 0.0, 0.45, color=MUTED, lw=1.2)
    ax.set_yticks([0.22, 0.78])
    ax.set_yticklabels(['dropped', 'kept'])
    ax.set_xlim(0, w1 - w0)
    ax.set_title(f'Cell {c["key"]} spikes: only the blue ones enter everything that follows', fontsize=9.5)

    ax = fig.add_subplot(gs[3])
    ax.plot(S.t[sel] - w0, S.x[sel], color=INK2, lw=1)
    ax.axhspan(FIELD_CENTER - 1.79 * FIELD_SIGMA, FIELD_CENTER + 1.79 * FIELD_SIGMA, color=BLUE, alpha=0.08)
    ax.text(0.1, FIELD_CENTER, 'place field', color=BLUE, fontsize=8, va='center')
    ax.set_xlim(0, w1 - w0)
    ax.set_ylabel('x position (cm)')
    ax.set_xlabel('Time in window (s)')
    n_kept = int(S.acg_ok.sum())
    fig.suptitle(f'Starting point: ACG theta-positive epochs  ({n_kept}/{len(S.acg_ok)} epochs kept in '
                 f'this session; cell {c["key"]}: {len(c["spk"])}/{len(c["spk_all"])} spikes kept)',
                 x=0.125, ha='left', fontsize=12, fontweight='bold', y=0.97)
    save(fig, 'fig01_acg_theta_epochs_recap.png')


def busiest_window(spk, width):
    edges = np.arange(spk.min(), spk.max(), width / 4)
    counts = np.array([np.sum((spk >= e) & (spk < e + width)) for e in edges])
    return edges[np.argmax(counts)]


def fig02_phase(S, c):
    w = 1.6
    w0 = busiest_window(c['spk_ph'][c['spk_ph'] < T0 + 300], w)
    w1 = w0 + w
    sel = (S.t >= w0) & (S.t <= w1)
    tt = S.t[sel]
    pk, _ = signal.find_peaks(S.filtered)
    pk_t = S.t[pk]
    pk_w = pk_t[(pk_t >= w0) & (pk_t <= w1)]
    spk = c['spk_ph'][(c['spk_ph'] >= w0) & (c['spk_ph'] <= w1)]
    sph = np.rad2deg(pp.assign_spike_phase(spk, S.t, S.filtered, FS))
    phase_trace = np.rad2deg(np.mod(pp.peak_interp_phase(S.filtered, S.t, tt), 2 * np.pi))

    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.3, 1], width_ratios=[1, 1, 1], hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tt - w0, S.lfp[sel], color=LIGHT, lw=0.8, label='raw LFP')
    ax.plot(tt - w0, S.filtered[sel], color=BLUE, lw=1.6, label='3-7 Hz band-passed LFP')
    ax.scatter(pk_w - w0, S.filtered[np.searchsorted(S.t, pk_w)], s=45, color=VIOLET, zorder=4,
               edgecolors='white', label='peaks (0 deg)')
    for p in pk_w:
        ax.axvline(p - w0, color=VIOLET, lw=0.6, ls=':', alpha=0.7)
    yl = ax.get_ylim()
    ax.vlines(spk - w0, yl[0], yl[0] + 0.15 * (yl[1] - yl[0]), color=ORANGE, lw=1.6, label='spikes')
    ax.set_xlim(0, w)
    ax.set_ylabel('LFP (uV)')
    ax.legend(loc='upper right', ncol=4, fontsize=8)
    ax.set_title('1) Band-pass the LFP to 3-7 Hz and find every peak')

    # worked example spike
    j = len(spk) // 2
    ts = spk[j]
    k = np.searchsorted(pk_t, ts) - 1
    tk, tk1 = pk_t[k], pk_t[k + 1]
    ax.annotate('', xy=(tk1 - w0, yl[1] * 0.92), xytext=(tk - w0, yl[1] * 0.92),
                arrowprops=dict(arrowstyle='<->', color=INK2, lw=1))
    ax.text((tk + tk1) / 2 - w0, yl[1] * 0.97, f'one theta cycle = {1000 * (tk1 - tk):.0f} ms',
            ha='center', fontsize=8, color=INK2)

    ax = fig.add_subplot(gs[1, :], sharex=ax)
    ax.plot(tt - w0, break_wraps(phase_trace, 180), color=BLUE, lw=1.6, label='interpolated phase')
    ax.scatter(spk - w0, sph, s=40, color=ORANGE, edgecolors='white', zorder=4, label='spike phase')
    for p in pk_w:
        ax.axvline(p - w0, color=VIOLET, lw=0.6, ls=':', alpha=0.7)
    ax.scatter([ts - w0], [sph[j]], s=140, facecolors='none', edgecolors=INK, lw=1.4, zorder=5)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel('Theta phase (deg)')
    ax.set_xlabel('Time in window (s)')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.set_title('2) Phase rises in a straight line from 0 deg (peak) to 360 deg (next peak); '
                 'read it off at each spike')

    frac = (ts - tk) / (tk1 - tk)
    note(fig.add_subplot(gs[2, 0:2]),
         'For a spike at time t that falls between peak k (time t_k) and peak k+1 (time t_k+1):\n'
         r'$\phi_{spike} = 360^\circ \times \frac{t - t_k}{t_{k+1} - t_k}$' '\n'
         f'Circled spike: it fired {1000 * (ts - tk):.0f} ms after the previous peak, inside a '
         f'{1000 * (tk1 - tk):.0f} ms cycle, so phase = 360 x {frac:.3f} = {360 * frac:.0f} deg.\n'
         '0 deg = LFP peak, 180 deg = LFP trough. Because each cycle is stretched to its own '
         'length, a slower or faster cycle does not distort the phase. Spikes before the first or '
         'after the last peak, or with no LFP sample within ~1 ms, get no phase and are dropped.',
         title='Formula (peak_interp_phase / assign_spike_phase)', width=95)
    note(fig.add_subplot(gs[2, 2]),
         'Why not the Hilbert transform? Interpolating between peaks gives the same answer for a '
         'clean sinusoid but is robust to the asymmetric (saw-tooth) theta waves seen in real '
         'hippocampal LFP. The SAME peak-interpolation rule is reused later to turn the '
         'field-index signal into the pass index.', title='Why this method', width=44)
    fig.suptitle('STEP 1 - giving every spike a theta phase', x=0.125, ha='left',
                 fontsize=12, fontweight='bold', y=0.97)
    save(fig, 'fig02_spike_theta_phase.png')


def fig03_locking(c):
    ph = np.deg2rad(c['phase_deg'])
    m = c['metrics']
    rng = np.random.default_rng(1)
    sub = rng.choice(len(ph), min(70, len(ph)), replace=False)
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0], projection='polar')
    for a in ph[sub]:
        ax.plot([a, a], [0, 1], color=BLUE, lw=0.6, alpha=0.35)
    ax.scatter(ph[sub], np.ones(len(sub)), s=14, color=BLUE, alpha=0.8, zorder=3)
    pref = np.deg2rad(m['PreferredPhase_deg'])
    ax.annotate('', xy=(pref, m['MRL']), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=3, shrinkA=0, shrinkB=0))
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(['0.5', '1'], fontsize=7)
    ax.set_title('a) every spike = an arrow of length 1\n    pointing at its phase', pad=14)
    ax.text(0.5, -0.13, f'red = average of all arrows: length MRL = {m["MRL"]:.2f}, '
            f'direction = {m["PreferredPhase_deg"]:.0f} deg', color=RED, fontsize=8.5, ha='center',
            transform=ax.transAxes)

    ax = fig.add_subplot(gs[0, 1], projection='polar')
    edges = np.radians(np.arange(0, 361, pp.PHASE_BIN_SIZE_DEG))
    counts, _ = np.histogram(ph, bins=edges)
    ax.bar(edges[:-1], counts, width=np.radians(pp.PHASE_BIN_SIZE_DEG), align='edge',
           color='#b7d3f6', edgecolor='white', lw=1.5)
    ax.plot([0, pref], [0, m['MRL'] * counts.max()], color=RED, lw=3)
    ax.set_title(f'b) polar histogram ({pp.PHASE_BIN_SIZE_DEG} deg bins)\n'
                 f'    preferred phase = {m["PreferredPhase_deg"]:.0f} deg', pad=14)

    ax = fig.add_subplot(gs[0, 2])
    e = np.arange(0, 361, pp.PHASE_BIN_SIZE_DEG)
    cnt, _ = np.histogram(c['phase_deg'], bins=e)
    ctr = e[:-1] + pp.PHASE_BIN_SIZE_DEG / 2
    ax.bar(np.concatenate([ctr, ctr + 360]), np.concatenate([cnt, cnt]), width=pp.PHASE_BIN_SIZE_DEG - 4,
           color='#b7d3f6')
    xx = np.linspace(0, 720, 400)
    ax.plot(xx, (0.5 + 0.5 * np.cos(np.deg2rad(xx))) * cnt.max() * 1.1, color=INK2, lw=1, ls='--',
            label='LFP shape (peak at 0/360)')
    ax.set_xticks(np.arange(0, 721, 180))
    ax.set_xlabel('Theta phase (deg), drawn twice')
    ax.set_ylabel('Spike count')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('c) same histogram, unrolled')

    note(fig.add_subplot(gs[1, :2]),
         'Phases are angles: 350 deg and 10 deg are only 20 deg apart, so an ordinary average '
         '(180 deg) would be wrong. Instead each spike is turned into an arrow of length 1 and the '
         'arrows are averaged.\n'
         r'$MRL = \bar{R} = \frac{1}{n}\sqrt{(\sum_j \cos\phi_j)^2 + (\sum_j \sin\phi_j)^2}$'
         r'$\qquad \bar{\phi} = \mathrm{atan2}(\sum_j \sin\phi_j,\ \sum_j \cos\phi_j)$' '\n'
         'MRL = 0: arrows cancel (no preferred phase). MRL = 1: every spike at exactly the same phase. '
         'The direction of the average arrow is the preferred phase. Rayleigh test of "is MRL bigger '
         'than chance?":\n'
         r'$R_n = n\bar{R},\qquad p = \exp(\sqrt{1 + 4n + 4(n^2 - R_n^2)} - (1 + 2n))$' '\n'
         f'Cell {c["key"]}: n = {m["n_spikes_theta"]} spikes, MRL = {m["MRL"]:.3f}, preferred phase = '
         f'{m["PreferredPhase_deg"]:.0f} deg, Rayleigh p = {m["Rayleigh_p"]:.2g}.',
         title='What is computed (circ_r, circ_mean, circ_rtest)', width=110)
    note(fig.add_subplot(gs[1, 2]),
         'These numbers (MRL, PreferredPhase_deg, Rayleigh_p, SignificantThetaModulation) go to the '
         'Excel sheet and colour the polar plot, but they do NOT decide whether Step 3 runs. '
         'That decision is made by the TMI shuffle test (next figure). A precessing cell spreads '
         'its spikes over many phases, so its MRL is only moderate even though its firing is '
         'strongly organised by theta.', title='Reported, not a gate', width=44)
    fig.suptitle(f'STEP 1b - how strongly does cell {c["key"]} prefer one theta phase?', x=0.125,
                 ha='left', fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig03_phase_locking_MRL_rayleigh.png')


def fig04_tmi(cA, cE):
    m = cA['metrics']
    phases = cA['phase_deg']
    tiled = np.concatenate([phases + j * 360 for j in range(5)])
    edges = np.arange(0, 5 * 360, 36)
    counts, edges = np.histogram(tiled, bins=edges)
    sm = pp._gauss_smooth_1d(counts.astype(float), n=7, sigma=0.5)
    left = edges[:-1]
    tmi, edges2, centres, norm = pp.calc_tmi(phases, return_hist=True)

    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    ax.bar(left + 18, counts, width=32, color='#cde2fb', label='counts (36 deg bins)')
    ax.plot(left + 18, sm, color=BLUE, lw=2, label='Gaussian-smoothed (7-pt, sigma 0.5 bin)')
    ax.axvspan(360, 1080 + 36, color=YELLOW, alpha=0.12, lw=0)
    ax.text(380, counts.max() * 1.2, 'middle part kept (edge-free)', fontsize=8, color=INK2)
    for k in range(6):
        ax.axvline(360 * k, color=GRID, lw=0.8)
    ax.set_ylim(0, counts.max() * 1.45)
    ax.set_xlim(0, 1800)
    ax.set_xticks(np.arange(0, 1801, 360))
    ax.set_xlabel('Phase copied over 5 cycles (deg)')
    ax.set_ylabel('Spikes')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title('a) Copy the phases over 5 cycles, histogram, smooth (copying avoids edge effects at 0/360)')

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(centres, norm, width=32, color='#cde2fb')
    ax.plot(centres, norm, color=BLUE, lw=2)
    imin = np.argmin(norm)
    ax.axhline(norm.min(), color=INK2, ls='--', lw=1)
    ax.annotate('', xy=(centres[imin], 1.0), xytext=(centres[imin], norm.min()),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=2))
    ax.text(centres[imin] + 25, (1 + norm.min()) / 2, f'TMI = 1 - {norm.min():.2f}\n     = {tmi:.2f}',
            color=RED, fontsize=9, fontweight='bold', va='center',
            bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.5))
    ax.set_ylim(0, 1.12)
    ax.set_xlabel('Phase (deg)')
    ax.set_ylabel('Normalised (max = 1)')
    ax.set_title('b) divide by max; TMI = 1 - trough')

    # shuffle example
    gid = pp._burst_groups(cA['spk_ph'], phases)
    rng = np.random.default_rng(99)
    sh = rng.uniform(0, 360, gid.max() + 1)[gid]
    tmi_s, _e, cen_s, norm_s = pp.calc_tmi(sh, return_hist=True)
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(cen_s, norm_s, width=32, color=GRID)
    ax.plot(cen_s, norm_s, color=INK2, lw=2)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel('Phase (deg)')
    ax.set_title(f'c) ONE shuffle: random phases\n    TMI = {tmi_s:.2f}')
    ax.text(0.02, 0.06, f'{len(phases)} spikes -> {gid.max() + 1} burst groups\n'
            '(spikes < 50 ms apart in the same 36 deg bin\n share one random phase)',
            transform=ax.transAxes, fontsize=7.5, color=INK2)

    for col, cc in ((1, cA), (2, cE)):
        mm = cc['metrics']
        ax = fig.add_subplot(gs[1, col])
        ax.hist(cc['shuffle_tmis'], bins=np.linspace(0, 1, 81), color=LIGHT, edgecolor='white')
        sig = mm['TMI_Significant']
        colr = BLUE if sig else MUTED
        ax.axvline(mm['TMI'], color=colr, lw=2.5)
        ax.text(mm['TMI'], ax.get_ylim()[1] * 0.9, f'  observed\n  TMI = {mm["TMI"]:.2f}', color=colr,
                fontsize=8.5, fontweight='bold')
        ax.set_xlim(0, max(1.0, mm['TMI'] + 0.15))
        ax.set_xlabel('TMI')
        ax.set_ylabel('Number of shuffles')
        verdict = 'theta-modulated -> continue' if sig else 'NOT modulated -> STOP'
        ax.set_title(f'{"d" if col == 1 else "e"}) Cell {cc["key"]}: 1000 shuffles\n'
                     f'    p = {mm["TMI_shuffle_p"]:.3g}  ({verdict})')

    note(fig.add_subplot(gs[2, :]),
         r'$TMI = 1 - \min\left(\frac{h_{smooth}}{\max(h_{smooth})}\right)$'
         '   - near 1: the cell is almost silent at its worst phase; near 0: it fires equally at all phases.\n'
         'Null test (Frank et al. 2001): every spike gets a random phase (0-360) and TMI is recomputed, '
         '1000 times. Bursts (spikes < 50 ms apart that fell in the same real 36-deg bin) keep one shared '
         'random phase, so bursting cannot fake modulation.\n'
         r'$p_{TMI} = \frac{1 + \mathrm{number\ of\ shuffles\ with}\ TMI_{shuffle} \geq TMI_{observed}}{1 + 1000}$'
         '\nGATE 1: only cells with p < 0.05 (TMI_Significant = True) go on to Step 3.\n'
         f'Cell {cA["key"]}: TMI = {m["TMI"]:.3f}, p = {m["TMI_shuffle_p"]:.3g}.   '
         f'Cell {cE["key"]}: TMI = {cE["metrics"]["TMI"]:.3f}, p = {cE["metrics"]["TMI_shuffle_p"]:.3g}. '
         'Note the shuffle TMIs are not near 0 - random phases still give a bumpy histogram - which is why '
         'a shuffle test is needed rather than a fixed TMI cut-off.',
         title='Formulae (calc_tmi, shuffle_tmi_significance)', width=150)
    fig.suptitle('STEP 2 - Theta Modulation Index and its shuffle test (GATE 1)', x=0.125, ha='left',
                 fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig04_TMI_and_shuffle_test.png')


def fig05_maps(S, c):
    st = c['steps']
    ext = [st.xe[0], st.xe[-1], st.ye[0], st.ye[-1]]
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], hspace=0.45, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(S.pos_xy[:, 0], S.pos_xy[:, 1], color=LIGHT, lw=0.3, alpha=0.4)
    ax.scatter(st.spk_xy[:, 0], st.spk_xy[:, 1], s=5, color=BLUE, alpha=0.6, edgecolors='none')
    ax.set_aspect('equal')
    ax.set_title(f'a) path (grey) + every spike of cell {c["key"]} (blue)')
    ax.set_ylabel('y (cm)')
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(st.rmap.T, origin='lower', extent=ext, cmap=SEQ_CMAP)
    fig.colorbar(im, ax=ax, label='Hz', fraction=0.025)
    ax.set_title(f'b) rate map: spikes / time per {st.binside:g} cm bin, smoothed')
    ax.set_ylabel('y (cm)')
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(st.fi_map.T, origin='lower', extent=ext, cmap=SEQ_CMAP, vmin=0, vmax=1)
    xc = 0.5 * (st.xe[:-1] + st.xe[1:])
    yc = 0.5 * (st.ye[:-1] + st.ye[1:])
    ax.contour(xc, yc, st.fi_map.T, levels=[0.2], colors=[ORANGE], linewidths=1.8)
    fig.colorbar(im, ax=ax, label='field index', fraction=0.025)
    ax.set_title('c) field-index map (0-1); orange outline = bins above 20 % of peak')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    note(fig.add_subplot(gs[:, 1]),
         '1. Occupancy: time spent in each bin (samples x tracking interval).\n'
         '2. Spike count per bin (position = nearest tracking sample to the spike).\n'
         r'$rate(bin) = \frac{spike\ count(bin)}{time(bin)}$' '\n'
         f'3. Smooth with a Gaussian (width {st.smth:g} cm). Unvisited bins = 0.\n'
         '   ALL spikes are used here (not only theta spikes) so the place field is defined from '
         'the full data.\n'
         '4. Field index = the rate map rescaled to 0-1 ("place" method):\n'
         r'$FI = \frac{rate - rate_{min}}{rate_{max} - rate_{min}}$' '\n'
         '   FI tells you, at every spot, "how deep inside the field am I?" (1 = at the peak).\n'
         '5. Field size for the next step: count bins with rate > 20 % of the peak.\n'
         r'$A = N_{bins} \times binside^2,\quad r = \sqrt{A/\pi}$' '\n'
         f'   Cell {c["key"]}: A = {st.area:.0f} cm2  ->  r = {st.radius:.1f} cm\n'
         '6. That radius sets the spatial band-pass filter used to find passes:\n'
         r'$band = (\frac{1}{6r},\ \frac{3}{r})$  cycles per cm of path' '\n'
         f'   = ({st.band[0]:.4f}, {st.band[1]:.3f}) cycles/cm, i.e. features between '
         f'{1 / st.band[1]:.0f} cm and {1 / st.band[0]:.0f} cm long.',
         title='What happens (rate_map, field_index_map, auto_filter_band)', width=58)
    fig.suptitle('STEP 3a - from spikes to a field-index map', x=0.125, ha='left',
                 fontsize=12, fontweight='bold', y=0.97)
    save(fig, 'fig05_ratemap_fieldindex.png')


def fig06_pass_index(S, c):
    st = c['steps']
    t_rest = S.t[np.flatnonzero((~S.running) & (S.t > T0 + 40))[0]]
    w0, w1 = t_rest - 16, t_rest + 14
    psel = (S.pos_ts >= w0) & (S.pos_ts <= w1)
    asel = (st.ts2 >= w0) & (st.ts2 <= w1)
    spk = c['spk_ov'][(c['spk_ov'] >= w0) & (c['spk_ov'] <= w1)]
    spk_pi = pass_index_at(st, spk)
    pk, _ = signal.find_peaks(st.ffi)
    pk = pk[(st.ts2[pk] >= w0) & (st.ts2[pk] <= w1)]
    pi_trace = wrap_pm_pi(pp.peak_interp_phase(st.ffi, st.ts2)) / np.pi

    fig = plt.figure(figsize=(13.5, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.15, 0.9], hspace=0.5)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(S.pos_ts[psel] - w0, S.pos_xy[psel, 0], color=INK2, lw=1)
    ax0.axhspan(FIELD_CENTER - 1.79 * FIELD_SIGMA, FIELD_CENTER + 1.79 * FIELD_SIGMA, color=BLUE, alpha=0.08)
    sx, _ = pp.spk_pos(S.pos_ts, S.pos_xy, spk)
    ax0.scatter(spk - w0, sx[:, 0], s=12, c=spk_pi, cmap='coolwarm', vmin=-1, vmax=1, zorder=3)
    ax0.set_ylabel('x (cm)')
    ax0.set_xlim(0, w1 - w0)
    ax0.set_title('a) position over time; spikes coloured by the pass index they will get '
                  '(blue = entering, red = leaving)')

    ax = fig.add_subplot(gs[1], sharex=ax0)
    ax.plot(st.ts2[asel] - w0, st.resampled[asel], color=LIGHT, lw=1.5, label='field index under the animal')
    ax.plot(st.ts2[asel] - w0, st.ffi[asel] + 0.5, color=BLUE, lw=1.6,
            label=f'band-passed ({st.band[0]:.4f}-{st.band[1]:.3f} cyc/cm), shifted +0.5')
    big = st.ffi[pk] > np.median(st.ffi)
    ax.scatter(st.ts2[pk[big]] - w0, st.ffi[pk[big]] + 0.5, color=VIOLET, s=40, zorder=4,
               edgecolors='white', label='peaks at field-centre crossings')
    ax.scatter(st.ts2[pk[~big]] - w0, st.ffi[pk[~big]] + 0.5, color=ORANGE, s=40, zorder=4,
               edgecolors='white', label='small peaks in the troughs (also counted!)')
    ax.set_ylabel('Field index')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=4)
    ax.set_title('b) field index along the path (equal distance steps), band-passed; every local peak',
                 pad=22)

    ax = fig.add_subplot(gs[2], sharex=ax0)
    ax.plot(st.ts2[asel] - w0, break_wraps(pi_trace[asel], 1.0), color=BLUE, lw=1.5)
    ax.scatter(spk - w0, spk_pi, s=14, c=spk_pi, cmap='coolwarm', vmin=-1, vmax=1, zorder=4,
               edgecolors='none')
    for p, isbig in zip(pk, big):
        ax.axvline(st.ts2[p] - w0, color=VIOLET if isbig else ORANGE, ls=':', lw=0.9)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel('Pass index')
    ax.set_xlabel('Time in window (s)')
    ax.set_title('c) between two peaks the phase rises linearly 0 -> 360 deg (same rule as theta); '
                 'wrap to -180..180 and divide by 180 -> PASS INDEX')

    note(fig.add_subplot(gs[3]),
         r'$\psi_{FI}(t) = 360^\circ \times \frac{t - t_{peak,k}}{t_{peak,k+1} - t_{peak,k}}$'
         r'$\qquad PI = \frac{\mathrm{wrap}_{[-180,180)}(\psi_{FI})}{180} \in [-1, 1)$' '\n'
         'PI = 0 at a peak. Just before the field centre the phase is ~350 deg -> wraps to -10 deg -> '
         'PI = -0.06 (entering side); just after it is ~10 deg -> PI = +0.06 (leaving side). PI = +/-1 is '
         'halfway (in time) between two peaks. Peaks are found with no minimum height, so the small bumps '
         'in the troughs (orange, here at the track ends) are peaks too: PI resets to 0 there and PI = +/-1 '
         'lands roughly at the field edges. During a pause (16-21 s) distance stops but time runs on, '
         'so that cycle is stretched.',
         title='Pass index formula (Climer et al. 2013)', width=175)
    fig.suptitle('STEP 3b - turning "where in the field" into a PASS INDEX', x=0.125, ha='left',
                 fontsize=12, fontweight='bold', y=0.97)
    save(fig, 'fig06_pass_index_construction.png')


def fig07_circular(c):
    r = c['res']
    x, th = r['spk_pass_index'], r['spk_theta_phase']
    s, b = r['s'], r['b']
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.plot([-1.1, 1.1], [1.25, 1.25], color=INK2, lw=2)
    for v in (-1, -0.5, 0, 0.5, 1):
        ax.plot([v * 1.1] * 2, [1.2, 1.3], color=INK2)
        ax.text(v * 1.1, 1.38, f'{v:+g}' if v else '0', ha='center', fontsize=8, color=INK2)
    ax.text(0, 1.0, 'pass index is a LINEAR number (x)', ha='center', fontsize=8.5, color=INK)
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.75 * np.cos(circ), 0.75 * np.sin(circ) - 0.55, color=LIGHT, lw=2)
    for v, lab in ((0, 'theta 0 = peak'), (90, '90'), (180, '180 = trough'), (270, '270')):
        a = np.deg2rad(v)
        ax.text(0.98 * np.cos(a), 0.98 * np.sin(a) - 0.55, lab, ha='center', va='center',
                fontsize=8, color=INK2)
    ax.text(0, -0.55, 'theta phase is a\nCIRCULAR number', ha='center', va='center', fontsize=8.5)
    ax.set_title('a) the two numbers we want to relate')

    ax = fig.add_subplot(gs[0, 1])
    draw_x = np.linspace(-1, 1, 400)
    for k, col in ((0, BLUE), (1, ORANGE)):
        ax.plot(draw_x, k * 360 + 0 * draw_x + 100, color=GRID, lw=0)
    ax.scatter(np.concatenate([x, x]), double(th), s=4, color=INK, alpha=0.35)
    ax.axhspan(360, 720, color=YELLOW, alpha=0.06, lw=0)
    ax.text(-0.97, 690, 'copy of the same spikes (+360)', fontsize=8, color=INK2,
            bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.5), zorder=5)
    phase_axes(ax)
    ax.set_xlabel('Pass index')
    ax.set_title('b) every spike drawn twice (0-360 and 360-720)')
    note_ax = fig.add_subplot(gs[0, 2])
    note(note_ax,
         'A phase of 359 deg and one of 1 deg are neighbours, but on a 0-360 axis they sit at the top '
         'and bottom. Drawing every spike a second time 360 deg higher lets the eye follow a band of '
         'points across the wrap. Nothing new is added - it is a display trick.\n\n'
         'The real geometry: pass index runs ALONG a cylinder and theta phase runs AROUND it (c). '
         'A straight "line" on that surface is a HELIX - a stripe that winds around while moving '
         'along. Fitting a line to circular data = finding the helix that the spikes hug most tightly. '
         'Unrolling the cylinder (d) turns the helix back into straight segments that wrap at 360.',
         title='Why circular data need special care', width=46)

    ax = fig.add_subplot(gs[1, 0:2], projection='3d')
    xx, aa = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(0, 2 * np.pi, 60))
    ax.plot_surface(xx, np.cos(aa), np.sin(aa), color=GRID, alpha=0.18, lw=0, shade=False)
    front = np.cos(th) > -0.2
    ax.scatter(x[front], np.cos(th[front]), np.sin(th[front]), s=4, color=BLUE, alpha=0.55, depthshade=False)
    ax.scatter(x[~front], np.cos(th[~front]), np.sin(th[~front]), s=3, color=BLUE, alpha=0.15, depthshade=False)
    xg = np.linspace(-1, 1, 400)
    hel = 2 * np.pi * s * xg + b
    ax.plot(xg, 1.01 * np.cos(hel), 1.01 * np.sin(hel), color=RED, lw=2.5)
    ax.plot(xg, np.ones_like(xg), np.zeros_like(xg), color=INK2, lw=1, ls='--')
    ax.text(1.05, 1.05, 0, '0 deg (LFP peak)', fontsize=8, color=INK2)
    ax.set_box_aspect((2.6, 1, 1))
    ax.view_init(elev=18, azim=-62)
    ax.set_xlabel('Pass index')
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title('c) the data live on a cylinder; the fitted "line" is a red helix', pad=0)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(np.concatenate([x, x]), double(th), s=4, color=INK, alpha=0.3)
    draw_fit(ax, s, b, label='fitted line (helix unrolled)')
    phase_axes(ax)
    ax.set_xlabel('Pass index')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('d) unrolled: helix = parallel red segments')
    fig.suptitle('STEP 3c (idea) - transforming the problem into circular data', x=0.08, ha='left',
                 fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig07_linear_to_circular.png')


def r_of_s(x, th, s_grid):
    ph = th[:, None] - 2 * np.pi * s_grid[None, :] * x[:, None]
    return np.abs(np.mean(np.exp(1j * ph), axis=0))


def fig08_fit(c):
    r = c['res']
    x, th = r['spk_pass_index'], np.mod(r['spk_theta_phase'], 2 * np.pi)
    s, b, fit_R = r['s'], r['b'], r['fit_R']
    lo, hi = pp.SLOPE_BNDS
    s_wide = np.linspace(-2.0, 2.0, 2001)
    R_wide = r_of_s(x, th, s_wide)
    grid_n = max(500, int(200 * (hi - lo) * max(np.ptp(x), 1.0)))
    grid = np.linspace(lo, hi, grid_n)
    R_grid = r_of_s(x, th, grid)
    # a competing (aliased) local maximum outside the chosen basin
    lm = signal.argrelmax(R_wide)[0]
    lm = lm[np.abs(s_wide[lm] - s) > 0.3]
    s_alias = s_wide[lm[np.argmax(R_wide[lm])]] if len(lm) else s + 0.8

    fig = plt.figure(figsize=(14.5, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.42, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(720 * s_wide, R_wide, color=LIGHT, lw=1.2, label='R(s) over a wide range')
    ax.plot(720 * grid, R_grid, color=BLUE, lw=2, label=f'searched: SLOPE_BNDS {lo:g} to {hi:g} cyc/unit')
    ax.axvspan(720 * -2, 720 * lo, color=GRID, alpha=0.5, lw=0)
    ax.axvspan(720 * hi, 720 * 2, color=GRID, alpha=0.5, lw=0)
    ax.scatter([720 * s], [fit_R], s=80, color=RED, zorder=5, edgecolors='white')
    ax.annotate(f'best slope s* = {s:.3f} cyc/unit\n= {720 * s:.0f} deg/pass\nR(s*) = fit_R = {fit_R:.3f}',
                xy=(720 * s, fit_R), xytext=(720 * hi + 60, fit_R * 0.75), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle='-', color=RED))
    ax.scatter([720 * s_alias], [r_of_s(x, th, np.array([s_alias]))[0]], s=50, color=ORANGE, zorder=5)
    ax.text(720 * s_alias, r_of_s(x, th, np.array([s_alias]))[0] + 0.015, 'a false (aliased) peak',
            fontsize=8, color=ORANGE, ha='center')
    ax.set_xlabel('Candidate slope (deg per pass = 720 x s)')
    ax.set_ylabel('R(s)  (how tightly residuals cluster)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(-1440, 1440)
    ax.set_title('a) score R(s) for every candidate slope (grid), refine near the best')

    ax = fig.add_subplot(gs[0, 2:4])
    ax.scatter(np.concatenate([x, x]), double(th), s=4, color=INK, alpha=0.3)
    draw_fit(ax, s, b)
    b_deg = np.rad2deg(np.mod(b, 2 * np.pi))
    ax.scatter([0, 0], [b_deg, b_deg + 360], s=70, color=VIOLET, zorder=6, edgecolors='white')
    wbox = dict(fc='white', ec='none', alpha=0.9, pad=1.5)
    ax.text(0.06, b_deg - 60, f'intercept b = {b_deg:.0f} deg\n(phase at PI = 0)', color=VIOLET,
            fontsize=8.5, bbox=wbox, zorder=7)
    x_a, x_b = -0.4, 0.4
    y_a = np.rad2deg(2 * np.pi * s * x_a + b) % 360 + 360
    y_b = y_a + np.rad2deg(2 * np.pi * s * (x_b - x_a))
    ax.plot([x_a, x_b, x_b], [y_a, y_a, y_b], color=INK2, lw=1, ls='--')
    ax.text(x_b + 0.03, (y_a + y_b) / 2, f'over 0.8 PI\nphase changes\n{y_b - y_a:.0f} deg', fontsize=8,
            color=INK2, va='center', bbox=wbox, zorder=7)
    phase_axes(ax)
    ax.set_title(f'b) the winning line: slope {720 * s:.0f} deg/pass, intercept {b_deg:.0f} deg')

    for k, (sv, lab, col) in enumerate(((0.0, 'flat line (s = 0)', MUTED),
                                        (s_alias, f'aliased s = {720 * s_alias:.0f}', ORANGE),
                                        (s, f'best s = {720 * s:.0f}', RED))):
        ax = fig.add_subplot(gs[1, k], projection='polar')
        res = th - 2 * np.pi * sv * x
        Rv = np.abs(np.mean(np.exp(1j * res)))
        ang = np.angle(np.mean(np.exp(1j * res)))
        jit = 1 + 0.06 * np.random.default_rng(k).standard_normal(len(res))
        ax.scatter(res, jit, s=3, color=col if col != MUTED else INK2, alpha=0.25)
        ax.annotate('', xy=(ang, Rv), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=2.5, shrinkA=0, shrinkB=0))
        ax.set_ylim(0, 1.25)
        ax.set_yticks([])
        ax.set_title(f'{"cde"[k]}) residuals for {lab}\n     R = {Rv:.3f}', fontsize=9, pad=12)

    note(fig.add_subplot(gs[1, 3]),
         'Residual of a spike = its phase minus what the line predicts:\n'
         r'$res_j = \theta_j - 2\pi s x_j$' '\n'
         'If the line is right, residuals pile up at one angle (long arrow). The fit picks the s '
         'with the longest arrow:\n'
         r'$R(s) = |\frac{1}{n}\sum_j e^{i(\theta_j - 2\pi s x_j)}|$' '\n'
         r'$b = \mathrm{atan2}(\sum \sin r_j, \sum \cos r_j)$' '\n'
         'fit_R = R(s*) is stored as goodness of fit (0..1). Least squares cannot be used because '
         'phase wraps around.',
         title='Fit (anglereg)', width=36, fontsize=8.2)
    fig.suptitle('STEP 3c - circular-linear regression (Kempter et al. 2012): fitting a line to wrapped phase',
                 x=0.08, ha='left', fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig08_circular_linear_fit.png')


def corr_parts(x, th, s):
    n = len(x)
    phi = np.mod(2 * np.pi * np.abs(s) * x, 2 * np.pi)
    thw = np.mod(th, 2 * np.pi)
    cm = lambda a: np.arctan2(np.sum(np.sin(a)), np.sum(np.cos(a)))

    def ray_p(a):
        Rn = np.hypot(np.sum(np.cos(a)), np.sum(np.sin(a)))
        return np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))
    sp, st = np.sin(phi - cm(phi)), np.sin(thw - cm(thw))
    l20, l02, l22 = np.mean(sp ** 2), np.mean(st ** 2), np.mean(sp ** 2 * st ** 2)
    return SimpleNamespace(phi=phi, thw=thw, phibar=cm(phi), thbar=cm(thw), sp=sp, st=st,
                           l20=l20, l02=l02, l22=l22, uniform=(ray_p(phi) > 0.5 or ray_p(thw) > 0.5))


def fig09_corr(c):
    r = c['res']
    x, th, s = r['spk_pass_index'], r['spk_theta_phase'], r['s']
    P = corr_parts(x, th, s)
    n = len(x)
    rho, p = r['rho'], r['p']
    z = rho * np.sqrt(n * P.l20 * P.l02 / P.l22)

    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x, np.rad2deg(P.phi), s=4, color=VIOLET, alpha=0.4)
    ax.axhline(np.rad2deg(np.mod(P.phibar, 2 * np.pi)), color=VIOLET, ls='--', lw=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_xlabel('Pass index x')
    ax.set_ylabel('phi (deg)')
    ax.set_title(f'a) POSITION turned into an angle\n    phi = 2 pi |s| x  (|s| = {abs(s):.3f})')

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(x, np.rad2deg(P.thw), s=4, color=INK, alpha=0.35)
    ax.axhline(np.rad2deg(np.mod(P.thbar, 2 * np.pi)), color=INK2, ls='--', lw=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_xlabel('Pass index x')
    ax.set_ylabel('theta (deg)')
    ax.set_title('b) the spike THETA PHASE\n    dashed = circular mean')

    ax = fig.add_subplot(gs[0, 2])
    pos = P.sp * P.st > 0
    ax.axhspan(0, 1.1, xmin=0.5, xmax=1, color=BLUE, alpha=0.06)
    ax.axhspan(-1.1, 0, xmin=0, xmax=0.5, color=BLUE, alpha=0.06)
    ax.axhspan(0, 1.1, xmin=0, xmax=0.5, color=ORANGE, alpha=0.06)
    ax.axhspan(-1.1, 0, xmin=0.5, xmax=1, color=ORANGE, alpha=0.06)
    ax.scatter(P.sp[pos], P.st[pos], s=5, color=BLUE, alpha=0.5, label=f'same-sign pairs ({pos.sum()})')
    ax.scatter(P.sp[~pos], P.st[~pos], s=5, color=ORANGE, alpha=0.5, label=f'opposite-sign ({(~pos).sum()})')
    ax.axhline(0, color=LIGHT, lw=0.8)
    ax.axvline(0, color=LIGHT, lw=0.8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('sin(phi - mean phi)')
    ax.set_ylabel('sin(theta - mean theta)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
    ax.set_title(f'c) do the two deviate together?\n    rho = {rho:.3f}')

    ax = fig.add_subplot(gs[1, 0])
    lim = max(4.0, abs(z) + 1)
    zz = np.linspace(-lim, lim, 600)
    ax.plot(zz, stats.norm.pdf(zz), color=INK2, lw=1.5)
    ax.fill_between(zz, stats.norm.pdf(zz), where=np.abs(zz) >= abs(z), color=RED, alpha=0.4)
    ax.axvline(z, color=RED, lw=2)
    ax.text(z, 0.36, f' z = {z:.1f}', color=RED, fontsize=9, fontweight='bold',
            ha='left' if z < lim - 2 else 'right')
    ax.set_xlabel('z (standard normal under "no relation")')
    ax.set_yticks([])
    ax.set_title(f'd) z-test: p = shaded tails = {fmt_p(p)}')

    note(fig.add_subplot(gs[1, 1:3]),
         'Step 1 - turn position into an angle using the fitted slope:  '
         r'$\varphi_j = 2\pi|s^*|x_j\ \mathrm{mod}\ 2\pi$' '\n'
         'Step 2 - centre both angles on their own circular means and take sines (a "circular deviation"):\n'
         r'$\rho = \frac{\sum_j \sin(\varphi_j - \bar{\varphi})\,\sin(\theta_j - \bar{\theta})}'
         r'{\sqrt{\sum_j \sin^2(\varphi_j - \bar{\varphi})\ \sum_j \sin^2(\theta_j - \bar{\theta})}}$' '\n'
         'Like Pearson r, -1..+1. |s| (not s) is used so the SIGN of rho comes from the data: negative = '
         'phase falls as the animal advances (precession).\n'
         'Step 3 - significance (asymptotic, no shuffling):\n'
         r'$\lambda_{ab} = \frac{1}{n}\sum_j \sin^a(\varphi_j - \bar{\varphi})\sin^b(\theta_j - \bar{\theta})$'
         r'$\qquad z = \rho\sqrt{\frac{n\,\lambda_{20}\lambda_{02}}{\lambda_{22}}}\qquad p = 1 - \mathrm{erf}(\frac{|z|}{\sqrt{2}})$'
         '\n'
         f'Cell {c["key"]}: n = {n}, lambda20 = {P.l20:.3f}, lambda02 = {P.l02:.3f}, lambda22 = {P.l22:.3f}, '
         f'rho = {rho:.3f}, z = {z:.2f}, p = {fmt_p(p)}.\n'
         'Special case: if phi or theta is almost uniform around the circle (Rayleigh p > 0.5) its mean '
         'is meaningless, and the numerator becomes  R(phi - theta) - R(phi + theta)  with R = |sum e^(i.)| '
         f'(used for this cell: {"yes" if P.uniform else "no"}).',
         title='Formulae (kempter_lincirc)', width=105, fontsize=8.4)
    fig.suptitle('STEP 3d - is the relationship real? Circular-linear correlation rho and its p-value',
                 x=0.08, ha='left', fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig09_correlation_rho_phi_pvalue.png')


def fig10_params(c):
    r = c['res']
    x, th, s, b = r['spk_pass_index'], r['spk_theta_phase'], r['s'], r['b']
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for k in (0, 360):
        ax.axhspan(70 + k, 250 + k, color=PREC_RANGE_C, alpha=0.7, lw=0)
        ax.axhspan(250 + k, 360 + k, color=PROC_RANGE_C, alpha=0.9, lw=0)
        ax.axhspan(0 + k, 70 + k, color=PROC_RANGE_C, alpha=0.9, lw=0)
    ax.scatter(np.concatenate([x, x]), double(th), s=3, color=INK, alpha=0.2)
    draw_fit(ax, s, b)
    for xv, key, lab in ((-1, 'pim1', 'PI -1'), (0, 'pi0', 'PI 0'), (1, 'pip1', 'PI +1')):
        v = r[f'phase_at_{key}_deg']
        ax.scatter([xv], [v + 360], s=80, color=VIOLET, zorder=6, edgecolors='white', clip_on=False)
        ax.text(xv + (0.06 if xv < 1 else -0.06), v + 360 - 75, f'{lab}: {v:.0f} deg\n({pp.phase_range_label(v)})',
                fontsize=7.5, color=VIOLET, ha='left' if xv < 1 else 'right', zorder=7,
                bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.5))
    phase_axes(ax)
    ax.legend(handles=[Patch(color=PREC_RANGE_C, label='precession range 70-250 deg'),
                       Patch(color=PROC_RANGE_C, label='procession range 250-70 deg')],
              loc='lower left', fontsize=7.5, frameon=True, facecolor='white', edgecolor='none',
              framealpha=0.92)
    ax.set_title('a) phase of the line at field entry / centre / exit')

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(np.concatenate([x, x]), double(th), s=3, color=INK, alpha=0.2)
    draw_fit(ax, s, b)
    xmin, xmax = x.min(), x.max()
    ax.axvspan(xmin, xmax, color=BLUE, alpha=0.07)
    ax.annotate('', xy=(xmax, 30), xytext=(xmin, 30), arrowprops=dict(arrowstyle='<->', color=BLUE))
    ax.text((xmin + xmax) / 2, 45, f'spikes span PI {xmin:.2f} to {xmax:.2f}\n(extent {xmax - xmin:.2f})',
            ha='center', fontsize=8, color=BLUE, bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.5))
    phase_axes(ax)
    ax.set_title(f'b) phase_range_deg (pipeline) = |slope| x extent\n    = {abs(r["slope_deg_per_pass"]):.0f} x '
                 f'{xmax - xmin:.2f} = {r["phase_range_deg"]:.0f} deg')
    ax.text(0.5, 0.97, f'NB: slope is per pass = per 2 PI units, so the phase the\nline actually sweeps over '
            f'this extent is half: {r["phase_range_deg"] / 2:.0f} deg', transform=ax.transAxes, ha='center',
            va='top', fontsize=7.8, color=RED, bbox=dict(fc='white', ec='none', alpha=0.92, pad=2), zorder=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot([-1, 1], [0, 0], color=INK2, lw=2)
    ax.text(-1, 0.06, 'one full pass: PI -1 -> +1 = 2 units', fontsize=8, color=INK2)
    ax.text(0, -0.25, r'$slope_{deg/pass} = \frac{180}{\pi}\times 2\pi s \times 2 = 720\,s$',
            ha='center', fontsize=11, color=INK)
    ax.text(0, -0.5, f's = {s:.4f} cycles per PI unit\n-> {r["slope_deg_per_pass"]:.1f} deg per pass',
            ha='center', fontsize=10, color=RED, fontweight='bold')
    ax.text(0, -0.85, 's is "cycles per unit of pass index"; one pass covers 2 units,\n'
            'one cycle is 360 deg -> 2 x 360 = 720.', ha='center', fontsize=8, color=INK2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.05, 0.2)
    ax.axis('off')
    ax.set_title('c) converting the slope to deg / pass')

    ax = fig.add_subplot(gs[1, 0])
    for sv, col, off in ((-0.42, BLUE, 0), (-2.0, VIOLET, 400)):
        xg = np.linspace(-1, 1, 800)
        ph = np.rad2deg(np.mod(2 * np.pi * sv * xg + 1.0, 2 * np.pi))
        ax.plot(xg, break_wraps(ph, 180) / 360 * 300 + off, color=col, lw=2)
        ends = 2 * np.pi * sv * np.array([-1, 1]) + 1.0
        nl = int(np.floor(ends.max() / (2 * np.pi)) - np.floor(ends.min() / (2 * np.pi)) + 1)
        ax.text(1.03, off + 150, f'{720 * sv:.0f} deg/pass\n-> {nl} segments\n'
                f'{"OK" if nl <= pp.MAX_FIT_LINES else "> 3: MultiLinesFit"}', fontsize=8, color=col,
                va='center')
    ax.set_xlim(-1, 1.6)
    ax.set_yticks([])
    ax.set_xlabel('Pass index')
    ax.set_title(f'd) n_fit_lines: how many wrapped pieces?\n    this cell: {r["n_fit_lines"]}')

    ax = fig.add_subplot(gs[1, 1])
    ph_c = np.rad2deg(0.5 * (r['ph_edges'][:-1] + r['ph_edges'][1:]))
    pi_c = 0.5 * (r['pi_edges'][:-1] + r['pi_edges'][1:])
    d = np.concatenate([r['density'], r['density']], axis=1)
    im = ax.pcolormesh(pi_c, np.concatenate([ph_c, ph_c + 360]), d.T, cmap=SEQ_CMAP, shading='auto')
    fig.colorbar(im, ax=ax, label='spikes / s', fraction=0.04)
    phase_axes(ax)
    ax.set_title('e) density map: spikes / time spent in each\n    (pass index, phase) cell, smoothed')

    lines = [
        ('n spikes (theta, with phase & PI)', f'{r["n_spikes"]}'),
        ('slope s (cycles / PI unit)', f'{s:.4f}'),
        ('slope (deg / pass) = 720 s', f'{r["slope_deg_per_pass"]:.1f}'),
        ('intercept b (deg)', f'{np.rad2deg(np.mod(b, 2 * np.pi)):.1f}'),
        ('fit_R  (residual MRL)', f'{r["fit_R"]:.3f}'),
        ('rho  (circ-linear corr.)', f'{r["rho"]:.3f}'),
        ('r^2 = rho^2', f'{r["r_squared"]:.3f}'),
        ('p  (Kempter z-test)', fmt_p(r["p"])),
        ('phase range (deg)', f'{r["phase_range_deg"]:.0f}'),
        ('phase at PI -1 / 0 / +1 (deg)', f'{r["phase_at_pim1_deg"]:.0f} / {r["phase_at_pi0_deg"]:.0f} / '
                                          f'{r["phase_at_pip1_deg"]:.0f}'),
        ('n_fit_lines', f'{r["n_fit_lines"]}'),
        ('PrecessionClass', f'{r["precession_class"]}'),
    ]
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0,rounding_size=0.03',
                                transform=ax.transAxes, fc=PANEL, ec='none'))
    ax.text(0.04, 0.95, f'f) every parameter stored for cell {c["key"]}', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    for k, (lab, val) in enumerate(lines):
        yy = 0.86 - k * 0.07
        ax.text(0.04, yy, lab, transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')
        ax.text(0.96, yy, val, transform=ax.transAxes, fontsize=8.5, color=INK, va='top', ha='right',
                fontweight='bold')
    fig.suptitle('STEP 3e - quantifying the fitted relationship', x=0.08, ha='left',
                 fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig10_parameters_quantified.png')


def fig11_classification(cells):
    thr = pp.SLOPE_THRESH_DEG_PER_PASS
    lo, hi = 720 * pp.SLOPE_BNDS[0], 720 * pp.SLOPE_BNDS[1]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.75, 1.25], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(lo, -thr, color=BLUE, alpha=0.10)
    ax.axvspan(-thr, thr, color=AQUA, alpha=0.18)
    ax.axvspan(thr, hi, color=ORANGE, alpha=0.10)
    for xx, txt, col in (((lo - thr) / 2, 'PHASE PRECESSING\nslope < -15 deg/pass', BLUE),
                         (0, 'PHASE\nLOCKED\n|slope| <= 15', AQUA),
                         ((thr + hi) / 2, 'PHASE SUCCEEDING\nslope > +15', ORANGE)):
        ax.text(xx, 0.82, txt, ha='center', va='top', fontsize=8.5, color=INK, fontweight='bold')
    ys = {'A': 0.4, 'B': 0.4, 'C': 0.52, 'D': 0.32, 'E': 0.12}
    for c in cells:
        r = c['res']
        sl = r['slope_deg_per_pass']
        sig = c['tested'] and np.isfinite(r['p']) and r['p'] < pp.ALPHA
        col = CLASS_STYLE[c['cls']][1]
        ax.scatter([sl], [ys[c['key']]], s=140, marker='o', facecolors=col if sig else 'white',
                   edgecolors=col if c['tested'] else LIGHT, lw=2.2, zorder=4)
        ax.text(sl + 14, ys[c['key']], f'{c["key"]}  ({sl:.0f})', ha='left', va='center', fontsize=9,
                fontweight='bold')
    ax.set_xlim(lo - 20, hi + 20)
    ax.set_ylim(0, 0.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Fitted slope (deg per pass)   -   searched range set by SLOPE_BNDS')
    ax.set_title('Rule: filled circle = significant fit (p < 0.05) -> class by slope.  '
                 'Open circle = p >= 0.05 -> NO PHASE RELATION wherever the slope is.  '
                 'Grey open = never tested (failed TMI).', fontsize=9.5)

    for k, c in enumerate(cells):
        r = c['res']
        ax = fig.add_subplot(gs[1, k])
        x, th = r['spk_pass_index'], r['spk_theta_phase']
        label, col = CLASS_STYLE[c['cls']]
        ax.scatter(np.concatenate([x, x]), double(th), s=2.5, color=INK, alpha=0.25)
        if c['tested']:
            draw_fit(ax, r['s'], r['b'], color=col if col not in (LIGHT, MUTED) else INK2)
        phase_axes(ax, ylabel='Theta phase (deg)' if k == 0 else '')
        ax.set_xlabel('Pass index')
        m = c['metrics']
        body = (f'TMI {m["TMI"]:.2f} (p {m["TMI_shuffle_p"]:.2g})\n' +
                (f'slope {r["slope_deg_per_pass"]:.0f} deg/pass\nrho {r["rho"]:.2f}, p {fmt_p(r["p"])}\n'
                 f'fit_R {r["fit_R"]:.2f}' if c['tested'] else 'Step 3 not run\n(points shown only\nfor comparison)'))
        ax.set_title(f'Cell {c["key"]}  -  {c["design"].replace("designed: ", "")}', fontsize=9)
        ax.text(0.03, 0.97, body, transform=ax.transAxes, fontsize=7.8, color=INK2, va='top',
                bbox=dict(fc='white', ec='none', alpha=0.93, pad=1.5), zorder=6)
        ax.text(0.5, -0.27, label, transform=ax.transAxes, ha='center', fontsize=9.5, fontweight='bold',
                color='white', bbox=dict(fc=col, ec='none', boxstyle='round,pad=0.35'))
    fig.suptitle('FINAL - classifying the cell (five simulated cells with known behaviour)', x=0.08,
                 ha='left', fontsize=12, fontweight='bold', y=0.98)
    save(fig, 'fig11_classification.png')


def fig12_formula_sheet():
    sections = [
        ('0  Starting point (ACG theta epochs)', [
            (r'$ED_{min} = \min_f \frac{\|ACG_{ref}(f) - ACG_{epoch}\|}{\|ACG_{ref}(f)\|},\quad f = 3.0, 3.1, \ldots, 7.0\ Hz;\quad keep\ if\ ED_{min} < 1$',
             'only spikes inside kept 1-s epochs are analysed'),
        ]),
        ('1  Spike theta phase', [
            (r'$\phi = 360^\circ \times \frac{t - t_k}{t_{k+1} - t_k}$', '0 = LFP peak, 180 = trough'),
        ]),
        ('1b Phase locking (reported)', [
            (r'$MRL = \frac{1}{n}\sqrt{(\sum_j \cos\phi_j)^2 + (\sum_j \sin\phi_j)^2},\quad '
             r'\bar{\phi} = \mathrm{atan2}(\sum_j \sin\phi_j, \sum_j \cos\phi_j)$', ''),
            (r'$R_n = n \cdot MRL,\quad p_{Rayleigh} = \exp(\sqrt{1 + 4n + 4(n^2 - R_n^2)} - (1 + 2n))$', ''),
        ]),
        ('2  Theta Modulation Index  (GATE 1)', [
            (r'$TMI = 1 - \min(h_{smooth} / \max(h_{smooth}))$', '36-deg bins, phases tiled over 5 cycles'),
            (r'$p_{TMI} = (1 + N[TMI_{shuffle} \geq TMI_{obs}]) / (1 + 1000)$', 'continue if p < 0.05'),
        ]),
        ('3a Field index', [
            (r'$rate = \frac{spikes}{time},\quad FI = \frac{rate - rate_{min}}{rate_{max} - rate_{min}},\quad '
             r'r = \sqrt{N_{bins > 20\%\,peak}\,binside^2 / \pi},\quad band = (\frac{1}{6r}, \frac{3}{r})$', ''),
        ]),
        ('3b Pass index', [
            (r'$PI = \mathrm{wrap}_{[-180,180)}(360^\circ \times \frac{t - t_{FIpeak,k}}{t_{FIpeak,k+1} - t_{FIpeak,k}}) / 180$',
             '-1 entering, 0 centre, +1 leaving'),
        ]),
        ('3c Circular-linear fit', [
            (r'$\hat{\theta}(x) = (2\pi s x + b)\ \mathrm{mod}\ 2\pi$', 'the model (a helix)'),
            (r'$R(s) = |\frac{1}{n}\sum_j e^{i(\theta_j - 2\pi s x_j)}|,\quad s^* = \arg\max R(s),\quad fit\_R = R(s^*)$',
             'grid search + refinement'),
            (r'$b = \mathrm{atan2}(\sum_j \sin(\theta_j - 2\pi s^* x_j),\ \sum_j \cos(\theta_j - 2\pi s^* x_j))$', ''),
        ]),
        ('3d Correlation and p-value', [
            (r'$\varphi_j = 2\pi |s^*| x_j\ \mathrm{mod}\ 2\pi$', 'position as an angle'),
            (r'$\rho = \frac{\sum \sin(\varphi_j - \bar{\varphi})\sin(\theta_j - \bar{\theta})}'
             r'{\sqrt{\sum \sin^2(\varphi_j - \bar{\varphi})\sum \sin^2(\theta_j - \bar{\theta})}}$', ''),
            (r'$z = \rho\sqrt{n\lambda_{20}\lambda_{02}/\lambda_{22}},\quad '
             r'\lambda_{ab} = \frac{1}{n}\sum \sin^a(\varphi_j - \bar{\varphi})\sin^b(\theta_j - \bar{\theta}),\quad '
             r'p = 1 - \mathrm{erf}(|z| / \sqrt{2})$', ''),
        ]),
        ('3e Parameters', [
            (r'$slope = 720\,s^*\ (deg/pass),\quad r^2 = \rho^2,\quad phase\ range = |slope| \times (\max x - \min x)$', ''),
            (r'$\theta(x_0) = \deg(2\pi s^* x_0 + b)\ \mathrm{mod}\ 360,\quad x_0 \in \{-1, 0, +1\};\quad '
             r'precession\ range: 70 \leq \theta < 250$', ''),
            (r'$n_{lines} = \lfloor \max(\hat{\theta}_{ends}) / 2\pi \rfloor - \lfloor \min(\hat{\theta}_{ends}) / 2\pi \rfloor + 1$', ''),
        ]),
        ('FINAL Classification (GATE 3)', [
            (r'$p \geq 0.05 \rightarrow$ no phase relation;   $n_{lines} > 3 \rightarrow$ non-precessing (MultiLinesFit)', ''),
            (r'$p < 0.05:\ slope < -15 \rightarrow$ PRECESSING;   $slope > +15 \rightarrow$ SUCCEEDING;   '
             r'$|slope| \leq 15 \rightarrow$ LOCKED', ''),
        ]),
    ]
    n_lines = sum(1 + len(v) for _, v in sections)
    fig = plt.figure(figsize=(14, 0.62 * n_lines + 1.2))
    y = 0.975
    fig.text(0.03, y, 'Formula sheet - phase-precession pipeline (ThetaMod_PhasePrec_v19)', fontsize=14,
             fontweight='bold', color=INK, va='top')
    dy = 0.93 / n_lines
    y -= dy * 1.1
    for head, items in sections:
        fig.text(0.03, y, head, fontsize=10.5, fontweight='bold', color=BLUE, va='top')
        y -= dy * 0.85
        for formula, comment in items:
            fig.text(0.06, y, formula, fontsize=11.5, color=INK, va='top')
            if comment:
                fig.text(0.97, y, comment, fontsize=8.5, color=MUTED, va='top', ha='right')
            y -= dy * 1.05
    save(fig, 'fig12_formula_sheet.png')


# ============================================================================
# Main
# ============================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print('Simulating session ...')
    S = build_session()
    print(f'  ACG theta-positive epochs: {int(S.acg_ok.sum())}/{len(S.acg_ok)}')
    rows = []
    for k, c in enumerate(CELLS):
        print(f'Running pipeline on cell {c["key"]} ...')
        run_cell(c, S, k)
        m, r = c['metrics'], c['res']
        rows.append(dict(cell=c['key'], design=c['design'], n_spikes_total=len(c['spk_all']),
                         n_spikes_ACGtheta=len(c['spk']), MRL=m['MRL'], PreferredPhase_deg=m['PreferredPhase_deg'],
                         Rayleigh_p=m['Rayleigh_p'], TMI=m['TMI'], TMI_shuffle_p=m['TMI_shuffle_p'],
                         TMI_Significant=m['TMI_Significant'], PrecessionTested=c['tested'],
                         PassIndex_n_spikes=r['n_spikes'], slope_s_cyc_per_unit=r['s'],
                         slope_deg_per_pass=r['slope_deg_per_pass'],
                         intercept_deg=np.rad2deg(np.mod(r['b'], 2 * np.pi)), fit_R=r['fit_R'],
                         rho=r['rho'], r_squared=r['r_squared'], precession_p=r['p'],
                         phase_range_deg=r['phase_range_deg'], phase_at_pim1_deg=r['phase_at_pim1_deg'],
                         phase_at_pi0_deg=r['phase_at_pi0_deg'], phase_at_pip1_deg=r['phase_at_pip1_deg'],
                         n_fit_lines=r['n_fit_lines'], PrecessionClass=c['cls']))
        print(f'  TMI={m["TMI"]:.3f} p={m["TMI_shuffle_p"]:.3g} | slope={r["slope_deg_per_pass"]:.1f} '
              f'rho={r["rho"]:.3f} p={r["p"]:.3g} -> {c["cls"]}')
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'demo_cell_results.csv', index=False)

    cA = next(c for c in CELLS if c['key'] == WALKTHROUGH_CELL)
    cE = next(c for c in CELLS if c['key'] == 'E')
    print('Drawing figures ...')
    fig00_overview()
    fig01_acg_recap(S, cA)
    fig02_phase(S, cA)
    fig03_locking(cA)
    fig04_tmi(cA, cE)
    fig05_maps(S, cA)
    fig06_pass_index(S, cA)
    fig07_circular(cA)
    fig08_fit(cA)
    fig09_corr(cA)
    fig10_params(cA)
    fig11_classification(CELLS)
    fig12_formula_sheet()
    print(f'Done. Output in {OUT_DIR}')


if __name__ == '__main__':
    main()
