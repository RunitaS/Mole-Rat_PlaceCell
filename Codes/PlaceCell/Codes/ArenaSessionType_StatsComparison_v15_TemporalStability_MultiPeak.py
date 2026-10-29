# -*- coding: utf-8 -*-
"""
Statistical comparison of place-cell characterization metrics, using the
workbook produced by PlaceCellChar_FieldDetect_Main_v3.py (the 'output_excel'
file), at three levels:

  1. Session types WITHIN each arena type -- the cells of one arena are
     grouped together and its session types are compared with each other:
         Open   : Cntrl, Rotate, Zero          (e.g. '1_Cntrl', '2Rotate')
         Linear : 0, 90, 180, 270              (digits at the end, e.g. '3_180')
         Circle : Stnd, NoRot, Rot             (e.g. '2NoRot', '4Stnd')
  2. Arena types as a whole (Circle / Linear / Open, all session types of an
     arena pooled), compared with each other.
  3. Animal-controlled arena types: not every animal was recorded in every
     arena, so level 2 mixes arena and animal differences. Here each
     comparison uses only the animals recorded in all of its arenas
     (ANIMAL_CONTROLLED_COMPARISONS):
         Circle vs Open : Fa8477, Fa5834
         Linear vs Open : Fa1059, Fa23BD
     Each plot shows these comparisons side by side (left to right in the
     order of ANIMAL_CONTROLLED_COMPARISONS), each panel with its own
     within-comparison statistics; the comparisons are not tested against
     each other.

All levels use the same tests (below). All statistics tables are saved to a
NEW workbook (+ one long-format CSV) in OUTPUT_DIR, together with the plots:

    OUTPUT_DIR\\SessionType\\<arena>\\              box/bar plots, session types of <arena>
    OUTPUT_DIR\\SessionType\\<arena>\\Histograms\\
    OUTPUT_DIR\\ArenaType\\                         box/bar plots, arena types
    OUTPUT_DIR\\ArenaType\\Histograms\\
    OUTPUT_DIR\\AnimalControlledArenaType\\         box/bar plots, the comparisons side by side
    OUTPUT_DIR\\AnimalControlledArenaType\\Histograms\\
    OUTPUT_DIR\\AnimalControlledArenaType\\ExampleMaps\\  temporal stability and
                                                    multi-peak ('MultiPeak_*') example maps
    OUTPUT_DIR\\AnimalControlledArenaType\\AnimalControlled_Stats.xlsx  level-3 tables only
    OUTPUT_DIR\\SessionArena_Stats.xlsx             sheets 'Ses<arena>_*', 'Arena_*' and
                                                    'AC<A><B>_*' (level 3)
    OUTPUT_DIR\\SessionArena_AllStats.csv
    OUTPUT_DIR\\RunSnapshots\\<YYYYmmdd_HHMMSS>\\       saved BEFORE the analysis runs:
        code\\                                      this script + PlaceCell_Main_v21,
                                                    CellClusteredStats_Utils, PlaceFieldShape_Utils
        RunMetadata.xlsx                            'Run' (time, git commit, versions)
                                                    and 'Parameters' (every setting)

INPUT_EXCEL is only read, not modified. The place-field shape columns
(PlaceFieldShape_Utils.SHAPE_COLUMNS) are written to a COPY of it,
FIELD_SHAPE_EXCEL ('<INPUT_EXCEL name>_FldShp.xlsx', same folder), in the
first empty columns of its 'PlaceFields' sheet.

Arena type and session type are both extracted from the folder path stored in
the 'session' column (column A) of the 'Full' and 'PlaceFields' sheets, e.g.
'Fa1059\\Linear\\Day10\\1_0' is arena 'Linear', session type '0' (the session
type comes from the label of the last folder). Rows whose arena is recognized
but whose session type isn't are kept in the arena comparison and left out of
the session-type comparison. Only rows with place_cell == True are analyzed
(except place-cell yield, which uses all recorded units).

Metrics compared (all restricted to place_cell == True):

  Sheet 'Full':
    - peak_fr          Peak firing rate
    - sir              Spatial information score
    - sparsity         Sparsity
    - coherence        Coherence score
    - stability_score  Stability score; also compared separately within each
                        stability type (STABILITY_TYPES): split-half p <= 0.05
                        with a positive score, p <= 0.05 with a negative
                        score, and p > 0.05
    - final_speed_score Speed score (mean of the binned and instantaneous
                        scores). Speed cells are those with speed_cell == True:
                        p-type if final_speed_score > 0, n-type if < 0. Cells
                        with speed_cell False / 'not tested' are non-speed.
    - mean_fr          Mean firing rate ('Extra' sheets)

  Sheet 'PlaceFields' (aggregated per cell):
    - n_fields         Number of place fields per cell
    - total_area_cm2   Area occupied by fields per cell (sum of field areas)
    - pct_area         Percentage area occupied by fields per cell (sum of
                        each field's pct_of_occupied_area)

  Sheet 'PlaceFields' (one value per field, from 'bin_coords'; see
  PlaceFieldShape_Utils.py):
    - fld_max_width_cm   Maximum field width (axis with the most bins)
    - fld_min_width_cm   Minimum field width (axis with the fewest bins)
    - fld_axis_angle_deg Angle between the max- and min-width axes
    - fld_width_ratio    Max width / min width
    - fld_mean_width_cm  Mean of the max and min widths
    Circular-track fields are 1-D (angular bins) and have no 2-D shape, so
    these are compared for Linear and Open only.

  Proportions: place-cell yield, theta modulation, coherence shuffle
  significance, split-half stability significance, speed cells, and the
  p-Speed / n-Speed / non-speed classification.

  Temporal stability (V15, merged from TemporalStabilityMap_v6_MapPassStability.py;
  one value per cell, TEMPORAL_STABILITY_METRICS, box plots / histograms
  'TemporalStability_<metric>.png'; sheets '*_TS_Desc', '*_TS_Omnibus',
  '*_TS_PostHoc'). Switched off with INCLUDE_TEMPORAL_STABILITY = False.
  The cells are every .ntt under TS_PLACE_CELL_FOLDER (the Output_PlaceTrue
  folder of a PlaceCell_Main_v21 run; all already place cells). With
  RUN_TS_MAPS = True, each cell is processed as follows (else the PerCell sheet
  of an existing TS_OUTPUT_EXCEL is used):
    1. Rebuild the session trajectory, spike positions and bin grid with
       PlaceCell_Main_v21.compute_metrics (same cleaning, smoothing, immobility
       filter and spike assignment as the run that detected the fields).
    2. Detect its place fields with PlaceCell_Main_v21's threshold method
       (detect_place_fields_threshold_2d / _circular) on those maps.
    3. For every field bin, split the trajectory into visits: a visit is one
       contiguous run of tracking frames inside that bin (entry -> exit), the
       same definition PlaceCell_Main_v21 uses for its visit count. Visits
       shorter than MIN_VISIT_S are ignored.
    4. Per field bin: % of visits during which the cell fired >= 1 spike
       (a spike belongs to a visit if it was assigned to one of its frames).
       mean_pct_across_bins / median_pct_across_bins summarise these per cell.
    5. Occupancy-weighted summaries, so that a bin the animal spent little time
       in (e.g. 1 visit, 1 spike -> 100%) counts less than a well-sampled bin:
         occ_wmean_pct_across_bins / occ_wmedian_pct_across_bins
                                 mean / median of the per-bin % weighted by
                                 each bin's dwell time (occupancy_s, moving-only
                                 frames); the weighted median is the smallest
                                 bin % at which the cumulative dwell reaches
                                 half the total
         occ_wmean_pct_mapnorm   sum over field bins of % x dwell, divided by
                                 the dwell of ALL valid bins (= the weighted
                                 mean x the fraction of map dwell spent in the
                                 field); PerCell + PerField
         occ_wmean_pct_all_bins  occupancy-weighted mean % over EVERY visited
                                 valid bin, in or out of a field; PerCell only
    6. Peak bin: the valid bin with the highest rate in the RAW (unsmoothed)
       ratemap (Circle track: raw angular map), whether or not it lies in a
       field; its % of visits with >= 1 spike is pct_at_peak_bin (white star
       on the raw and stability maps).
    7. Lap stability: temporal stability per PASS through a field. A pass is
       one contiguous run of moving-only frames inside the field: it starts
       when the animal enters any bin of the field and ends when it leaves the
       field through any bin (moving between bins of the same field does not
       end it). Passes shorter than MIN_PASS_S, and passes cut by the start /
       end of the session (no observed entry or exit), are dropped. A pass
       'fired' if >= MIN_SPIKES_PER_PASS spikes were assigned to its frames.
         lap_stability_score   n passes with >= MIN_SPIKES_PER_PASS spikes /
                               n passes (0-1), per field (PerField); on PerCell
                               it pools the passes through every field
         mean_lap_stability_across_fields   PerCell: mean of the cell's
                               per-field lap_stability_score
  Figure (one per cell, saved next to its .ntt in TEMPORAL_STABILITY_SUBDIR):
    trajectory + spikes | raw ratemap | Gaussian-smoothed ratemap | temporal
    stability map (field bins coloured by %, each field labelled with its lap
    stability; visited non-field bins gray; never-visited bins blank; Circle
    track drawn as a ring of angular bins).
  Per-cell / per-field / per-bin results: TS_OUTPUT_EXCEL (sheets PerCell,
  PerField, PerBin). Example maps (animal-controlled comparisons only): per
  arena, the N_EXAMPLE_MAPS cells with the highest (Best) and lowest (Worst)
  EXAMPLE_METRIC and the N_EXAMPLE_MAPS closest to the arena mean (Mean),
  copied with a banner naming the animal, session and unit to
  OUTPUT_DIR\\AnimalControlledArenaType\\ExampleMaps\\
  ('<comparison>_<arena>_<Best|Mean|Worst><rank>_<session>_<unit>.png').
  Analysis settings (bin size, smoothing, filters, field thresholds,
  COORD_UNITS, ...) come from PlaceCell_Main_v21.

  Multi-peak fields (V15, merged from PlaceCell_MultiPeak_KDE_v2.py and
  MultiPeak_StatsComparison.py; switched off with INCLUDE_MULTIPEAK = False).
  Whether one isolated place field has two or more peaks inside it (not
  whether a cell has several fields). Runs on the place_cell == True cells of
  INPUT_EXCEL ('Full'), whose .ntt + tracking files are in MP_PLACE_CELL_FOLDER;
  the fields PlaceCell_Main isolated ('PlaceFields', bin_coords) are reused
  as-is. With RUN_MULTIPEAK_KDE = True each cell's full-session ratemap is
  rebuilt exactly as in PlaceCell_Main (the MP_ ratemap settings must match
  that run; ratemap_check flags any field whose recomputed smoothed peak does
  not reproduce the Excel peak_fr_hz), else the 'Fields' sheet of an existing
  MULTIPEAK_EXCEL is used. Per field:
    1. Fit a weighted Gaussian KDE (scipy.stats.gaussian_kde, bandwidth
       KDE_BW_METHOD) on the field's bin centres, each bin weighted by its rate
       in the KDE_RATEMAP_TYPE map (raw, occupancy-weighted smoothed,
       Gaussian-smoothed, Yartsev or Skaggs adaptive-binned). Scaled by the
       summed rate, the KDE is in Hz. Circle track: 1-D along the track angle
       (adaptive discs become intervals, wrapping around theta = 0/2*pi).
    2. Find the KDE's local maxima inside the field (8-neighbourhood on the
       2-D grid; left/right neighbour on the Circle track), also evaluating a
       1-bin ring around the field, so a field-edge bin only counts if the KDE
       really falls off outside it. A maximum is counted as a peak if its
       height is >= PEAK_REL_HEIGHT_FRAC of the field's highest peak AND the
       KDE dips below (1 - MIN_TROUGH_DIP_FRAC) x its height on the way to
       every higher peak. Fields with < MIN_KDE_POINTS non-zero-rate bins get
       no KDE (kde_note) and are single-peak.
    3. >= 2 counted peaks -> multi_peak = True; the field is cut at the
       troughs between its peaks by marker-based watershed on the KDE.
  Fields with no KDE (EXCLUDE_NO_KDE_FIELDS) or a failed ratemap_check
  (EXCLUDE_RATEMAP_MISMATCH) are left out of the comparison; the per-cell
  values are computed from the fields kept. Metrics compared (MULTIPEAK_METRICS
  and MULTIPEAK_CATEGORICAL; plots 'MultiPeak_<Cell|Field|MultiField>_<metric>.png'
  and 'MultiPeak_Categorical_<metric>.png', histograms 'Hist_MultiPeak_*';
  sheets '*_MP_Desc', '*_MP_Omnibus', '*_MP_PostHoc', '*_MP_CatDesc', ...):
    per cell        n_multi_peak_fields, pct_multi_peak_fields, max_n_peaks,
                    % of place cells with >= 1 multi-peak field
    per field       n_peaks, % of place fields that are multi-peak
    multi-peak      shallowest_trough_rel (saddle / lower peak of the weakest
    fields only     split), lowest_counted_peak_rel (lowest / highest peak;
                    >= PEAK_REL_HEIGHT_FRAC by construction),
                    peak_separation_cm (mean distance between neighbouring peaks)
  Several fields of a cell are repeated measures ('KW' treats them as
  independent; the LMM's cell random effect accounts for them).
  Per-field / per-sub-field / per-cell results: MULTIPEAK_EXCEL (sheets
  Fields, SubFields, Cells, Settings). Figure per cell, in MP_RATEMAP_DIR\\
  MultiPeak or SinglePeak: Gaussian-smoothed ratemap (field bins only, field
  outlines black, trough cuts white, counted peaks as stars), the KDE input
  map, and the field KDE. Example maps (animal-controlled comparisons only):
  per arena, best / mean / worst N_EXAMPLE_MAPS cells by MP_EXAMPLE_METRIC,
  saved as 'MultiPeak_<comparison>_<arena>_<Best|Mean|Worst><rank>_...png' in
  OUTPUT_DIR\\AnimalControlledArenaType\\ExampleMaps\\.

Box-plot points: each point (one unit in one session; for field shape, one
field) is filled with its group's hue, darker for a higher spatial
information score (SHADE_METRIC, ranked within the comparison; the lightest
shade is a tint, SHADE_LIGHT_MIX, so it stays visible). The shades
are computed separately for every comparison -- each SessionType\\<arena>
folder, ArenaType, and each AnimalControlledArenaType pair -- and a cell
keeps its shade in all box plots of that comparison. Significance is shown by
the marker: o passed / X failed (coherence shuffle, split-half stability),
^ / v p- / n-type speed cell, X not a speed cell.

Repeated measures: every row is one unit in one session, and a unit recorded
in several sessions of a day (same animal + day + unit label) appears once per
session -- within an arena (e.g. Linear: 332 place-cell rows from 195 cells)
and, for units recorded in two arenas on the same day, across arenas. Those
rows are not independent. Tests used (see CellClusteredStats_Utils.py):

  - continuous metrics (CONTINUOUS_TEST below):
      'KW'  : Kruskal-Wallis omnibus test, pairwise Mann-Whitney U post hoc
              tests. These treat every row as independent (no correction for
              repeated cells); the 'model_note' column reports how many cells
              repeat.
      'LMM' : linear mixed model (on ranks, METRIC_TRANSFORM)
                metric ~ group + (1|animal) + (1|animal:day) + (1|cell)
              omnibus likelihood-ratio test, pairwise Wald contrasts
              (CellClusteredStats_Utils.compare_continuous_nested).
  - % outcomes and speed direction (p/n/non): chi-square test of
    independence, pairwise chi-square post hoc tests (Yates-corrected for
    2 x 2 tables). Also treat rows as independent; 'model_note' flags
    tables with expected counts < 5.

Pairwise p-values are Holm-Bonferroni corrected within each metric (and, for
session types, within each arena).

Parameters to edit (below):
    INPUT_EXCEL  = path to the output_excel file from
                   PlaceCellChar_FieldDetect_Main_v3.py (read only)
    OUTPUT_DIR   = new folder for all statistics tables and plots
    INCLUDE_TEMPORAL_STABILITY, RUN_TS_MAPS, TS_PLACE_CELL_FOLDER,
    TS_OUTPUT_EXCEL = temporal stability (see above)
    INCLUDE_MULTIPEAK, RUN_MULTIPEAK_KDE, MP_PLACE_CELL_FOLDER, the KDE
    peak / trough / bandwidth settings and the MP_ ratemap settings =
    multi-peak fields (see above)
"""

import os
import re
import sys
import heapq
import shutil
import platform
import functools
import subprocess
import tempfile
import importlib.metadata
import textwrap
import colorsys
import threading
import concurrent.futures
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import convolve, gaussian_filter1d
from scipy.stats import gaussian_kde
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_hex, to_rgb, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable

import CellClusteredStats_Utils as ccs
import PlaceFieldShape_Utils as pfs
import PlaceCell_Main_v21 as pc   # temporal stability maps

# ── Parameters ──────────────────────────────────────────────────────────────

INPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\PC_SmoothCompare\CrossCheck_DoubleRun\All_TT_PlaceChar_SiSkaggsAdptBin_Alpha200.xlsx'
OUTPUT_DIR  = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\PlaceCellChar_v2'

# Copy of INPUT_EXCEL with the place-field shape columns added to its
# 'PlaceFields' sheet.
FIELD_SHAPE_EXCEL = os.path.splitext(INPUT_EXCEL)[0] + '_FldShp.xlsx'

STATS_XLSX = os.path.join(OUTPUT_DIR, 'SessionArena_Stats.xlsx')
STATS_CSV  = os.path.join(OUTPUT_DIR, 'SessionArena_AllStats.csv')

# Before the analysis starts, a snapshot of the code (this script + the local
# modules it imports) and RunMetadata.xlsx (run info + every parameter) are
# saved to OUTPUT_DIR\RUN_SNAPSHOT_SUBDIR\<YYYYmmdd_HHMMSS>.
RUN_SNAPSHOT_SUBDIR = 'RunSnapshots'

# ── Temporal stability (merged from TemporalStabilityMap_v6_MapPassStability.py) ──

# False: no temporal stability comparison at all.
INCLUDE_TEMPORAL_STABILITY = True

# True: build the temporal stability maps + TS_OUTPUT_EXCEL, then compare them.
# False: skip the maps and compare the PerCell sheet of an existing TS_OUTPUT_EXCEL.
RUN_TS_MAPS = True

# Place-cell folder (Output_PlaceTrue of a PlaceCell_Main_v21 run): every .ntt under
# it is analysed; each session folder also holds its tracking file.
TS_PLACE_CELL_FOLDER = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200'

# Per-cell / per-field / per-bin temporal stability table.
TS_OUTPUT_EXCEL = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Alpha200\Results\TemporalStability.xlsx'

# Temporal stability figures go in this subfolder of each session folder (next to the .ntt).
TEMPORAL_STABILITY_SUBDIR = 'temporal_stability'

MIN_VISIT_S     = 0.1      # visits shorter than this (s) are ignored; 0 = keep every visit
MIN_PASS_S      = 0.1      # field passes shorter than this (s) are ignored; 0 = keep every pass
MIN_SPIKES_PER_PASS = 5    # a pass counts as stable with >= this many spikes
STABILITY_CMAP  = 'jet'
NON_FIELD_COLOR = '0.75'   # gray for visited bins outside every place field

TS_MAX_WORKERS = pc.MAX_WORKERS

# Example maps for each animal-controlled comparison: per arena, the
# N_EXAMPLE_MAPS cells with the highest (best) and lowest (worst)
# EXAMPLE_METRIC and the N_EXAMPLE_MAPS closest to the arena's mean (mean),
# saved in AnimalControlledArenaType\EXAMPLE_MAPS_SUBDIR.
EXAMPLE_METRIC      = 'mean_pct_across_bins'
N_EXAMPLE_MAPS      = 3
EXAMPLE_MAPS_SUBDIR = 'ExampleMaps'

# ── Multi-peak fields (merged from PlaceCell_MultiPeak_KDE_v2.py) ───────────

# False: no multi-peak comparison at all.
INCLUDE_MULTIPEAK = True

# True: run the KDE multi-peak analysis (per-cell figures + MULTIPEAK_EXCEL),
# then compare it. False: skip it and compare the 'Fields' sheet of an
# existing MULTIPEAK_EXCEL.
RUN_MULTIPEAK_KDE = True

# Output_PlaceTrue folder of the PlaceCell_Main run that wrote INPUT_EXCEL
# (.ntt + tracking file of every place cell). The place cells and their
# fields are taken from INPUT_EXCEL ('Full' and 'PlaceFields' sheets).
MP_PLACE_CELL_FOLDER = TS_PLACE_CELL_FOLDER

# Peak detection on each field's KDE
PEAK_REL_HEIGHT_FRAC = 0.75     # a KDE peak is counted if its height is >= this
                                # fraction of the field's highest KDE peak
MIN_TROUGH_DIP_FRAC  = 0.2      # ...and if the KDE dips below (1 - this) x its height on the
                                # way to any higher peak (0.0: any real dip; flat ridges and
                                # plateaus never count as a second peak)
KDE_BW_METHOD        = 0.35     # gaussian_kde bw_method: 'scott', 'silverman' or a scalar
                                # factor (kernel sigma = factor x the field's weighted std)
MIN_KDE_POINTS       = 5        # fields with fewer bins of non-zero rate (in the KDE input map)
                                # get no KDE and are reported as single-peak (kde_note says why)

# Ratemap the KDE is fitted on (weights of the field bins); same options as
# PlaceCell_Main's SI_RATEMAP_TYPE:
#   'raw'                     : spikes / occupancy, unsmoothed
#   'occ_weighted_smooth'     : smooth(spikes) / smooth(occupancy), Gaussian sigma MP_GAUSSIAN_SIGMA_CM
#   'gaussian_smooth'         : Gaussian-smoothed ratemap (the map the fields were isolated on)
#   'Yartsev_adaptive_binned' : adaptive disc grows until it holds MP_YARTSEV_DISC_OCC_S of occupancy
#   'Skaggs_adaptive_binned'  : adaptive disc grows until r_cm * occ_s * sqrt(spikes) >= MP_SKAGGS_ALPHA
KDE_RATEMAP_TYPE = 'Skaggs_adaptive_binned'
_KDE_RATEMAP_TYPES = ('raw', 'occ_weighted_smooth', 'gaussian_smooth',
                      'Yartsev_adaptive_binned', 'Skaggs_adaptive_binned')
if KDE_RATEMAP_TYPE not in _KDE_RATEMAP_TYPES:
    raise ValueError(f"KDE_RATEMAP_TYPE must be one of {_KDE_RATEMAP_TYPES}, "
                     f"got {KDE_RATEMAP_TYPE!r}")
# Adaptive-map settings -- keep them equal to the PlaceCell_Main run that
# wrote INPUT_EXCEL so the KDE sees the same map as its SIR (Alpha200 run).
MP_ADAPTIVE_MAX_RSQ   = 200     # disc stops growing at r^2 = MP_ADAPTIVE_MAX_RSQ - 1 (bins^2)
MP_ADAPTIVE_MIN_OCC_S = 0.5     # bins whose final disc holds < this occupancy (s) get rate 0
MP_YARTSEV_DISC_OCC_S = 1.0     # Yartsev: occupancy (s) a disc must enclose
MP_SKAGGS_ALPHA       = 200.0   # Skaggs: cm * s * sqrt(spike)

# Fields left out of the multi-peak comparison:
EXCLUDE_NO_KDE_FIELDS    = True   # no KDE fitted (too few non-zero bins, see kde_note)
EXCLUDE_RATEMAP_MISMATCH = True   # rebuilt ratemap did not reproduce the Excel peak

# Ratemap rebuild (must match the PlaceCell_Main run that wrote INPUT_EXCEL)
MP_FPS                  = 30       # tracking frame rate (Hz)
MP_BIN_CM               = 2.0      # bin size in cm
MP_ARENA_WIDTH_CM       = 80.0     # physical arena width in cm (pixel coordinates only)
MP_USE_MIN_OCC_S        = False    # valid bins: >= MP_MIN_OCC_S seconds occupancy
MP_MIN_OCC_S            = 0.5
MP_USE_MIN_VISITS       = True     # valid bins: visited >= MP_MIN_VISITS times
MP_MIN_VISITS           = 2
MP_MAX_GAP_US           = 50_000   # max spike–position gap in µs (50 ms)
MP_POS_JUMP_THRESH_CMS  = 90.0     # tracking-artifact speed threshold (cm/s)
MP_POS_SMOOTH_SIGMA_SMP = 5.0      # Gaussian smoothing sigma (samples) of x/y position
MP_IMMOBILITY_SPEED_CMS = 0.5      # frames slower than this (cm/s) are dropped
MP_COORD_UNITS          = 'cm'     # 'cm': *_cm.csv (time col A, x col D, y col E); 'pixel': x/y/time columns
MP_GAUSSIAN_SIGMA_CM    = 3        # ratemap Gaussian smoothing sigma (cm)
MP_RADIAL_RANGE_CLIP_PCTILE = 0.5  # Circle track: percentile trimmed off each end of radial distance

MP_MAX_WORKERS = 4

# Example maps for each animal-controlled comparison (as EXAMPLE_METRIC, with
# N_EXAMPLE_MAPS), ranked by this per-cell multi-peak metric.
MP_EXAMPLE_METRIC = 'pct_multi_peak_fields'

# Output folder (inside MP_PLACE_CELL_FOLDER), named after the KDE settings
# so runs with different settings don't overwrite each other.
_MP_BW_TAG = KDE_BW_METHOD if isinstance(KDE_BW_METHOD, str) else f'{KDE_BW_METHOD:g}'
MULTIPEAK_SUBDIR = (f'MultiPeak_KDE_Pk{PEAK_REL_HEIGHT_FRAC * 100:g}_Tr{MIN_TROUGH_DIP_FRAC:g}'
                    f'_Sm{_MP_BW_TAG}_{KDE_RATEMAP_TYPE}').replace('.', '')
MULTIPEAK_DIR   = os.path.join(MP_PLACE_CELL_FOLDER, MULTIPEAK_SUBDIR)
MULTIPEAK_EXCEL = os.path.join(MULTIPEAK_DIR, 'MultiPeak_KDE_Fields.xlsx')
MP_RATEMAP_DIR  = os.path.join(MULTIPEAK_DIR, 'ratemaps_GaussianSmoothed')

# Saved to the 'Settings' sheet of MULTIPEAK_EXCEL.
MP_RUN_CONFIG_VARS = [
    'MP_PLACE_CELL_FOLDER', 'INPUT_EXCEL', 'MULTIPEAK_EXCEL', 'MP_RATEMAP_DIR',
    'PEAK_REL_HEIGHT_FRAC', 'MIN_TROUGH_DIP_FRAC', 'KDE_BW_METHOD', 'MIN_KDE_POINTS',
    'KDE_RATEMAP_TYPE', 'MP_ADAPTIVE_MAX_RSQ', 'MP_ADAPTIVE_MIN_OCC_S', 'MP_YARTSEV_DISC_OCC_S',
    'MP_SKAGGS_ALPHA', 'MP_MAX_WORKERS',
    'MP_FPS', 'MP_BIN_CM', 'MP_ARENA_WIDTH_CM',
    'MP_USE_MIN_OCC_S', 'MP_MIN_OCC_S', 'MP_USE_MIN_VISITS', 'MP_MIN_VISITS',
    'MP_MAX_GAP_US', 'MP_POS_JUMP_THRESH_CMS', 'MP_POS_SMOOTH_SIGMA_SMP', 'MP_IMMOBILITY_SPEED_CMS',
    'MP_COORD_UNITS', 'MP_GAUSSIAN_SIGMA_CM', 'MP_RADIAL_RANGE_CLIP_PCTILE',
]

MP_NTT_DTYPE = np.dtype([
    ('timestamp',   '<u8'),
    ('sc_number',   '<u4'),
    ('cell_number', '<u4'),
    ('params',      '<u4', (8,)),
    ('waveforms',   '<i2', (32, 4)),
])

ARENA_TYPES = ['Circle', 'Linear', 'Open']

# Session types within each arena, in plotting order.
ARENA_SESSION_TYPES = {
    'Open':   ['Cntrl', 'Rotate', 'Zero'],
    'Linear': ['0', '90', '180', '270'],
    'Circle': ['Stnd', 'NoRot', 'Rot'],
}
ALPHA       = 0.05

# Animal-controlled arena comparisons (level 3): {name: (animals, arenas)}.
# Each compares the given arenas using only the cells of the given animals,
# which were recorded in all of those arenas. Animal IDs are matched
# case-insensitively against the animal parsed from the session path
# (CellClusteredStats_Utils.ANIMAL_PATTERN). The plots of all comparisons are
# saved side by side (left to right in this order) in ANIMAL_CONTROLLED_DIR.
ANIMAL_CONTROLLED_DIR = 'AnimalControlledArenaType'
ANIMAL_CONTROLLED_COMPARISONS = {
    'Circle_vs_Open': (['Fa8477', 'Fa5834'], ['Circle', 'Open']),
    'Linear_vs_Open': (['Fa1059', 'Fa23BD'], ['Linear', 'Open']),
}
PANEL_GAP_IN = 0.3   # white gap between the side-by-side panels (inches)

# True: a unit with the same label recorded in two arenas on the same day
# (same animal + day + unit) is treated as the SAME cell. Only correct if that
# day's sessions in both arenas were spike-sorted together; set False if each
# arena was sorted separately (labels then coincide only by chance).
LINK_CELLS_ACROSS_ARENAS = True

# Test for the continuous metrics:
#   'LMM' : nested linear mixed model (accounts for repeated cells, days and
#           animals; see CellClusteredStats_Utils.compare_continuous_nested)
#   'KW'  : Kruskal-Wallis omnibus + pairwise Mann-Whitney U post-hoc, Holm-
#           Bonferroni corrected -- ASSUMES every row is independent (ignores
#           cells recorded in several sessions / arenas of one day)
CONTINUOUS_TEST = 'KW'

# With CONTINUOUS_TEST = 'LMM', the model is fitted to 'rank' (the ranks of
# the metric; robust, the mixed-model analogue of Kruskal-Wallis) or 'none'
# (the raw values).
METRIC_TRANSFORM = 'rank'

FULL_METRICS = {
    'peak_fr':         'Peak firing rate',
    'sir':             'Spatial information score',
    'sparsity':        'Sparsity',
    'coherence':       'Coherence score',
    'stability_score':   'Stability score',
    'final_speed_score': 'Speed score',
}

FIELD_METRICS = {
    'n_fields':       'Number of place fields per cell',
    'total_area_cm2': 'Area occupied by fields per cell',
    'pct_area':       'Percentage area occupied by fields per cell',
}

# Place-field shape, one value per field (PlaceFieldShape_Utils.py).
FIELD_SHAPE_METRICS = {
    'fld_max_width_cm':   'Maximum place-field width (cm)',
    'fld_min_width_cm':   'Minimum place-field width (cm)',
    'fld_axis_angle_deg': 'Angle between max- and min-width axes (deg)',
    'fld_width_ratio':    'Place-field width ratio (max / min)',
    'fld_mean_width_cm':  'Mean place-field width (cm)',
}

# Stability score is also compared within each stability type (split-half
# significance, stability_p_value <= ALPHA, and the sign of the score); one
# extra box plot per type, 'Full_stability_score_<type>.png'.
STABILITY_TYPES = {
    'SigPos': 'significant, positive',
    'SigNeg': 'significant, negative',
    'NonSig': 'not significant',
}

# Extra continuous metric generated by the characterization pipeline but not
# covered by FULL_METRICS above.
EXTRA_FULL_METRICS = {'mean_fr': 'Mean firing rate'}

# Place-cell yield uses every recorded unit (place_cell True/False/None), not
# just confirmed place cells.
CATEGORICAL_METRICS_ALL_UNITS = {
    'place_cell': 'Place-cell yield (% of recorded units classified as place cells)',
}
CATEGORICAL_METRICS_PLACE_CELLS = {
    'theta_modulated':          'Theta modulation (% of place cells)',
    'coherence_bootstrap_sig':  'Coherence shuffle significance (% of place cells)',
    'stability_significant':    'Split-half stability significance, p <= 0.05 (% of place cells)',
    'speed_cell_final':         'Speed cells (% of place cells)',
}

# Temporal stability, one value per cell (PerCell columns of TS_OUTPUT_EXCEL).
TEMPORAL_STABILITY_METRICS = {
    'mean_pct_across_bins':   'Mean temporal stability across field bins (% visits with >= 1 spike)',
    'median_pct_across_bins': 'Median temporal stability across field bins (% visits with >= 1 spike)',
    'occ_wmean_pct_across_bins':   'Occupancy-weighted mean temporal stability across field bins (% visits with >= 1 spike)',
    'occ_wmedian_pct_across_bins': 'Occupancy-weighted median temporal stability across field bins (% visits with >= 1 spike)',
    'occ_wmean_pct_mapnorm':  'Field temporal stability weighted by dwell, normalised to whole-map dwell (%)',
    'occ_wmean_pct_all_bins': 'Occupancy-weighted mean temporal stability across all visited map bins (% visits with >= 1 spike)',
    'pct_at_peak_bin':       'Temporal stability at the raw-ratemap peak bin (% visits with >= 1 spike)',
    'lap_stability_score':   f'Lap stability, passes pooled over all fields '
                             f'(fraction of passes with >= {MIN_SPIKES_PER_PASS} spikes)',
    'mean_lap_stability_across_fields':
                             f'Mean lap stability across fields '
                             f'(fraction of passes with >= {MIN_SPIKES_PER_PASS} spikes)',
}

# Multi-peak fields, by table (build_multipeak_dfs): 'Cell' one value per
# place cell, 'Field' one per place field, 'MultiField' one per multi-peak field.
MULTIPEAK_METRICS = {
    'Cell': {
        'n_multi_peak_fields':   'Multi-peak fields per cell',
        'pct_multi_peak_fields': 'Percentage of fields per cell that are multi-peak',
        'max_n_peaks':           'Most peaks in a field of the cell',
    },
    'Field': {
        'n_peaks': 'Number of peaks per field',
    },
    'MultiField': {
        'shallowest_trough_rel':   'Trough / lower peak (multi-peak fields)',
        'lowest_counted_peak_rel': 'Lowest peak / highest peak (multi-peak fields)',
        'peak_separation_cm':      'Distance between neighbouring peaks, cm (multi-peak fields)',
    },
}
MULTIPEAK_CATEGORICAL = {
    'Cell':  {'has_multi_peak_field': 'Cells with a multi-peak field (% of place cells)'},
    'Field': {'multi_peak':           'Multi-peak fields (% of place fields)'},
}


@dataclass
class Grouping:
    """What the groups being compared are: the column holding the group label,
    the groups in plotting order, a plural noun for the stats text, and an
    optional prefix for plot titles (e.g. 'Linear arena'). `shade` is the
    per-cell shading of this comparison, {(session, unit): t} (cell_shades),
    set by run_comparison."""
    col: str
    order: list
    noun: str
    title: str = ''
    shade: dict = None


ARENA_GROUPING = Grouping('arena_type', ARENA_TYPES, 'arena types')


def session_grouping(arena: str) -> Grouping:
    return Grouping('session_type', ARENA_SESSION_TYPES[arena], 'session types',
                    f'{arena} arena')


def animal_controlled_grouping(animals: list, arenas: list) -> Grouping:
    return Grouping('arena_type', [a for a in ARENA_TYPES if a in arenas], 'arena types',
                    f'{" + ".join(animals)} only')

# ── Helpers ───────────────────────────────────────────────────────────────

def _to_bool(x) -> bool:
    """Normalize place_cell values that may come back from Excel as bool,
    numpy bool, or string ('True'/'False')."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == 'true'
    if pd.isna(x):
        return False
    return bool(x)


def _to_tri_bool(x):
    """Normalize a boolean-ish Excel cell (bootstrap_sig / speed_cell / ...)
    to True, False, or None (unknown/blank/NaN/'not tested'), since
    these columns may come back from Excel as bool, string, or a blank cell
    that pandas reads as NaN."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s == 'true':
            return True
        if s == 'false':
            return False
        return None
    if pd.isna(x):
        return None
    return bool(x)


def _speed_class(score, cell):
    """'p' (p-type speed cell), 'n' (n-type), 'non' (non-speed), or None if
    final_speed_score is missing. A cell is a speed cell only when speed_cell
    is True; its type follows the sign of final_speed_score. speed_cell False
    or 'not tested' (the shuffle was skipped because the initial fit wasn't
    significant) both count as non-speed."""
    s = pd.to_numeric(score, errors='coerce')
    if pd.isna(s):
        return None
    if _to_tri_bool(cell) is True and s != 0:
        return 'p' if s > 0 else 'n'
    return 'non'


def add_speed_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'speed_class' ('p'/'n'/'non'/None) and 'speed_cell_final'
    (True/False/None, for the % speed cells comparison) columns."""
    df = df.copy()
    df['speed_class'] = [_speed_class(s, c)
                         for s, c in zip(df['final_speed_score'], df['speed_cell'])]
    df['speed_cell_final'] = df['speed_class'].map(
        lambda k: None if k is None else k in ('p', 'n'))
    return df


def _stability_sig(v):
    v = pd.to_numeric(v, errors='coerce')
    return bool(v <= ALPHA) if pd.notna(v) else None


def _stability_type(score, p):
    """'SigPos' / 'SigNeg' (split-half p <= ALPHA with a score > 0 / < 0),
    'NonSig' (p > ALPHA), or None if the score or p-value is missing (or the
    score is exactly 0 and significant)."""
    s = pd.to_numeric(score, errors='coerce')
    sig = _stability_sig(p)
    if pd.isna(s) or sig is None:
        return None
    if not sig:
        return 'NonSig'
    return 'SigPos' if s > 0 else 'SigNeg' if s < 0 else None


def extract_arena_type(session_path: str):
    """Return 'Open', 'Linear', or 'Circle' if one of those appears as a
    path component of `session_path`, else None."""
    parts = re.split(r'[\\/]+', str(session_path))
    for part in parts:
        for arena in ARENA_TYPES:
            if part.strip().lower() == arena.lower():
                return arena
    return None


def extract_session_type(session_path: str, arena: str):
    """Return the session type named in the last folder of `session_path`
    (e.g. '1_Cntrl' -> 'Cntrl' for Open, '3_180' -> '180' for Linear,
    '2NoRot' -> 'NoRot' for Circle), or None if it can't be recognized."""
    if arena is None:
        return None
    label = re.split(r'[\\/]+', str(session_path).strip())[-1].strip().lower()

    if arena == 'Open':
        for t in ARENA_SESSION_TYPES['Open']:
            if t.lower() in label:
                return t
    elif arena == 'Linear':
        m = re.search(r'_(270|180|90|0)$', label)
        if m:
            return m.group(1)
    elif arena == 'Circle':
        # 'norot' must be tested before 'rot', which it contains.
        if 'norot' in label:
            return 'NoRot'
        if 'stnd' in label:
            return 'Stnd'
        if 'rot' in label:
            return 'Rot'
    return None


def _label(group: str) -> str:
    """Display label for plots: Linear session types are angles."""
    return f'{group}°' if str(group).isdigit() else str(group)


def _tag_groups(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Add 'arena_type', 'session_type' and 'cell_id' (+ 'animal', 'rec_day')
    columns to a dataframe with 'session' and 'unit' columns. Rows with no
    recognizable arena are dropped; rows with an arena but no recognizable
    session type are kept (session_type None) for the arena comparison."""
    df = df.copy()
    df['arena_type'] = df['session'].apply(extract_arena_type)
    unknown = df[df['arena_type'].isna()]
    if len(unknown):
        print(f'WARNING: {len(unknown)} {what} had a session path with no '
              f'recognizable arena type ({ARENA_TYPES}); excluded from analysis:')
        for s in sorted(unknown['session'].astype(str).unique()):
            print(f'    {s}')
    df = df.dropna(subset=['arena_type']).copy()

    df['session_type'] = [extract_session_type(s, a)
                          for s, a in zip(df['session'], df['arena_type'])]
    no_type = df[df['session_type'].isna()]
    if len(no_type):
        print(f'WARNING: {len(no_type)} {what} had no recognizable session type; '
              f'kept in the arena comparison, excluded from the session-type comparison:')
        for s in sorted(no_type['session'].astype(str).unique()):
            print(f'    {s}')
    return ccs.add_cell_ids(df, include_arena=not LINK_CELLS_ACROSS_ARENAS)


def load_sheets():
    """Read 'Full' and 'PlaceFields', add the field-shape columns to the
    latter and save them to FIELD_SHAPE_EXCEL (a copy of INPUT_EXCEL)."""
    full   = pd.read_excel(INPUT_EXCEL, sheet_name='Full')
    fields = pfs.add_field_shape(pd.read_excel(INPUT_EXCEL, sheet_name='PlaceFields'))
    pfs.save_shape_columns(INPUT_EXCEL, FIELD_SHAPE_EXCEL, fields)
    print(f'Place-field shape columns saved to {FIELD_SHAPE_EXCEL}')
    return full, fields


def build_full_metrics_df(full: pd.DataFrame) -> pd.DataFrame:
    df = full[full['place_cell'].apply(_to_bool)]
    df = add_speed_class(_tag_groups(df, 'place cell(s)'))
    df['stability_significant'] = df['stability_p_value'].apply(_stability_sig)
    df['stability_type'] = [_stability_type(s, p) for s, p in
                            zip(df['stability_score'], df['stability_p_value'])]
    return df


def build_field_metrics_df(full: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    place_cells = full[full['place_cell'].apply(_to_bool)][
        ['session', 'unit', 'n_fields_detected']].drop_duplicates()

    agg = (fields.groupby(['session', 'unit'])
                 .agg(n_fields=('field_number', 'count'),
                      total_area_cm2=('area_cm2', 'sum'),
                      pct_area=('pct_of_occupied_area', 'sum'))
                 .reset_index())

    merged = place_cells.merge(agg, on=['session', 'unit'], how='left')
    # Place cells with no rows in 'PlaceFields' (n_fields_detected == 0) get 0s.
    for col in ('n_fields', 'total_area_cm2', 'pct_area'):
        merged[col] = merged[col].fillna(0.0)

    return _tag_groups(merged, 'place cell(s) (field analysis)')


def build_field_shape_df(full: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    """One row per place field of a place cell, with its shape columns.
    Fields with no 2-D shape (circular track) are dropped."""
    place_cells = full[full['place_cell'].apply(_to_bool)][['session', 'unit']].drop_duplicates()
    df = fields[['session', 'unit', 'field_number'] + pfs.SHAPE_COLUMNS].merge(
        place_cells, on=['session', 'unit'], how='inner')
    df = df.dropna(subset=['fld_max_width_cm'])
    return _tag_groups(df, 'place field(s) (field-shape analysis)')


def build_all_units_df(full: pd.DataFrame) -> pd.DataFrame:
    """Like build_full_metrics_df, but keeps every recorded unit (not just
    confirmed place cells) -- needed to compute place-cell yield per group."""
    return _tag_groups(full, 'unit(s)')


def compare_groups(df: pd.DataFrame, metric_col: str, metric_label: str, grp: Grouping):
    """Descriptive stats, omnibus test, and post-hoc pairwise tests for one
    metric across the groups of `grp`. CONTINUOUS_TEST = 'KW': Kruskal-Wallis,
    then pairwise Mann-Whitney U (Holm-Bonferroni corrected), rows treated as
    independent; 'LMM': nested linear mixed model accounting for repeated
    cells, days and animals (CellClusteredStats_Utils.compare_continuous_nested).
    Returns (desc_df, omnibus_row, posthoc_df)."""
    groups, n_cells = {}, {}
    for g in grp.order:
        sub = df.loc[df[grp.col] == g, [metric_col, 'cell_id']].copy()
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
        sub = sub.dropna(subset=[metric_col])
        if len(sub):
            groups[g] = sub[metric_col].to_numpy(dtype=float)
            n_cells[g] = sub['cell_id'].nunique()

    desc_rows = []
    for g, vals in groups.items():
        desc_rows.append({
            'metric': metric_label, grp.col: g, 'n': len(vals),
            'n_cells': n_cells[g],
            'mean': np.mean(vals), 'median': np.median(vals),
            'q1': np.percentile(vals, 25), 'q3': np.percentile(vals, 75),
            'std': np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            'sem': stats.sem(vals) if len(vals) > 1 else np.nan,
            'min': np.min(vals), 'max': np.max(vals),
        })
    desc_df = pd.DataFrame(desc_rows)

    omnibus = {'metric': metric_label, 'n_groups': len(groups),
               'groups': ', '.join(f'{g} (n={len(v)})' for g, v in groups.items())}

    if CONTINUOUS_TEST == 'KW':
        if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
            omni, posthoc_rows = None, []
        else:
            sub = df[df[grp.col].isin(list(groups))].copy()
            sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
            summary = ccs.repetition_summary(sub.dropna(subset=[metric_col]), grp.col)
            try:
                omni, posthoc_rows = ccs._classic_continuous(
                    groups, f'Kruskal-Wallis + Mann-Whitney U (rows treated as independent): {summary}')
            except ValueError:  # e.g. all values identical
                omni, posthoc_rows = None, []
    elif CONTINUOUS_TEST == 'LMM':
        omni, posthoc_rows = ccs.compare_continuous_nested(df, grp.col, grp.order,
                                                           metric_col, transform=METRIC_TRANSFORM)
    else:
        raise ValueError(f"CONTINUOUS_TEST must be 'LMM' or 'KW', not {CONTINUOUS_TEST!r}")
    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    omnibus.update(omni)
    omnibus['significant'] = bool(omni['p_value'] < ALPHA)
    posthoc_df = pd.DataFrame([{'metric': metric_label, **r} for r in posthoc_rows])
    return desc_df, omnibus, posthoc_df



# ── Figure style (matches the Figure 3 place-cell histograms) ────────────────
# Arial, bold axis labels, small plain tick labels, thin black left/bottom
# axes with short outward ticks, solid-filled bars separated by thin white
# gaps, and a dashed median line labelled 'med=...'.
PAL_MAGENTA = '#FA7FFA'
PAL_CYAN    = '#7FFAFA'
PAL_BLUE    = '#7FB3E5'
PAL_DKBLUE  = '#0066CC'
PAL_GRAY    = '#7F7F7F'
PAL_GREEN   = '#00C000'
PAL_ORANGE  = '#FF8000'
PAL_RED     = '#FF0000'
PAL_BLACK   = '#000000'
FIG_FONT    = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

FS_LABEL, FS_TITLE, FS_TICK, FS_LEGEND, FS_STATS, FS_MED = 14, 15, 11, 11, 9, 10
STATS_LINE_IN = 0.2    # panel height per line of stats text (inches)
LEGEND_ROW_IN = 0.3    # panel height per row of legend entries (inches)
FIG_DPI = 500

AXIS_LW     = 0.8      # spines, ticks and the dashed median line
BOX_LW      = 1.0      # box-plot outlines, whiskers, caps; significance lines
MEDIAN_LW   = 1.5      # box-plot median
BAR_EDGE    = 'white'  # bar/histogram outlines: a thin white gap between bars
BAR_EDGE_LW = 0.6
BAR_WIDTH   = 0.6      # categorical bars, as a fraction of the bar spacing
GROUP_W     = 1.1      # box/bar plots: x-axis width per condition (inches)
AXIS_PAD_W  = 1.4      # box/bar plots: extra width for the y-axis label/ticks
STATS_CHARS_PER_IN = 15  # approx. characters of stats text per inch of width

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    FIG_FONT,
    'pdf.fonttype':       42,       # keep text editable in Illustrator
    'ps.fonttype':        42,
    'svg.fonttype':       'none',
    'text.color':         PAL_BLACK,
    'axes.labelcolor':    PAL_BLACK,
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.edgecolor':     PAL_BLACK,
    'axes.linewidth':     AXIS_LW,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.color':        PAL_BLACK,
    'ytick.color':        PAL_BLACK,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.width':  AXIS_LW,
    'ytick.major.width':  AXIS_LW,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'axes.grid':          False,
    'legend.frameon':     False,
})

# Per-group colors. face = fill for boxes/bars/histograms/points; dark / light
# = deeper and paler tones of that shade, used for p-type / n-type speed cells
# (see plot_speed_direction); edge = point outline.
def _shade(face, dark, light):
    return {'face': face, 'edge': PAL_BLACK, 'dark': dark, 'light': light}

# Arena types: magenta = circular track, cyan = linear track, blue = open arena.
ARENA_COLORS = {
    'Open':   _shade(PAL_BLUE,    PAL_DKBLUE, '#CCE1F5'),
    'Linear': _shade(PAL_CYAN,    '#00B3B3',  '#CCFDFD'),
    'Circle': _shade(PAL_MAGENTA, '#C000C0',  '#FDCCFD'),
}

# Session types: shades of their arena's hue.
SESSION_COLORS = {
    # Open: Cntrl light blue, Rotate dark blue, Zero gray
    'Cntrl':  _shade(PAL_BLUE,   PAL_DKBLUE, '#CCE1F5'),
    'Rotate': _shade(PAL_DKBLUE, '#003D7A',  '#99C2EB'),
    'Zero':   _shade('#BFBFBF',  '#404040',  '#E6E6E6'),
    # Linear: 0 darkest cyan -> 270 lightest
    '0':      _shade('#008080',  '#004D4D',  '#99CCCC'),
    '90':     _shade('#00B3B3',  '#006B6B',  '#99E1E1'),
    '180':    _shade('#33D6D6',  '#1F8080',  '#ADEFEF'),
    '270':    _shade(PAL_CYAN,   '#00B3B3',  '#CCFDFD'),
    # Circle: Stnd lightest magenta, NoRot lighter, Rot dark
    'Stnd':   _shade('#FDC4FD',  '#D94FD9',  '#FEE7FE'),
    'NoRot':  _shade(PAL_MAGENTA, '#C000C0', '#FDCCFD'),
    'Rot':    _shade('#C000C0',  '#730073',  '#E699E6'),
}

# Arena and session-type labels don't overlap, so one lookup serves both.
GROUP_COLORS = {**ARENA_COLORS, **SESSION_COLORS}
_DEFAULT_COLOR = {'face': PAL_GRAY, 'edge': PAL_BLACK, 'dark': PAL_GRAY, 'light': '#D4D4D4'}

# Points that failed their significance/shuffle test are always drawn in this
# neutral gray (the reference figures' "discarded" gray), regardless of group.
SIG_GRAY = {'face': PAL_GRAY, 'edge': PAL_BLACK}

# Per-cell shading: every point in the box plots is filled with its group's
# hue, darker for a higher SHADE_METRIC (spatial information score). The
# shades are computed once per comparison -- each SessionType\<arena> folder,
# ArenaType, and each animal-controlled pair -- from that comparison's place
# cells, and a cell (one unit in one session) keeps its shade in every box
# plot of that comparison (all fields of a cell share the cell's shade).
SHADE_METRIC    = 'sir'
SHADE_SCALE     = 'rank'          # 'rank': percentile of the score; 'linear': min-max of the score
# Lowest score: the group's pure hue mixed with SHADE_LIGHT_MIX white (kept
# well below 1 so it stays visible); highest score: the pure hue scaled to
# SHADE_DARK_KEEP of its brightness. Shades in between are blended linearly.
SHADE_LIGHT_MIX = 0.45
SHADE_DARK_KEEP = 0.35

# Point markers for the significance metrics (the fill stays the cell's shade).
MARKER_PASS, MARKER_FAIL, MARKER_P, MARKER_N = 'o', 'X', '^', 'v'
MARKER_SIZE = {MARKER_PASS: 22}   # others: 30 (they look smaller at equal area)


def cell_shades(df: pd.DataFrame) -> dict:
    """{(session, unit): t}, t from 0 (lowest SHADE_METRIC in `df`) to 1
    (highest); rows with no score are left out."""
    v = pd.to_numeric(df[SHADE_METRIC], errors='coerce')
    ok = v.notna()
    v = v[ok]
    if SHADE_SCALE == 'rank':
        t = (v.rank(method='average') - 1) / max(len(v) - 1, 1)
    elif SHADE_SCALE == 'linear':
        span = v.max() - v.min()
        t = (v - v.min()) / span if span > 0 else v * 0 + 0.5
    else:
        raise ValueError(f"SHADE_SCALE must be 'rank' or 'linear', not {SHADE_SCALE!r}")
    return dict(zip(zip(df.loc[ok, 'session'], df.loc[ok, 'unit']), t))


def _cell_color(group: str, t: float) -> str:
    """The group's hue at shade position t (0 = lightest tint, 1 = darkest)."""
    h, _l, s = colorsys.rgb_to_hls(*to_rgb(_color(group)['face']))
    pure  = np.array(colorsys.hls_to_rgb(h, 0.5, s))
    light = pure + (1.0 - pure) * SHADE_LIGHT_MIX
    dark  = pure * SHADE_DARK_KEEP
    return to_hex(light + (dark - light) * t)


def _color(group: str) -> dict:
    return GROUP_COLORS.get(group, _DEFAULT_COLOR)


def _new_fig(figsize, legend_rows=0, stats_lines=None):
    """Figure + main axes. If there is a legend and/or stats text, a second
    borderless axes sits below the plot to hold them (legend on top, stats
    underneath), so neither overlaps the data. The panel is sized to fit."""
    w, h = figsize
    panel_h = legend_rows * LEGEND_ROW_IN + len(stats_lines or []) * STATS_LINE_IN
    if panel_h > 0:
        panel_h += 0.2
        fig, (ax, ax_info) = plt.subplots(2, 1, figsize=(w, h + panel_h),
                                          gridspec_kw={'height_ratios': [h, panel_h]})
        ax_info.axis('off')
        return fig, ax, ax_info
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, None


def _bar_fig_width(n_groups: int) -> float:
    """Figure width for a box/bar plot with `n_groups` conditions: a fixed
    width per condition, so boxes/bars and the gaps between them keep their
    proportions whatever the number of conditions."""
    return n_groups * GROUP_W + AXIS_PAD_W


def _wrap_stats(stats_lines, fig_w: float) -> list:
    """Re-wrap the stats text lines to fit a figure `fig_w` inches wide."""
    width = max(40, int(fig_w * STATS_CHARS_PER_IN))
    return [w for line in (stats_lines or []) for w in (textwrap.wrap(line, width) or [''])]


def _draw_stats(ax_info, stats_lines):
    """Write the stats text at the bottom of the panel (below any legend)."""
    if ax_info is None or not stats_lines:
        return
    ax_info.text(0.5, 0.0, '\n'.join(stats_lines), transform=ax_info.transAxes,
                 fontsize=FS_STATS, va='bottom', ha='center', linespacing=1.3)


# ── Stats text shown under each plot ─────────────────────────────────────────

def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _stars(p) -> str:
    if p is None or pd.isna(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < ALPHA else 'n.s.'


def _pair_label(cmp: str) -> str:
    return ' vs '.join(_label(g) for g in str(cmp).split(' vs '))


def _desc_str(vals) -> str:
    """'mean ± SD | median [Q1, Q3]' for an array of values."""
    vals = np.asarray(vals, dtype=float)
    if not len(vals):
        return 'n/a'
    sd = f'{np.std(vals, ddof=1):.3g}' if len(vals) > 1 else 'n/a'
    return (f'mean {np.mean(vals):.3g} ± {sd} SD | median {np.median(vals):.3g} '
            f'[Q1 {np.percentile(vals, 25):.3g}, Q3 {np.percentile(vals, 75):.3g}]')


def _test_lines(omnibus: dict, posthoc_df: pd.DataFrame) -> list:
    """Omnibus test (statistic, dof, p) and post-hoc pairwise tests (statistic,
    raw p, Holm-corrected p) as text lines."""
    test = omnibus.get('test', '')
    if test == 'insufficient data':
        return ['Insufficient data for a statistical test']
    dof = omnibus.get('dof')
    dof_str = f', dof = {int(dof)}' if dof is not None and pd.notna(dof) else ''
    p = omnibus.get('p_value')
    lines = [f'{test}: {omnibus.get("stat_name", "stat")} = {omnibus["statistic"]:.3f}{dof_str}, '
             f'p = {_fmt_p(p)} {_stars(p)}']
    if posthoc_df is not None and len(posthoc_df):
        lines.append(f'Post-hoc ({posthoc_df["test"].iloc[0]}, Holm-Bonferroni corrected):')
        for _, r in posthoc_df.iterrows():
            fmt = '.3g' if r['stat_name'] == 'OR' else '.1f' if r['stat_name'] == 'U' else '.3f'
            stat_str = f'{r["stat_name"]} = {r["statistic"]:{fmt}}'
            lines.append(f'{_pair_label(r["comparison"])}: {stat_str}, p = {_fmt_p(r["p_raw"])}, '
                         f'p(Holm) = {_fmt_p(r["p_holm"])} {_stars(r["p_holm"])}')
    return lines


def _model_note_lines(omnibus: dict) -> list:
    """Which model was used, and on how many cells (one line; wrapped to the
    figure width by _wrap_stats)."""
    note = omnibus.get('model_note')
    return [note] if isinstance(note, str) and note else []


def _continuous_stats_lines(desc_df, omnibus, posthoc_df, grp: Grouping) -> list:
    lines = []
    for _, r in desc_df.iterrows():
        sd = f'{r["std"]:.3g}' if pd.notna(r['std']) else 'n/a'
        lines.append(f'{_label(r[grp.col])} (n={int(r["n"])}): mean {r["mean"]:.3g} ± {sd} SD | '
                     f'median {r["median"]:.3g} [Q1 {r["q1"]:.3g}, Q3 {r["q3"]:.3g}]')
    return lines + _test_lines(omnibus, posthoc_df) + _model_note_lines(omnibus)


def _categorical_stats_lines(desc_df, omnibus, posthoc_df, grp: Grouping) -> list:
    lines = [f'{_label(r[grp.col])} (n={int(r["n"])}): {int(r["n_true"])} yes / '
             f'{int(r["n_false"])} no = {r["pct_true"]:.1f} %'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df) + _model_note_lines(omnibus)


def _speed_dir_stats_lines(desc_df, omnibus, posthoc_df, grp: Grouping) -> list:
    lines = [f'{_label(r[grp.col])} (n={int(r["n"])}): '
             f'p-Speed {int(r["n_p_speed"])} ({r["pct_p_speed"]:.1f} %), '
             f'n-Speed {int(r["n_n_speed"])} ({r["pct_n_speed"]:.1f} %), '
             f'non-speed {int(r["n_non_speed"])} ({r["pct_non_speed"]:.1f} %)'
             for _, r in desc_df.iterrows()]
    return lines + _test_lines(omnibus, posthoc_df) + _model_note_lines(omnibus)


def _style_axes(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines[['top', 'right']].set_visible(False)


def _titled(text: str, grp: Grouping) -> str:
    return f'{grp.title}: {text}' if grp.title else text


def _set_title(ax, text):
    # ~7 bold title characters per inch of figure width
    ax.set_title(textwrap.fill(text, max(20, int(ax.figure.get_figwidth() * 7))),
                 fontsize=FS_TITLE - 2, pad=12)


def _save(fig, out_path):
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def _draw_sig_brackets(ax, pos_by_group: dict, posthoc_df, y_base: float,
                       y_step: float, y_text: float):
    """For every pairwise comparison with Holm-corrected p < ALPHA, draw a
    single horizontal line above the two groups with the number of stars
    (* p<0.05, ** p<0.01, *** p<0.001) centered on it. Narrower spans are
    stacked lowest so lines don't cross. Returns the y of the highest line
    drawn, or None if no comparison was significant."""
    if posthoc_df is None or not len(posthoc_df):
        return None
    sig = []
    for _, r in posthoc_df.iterrows():
        p = r.get('p_holm')
        if p is None or pd.isna(p) or p >= ALPHA:
            continue
        g1, g2 = [g.strip() for g in str(r['comparison']).split(' vs ')]
        if g1 not in pos_by_group or g2 not in pos_by_group:
            continue
        x1, x2 = sorted((pos_by_group[g1], pos_by_group[g2]))
        sig.append((x2 - x1, x1, x2, _stars(p)))
    y = y_base
    for _span, x1, x2, stars in sorted(sig):
        ax.plot([x1, x2], [y, y], color=PAL_BLACK, linewidth=BOX_LW,
                solid_capstyle='butt', clip_on=False, zorder=5)
        ax.text((x1 + x2) / 2, y + y_text, stars, ha='center', va='center',
                fontsize=FS_LABEL + 3, fontweight='bold', clip_on=False, zorder=5)
        y += y_step
    return y - y_step if sig else None


# Metrics whose individual points get a marker by p-value/shuffle-test status
# -- see _point_styles. Also drives which plots get a significance legend.
SIGNIFICANCE_METRICS = {'coherence', 'stability_score', 'final_speed_score'}


def _point_styles(metric_col: str, sub: pd.DataFrame, group: str, shades: dict) -> list:
    """Per-point (face, marker) for one group's jittered scatter points,
    aligned row-for-row with `sub` (which must still hold the original
    columns, e.g. 'coherence_bootstrap_sig').

    face: the cell's shade (`shades`, see cell_shades); the plain group color
    for cells with no shade. marker:
    - coherence: MARKER_FAIL only if the cell's coherence failed its shuffle
      test (coherence_bootstrap_sig is False); untested/blank cells keep
      MARKER_PASS.
    - stability_score: MARKER_FAIL unless its split-half p-value is <= ALPHA.
    - final_speed_score: MARKER_P / MARKER_N for p-type (score > 0) / n-type
      (score < 0) speed cells, MARKER_FAIL for the rest (speed_cell False /
      'not tested').
    - everything else: MARKER_PASS.
    """
    shades = shades or {}
    base = _color(group)['face']
    faces = [_cell_color(group, shades[k]) if k in shades else base
             for k in zip(sub['session'], sub['unit'])]

    if metric_col == 'coherence':
        markers = [MARKER_FAIL if _to_tri_bool(s) is False else MARKER_PASS
                   for s in sub['coherence_bootstrap_sig']]
    elif metric_col == 'stability_score':
        p = pd.to_numeric(sub['stability_p_value'], errors='coerce')
        markers = [MARKER_PASS if (pd.notna(v) and v <= ALPHA) else MARKER_FAIL for v in p]
    elif metric_col == 'final_speed_score':
        markers = [{'p': MARKER_P, 'n': MARKER_N}.get(k, MARKER_FAIL) for k in sub['speed_class']]
    else:
        markers = [MARKER_PASS] * len(sub)
    return list(zip(faces, markers))


_SIGNIFICANCE_LEGENDS = {
    'coherence': [
        (MARKER_PASS, 'passed shuffle test (p < 0.05) / not tested'),
        (MARKER_FAIL, 'failed shuffle test'),
    ],
    'stability_score': [
        (MARKER_PASS, f'p ≤ {ALPHA:g}'),
        (MARKER_FAIL, f'p > {ALPHA:g}'),
    ],
    'final_speed_score': [
        (MARKER_P,    'p-type speed cell (positive)'),
        (MARKER_N,    'n-type speed cell (negative)'),
        (MARKER_FAIL, 'not a speed cell / not tested'),
    ],
}


def _add_significance_legend(ax_info, metric_col: str, face: str):
    """Draw the significance (marker) legend into the borderless panel below
    the plot, with `face` as the example fill."""
    entries = _SIGNIFICANCE_LEGENDS.get(metric_col)
    if not entries:
        return
    handles = [Line2D([0], [0], marker=marker, linestyle='None', markersize=8,
                       markerfacecolor=face, markeredgecolor=PAL_BLACK,
                       markeredgewidth=0.5, label=label)
               for marker, label in entries]
    ax_info.legend(handles=handles, loc='upper center', fontsize=FS_LEGEND, ncol=1)


def plot_metric(df: pd.DataFrame, metric_col: str, metric_label: str, out_path: str,
                grp: Grouping, stats_lines=None, posthoc_df=None):
    present = []
    subsets = {}
    for g in grp.order:
        sub = df.loc[df[grp.col] == g].copy()
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors='coerce')
        sub = sub.dropna(subset=[metric_col])
        if len(sub):
            present.append(g)
            subsets[g] = sub
    if not present:
        return

    data = [subsets[g][metric_col].to_numpy(dtype=float) for g in present]
    positions = np.arange(1, len(present) + 1)

    has_legend = metric_col in SIGNIFICANCE_METRICS
    fig_w = _bar_fig_width(len(present))
    if grp.shade:
        stats_lines = list(stats_lines or []) + [
            f'Point shade: the cell\'s {FULL_METRICS[SHADE_METRIC].lower()} '
            f'({"rank" if SHADE_SCALE == "rank" else "value"} within this comparison), '
            f'darker = higher']
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig(
        (fig_w, 5.5), stats_lines=stats_lines,
        legend_rows=len(_SIGNIFICANCE_LEGENDS[metric_col]) if has_legend else 0)

    bp = ax.boxplot(data, positions=positions, showfliers=False,
                     patch_artist=True, widths=0.55,
                     boxprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                     whiskerprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                     capprops=dict(linewidth=BOX_LW, color=PAL_BLACK),
                     medianprops=dict(linewidth=MEDIAN_LW, color=PAL_BLACK),
                     zorder=2)
    for patch, g in zip(bp['boxes'], present):
        patch.set_facecolor(_color(g)['face'])
        patch.set_edgecolor(PAL_BLACK)

    rng = np.random.default_rng(0)
    for pos, g, vals in zip(positions, present, data):
        styles  = _point_styles(metric_col, subsets[g], g, grp.shade)
        faces   = np.array([fc for fc, _m in styles])
        markers = np.array([m for _fc, m in styles])
        x = pos + rng.uniform(-0.12, 0.12, size=len(vals))
        for m in dict.fromkeys(markers):
            sel = markers == m
            ax.scatter(x[sel], vals[sel], s=MARKER_SIZE.get(m, 30), marker=m,
                       facecolors=faces[sel], edgecolors=PAL_BLACK, linewidths=0.5, zorder=3)

    # Significance lines above significantly different pairs of groups.
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    y_data_max = max(np.max(v) for v in data)
    top = _draw_sig_brackets(ax, dict(zip(present, positions)), posthoc_df,
                             y_base=y_data_max + 0.06 * span,
                             y_step=0.10 * span, y_text=0.03 * span)
    if top is not None:
        ax.set_ylim(y_lo, max(y_hi, top + 0.08 * span))

    if has_legend:
        _add_significance_legend(ax_info, metric_col, _cell_color(present[0], 0.5))
    _draw_stats(ax_info, stats_lines)

    tick_labels = [f'{_label(g)}\n(n={len(subsets[g])})' for g in present]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(textwrap.fill(metric_label, 30), fontsize=FS_LABEL, labelpad=8)
    _set_title(ax, _titled(metric_label, grp))
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, out_path)


def run_all_comparisons(df: pd.DataFrame, metrics: dict, plot_prefix: str,
                        grp: Grouping, plots_dir: str):
    all_desc, all_omnibus, all_posthoc = [], [], []
    os.makedirs(plots_dir, exist_ok=True)
    for col, label in metrics.items():
        desc_df, omnibus, posthoc_df = compare_groups(df, col, label, grp)
        all_desc.append(desc_df)
        all_omnibus.append(omnibus)
        if len(posthoc_df):
            all_posthoc.append(posthoc_df)
        plot_metric(df, col, label,
                    os.path.join(plots_dir, f'{plot_prefix}_{col}.png'), grp,
                    stats_lines=_continuous_stats_lines(desc_df, omnibus, posthoc_df, grp),
                    posthoc_df=posthoc_df)

    desc_df    = pd.concat(all_desc, ignore_index=True) if all_desc else pd.DataFrame()
    omnibus_df = pd.DataFrame(all_omnibus)
    posthoc_df = pd.concat(all_posthoc, ignore_index=True) if all_posthoc else pd.DataFrame()
    return desc_df, omnibus_df, posthoc_df


# ── Categorical (proportion) comparisons ────────────────────────────────────
# Boolean/tri-state outcomes generated by the characterization pipeline
# (theta modulation, shuffle/stability significance, speed modulation) that
# weren't covered above because they're proportions, not continuous metrics.
# Compared with a chi-square test of independence, then pairwise chi-square
# tests between groups, Holm-Bonferroni corrected (_chi_square_tests).

def _chi_square_tests(table: list, present: list, summary: str):
    """Chi-square test of independence on the groups x categories count
    `table`, then pairwise chi-square tests between every pair of groups
    (Yates-corrected for 2 x 2 tables), Holm-Bonferroni corrected.
    Categories with no counts are dropped. Returns (omnibus_dict,
    posthoc_rows), or (None, []) if the table has < 2 groups or categories."""
    arr = np.array(table, dtype=float)
    arr = arr[:, arr.sum(axis=0) > 0] if arr.size else arr
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None, []

    chi2, p, dof, expected = stats.chi2_contingency(arr)
    note = f'Chi-square (rows treated as independent): {summary}'
    n_low = int((expected < 5).sum())
    if n_low:
        note += (f'; {n_low} of {expected.size} expected counts < 5, '
                 f'chi-square approximation may be unreliable')
    omni = {'test': 'Chi-square test of independence', 'stat_name': 'χ²',
            'statistic': chi2, 'dof': dof, 'p_value': p, 'model_note': note}

    rows = []
    for i, j in combinations(range(len(present)), 2):
        sub_t = arr[[i, j]]
        sub_t = sub_t[:, sub_t.sum(axis=0) > 0]
        if sub_t.shape[1] < 2:
            c2, pp, d = np.nan, np.nan, np.nan
        else:
            c2, pp, d, _e = stats.chi2_contingency(sub_t)
        rows.append({'comparison': f'{present[i]} vs {present[j]}',
                     'test': 'Pairwise chi-square' + (' (Yates)' if sub_t.shape == (2, 2) else ''),
                     'stat_name': 'χ²', 'statistic': c2, 'dof': d, 'p_raw': pp})
    return omni, ccs._finish_posthoc(rows)


def compare_categorical(df: pd.DataFrame, col: str, label: str, grp: Grouping):
    """Descriptive counts, chi-square omnibus test and Holm-corrected pairwise
    chi-square tests for one True/False outcome across the groups of `grp`."""
    sub = df[df[grp.col].isin(grp.order)].copy()
    sub['_y'] = sub[col].apply(_to_tri_bool)
    sub = sub.dropna(subset=['_y'])
    present = [g for g in grp.order if (sub[grp.col] == g).any()]
    table = []
    for g in present:
        yg = sub.loc[sub[grp.col] == g, '_y'].astype(bool)
        table.append([int(yg.sum()), int((~yg).sum())])
    omni, posthoc_rows = (_chi_square_tests(table, present,
                                            ccs.repetition_summary(sub, grp.col))
                          if len(present) >= 2 else (None, []))
    rows = [{'metric': label, grp.col: g, 'n': t + f, 'n_true': t, 'n_false': f,
             'pct_true': 100.0 * t / (t + f)}
            for g, (t, f) in zip(present, table)]
    desc_df = pd.DataFrame(rows)

    omnibus = {'metric': label, 'n_groups': len(present),
               'groups': ', '.join(f'{g} (n={r["n"]})' for g, r in zip(present, rows))}

    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    omnibus.update(omni)
    omnibus['significant'] = bool(pd.notna(omni['p_value']) and omni['p_value'] < ALPHA)
    posthoc_df = pd.DataFrame([{'metric': label, **r} for r in posthoc_rows])
    return desc_df, omnibus, posthoc_df


def _set_percent_ylim(ax, pos_by_group: dict, posthoc_df, bar_top: float):
    """Draw significance lines above 100 %-scale bars and extend the y-axis to
    fit them, keeping the ticks at 0-100."""
    top = _draw_sig_brackets(ax, pos_by_group, posthoc_df, y_base=bar_top + 4,
                             y_step=8, y_text=3)
    ax.set_ylim(0, 100 if top is None else max(100, top + 8))
    ax.set_yticks(np.arange(0, 101, 20))


def plot_categorical(df: pd.DataFrame, col: str, label: str, out_path: str,
                     grp: Grouping, stats_lines=None, posthoc_df=None,
                     ylabel: str = '% of cells'):
    present, pct_true, ns = [], [], []
    for g in grp.order:
        sub = df.loc[df[grp.col] == g, col].apply(_to_tri_bool).dropna()
        if len(sub) == 0:
            continue
        present.append(g)
        ns.append(len(sub))
        pct_true.append(100.0 * (sub == True).sum() / len(sub))  # noqa: E712
    if not present:
        return

    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), stats_lines=stats_lines)
    _draw_stats(ax_info, stats_lines)

    positions = np.arange(len(present))
    faces = [_color(g)['face'] for g in present]
    ax.bar(positions, pct_true, color=faces, edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW,
           width=BAR_WIDTH, zorder=2)

    tick_labels = [f'{_label(g)}\n(n={n})' for g, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=8)
    _set_percent_ylim(ax, dict(zip(present, positions)), posthoc_df, max(pct_true))
    _set_title(ax, _titled(label, grp))
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, out_path)


def compare_speed_direction(df: pd.DataFrame, grp: Grouping):
    """3-way (p-Speed / n-Speed / non-speed) classification from the final
    speed score / speed_cell across the groups of `grp` -- a groups x 3
    chi-square test with Holm-corrected pairwise chi-square tests, since
    'speed direction' isn't a simple True/False outcome."""
    label = 'Speed-direction classification (p-Speed / n-Speed / non-speed, % of place cells)'
    categories = ['p', 'n', 'non']
    sub = df[df[grp.col].isin(grp.order) & df['speed_class'].isin(categories)]
    present = [g for g in grp.order if (sub[grp.col] == g).any()]
    table = [[int(((sub[grp.col] == g) & (sub['speed_class'] == c)).sum())
              for c in categories] for g in present]
    omni, posthoc_rows = (_chi_square_tests(table, present,
                                            ccs.repetition_summary(sub, grp.col))
                          if len(present) >= 2 else (None, []))
    rows = []
    for g, (n_p, n_n, n_non) in zip(present, table):
        n_total = n_p + n_n + n_non
        rows.append({'metric': label, grp.col: g, 'n': n_total,
                      'n_p_speed': n_p, 'n_n_speed': n_n, 'n_non_speed': n_non,
                      'pct_p_speed': 100.0 * n_p / n_total,
                      'pct_n_speed': 100.0 * n_n / n_total,
                      'pct_non_speed': 100.0 * n_non / n_total})
    desc_df = pd.DataFrame(rows)

    omnibus = {'metric': label, 'n_groups': len(present),
               'groups': ', '.join(f'{g} (n={r["n"]})' for g, r in zip(present, rows))}

    if omni is None:
        omnibus.update({'test': 'insufficient data', 'statistic': np.nan,
                         'p_value': np.nan, 'dof': np.nan, 'significant': False})
        return desc_df, omnibus, pd.DataFrame()

    omnibus.update(omni)
    omnibus['significant'] = bool(pd.notna(omni['p_value']) and omni['p_value'] < ALPHA)
    posthoc_df = pd.DataFrame([{'metric': label, **r} for r in posthoc_rows])
    return desc_df, omnibus, posthoc_df


def plot_speed_direction(df: pd.DataFrame, out_path: str, grp: Grouping,
                         stats_lines=None, posthoc_df=None):
    present, pct_p, pct_n, pct_non, ns = [], [], [], [], []
    for g in grp.order:
        cls = df.loc[df[grp.col] == g, 'speed_class'].dropna()
        n_total = len(cls)
        if n_total == 0:
            continue
        n_p = int((cls == 'p').sum())
        n_n = int((cls == 'n').sum())
        present.append(g)
        ns.append(n_total)
        pct_p.append(100.0 * n_p / n_total)
        pct_n.append(100.0 * n_n / n_total)
        pct_non.append(100.0 * (n_total - n_p - n_n) / n_total)
    if not present:
        return

    fig_w = _bar_fig_width(len(present))
    stats_lines = _wrap_stats(stats_lines, fig_w)
    fig, ax, ax_info = _new_fig((fig_w, 5.5), legend_rows=1, stats_lines=stats_lines)

    positions = np.arange(len(present))
    dark   = [_color(g)['dark']  for g in present]
    light  = [_color(g)['light'] for g in present]
    bottoms2 = [p + n for p, n in zip(pct_p, pct_n)]
    bar_kw = dict(edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, width=BAR_WIDTH, zorder=2)
    ax.bar(positions, pct_p, color=dark, label='p-Speed', **bar_kw)
    ax.bar(positions, pct_n, bottom=pct_p, color=light, label='n-Speed', **bar_kw)
    ax.bar(positions, pct_non, bottom=bottoms2, color=SIG_GRAY['face'],
           label='non-speed', **bar_kw)

    tick_labels = [f'{_label(g)}\n(n={n})' for g, n in zip(present, ns)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('% of place cells', fontsize=FS_LABEL, labelpad=8)
    _set_percent_ylim(ax, dict(zip(present, positions)), posthoc_df, 100)
    _set_title(ax, _titled('Speed-direction classification (speed score)', grp))
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc='upper center', fontsize=FS_LEGEND, ncol=3)
    _draw_stats(ax_info, stats_lines)
    fig.tight_layout()
    _save(fig, out_path)


# ── Histograms by group ─────────────────────────────────────────────────────
# Histogram plots of the place-cell characterization metrics, split by group
# (arena type or session type), on the SAME place-cell populations analyzed
# above.

N_BINS   = 15

SPEED_METRICS = {
    'final_speed_score': 'Speed score',
}
HIST_FULL_METRICS = {k: v for k, v in FULL_METRICS.items() if k not in SPEED_METRICS}

# Explicit p-type/n-type speed-cell colors (green/red) -- independent of the
# per-group GROUP_COLORS palette used elsewhere.
P_COLOR = {'face': PAL_GREEN, 'edge': PAL_BLACK}  # p-type (positive) speed cells
N_COLOR = {'face': PAL_RED,   'edge': PAL_BLACK}  # n-type (negative) speed cells

HIST_PANEL_W, HIST_PANEL_H = 3.6, 3.4   # size of one histogram panel (inches)


def _new_panel_fig(n_pan: int, stats_lines=None):
    """Figure with a row of `n_pan` histogram panels (one per group, as in the
    Figure 3 histograms) on a shared x-axis, and a borderless panel underneath
    for the stats text. Returns (fig, axes, ax_info)."""
    info_h = len(stats_lines or []) * STATS_LINE_IN + 0.4
    fig = plt.figure(figsize=(max(HIST_PANEL_W * n_pan, 7), HIST_PANEL_H + info_h))
    gs = fig.add_gridspec(2, n_pan, height_ratios=[HIST_PANEL_H, info_h])
    axes = []
    for i in range(n_pan):
        axes.append(fig.add_subplot(gs[0, i], sharex=axes[0] if axes else None))
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    return fig, axes, ax_info


def _hist_panel(ax, vals, bins, face: str):
    """Solid histogram with thin white gaps between the bars, plus a dashed
    line at the median labelled 'med=...' in the top-right corner."""
    counts, _, _ = ax.hist(vals, bins=bins, color=face, edgecolor=BAR_EDGE,
                           linewidth=BAR_EDGE_LW, zorder=2)
    ax.set_ylim(0, max(counts.max(), 1) * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    med = np.median(vals)
    ax.axvline(med, color=PAL_BLACK, linestyle=(0, (3, 2)), linewidth=AXIS_LW, zorder=4)
    ax.text(0.98, 0.99, f'med={med:.3g}', transform=ax.transAxes,
            ha='right', va='top', fontsize=FS_MED)


def plot_histogram_by_group(df: pd.DataFrame, metric_col: str, metric_label: str,
                            out_path: str, grp: Grouping):
    """One figure per metric: one histogram panel per group present, on
    shared bins so the distributions are directly comparable."""
    values = {}
    for g in grp.order:
        vals = pd.to_numeric(df.loc[df[grp.col] == g, metric_col],
                              errors='coerce').dropna().to_numpy(dtype=float)
        if len(vals):
            values[g] = vals
    if not values:
        return

    all_vals = np.concatenate(list(values.values()))
    bins = np.histogram_bin_edges(all_vals, bins=N_BINS)

    desc_df, omnibus, posthoc_df = compare_groups(df, metric_col, metric_label, grp)
    stats_lines = _wrap_stats(_continuous_stats_lines(desc_df, omnibus, posthoc_df, grp),
                              max(HIST_PANEL_W * len(values), 7))
    fig, axes, ax_info = _new_panel_fig(len(values), stats_lines)

    for ax, (g, vals) in zip(axes, values.items()):
        _hist_panel(ax, vals, bins, _color(g)['face'])
        ax.set_title(f'{_label(g)} (n={len(vals)})', fontsize=FS_LABEL, pad=10)
        ax.set_xlabel(textwrap.fill(metric_label, 22), fontsize=FS_LABEL, labelpad=6)
        _style_axes(ax)
    axes[0].set_ylabel('Count', fontsize=FS_LABEL, labelpad=6)
    _draw_stats(ax_info, stats_lines)
    if grp.title:
        fig.suptitle(grp.title, fontsize=FS_TITLE, fontweight='bold')

    fig.tight_layout()
    _save(fig, out_path)


def _speed_type_masks(sub: pd.DataFrame):
    """Boolean (p_mask, n_mask) numpy arrays for a sub-dataframe, from its
    speed_class column."""
    return (sub['speed_class'] == 'p').to_numpy(), (sub['speed_class'] == 'n').to_numpy()


def plot_speed_histograms_by_group(df: pd.DataFrame, metric_col: str,
                                   metric_label: str, out_path: str, grp: Grouping):
    """Speed-score histogram restricted to speed cells (speed_cell ==
    True), with one panel per group and p-type (green) / n-type (red)
    speed cells stacked within each panel."""
    passed = df['speed_class'].isin(['p', 'n'])

    sub_all = df.loc[passed].copy()
    sub_all[metric_col] = pd.to_numeric(sub_all[metric_col], errors='coerce')
    sub_all = sub_all.dropna(subset=[metric_col])

    present = [g for g in grp.order if (sub_all[grp.col] == g).any()]
    if not present:
        print(f'  No speed cells for {metric_label}; skipping plot.')
        return

    bins = np.histogram_bin_edges(sub_all[metric_col].to_numpy(dtype=float), bins=N_BINS)

    # Stats text: descriptives per group and p/n type, then a
    # Kruskal-Wallis / Mann-Whitney U (compare_groups)
    # comparison of the pooled speed-modulated cells across groups.
    split = {}
    stats_lines = []
    for g in present:
        sub = sub_all.loc[sub_all[grp.col] == g]
        p_mask, n_mask = _speed_type_masks(sub)
        split[g] = (sub.loc[p_mask, metric_col].to_numpy(dtype=float),
                    sub.loc[n_mask, metric_col].to_numpy(dtype=float))
        for name, v in zip(('p-type', 'n-type'), split[g]):
            stats_lines.append(f'{_label(g)} {name} (n={len(v)}): {_desc_str(v)}')
    _, omnibus, posthoc_df = compare_groups(sub_all, metric_col, metric_label, grp)
    stats_lines.append(f'Across {grp.noun} (all speed-modulated cells pooled):')
    stats_lines += _test_lines(omnibus, posthoc_df)

    fig, axes, ax_info = _new_panel_fig(len(present), stats_lines)
    _draw_stats(ax_info, stats_lines)

    for ax, g in zip(axes, present):
        sub = sub_all.loc[sub_all[grp.col] == g]
        p_vals, n_vals = split[g]

        ax.hist([p_vals, n_vals], bins=bins, stacked=True,
                color=[P_COLOR['face'], N_COLOR['face']],
                edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, zorder=2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f'{_label(g)}\n(n={len(sub)}: {len(p_vals)} p-type, {len(n_vals)} n-type)',
                     fontsize=FS_LABEL, pad=10)
        ax.set_xlabel(textwrap.fill(metric_label, 22), fontsize=FS_LABEL, labelpad=6)
        _style_axes(ax)

    axes[0].set_ylabel('Count', fontsize=FS_LABEL, labelpad=6)
    handles = [Patch(facecolor=P_COLOR['face'], edgecolor='none', label='p-type speed cell'),
               Patch(facecolor=N_COLOR['face'], edgecolor='none', label='n-type speed cell')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'{_titled(metric_label, grp)}\n(speed cells only)',
                 fontsize=FS_TITLE, y=1.2)
    fig.tight_layout()
    _save(fig, out_path)


# ── Temporal stability maps (from TemporalStabilityMap_v6_MapPassStability.py) ──

_print_lock = threading.Lock()


def _find_tracking_csv(dirpath: str) -> str | None:
    """Same tracking-file selection as PlaceCell_Main_v21's batch scan."""
    tracking = [f for f in os.listdir(dirpath) if f.lower().endswith(('.csv', '.xlsx'))]
    if pc.COORD_UNITS == 'cm':
        tracking = [f for f in tracking if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in tracking if not f.lower().endswith('_cm.csv')]
    return os.path.join(dirpath, tracking[0]) if len(tracking) == 1 else None


def _visit_stats(flat_bin: np.ndarray, dt_frames: np.ndarray, spike_frame: np.ndarray,
                 n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin visit counts on the moving-only frame sequence.

    A visit starts whenever the bin index differs from the previous frame and
    lasts until the animal leaves that bin. A visit 'fired' if >= 1 spike was
    assigned (nearest frame) to one of its frames. Visits shorter than
    MIN_VISIT_S are dropped.

    Returns (n_visits, n_visits_fired), each of length n_bins.
    """
    if flat_bin.size == 0:
        return np.zeros(n_bins, dtype=np.int64), np.zeros(n_bins, dtype=np.int64)

    starts = np.flatnonzero(np.r_[True, flat_bin[1:] != flat_bin[:-1]])
    ends   = np.r_[starts[1:] - 1, flat_bin.size - 1]
    visit_bin = flat_bin[starts]
    visit_dur = np.add.reduceat(dt_frames, starts)

    spk = np.sort(spike_frame)
    n_spk = (np.searchsorted(spk, ends, side='right')
             - np.searchsorted(spk, starts, side='left'))

    keep = visit_dur >= MIN_VISIT_S
    n_visits = np.bincount(visit_bin[keep], minlength=n_bins)
    n_fired  = np.bincount(visit_bin[keep & (n_spk > 0)], minlength=n_bins)
    return n_visits, n_fired


def _pass_stats(in_field: np.ndarray, dt_frames: np.ndarray,
                spike_frame: np.ndarray) -> tuple[int, int]:
    """Passes through one field on the moving-only frame sequence.

    `in_field` is True on frames inside any bin of the field. A pass is one
    contiguous run of True frames (entry -> exit through any bin). Passes
    shorter than MIN_PASS_S, or touching the first / last frame (entry or exit
    not observed), are dropped. A pass is stable if >= MIN_SPIKES_PER_PASS
    spikes were assigned (nearest frame) to its frames.

    Returns (n_passes, n_passes_with_min_spikes).
    """
    if not in_field.any():
        return 0, 0
    edges  = np.diff(np.r_[0, in_field.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1) - 1
    cum_dt = np.r_[0.0, np.cumsum(dt_frames)]
    dur    = cum_dt[ends + 1] - cum_dt[starts]

    spk = np.sort(spike_frame)
    n_spk = (np.searchsorted(spk, ends, side='right')
             - np.searchsorted(spk, starts, side='left'))

    keep = (dur >= MIN_PASS_S) & (starts > 0) & (ends < in_field.size - 1)
    return int(keep.sum()), int((keep & (n_spk >= MIN_SPIKES_PER_PASS)).sum())


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value at which the cumulative weight (values sorted ascending)
    reaches half the total weight. NaN if the weights sum to 0."""
    total = weights.sum()
    if total <= 0:
        return np.nan
    order = np.argsort(values, kind='stable')
    cum_w = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum_w, 0.5 * total)])


def _field_bins(bin_coords: str, is_circle: bool) -> list[tuple[int, ...]]:
    """bin_coords string -> list of bin tuples: (bx, by) for 2-D, (bt,) for Circle."""
    if is_circle:
        return [(int(b),) for b in bin_coords.split(';')]
    return [tuple(int(v) for v in pair.split('-')) for pair in bin_coords.split(';')]


def _contour_field(ax, field_mask: np.ndarray, bin_cm: float, upsample: int = 8) -> None:
    """Outline a 2-D field mask along bin edges on a cm-extent imshow. The mask
    is padded (so fields touching the grid edge still close) and upsampled (so
    the contour follows bin edges instead of cutting corners between centres)."""
    up = np.kron(np.pad(field_mask.astype(float), 1), np.ones((upsample, upsample)))
    xc = ((np.arange(up.shape[0]) + 0.5) / upsample - 1) * bin_cm
    yc = ((np.arange(up.shape[1]) + 0.5) / upsample - 1) * bin_cm
    ax.contour(xc, yc, up.T, levels=[0.5], colors='black', linewidths=1.2)


def _colorbar(fig, ax, mappable, label: str) -> None:
    """Colorbar matched to the axes height (thin linear-track maps included)."""
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.1)
    fig.colorbar(mappable, cax=cax, label=label)


def _mark_peak(ax, x: float, y: float) -> None:
    """White star on the raw-ratemap peak bin (x, y in the axes' data units)."""
    ax.plot(x, y, marker='*', color='white', markeredgecolor='black', markersize=14,
            linestyle='none')


def _plot_rate(ax, fig, rate: np.ndarray, valid: np.ndarray, extent: list, title: str):
    im = ax.imshow(np.ma.masked_where(~valid, rate).T, origin='lower', extent=extent,
                   cmap='jet', interpolation='nearest')
    _colorbar(fig, ax, im, 'firing rate (Hz)')
    ax.set_title(title)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')


def _plot_stability_2d(ax, fig, pct: np.ndarray, field_mask: np.ndarray,
                       valid: np.ndarray, extent: list):
    gray = np.ma.masked_where(~(valid & ~field_mask), np.ones_like(pct))
    ax.imshow(gray.T, origin='lower', extent=extent,
              cmap=ListedColormap([NON_FIELD_COLOR]), interpolation='nearest')
    im = ax.imshow(np.ma.masked_where(~field_mask, pct).T, origin='lower', extent=extent,
                   cmap=STABILITY_CMAP, vmin=0, vmax=100, interpolation='nearest')
    _colorbar(fig, ax, im, '% visits with >= 1 spike')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')


def _plot_stability_ring(ax, fig, pct: np.ndarray, field_mask: np.ndarray,
                         valid: np.ndarray, ctx_circ: dict):
    """Angular analogue of _plot_stability_2d, drawn as a ring like
    PlaceCell_Main_v21._plot_ratemap_ring."""
    n_theta, bw = ctx_circ['n_bins_theta'], ctx_circ['bin_width_rad']
    r_min, r_max = ctx_circ['r_min'], ctx_circ['r_max']
    theta_mesh, r_mesh = np.meshgrid(np.arange(n_theta + 1) * bw, [r_min, r_max])

    gray = np.ma.masked_where(~(valid & ~field_mask), np.ones(n_theta)).reshape(1, n_theta)
    ax.pcolormesh(theta_mesh, r_mesh, gray, cmap=ListedColormap([NON_FIELD_COLOR]),
                  shading='flat')
    field = np.ma.masked_where(~field_mask, pct).reshape(1, n_theta)
    mesh = ax.pcolormesh(theta_mesh, r_mesh, field, cmap=STABILITY_CMAP,
                         vmin=0, vmax=100, shading='flat')
    fig.colorbar(mesh, ax=ax, label='% visits with >= 1 spike', fraction=0.046, pad=0.1)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_max * 1.1)
    ax.set_rticks([])


def _ts_run_cell(args) -> tuple[dict | None, list[dict], list[dict]]:
    """Temporal stability map for one place cell.

    Returns (cell_row, field_rows, bin_rows); cell_row is None if skipped.
    """
    idx, total, dirpath, unit = args
    session  = os.path.relpath(dirpath, TS_PLACE_CELL_FOLDER)
    ntt_path = os.path.join(dirpath, unit)
    with _print_lock:
        print(f'[{idx}/{total}]  {session}  |  {unit}')

    csv_path = _find_tracking_csv(dirpath)
    if csv_path is None:
        with _print_lock:
            print(f'  [SKIP] no single tracking file found in {dirpath}')
        return None, [], []

    metrics, ctx = pc.compute_metrics(csv_path, ntt_path, pc.arena_width_cm, pc.target_bin_cm)
    if not ctx or not ctx['valid_mask'].any():
        with _print_lock:
            print(f'  [SKIP] no valid occupancy for {unit}')
        return None, [], []

    bin_cm      = pc.target_bin_cm
    n_bx, n_by  = ctx['n_bins_x'], ctx['n_bins_y']
    is_circle   = pc._is_circle_arena(dirpath)
    arena_type  = 'Circle' if is_circle else '2D'
    spike_frame = ctx['spike_frame']

    # ── Place fields + the bin sequence / grid they are defined on ───────────
    # Same detection as PlaceCell_Main_v21._run_job (Circle track: angular bins).
    if is_circle:
        ctx_circ  = pc._build_ratemap_circular(ctx, bin_cm)
        fields, _ = pc.detect_place_fields_threshold_circular(ctx_circ, bin_cm)
        flat_bin  = ctx_circ['beh_bt']
        grid      = (ctx_circ['n_bins_theta'],)
        valid     = ctx_circ['valid_mask']
        occ       = ctx_circ['occ_map']
    else:
        ctx_circ  = None
        fields, _ = pc.detect_place_fields_threshold_2d(metrics, ctx, bin_cm)
        flat_bin  = ctx['beh_bx'] * n_by + ctx['beh_by']
        grid      = (n_bx, n_by)
        valid     = ctx['valid_mask']
        occ       = ctx['occ_map']

    if not fields:
        with _print_lock:
            print(f'  [SKIP] no place field detected for {unit}')
        return None, [], []

    field_mask = np.zeros(grid, dtype=bool)
    field_id   = np.zeros(grid, dtype=np.int64)       # field_number per bin, 0 = no field
    for f in fields:
        for bn in _field_bins(f['bin_coords'], is_circle):
            field_mask[bn] = True
            field_id[bn]   = f['field_number']

    # ── Visits per bin + fraction with >= 1 spike ────────────────────────────
    n_visits, n_fired = _visit_stats(flat_bin, ctx['dt_frames'], spike_frame, int(np.prod(grid)))
    n_visits = n_visits.reshape(grid)
    n_fired  = n_fired.reshape(grid)
    spk_bins = np.bincount(flat_bin[spike_frame], minlength=int(np.prod(grid))).reshape(grid)

    pct = np.full(grid, np.nan)
    has_visits = field_mask & (n_visits > 0)
    pct[has_visits] = 100.0 * n_fired[has_visits] / n_visits[has_visits]

    # ── Temporal stability at the raw-ratemap peak bin ───────────────────────
    # Peak over all valid bins of the RAW (unsmoothed) ratemap, first bin on ties.
    # Circle track: the raw angular map (spikes / occupancy per angular bin, as in
    # PlaceCell_Main_v21._ratemap_circular). The peak need not lie in a field.
    if is_circle:
        raw_rate = np.zeros(grid)
        np.divide(spk_bins, occ, out=raw_rate, where=valid)
    else:
        raw_rate = ctx['fr_raw']
    peak_bn = tuple(int(b) for b in
                    np.unravel_index(np.argmax(np.where(valid, raw_rate, -np.inf)), grid))
    peak_pct = (100.0 * n_fired[peak_bn] / n_visits[peak_bn]) if n_visits[peak_bn] else np.nan

    # ── Tables ────────────────────────────────────────────────────────────────
    bin_rows = []
    for bn in zip(*np.nonzero(field_mask)):
        row = {'session': session, 'unit': unit, 'arena_type': arena_type,
               'field_number': int(field_id[bn])}
        if is_circle:
            row['bin_theta'] = int(bn[0])
        else:
            row['bin_x'], row['bin_y'] = int(bn[0]), int(bn[1])
        row.update({'n_visits': int(n_visits[bn]), 'n_visits_with_spike': int(n_fired[bn]),
                    'pct_visits_with_spike': round(float(pct[bn]), 2) if has_visits[bn] else np.nan,
                    'n_spikes': int(spk_bins[bn]), 'occupancy_s': round(float(occ[bn]), 3)})
        bin_rows.append(row)

    occ_map_total = float(occ[valid].sum())                # dwell (s) over the whole ratemap

    def _summary(mask: np.ndarray) -> dict:
        p = pct[mask & has_visits]
        w = occ[mask & has_visits]                         # dwell time (s) per bin
        v, fired = int(n_visits[mask].sum()), int(n_fired[mask].sum())
        n_pass, n_pass_ok = _pass_stats(mask.ravel()[flat_bin], ctx['dt_frames'], spike_frame)
        return {'n_field_bins': int(mask.sum()),
                'mean_pct_across_bins':   round(float(np.mean(p)), 2)   if p.size else np.nan,
                'median_pct_across_bins': round(float(np.median(p)), 2) if p.size else np.nan,
                'occ_wmean_pct_across_bins':
                    round(float(np.sum(p * w) / w.sum()), 2) if w.sum() > 0 else np.nan,
                'occ_wmedian_pct_across_bins': round(_weighted_median(p, w), 2),
                'occ_wmean_pct_mapnorm':
                    round(float(np.sum(p * w) / occ_map_total), 2) if occ_map_total > 0 else np.nan,
                'occupancy_s_field_bins': round(float(occ[mask].sum()), 3),
                'n_visits_total': v, 'n_visits_with_spike_total': fired,
                'pooled_pct_visits_with_spike': round(100.0 * fired / v, 2) if v else np.nan,
                'n_passes': n_pass, 'n_passes_with_min_spikes': n_pass_ok,
                'lap_stability_score': round(n_pass_ok / n_pass, 3) if n_pass else np.nan}

    field_rows = [{'session': session, 'unit': unit, 'arena_type': arena_type,
                   'field_number': int(fn), **_summary(field_id == fn)}
                  for fn in sorted(np.unique(field_id[field_mask]))]
    cell_row = {'session': session, 'unit': unit, 'arena_type': arena_type,
                'n_fields': len(field_rows), 'n_spikes': metrics.get('n_spikes'),
                'peak_fr': metrics.get('peak_fr'), 'mean_fr': metrics.get('mean_fr'),
                'sir': metrics.get('sir'), **_summary(field_mask)}
    field_laps = [fr['lap_stability_score'] for fr in field_rows
                  if not np.isnan(fr['lap_stability_score'])]
    cell_row['mean_lap_stability_across_fields'] = (round(float(np.mean(field_laps)), 3)
                                                    if field_laps else np.nan)

    # Whole-map occupancy-weighted stability: every visited valid bin, in or out of a field.
    map_bins = valid & (n_visits > 0)
    p_map = 100.0 * n_fired[map_bins] / n_visits[map_bins]
    w_map = occ[map_bins]
    cell_row.update({'n_map_bins_visited': int(map_bins.sum()),
                     'occupancy_s_map': round(occ_map_total, 3),
                     'occ_wmean_pct_all_bins':
                         round(float(np.sum(p_map * w_map) / w_map.sum()), 2)
                         if w_map.sum() > 0 else np.nan})
    if is_circle:
        cell_row['peak_bin_theta'] = peak_bn[0]
    else:
        cell_row['peak_bin_x'], cell_row['peak_bin_y'] = peak_bn
    cell_row.update({'peak_raw_fr': round(float(raw_rate[peak_bn]), 3),
                     'peak_bin_in_field': bool(field_mask[peak_bn]),
                     'peak_bin_n_visits': int(n_visits[peak_bn]),
                     'peak_bin_n_visits_with_spike': int(n_fired[peak_bn]),
                     'pct_at_peak_bin': round(float(peak_pct), 2)})

    # ── Figure ────────────────────────────────────────────────────────────────
    # Wide arenas (linear track) stack the four panels vertically; others sit in one row.
    aspect = n_by / n_bx
    if aspect < 0.5:
        n_rows, n_cols = 4, 1
        fig = Figure(figsize=(14, 4 * (14 * aspect + 1.5) + 1))
    else:
        n_rows, n_cols = 1, 4
        fig = Figure(figsize=(26, 6.5))
    FigureCanvasAgg(fig)
    extent = [0, n_bx * bin_cm, 0, n_by * bin_cm]

    ax_traj = fig.add_subplot(n_rows, n_cols, 1)
    ax_traj.plot(ctx['x_cm_all'], ctx['y_cm_all'], color='0.6', linewidth=0.5, label='trajectory')
    ax_traj.plot(ctx['x_cm'][spike_frame], ctx['y_cm'][spike_frame], '.', color='red',
                 markersize=3, label='spikes')
    ax_traj.set_xlim(extent[0], extent[1])
    ax_traj.set_ylim(extent[2], extent[3])
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_xlabel('x (cm)')
    ax_traj.set_ylabel('y (cm)')
    ax_traj.set_title('Trajectory + spikes')
    ax_traj.legend(loc='upper right', ncol=2, fontsize=8)
    make_axes_locatable(ax_traj).append_axes('right', size='3%', pad=0.1).axis('off')  # align with maps

    ax_raw = fig.add_subplot(n_rows, n_cols, 2)
    _plot_rate(ax_raw, fig, ctx['fr_raw'], ctx['valid_mask'], extent, 'Raw ratemap')
    ax_smooth = fig.add_subplot(n_rows, n_cols, 3)
    _plot_rate(ax_smooth, fig, ctx['fr_smooth'], ctx['valid_mask'], extent,
               f'Gaussian-smoothed (sigma = {pc.GAUSSIAN_SIGMA_CM:g} cm)')

    stab_title = 'Temporal stability (field bins)'
    if is_circle:
        assert ctx_circ is not None
        ax_stab = fig.add_subplot(n_rows, n_cols, 4, projection='polar')
        _plot_stability_ring(ax_stab, fig, pct, field_mask, valid, ctx_circ)
        _mark_peak(ax_stab, (peak_bn[0] + 0.5) * ctx_circ['bin_width_rad'],
                   0.5 * (ctx_circ['r_min'] + ctx_circ['r_max']))
        stab_title += ', angular'
    else:
        ax_stab = fig.add_subplot(n_rows, n_cols, 4)
        _plot_stability_2d(ax_stab, fig, pct, field_mask, valid, extent)
        for fn in np.unique(field_id[field_mask]):
            for ax in (ax_raw, ax_smooth, ax_stab):
                _contour_field(ax, field_id == fn, bin_cm)
        for ax in (ax_raw, ax_stab):
            _mark_peak(ax, (peak_bn[0] + 0.5) * bin_cm, (peak_bn[1] + 0.5) * bin_cm)
    ax_stab.set_title(f'{stab_title}\n* = raw-ratemap peak bin: {cell_row["pct_at_peak_bin"]}% '
                      f'of {cell_row["peak_bin_n_visits"]} visits')

    # Lap stability label on each field (2-D: bin centroid; Circle: circular-mean angle).
    for fr in field_rows:
        bins = np.argwhere(field_id == fr['field_number'])
        if is_circle:
            assert ctx_circ is not None
            ang = (bins[:, 0] + 0.5) * ctx_circ['bin_width_rad']
            xy = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi),
                  0.5 * (ctx_circ['r_min'] + ctx_circ['r_max']))
        else:
            xy = tuple((bins.mean(axis=0) + 0.5) * bin_cm)
        ax_stab.annotate(f"F{fr['field_number']}: {fr['n_passes_with_min_spikes']}/{fr['n_passes']}"
                         f" = {fr['lap_stability_score']}", xy=xy, ha='center', va='center',
                         fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8))

    ntt_name = os.path.splitext(unit)[0]
    lap_line = '  |  '.join(f"F{fr['field_number']}: {fr['n_passes_with_min_spikes']}/"
                            f"{fr['n_passes']} passes = {fr['lap_stability_score']}"
                            for fr in field_rows)
    fig.suptitle(f"{session}  |  {ntt_name}\n"
                 f"{cell_row['n_fields']} field(s), {cell_row['n_field_bins']} bins  |  "
                 f"mean across bins = {cell_row['mean_pct_across_bins']}%, "
                 f"median = {cell_row['median_pct_across_bins']}%, "
                 f"occ-weighted mean = {cell_row['occ_wmean_pct_across_bins']}%, "
                 f"occ-weighted median = {cell_row['occ_wmedian_pct_across_bins']}%, "
                 f"pooled = {cell_row['pooled_pct_visits_with_spike']}% of "
                 f"{cell_row['n_visits_total']} visits with >= 1 spike\n"
                 f"Lap stability (passes with >= {MIN_SPIKES_PER_PASS} spikes): {lap_line}  ||  "
                 f"all fields pooled = {cell_row['lap_stability_score']}, "
                 f"mean across fields = {cell_row['mean_lap_stability_across_fields']}",
                 fontsize=10)
    fig.tight_layout()

    save_dir = os.path.join(dirpath, TEMPORAL_STABILITY_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{ntt_name}_temporal_stability.png')
    fig.savefig(save_path, dpi=150)
    with _print_lock:
        print(f'  [SAVED] {save_path}')

    return cell_row, field_rows, bin_rows


def run_ts_maps() -> pd.DataFrame:
    """Temporal stability map for every .ntt under TS_PLACE_CELL_FOLDER; writes
    TS_OUTPUT_EXCEL and returns its PerCell table."""
    jobs = []
    for dirpath, _, filenames in os.walk(TS_PLACE_CELL_FOLDER):
        jobs += [(dirpath, unit) for unit in sorted(filenames) if unit.lower().endswith('.ntt')]
    job_args = [(i, len(jobs), *job) for i, job in enumerate(jobs, start=1)]
    print(f'Found {len(jobs)} place cell(s) in {TS_PLACE_CELL_FOLDER}\n')

    results = []
    # The maps keep the full axes box (the stats-plot style above hides the top
    # and right spines).
    with matplotlib.rc_context({'axes.spines.top': True, 'axes.spines.right': True}), \
            concurrent.futures.ThreadPoolExecutor(max_workers=TS_MAX_WORKERS) as executor:
        futures = {executor.submit(_ts_run_cell, a): a for a in job_args}
        for future in concurrent.futures.as_completed(futures):
            a = futures[future]
            try:
                results.append((a[0], future.result()))
            except Exception as e:
                print(f'  ERROR in {a[2]} | {a[3]}: {e}')
    results.sort(key=lambda r: r[0])

    cell_rows  = [r[0] for _, r in results if r[0] is not None]
    field_rows = [f for _, r in results for f in r[1]]
    bin_rows   = [b for _, r in results for b in r[2]]

    cell_df = pd.DataFrame(cell_rows)
    os.makedirs(os.path.dirname(TS_OUTPUT_EXCEL), exist_ok=True)
    with pd.ExcelWriter(TS_OUTPUT_EXCEL, engine='openpyxl') as writer:
        cell_df.to_excel(writer,                  sheet_name='PerCell',  index=False)
        pd.DataFrame(field_rows).to_excel(writer, sheet_name='PerField', index=False)
        pd.DataFrame(bin_rows).to_excel(writer,   sheet_name='PerBin',   index=False)

    print(f'\nDone. {len(cell_rows)}/{len(jobs)} temporal stability map(s) plotted.')
    print(f'Temporal stability results saved to {TS_OUTPUT_EXCEL}')
    return cell_df


def build_temporal_stability_df() -> pd.DataFrame:
    """PerCell temporal stability table (rebuilt with run_ts_maps if
    RUN_TS_MAPS, else read from TS_OUTPUT_EXCEL), tagged with arena type,
    session type, animal and cell_id like the other dataframes."""
    if RUN_TS_MAPS:
        cell_df = run_ts_maps()
    else:
        print(f'Loading PerCell sheet of {TS_OUTPUT_EXCEL} ...')
        cell_df = pd.read_excel(TS_OUTPUT_EXCEL, sheet_name='PerCell')
    missing = [c for c in TEMPORAL_STABILITY_METRICS if c not in cell_df.columns]
    if missing:
        raise KeyError(f'PerCell table has no column(s) {missing}; '
                       f'rebuild it with RUN_TS_MAPS = True.')
    return _tag_groups(cell_df, 'place cell(s) (temporal stability)')


def _map_path(session: str, unit: str) -> str:
    """Temporal stability figure written by _ts_run_cell for this cell."""
    return os.path.join(TS_PLACE_CELL_FOLDER, session, TEMPORAL_STABILITY_SUBDIR,
                        f'{os.path.splitext(unit)[0]}_temporal_stability.png')


def _pick_examples(d: pd.DataFrame, n: int, metric: str) -> list[tuple[str, int, pd.Series]]:
    """(category, rank, row) for the n best, n closest-to-mean and n worst rows
    of d by `metric`. A cell is picked once: Best first, then Worst, then
    Mean from the rest (small groups give fewer maps)."""
    d = d.dropna(subset=[metric]).sort_values(
        [metric, 'session', 'unit'], ascending=[False, True, True], kind='stable')
    best  = d.head(n)
    worst = d.drop(best.index).tail(n).iloc[::-1]
    rest  = d.drop(best.index.union(worst.index))
    mean  = rest.loc[(rest[metric] - d[metric].mean()).abs()
                     .sort_values(kind='stable').index[:n]]
    return [(cat, rank, row) for cat, part in (('Best', best), ('Mean', mean), ('Worst', worst))
            for rank, (_, row) in enumerate(part.iterrows(), start=1)]


def _save_example_maps(grp: Grouping, df: pd.DataFrame, out_dir: str, prefix: str,
                       metric: str, map_path) -> None:
    """Copy the best / mean / worst N_EXAMPLE_MAPS per-cell figures
    (map_path(session, unit): temporal stability map or multi-peak ratemap) of
    each group of `grp`, ranked by `metric`, into `out_dir` as
    '<prefix>_<group>_<cat><rank>_...png', each with a banner naming the
    animal, session, unit and its `metric` value."""
    os.makedirs(out_dir, exist_ok=True)
    unit_str = '%' if 'pct' in metric else ''
    for g in grp.order:
        d = df[df[grp.col] == g]
        group_mean = d[metric].mean()
        for cat, rank, row in _pick_examples(d, N_EXAMPLE_MAPS, metric):
            src = map_path(row['session'], row['unit'])
            if not os.path.isfile(src):
                print(f'  [SKIP example] map not found: {src}')
                continue
            img = imread(src)
            h_in, w_in = img.shape[0] / 150, img.shape[1] / 150   # maps are saved at dpi=150
            banner_in = 1.0
            fig = Figure(figsize=(w_in, h_in + banner_in))
            FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, h_in / (h_in + banner_in)))
            ax.imshow(img)
            ax.axis('off')
            ntt_name = os.path.splitext(row['unit'])[0]
            fig.text(0.5, 1 - 0.5 * banner_in / (h_in + banner_in),
                     f'{grp.title}  |  {_label(g)}  |  {cat} {rank} of {N_EXAMPLE_MAPS}  |  '
                     f'Animal: {row["animal"]}  |  Session: {row["session"]}  |  Unit: {ntt_name}\n'
                     f'{metric} = {row[metric]:.2f}{unit_str}  '
                     f'({_label(g)} mean = {group_mean:.2f}{unit_str})',
                     ha='center', va='center', fontsize=14, fontweight='bold')
            session_tag = '_'.join(p for p in str(row['session']).replace('/', '\\').split('\\') if p)
            fig.savefig(os.path.join(out_dir, f'{prefix}_{g}_{cat}{rank}_{session_tag}_{ntt_name}.png'),
                        dpi=150)
    print(f'Example maps ({prefix}) saved to {out_dir}')


# ── Multi-peak: ratemap reconstruction (copied from PlaceCell_Main, CPU only) ─

def _mp_bin_visit_counts(flat_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Number of visits per bin: a visit starts whenever the animal enters a bin
    (first frame, or bin index differs from the previous frame)."""
    flat_idx = np.asarray(flat_idx)
    if flat_idx.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    starts = np.ones(flat_idx.size, dtype=bool)
    starts[1:] = flat_idx[1:] != flat_idx[:-1]
    return np.bincount(flat_idx[starts], minlength=n_bins)


def _mp_valid_bin_mask(occ: np.ndarray, visits: np.ndarray) -> np.ndarray:
    """Valid bins under the enabled criteria (MP_USE_MIN_OCC_S / MP_USE_MIN_VISITS)."""
    mask = occ > 0
    if MP_USE_MIN_OCC_S:
        mask &= occ >= MP_MIN_OCC_S
    if MP_USE_MIN_VISITS:
        mask &= visits >= MP_MIN_VISITS
    return mask


def _mp_gaussian_kernel(sigma_bins: float) -> np.ndarray:
    """2D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return kernel


def _mp_gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    """1D Gaussian kernel, sigma given in bins, truncated at 3 sigma."""
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return kernel


def _mp_gaussian_smooth(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Gaussian smoothing restricted to valid bins, edge-corrected by dividing
    the convolved rates by the convolved mask."""
    kernel = _mp_gaussian_kernel(MP_GAUSSIAN_SIGMA_CM / bin_cm)
    fr_in   = np.where(valid_mask, fr_map, 0.0)
    mask_in = valid_mask.astype(np.float64)
    smoothed_fr      = convolve(fr_in,   kernel, mode='constant', cval=0.0)
    smoothed_weights = convolve(mask_in, kernel, mode='constant', cval=0.0)

    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


def _mp_gaussian_smooth_circular(fr_map: np.ndarray, valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """Gaussian smoothing on a 1D angular rate map, padded by wrapping (the
    track is a closed loop)."""
    kernel = _mp_gaussian_kernel_1d(MP_GAUSSIAN_SIGMA_CM / bin_cm)
    kr = kernel.shape[0] // 2
    n_theta = fr_map.shape[0]

    fr_pad   = np.pad(np.where(valid_mask, fr_map, 0.0), (kr, kr), mode='wrap')
    mask_pad = np.pad(valid_mask.astype(np.float64),     (kr, kr), mode='wrap')
    smoothed_fr      = convolve(fr_pad,   kernel, mode='constant', cval=0.0)[kr:kr + n_theta]
    smoothed_weights = convolve(mask_pad, kernel, mode='constant', cval=0.0)[kr:kr + n_theta]

    smoothed = np.zeros_like(smoothed_fr)
    valid_weights = smoothed_weights > 0
    smoothed[valid_weights] = smoothed_fr[valid_weights] / smoothed_weights[valid_weights]
    smoothed[~valid_mask] = 0.0
    return smoothed


def _mp_load_tracking(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, and pixel→cm-convert one tracking file (same as
    PlaceCell_Main._load_tracking, full session). Returns (x_cm, y_cm, t_us)."""
    data = (pd.read_excel(csv_path) if csv_path.lower().endswith('.xlsx')
            else pd.read_csv(csv_path))

    if MP_COORD_UNITS == 'cm':
        t = np.asarray(data.iloc[:, 0], dtype=float)
        x = np.asarray(data.iloc[:, 3], dtype=float)
        y = np.asarray(data.iloc[:, 4], dtype=float)
    else:
        x = np.asarray(data['x'],    dtype=float)
        y = np.asarray(data['y'],    dtype=float)
        t = np.asarray(data['time'], dtype=float)

    mask = ~np.isin(x, [1, -1])
    x, y, t = x[mask], y[mask], t[mask]

    if len(t) >= 2:
        if MP_COORD_UNITS == 'cm':
            units_per_cm = 1.0
        else:
            x_span = np.percentile(x, 99.5) - np.percentile(x, 0.5)
            y_span = np.percentile(y, 99.5) - np.percentile(y, 0.5)
            units_per_cm = max(x_span, y_span) / MP_ARENA_WIDTH_CM
        jump_thresh_per_us = MP_POS_JUMP_THRESH_CMS * units_per_cm * 1e-6

        dt_step   = np.diff(t)
        dxy_step  = np.hypot(np.diff(x), np.diff(y))
        valid_step = dt_step > 0
        speed_step = np.zeros_like(dxy_step)
        speed_step[valid_step] = dxy_step[valid_step] / dt_step[valid_step]

        speed    = np.append(speed_step, speed_step[-1])
        valid_dt = np.append(valid_step, valid_step[-1])

        keep = np.where(valid_dt & (speed < jump_thresh_per_us))[0]
        x, y, t = x[keep], y[keep], t[keep]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    if len(t) == 0:
        return x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)

    if MP_COORD_UNITS == 'cm':
        x_cm = x - x.min()
        y_cm = y - y.min()
    else:
        px_per_cm = max(x.max() - x.min(), y.max() - y.min()) / MP_ARENA_WIDTH_CM
        x_cm = (x - x.min()) / px_per_cm
        y_cm = (y - y.min()) / px_per_cm

    return x_cm, y_cm, t


def _mp_smooth_tracking_position(x_cm: np.ndarray, y_cm: np.ndarray,
                              t_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iterative jump removal, linear interpolation over artifact frames, then
    Gaussian smoothing (same as PlaceCell_Main._smooth_tracking_position)."""
    n = len(x_cm)
    if n < 2:
        return x_cm.copy(), y_cm.copy()

    bad = np.zeros(n, dtype=bool)
    for _ in range(n):
        good_idx = np.where(~bad)[0]
        if len(good_idx) < 2:
            break
        dt_good = np.diff(t_us[good_idx]) * 1e-6
        with np.errstate(invalid='ignore', divide='ignore'):
            step_speed = np.hypot(np.diff(x_cm[good_idx]), np.diff(y_cm[good_idx])) / dt_good
        step_speed[dt_good <= 0] = 0.0

        newly_bad = step_speed > MP_POS_JUMP_THRESH_CMS
        if not newly_bad.any():
            break
        bad[good_idx[1:][newly_bad]] = True

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0 or len(good_idx) == n:
        x_clean, y_clean = x_cm.copy(), y_cm.copy()
    else:
        x_clean = np.interp(t_us, t_us[good_idx], x_cm[good_idx])
        y_clean = np.interp(t_us, t_us[good_idx], y_cm[good_idx])

    x_smooth = gaussian_filter1d(x_clean, sigma=MP_POS_SMOOTH_SIGMA_SMP, mode='nearest')
    y_smooth = gaussian_filter1d(y_clean, sigma=MP_POS_SMOOTH_SIGMA_SMP, mode='nearest')
    return x_smooth, y_smooth


def _mp_instantaneous_speed(x_cm: np.ndarray, y_cm: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Per-frame running speed (cm/s); the last frame gets the second-last frame's speed."""
    dt_s = np.diff(t_us) * 1e-6
    with np.errstate(invalid='ignore', divide='ignore'):
        speed = np.hypot(np.diff(x_cm), np.diff(y_cm)) / dt_s
    speed[dt_s <= 0] = np.nan
    return np.append(speed, speed[-1])


def _mp_load_session_track(csv_path: str) -> dict | None:
    """Tracking part of PlaceCell_Main.compute_metrics (steps 1-3), shared by
    every unit of a session: cleaned + smoothed position, per-frame occupancy
    time, immobility mask and position bins."""
    x_cm, y_cm, t = _mp_load_tracking(csv_path)
    if len(t) == 0:
        return None
    x_cm, y_cm = _mp_smooth_tracking_position(x_cm, y_cm, t)

    dt_frames     = np.empty(len(t), dtype=np.float64)
    dt_frames[0]  = 1.0 / MP_FPS
    dt_frames[1:] = np.minimum(np.diff(t) * 1e-6, 2.0 / MP_FPS)

    if len(t) >= 2:
        moving = ~(_mp_instantaneous_speed(x_cm, y_cm, t) < MP_IMMOBILITY_SPEED_CMS)   # NaN speed kept
    else:
        moving = np.ones(len(t), dtype=bool)
    if not moving.any():
        return None

    n_bins_x = max(1, int(np.ceil(x_cm.max() / MP_BIN_CM)))
    n_bins_y = max(1, int(np.ceil(y_cm.max() / MP_BIN_CM)))
    beh_bx = np.clip((x_cm / MP_BIN_CM).astype(int), 0, n_bins_x - 1)
    beh_by = np.clip((y_cm / MP_BIN_CM).astype(int), 0, n_bins_y - 1)

    return dict(x_cm=x_cm, y_cm=y_cm, t=t, dt_frames=dt_frames, moving=moving,
                beh_bx=beh_bx, beh_by=beh_by, n_bins_x=n_bins_x, n_bins_y=n_bins_y)


def _mp_build_unit_ratemap(track: dict, ntt_path: str) -> dict:
    """Spike part of PlaceCell_Main.compute_metrics (steps 4-7): 50 ms
    nearest-frame assignment, immobility filter, occupancy / spike maps, valid
    bins, raw and Gaussian-smoothed ratemaps (moving frames only)."""
    t, moving = track['t'], track['moving']
    n_bins_x, n_bins_y = track['n_bins_x'], track['n_bins_y']

    spike_data = np.memmap(ntt_path, dtype=MP_NTT_DTYPE, mode='r', offset=16 * 1024)
    spike_data = spike_data[spike_data['cell_number'] != 0]  # drop unsorted/discarded cluster 0
    spike_ts   = np.sort(spike_data['timestamp'].astype(np.float64))

    idx   = np.searchsorted(t, spike_ts, side='left')
    idx_l = np.clip(idx - 1, 0, len(t) - 1)
    idx_r = np.clip(idx,     0, len(t) - 1)
    dist_l  = np.abs(spike_ts - t[idx_l])
    dist_r  = np.abs(spike_ts - t[idx_r])
    nearest = np.where(dist_l <= dist_r, idx_l, idx_r)
    valid_spike  = np.minimum(dist_l, dist_r) <= MP_MAX_GAP_US
    moving_spike = valid_spike & moving[nearest]

    new_frame_idx = np.cumsum(moving) - 1
    spike_frame   = new_frame_idx[nearest[moving_spike]]
    beh_bx    = track['beh_bx'][moving]
    beh_by    = track['beh_by'][moving]
    dt_frames = track['dt_frames'][moving]

    occ_map   = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    spike_map = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    np.add.at(occ_map,   (beh_bx, beh_by), dt_frames)
    np.add.at(spike_map, (beh_bx[spike_frame], beh_by[spike_frame]), 1.0)

    visit_map  = _mp_bin_visit_counts(beh_bx * n_bins_y + beh_by,
                                   n_bins_x * n_bins_y).reshape(n_bins_x, n_bins_y)
    valid_mask = _mp_valid_bin_mask(occ_map, visit_map)

    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _mp_gaussian_smooth(fr_raw, valid_mask, MP_BIN_CM)

    return dict(x_cm=track['x_cm'][moving], y_cm=track['y_cm'][moving],
                spike_frame=spike_frame, dt_frames=dt_frames,
                occ_map=occ_map, spike_map=spike_map, valid_mask=valid_mask,
                fr_raw=fr_raw, fr_smooth=fr_smooth,
                n_bins_x=n_bins_x, n_bins_y=n_bins_y)


def _mp_fit_ring_centre(x_cm: np.ndarray, y_cm: np.ndarray) -> tuple:
    """Algebraic (Kasa) circle fit to the tracked positions -> (cx, cy)."""
    A = np.column_stack([x_cm, y_cm, np.ones_like(x_cm)])
    b = x_cm ** 2 + y_cm ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[0] / 2.0, sol[1] / 2.0


def _mp_build_ratemap_circular(ctx: dict) -> dict:
    """1-D angular ratemap of the Circle track (same as
    PlaceCell_Main._build_ratemap_circular)."""
    x_cm, y_cm  = ctx['x_cm'], ctx['y_cm']
    spike_frame = ctx['spike_frame']
    dt_frames   = ctx['dt_frames']

    cx, cy = _mp_fit_ring_centre(x_cm, y_cm)
    r_cm        = np.hypot(x_cm - cx, y_cm - cy)
    theta_0_2pi = np.mod(np.arctan2(y_cm - cy, x_cm - cx), 2.0 * np.pi)

    r_min = float(np.percentile(r_cm, MP_RADIAL_RANGE_CLIP_PCTILE))
    r_max = float(np.percentile(r_cm, 100.0 - MP_RADIAL_RANGE_CLIP_PCTILE))

    bin_width_rad = MP_BIN_CM / max(float(np.median(r_cm)), 1e-6)
    n_bins_theta  = max(1, int(round(2.0 * np.pi / bin_width_rad)))
    bin_width_rad = 2.0 * np.pi / n_bins_theta

    beh_bt = np.clip((theta_0_2pi / bin_width_rad).astype(int), 0, n_bins_theta - 1)

    occ_map   = np.zeros(n_bins_theta, dtype=np.float64)
    spike_map = np.zeros(n_bins_theta, dtype=np.float64)
    np.add.at(occ_map,   beh_bt, dt_frames)
    np.add.at(spike_map, beh_bt[spike_frame], 1.0)

    valid_mask = _mp_valid_bin_mask(occ_map, _mp_bin_visit_counts(beh_bt, n_bins_theta))
    fr_raw = np.zeros_like(occ_map)
    fr_raw[valid_mask] = spike_map[valid_mask] / occ_map[valid_mask]
    fr_smooth = _mp_gaussian_smooth_circular(fr_raw, valid_mask, MP_BIN_CM)

    return dict(occ_map=occ_map, spike_map=spike_map,
                valid_mask=valid_mask, fr_raw=fr_raw, fr_smooth=fr_smooth,
                n_bins_theta=n_bins_theta, bin_width_rad=bin_width_rad,
                r_min=r_min, r_max=r_max, track_width_cm=max(r_max - r_min, 0.0))


# ── Multi-peak: KDE input ratemap (KDE_RATEMAP_TYPE) ────────────────────────
# 2-D grid: copied from PlaceCell_Main (_mp_occ_weighted_smooth and the
# Yartsev / Skaggs adaptive-disc ratemaps, CPU only).

def _mp_occ_weighted_smooth(spike_map: np.ndarray, occ_map: np.ndarray,
                         valid_mask: np.ndarray, bin_cm: float) -> np.ndarray:
    """smooth(spikes) / smooth(occupancy), both restricted to valid bins."""
    kernel = _mp_gaussian_kernel(MP_GAUSSIAN_SIGMA_CM / bin_cm)
    spk_smooth = convolve(np.where(valid_mask, spike_map, 0.0), kernel, mode='constant', cval=0.0)
    occ_smooth = convolve(np.where(valid_mask, occ_map,   0.0), kernel, mode='constant', cval=0.0)
    rate = np.zeros_like(spk_smooth)
    ok = valid_mask & (occ_smooth > 0)
    rate[ok] = spk_smooth[ok] / occ_smooth[ok]
    return rate


@functools.lru_cache(maxsize=16)
def _mp_adaptive_disc_geometry(n_bins_x: int, n_bins_y: int,
                            max_rsq: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Grid geometry for the adaptive-disc ratemaps, cached per grid shape:
    (pad, gather_idx, n_enclosed, disc_r_bins) -- see PlaceCell_Main."""
    pad = int(np.floor(np.sqrt(max_rsq - 1)))
    ax  = np.arange(-pad, pad + 1)
    dx, dy = np.meshgrid(ax, ax, indexing='ij')
    dx, dy = dx.ravel(), dy.ravel()
    d2     = dx ** 2 + dy ** 2
    keep   = d2 <= max_rsq - 1
    order  = np.argsort(d2[keep], kind='stable')
    dx, dy, d2 = dx[keep][order], dy[keep][order], d2[keep][order]

    bx, by = np.meshgrid(np.arange(n_bins_x), np.arange(n_bins_y), indexing='ij')
    px = bx.ravel()[None, :] + pad + dx[:, None]
    py = by.ravel()[None, :] + pad + dy[:, None]
    gather_idx = px * (n_bins_y + 2 * pad) + py

    n_enclosed  = np.searchsorted(d2, np.arange(max_rsq), side='right')
    disc_r_bins = np.sqrt(d2[n_enclosed - 1])
    return pad, gather_idx, n_enclosed, disc_r_bins


def _mp_adaptive_disc_sums(spike_map: np.ndarray, occ_map: np.ndarray,
                        valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spikes and occupancy inside every candidate disc of every bin:
    (spk_disc, occ_disc, disc_r_bins), discs indexed by r^2 = 0..ADAPTIVE_MAX_RSQ-1."""
    n_bins_x, n_bins_y = occ_map.shape
    pad, gather_idx, n_enclosed, disc_r_bins = _mp_adaptive_disc_geometry(
        n_bins_x, n_bins_y, MP_ADAPTIVE_MAX_RSQ)
    spk_pad = np.pad(np.where(valid_mask, spike_map, 0.0), pad).ravel()
    occ_pad = np.pad(np.where(valid_mask, occ_map,   0.0), pad).ravel()
    spk_disc = np.cumsum(spk_pad[gather_idx], axis=0)[n_enclosed - 1]
    occ_disc = np.cumsum(occ_pad[gather_idx], axis=0)[n_enclosed - 1]
    return spk_disc, occ_disc, disc_r_bins


def _mp_adaptive_rate_at(k: np.ndarray, spk_disc: np.ndarray, occ_disc: np.ndarray,
                      valid_mask: np.ndarray) -> np.ndarray:
    """Rate from the chosen disc index `k` of every (flattened) bin; bins whose
    final disc holds < MP_ADAPTIVE_MIN_OCC_S of occupancy get rate 0. Works for
    the 2-D grid and the 1-D angular map alike."""
    cols    = np.arange(valid_mask.size)
    spk_fin = spk_disc[k, cols].reshape(valid_mask.shape)
    occ_fin = occ_disc[k, cols].reshape(valid_mask.shape)
    rate = np.zeros(valid_mask.shape, dtype=np.float64)
    ok = valid_mask & (occ_fin >= MP_ADAPTIVE_MIN_OCC_S)
    rate[ok] = spk_fin[ok] / occ_fin[ok]
    return rate


def _mp_first_satisfied(satisfied: np.ndarray) -> np.ndarray:
    """Index of the first disc meeting the criterion, else the largest disc."""
    return np.where(satisfied.any(axis=0), satisfied.argmax(axis=0), satisfied.shape[0] - 1)


def _mp_adaptive_ratemap(spike_map: np.ndarray, occ_map: np.ndarray, valid_mask: np.ndarray,
                      spk_disc: np.ndarray, occ_disc: np.ndarray,
                      disc_r_bins: np.ndarray, bin_cm: float) -> np.ndarray:
    """Yartsev or Skaggs adaptive ratemap (KDE_RATEMAP_TYPE) from disc sums."""
    if not valid_mask.any() or spike_map[valid_mask].sum() <= 0:
        return np.zeros(valid_mask.shape, dtype=np.float64)
    if KDE_RATEMAP_TYPE == 'Yartsev_adaptive_binned':
        satisfied = occ_disc >= MP_YARTSEV_DISC_OCC_S
    else:   # Skaggs & McNaughton 1998, r = disc radius + half a bin, in cm
        r_cm = (disc_r_bins + 0.5) * bin_cm
        satisfied = r_cm[:, None] * occ_disc * np.sqrt(spk_disc) >= MP_SKAGGS_ALPHA
    return _mp_adaptive_rate_at(_mp_first_satisfied(satisfied), spk_disc, occ_disc, valid_mask)


def _mp_kde_input_map_2d(ctx: dict) -> np.ndarray:
    """KDE_RATEMAP_TYPE ratemap on the 2-D grid."""
    if KDE_RATEMAP_TYPE == 'raw':
        return ctx['fr_raw']
    if KDE_RATEMAP_TYPE == 'gaussian_smooth':
        return ctx['fr_smooth']
    spike_map, occ_map, valid_mask = ctx['spike_map'], ctx['occ_map'], ctx['valid_mask']
    if KDE_RATEMAP_TYPE == 'occ_weighted_smooth':
        return _mp_occ_weighted_smooth(spike_map, occ_map, valid_mask, MP_BIN_CM)
    spk_disc, occ_disc, disc_r_bins = _mp_adaptive_disc_sums(spike_map, occ_map, valid_mask)
    return _mp_adaptive_ratemap(spike_map, occ_map, valid_mask,
                             spk_disc, occ_disc, disc_r_bins, MP_BIN_CM)


# 1-D angular map (Circle track): the same maps, with the disc replaced by an
# interval of +-r bins along the track and smoothing / intervals wrapping
# around the theta=0/2*pi seam. The interval grows in whole bins up to the
# 2-D maximum radius, floor(sqrt(MP_ADAPTIVE_MAX_RSQ - 1)) bins.

def _mp_kde_input_map_circular(cc: dict) -> np.ndarray:
    """KDE_RATEMAP_TYPE ratemap on the 1-D angular map of the Circle track."""
    if KDE_RATEMAP_TYPE == 'raw':
        return cc['fr_raw']
    if KDE_RATEMAP_TYPE == 'gaussian_smooth':
        return cc['fr_smooth']
    spike_map, occ_map, valid_mask = cc['spike_map'], cc['occ_map'], cc['valid_mask']
    spk_in = np.where(valid_mask, spike_map, 0.0)
    occ_in = np.where(valid_mask, occ_map,   0.0)
    n = len(occ_map)

    if KDE_RATEMAP_TYPE == 'occ_weighted_smooth':
        kernel = _mp_gaussian_kernel_1d(MP_GAUSSIAN_SIGMA_CM / MP_BIN_CM)
        kr = kernel.shape[0] // 2
        spk_s = convolve(np.pad(spk_in, (kr, kr), mode='wrap'), kernel, mode='constant')[kr:kr + n]
        occ_s = convolve(np.pad(occ_in, (kr, kr), mode='wrap'), kernel, mode='constant')[kr:kr + n]
        rate = np.zeros(n)
        ok = valid_mask & (occ_s > 0)
        rate[ok] = spk_s[ok] / occ_s[ok]
        return rate

    r_max = min(int(np.floor(np.sqrt(MP_ADAPTIVE_MAX_RSQ - 1))), (n - 1) // 2)
    spk_disc = np.empty((r_max + 1, n))
    occ_disc = np.empty((r_max + 1, n))
    spk_disc[0], occ_disc[0] = spk_in, occ_in
    for r in range(1, r_max + 1):
        spk_disc[r] = spk_disc[r - 1] + np.roll(spk_in, r) + np.roll(spk_in, -r)
        occ_disc[r] = occ_disc[r - 1] + np.roll(occ_in, r) + np.roll(occ_in, -r)
    return _mp_adaptive_ratemap(spike_map, occ_map, valid_mask, spk_disc, occ_disc,
                             np.arange(r_max + 1, dtype=float), MP_BIN_CM)


# ── Multi-peak: detection (KDE + watershed) ─────────────────────────────────

def _watershed(values: np.ndarray, adj: list, seeds: list) -> np.ndarray:
    """Marker-based watershed (priority flood, highest value first) over the
    field nodes 0..len(values)-1. `adj[i]` lists node i's neighbours (indices
    >= len(values) are outside the field and ignored); seeds[k] gets label k+1.
    Each node takes the label of the basin that reaches it first while
    descending the surface, so basins meet at the troughs."""
    n_field = len(values)
    labels = np.zeros(n_field, dtype=int)
    heap = []
    for k, s in enumerate(seeds, start=1):
        labels[s] = k
        heapq.heappush(heap, (-values[s], s))
    while heap:
        _, i = heapq.heappop(heap)
        for j in adj[i]:
            if j < n_field and labels[j] == 0:
                labels[j] = labels[i]
                heapq.heappush(heap, (-values[j], j))
    return labels


def _merge_levels(rate: np.ndarray, adj: list) -> np.ndarray:
    """Union-find over the field bins in descending KDE order (0-D topological
    persistence). For every bin, the KDE level at which the region it tops
    merges into a region with a higher top, i.e. the trough that has to be
    crossed to reach a higher peak. Non-peak bins merge at their own height;
    the field's highest bin never merges (inf)."""
    n_field = len(rate)
    parent  = np.full(n_field, -1)
    top     = np.arange(n_field)                 # highest bin of each root's region
    merge   = np.full(n_field, np.inf)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.argsort(-rate, kind='stable'):
        parent[i] = i
        for j in adj[i]:
            if j >= n_field or parent[j] < 0:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # Higher top survives; exact ties go to the lower bin index, the
            # same order the local maxima are ranked in (_kde_multipeak).
            ti, tj = top[ri], top[rj]
            hi, lo = (ri, rj) if (rate[ti], -ti) >= (rate[tj], -tj) else (rj, ri)
            merge[top[lo]] = rate[i]
            parent[lo] = hi
    return merge


def _no_kde_result(n_field: int, w: np.ndarray, note: str) -> dict:
    """Single-peak result for a field the KDE can't be fitted on (too few
    non-zero-rate bins): the whole field is one sub-field, peak at the
    highest raw-rate bin."""
    peak = int(np.argmax(w))
    return dict(rate=np.full(n_field, np.nan), labels=np.ones(n_field, dtype=int),
                local_max=[peak], counted=[peak], trough_rel={peak: np.nan},
                ref_hz=np.nan, peak_fallback=False, saddles={},
                kde_axes='', kde_factor=np.nan, kde_neff=np.nan,
                kde_bw_bins=np.array([np.nan]), kde_n_points=int((w > 0).sum()),
                kde_note=note)


def _kde_multipeak(pts_ext: np.ndarray, n_field: int, adj: list, w: np.ndarray) -> dict:
    """KDE peak search + trough cut for one field.

    pts_ext : (d, n_ext) bin-centre coordinates (in bins) -- the field's n_field
              bins first, then a 1-bin ring around the field.
    adj     : for each field bin, the indices (into pts_ext) of its neighbours.
    w       : firing rate of each field bin in the KDE_RATEMAP_TYPE map (KDE weights).
    """
    d   = pts_ext.shape[0]
    pos = w > 0
    if pos.sum() < MIN_KDE_POINTS:
        return _no_kde_result(n_field, w, f'{int(pos.sum())} non-zero-rate bin(s) < '
                                          f'MIN_KDE_POINTS: no KDE, single peak')

    # Collinear non-zero-rate bins (a 1-bin-wide strip, or too few points)
    # make the 2-D covariance singular: fit a 1-D KDE along the field's long
    # (first principal) axis instead.
    X   = pts_ext[:, :n_field][:, pos]
    cov = np.atleast_2d(np.cov(X, aweights=w[pos]))
    evals, evecs = np.linalg.eigh(cov)
    if evals.max() <= 0:
        return _no_kde_result(n_field, w, 'all non-zero-rate bins at one location: no KDE, single peak')
    if d > 1 and evals.min() <= 1e-6 * evals.max():
        axis      = evecs[:, -1]
        proj      = axis[None, :]
        kde_axes  = f'long axis ({axis[0]:+.2f}, {axis[1]:+.2f})'
    else:
        proj      = np.eye(d)
        kde_axes  = 'theta' if d == 1 else 'x,y'
    kde = gaussian_kde(proj @ X, bw_method=KDE_BW_METHOD, weights=w[pos])

    # Σ r_i K(x - x_i) with K integrating to 1 over one bin -> rate in Hz.
    rate_ext = kde(proj @ pts_ext) * float(w.sum())
    rate     = rate_ext[:n_field]

    # Local maxima inside the field (>= every neighbour, ring bins included,
    # so a field-edge bin only counts if the KDE really falls off outside it).
    # Adjacent maxima can only be exact ties -- keep one of them.
    cand = [i for i in range(n_field) if all(rate_ext[i] >= rate_ext[j] for j in adj[i])]
    cand.sort(key=lambda i: -rate[i])
    local_max = []
    for i in cand:
        if not any(i in adj[p] for p in local_max):
            local_max.append(i)

    # The KDE's maximum can sit just outside a non-convex field; then fall back
    # to the field's highest KDE bin as its single peak.
    peak_fallback = not local_max
    if peak_fallback:
        local_max = [int(np.argmax(rate))]

    # A peak counts if it is tall enough AND a real trough separates it from
    # every higher peak (trough_rel = deepest-needed crossing / own height).
    merge      = _merge_levels(rate, adj)
    ref_hz     = float(rate[local_max[0]])
    trough_rel = {i: (np.nan if i == local_max[0] else float(merge[i] / rate[i]))
                  for i in local_max}
    counted = [local_max[0]] + [
        i for i in local_max[1:]
        if rate[i] >= PEAK_REL_HEIGHT_FRAC * ref_hz
        and merge[i] < (1.0 - MIN_TROUGH_DIP_FRAC) * rate[i]]
    labels  = _watershed(rate, adj, counted)

    # Saddle between two touching sub-fields = highest crossing point between
    # them (max over touching bin pairs of the lower of the two KDE values).
    saddles: dict[tuple[int, int], float] = {}
    for i in range(n_field):
        for j in adj[i]:
            if j < n_field and labels[i] and labels[j] and labels[i] < labels[j]:
                key = (int(labels[i]), int(labels[j]))
                saddles[key] = max(saddles.get(key, -np.inf), float(min(rate[i], rate[j])))

    return dict(rate=rate, labels=labels, local_max=local_max, counted=counted,
                trough_rel=trough_rel, ref_hz=ref_hz, peak_fallback=peak_fallback,
                saddles=saddles, kde_axes=kde_axes, kde_factor=float(kde.factor),
                kde_neff=float(kde.neff),
                kde_bw_bins=np.sqrt(np.diag(np.atleast_2d(kde.covariance))),
                kde_n_points=int(pos.sum()), kde_note='')


def _analyse_field_2d(nodes: list, kde_map: np.ndarray) -> dict:
    """KDE multi-peak analysis of one 2-D grid field (nodes = (bx, by) bins),
    bins weighted by the KDE input ratemap `kde_map`."""
    index = {n: k for k, n in enumerate(nodes)}
    ext_nodes = list(nodes)
    adj = []
    for bx, by in nodes:
        nbrs = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                nb = (bx + ddx, by + ddy)
                k = index.get(nb)
                if k is None:                     # ring bin just outside the field
                    k = len(ext_nodes)
                    index[nb] = k
                    ext_nodes.append(nb)
                nbrs.append(k)
        adj.append(nbrs)

    pts_ext = np.asarray(ext_nodes, dtype=float).T + 0.5
    idx = np.asarray(nodes)
    return _kde_multipeak(pts_ext, len(nodes), adj, kde_map[idx[:, 0], idx[:, 1]])


def _order_ring_run(bts: list) -> np.ndarray:
    """Angular bins of one Circle field in track order: a field straddling the
    theta=0/2*pi seam is rotated to start after its gap, so positions along
    the returned array are contiguous."""
    bts = np.sort(np.asarray(bts, dtype=int))
    gaps = np.where(np.diff(bts) > 1)[0]
    if len(gaps) == 0:
        return bts
    g = gaps[0]
    return np.concatenate([bts[g + 1:], bts[:g + 1]])


def _analyse_field_circular(ordered: np.ndarray, kde_map: np.ndarray) -> dict:
    """KDE multi-peak analysis of one Circle field along the track (1-D,
    positions 0..L-1 in track order, plus one ring bin at each end), bins
    weighted by the KDE input angular ratemap `kde_map`."""
    L = len(ordered)
    pos_ext = np.concatenate([np.arange(L), [-1, L]]).astype(float) + 0.5
    adj = [[i - 1 if i > 0 else L, i + 1 if i < L - 1 else L + 1] for i in range(L)]
    return _kde_multipeak(pos_ext[None, :], L, adj, kde_map[ordered])


def _join(vals, fmt='{:.4f}') -> str:
    return ';'.join(fmt.format(v) for v in vals)


def _analyse_cell(ctx: dict, cc: dict | None, fdf: pd.DataFrame,
                  arena_type: str, session: str, unit: str) -> tuple[list, list, list]:
    """Runs the multi-peak analysis on every field of one cell.
    Returns (field_rows, sub_rows, plot_fields)."""
    field_rows, sub_rows, plot_fields = [], [], []
    circle = arena_type == 'Circle'

    for _, f in fdf.sort_values('field_number').iterrows():
        fn = int(f['field_number'])
        coords = str(f['bin_coords'])

        if circle:
            n_theta = cc['n_bins_theta']
            bw_rad  = cc['bin_width_rad']
            bts = [int(b) for b in coords.split(';')]
            if min(bts) < 0 or max(bts) >= n_theta:
                raise ValueError(f'field {fn}: bin_coords outside the {n_theta}-bin angular map')
            nodes = _order_ring_run(bts)
            res   = _analyse_field_circular(nodes, cc['kde_in'])
            pos   = np.arange(len(nodes), dtype=float)[:, None]   # position along the track (bins)
            sm, raw = cc['fr_smooth'][nodes], cc['fr_raw'][nodes]
            bin_str = [str(int(b)) for b in nodes]
            theta_c = (nodes + 0.5) * bw_rad
        else:
            nodes = sorted(tuple(int(v) for v in p.split('-')) for p in coords.split(';'))
            idx = np.asarray(nodes)
            if (idx.min() < 0 or idx[:, 0].max() >= ctx['n_bins_x']
                    or idx[:, 1].max() >= ctx['n_bins_y']):
                raise ValueError(f'field {fn}: bin_coords outside the '
                                 f'{ctx["n_bins_x"]}x{ctx["n_bins_y"]} ratemap grid')
            res = _analyse_field_2d(nodes, ctx['kde_in'])
            pos = idx.astype(float)
            sm, raw = ctx['fr_smooth'][idx[:, 0], idx[:, 1]], ctx['fr_raw'][idx[:, 0], idx[:, 1]]
            bin_str = [f'{bx}-{by}' for bx, by in nodes]

        rate, labels, counted = res['rate'], res['labels'], res['counted']
        ref_hz   = res['ref_hz']
        n_peaks  = len(counted)
        multi    = n_peaks >= 2
        n_field  = len(nodes)

        # Sanity check: the rebuilt smoothed map must reproduce PlaceCell_Main's
        # field peak (peak_fr_hz, rounded to 4 dp in the Excel).
        recomputed_peak = float(sm.max())
        excel_peak      = float(f['peak_fr_hz'])
        ratemap_check   = bool(np.isclose(recomputed_peak, excel_peak, rtol=1e-3, atol=1e-3))

        peak_hz = [float(rate[c]) for c in counted]
        troughs, separations = {}, {}
        for (a, b), s in sorted(res['saddles'].items()):
            troughs[(a, b)] = s / min(peak_hz[a - 1], peak_hz[b - 1])
            # Distance between the two peaks either side of this trough (arc
            # length along the track on the Circle; ~MP_BIN_CM per bin).
            d_bins = float(np.hypot(*(pos[counted[a - 1]] - pos[counted[b - 1]]))) \
                if pos.shape[1] == 2 else float(abs(pos[counted[a - 1], 0] - pos[counted[b - 1], 0]))
            separations[(a, b)] = d_bins * MP_BIN_CM

        field_rows.append({
            'session': session, 'unit': unit, 'arena_type': arena_type,
            'field_number': fn,
            'multi_peak': multi,
            'n_peaks': n_peaks,
            'n_kde_local_maxima': len(res['local_max']),
            'peak_fallback': res['peak_fallback'],
            'kde_peak_hz': round(ref_hz, 4),
            'peak_threshold_hz': round(PEAK_REL_HEIGHT_FRAC * ref_hz, 4),
            'peak_kde_hz': _join(peak_hz),
            'peak_bins': ';'.join(bin_str[c] for c in counted),
            'local_max_rel_heights': _join([rate[i] / ref_hz for i in res['local_max']], '{:.3f}'),
            'local_max_trough_rel': _join([res['trough_rel'][i] for i in res['local_max']], '{:.3f}'),
            'shallowest_trough_rel': round(max(troughs.values()), 4) if troughs else np.nan,
            'troughs_rel': ';'.join(f'{a}-{b}:{v:.3f}' for (a, b), v in troughs.items()),
            'lowest_counted_peak_rel': round(min(peak_hz) / ref_hz, 4) if multi else np.nan,
            'peak_separation_cm': (round(float(np.mean(list(separations.values()))), 2)
                                   if separations else np.nan),
            'peak_separations_cm': ';'.join(f'{a}-{b}:{v:.1f}' for (a, b), v in separations.items()),
            'kde_n_points': res['kde_n_points'],
            'kde_n_eff': round(res['kde_neff'], 2),
            'kde_factor': round(res['kde_factor'], 4),
            'kde_axes': res['kde_axes'],
            'kde_bw_cm': _join(res['kde_bw_bins'] * MP_BIN_CM, '{:.2f}'),
            'kde_note': res['kde_note'],
            'n_bins': n_field,
            'area_cm2': f.get('area_cm2'),
            'pct_of_occupied_area': f.get('pct_of_occupied_area'),
            'peak_fr_hz': excel_peak,
            'recomputed_peak_fr_hz': round(recomputed_peak, 4),
            'ratemap_check': ratemap_check,
            'bin_coords': ';'.join(sorted(bin_str, key=lambda s: tuple(int(v) for v in s.split('-')))),
        })

        plot_fields.append(dict(field_number=fn, nodes=nodes, rate=rate, labels=labels,
                                counted=counted, local_max=res['local_max'],
                                ref_hz=ref_hz, multi=multi))

        if not multi:
            continue

        # ── Sub-field rows (multi-peak fields only) ───────────────────────────
        for k in range(1, n_peaks + 1):
            members = np.where(labels == k)[0]
            seed    = counted[k - 1]
            loc     = members[int(np.argmax(sm[members]))]
            sm_m    = sm[members]
            touching = [s for (a, b), s in res['saddles'].items() if k in (a, b)]
            max_saddle = max(touching) if touching else np.nan

            row = {
                'session': session, 'unit': unit, 'arena_type': arena_type,
                'field_number': fn, 'sub_field_number': k, 'sub_field_id': f'{fn}.{k}',
                'kde_peak_hz': round(peak_hz[k - 1], 4),
                'kde_peak_rel': round(peak_hz[k - 1] / ref_hz, 4),
                'peak_fr_hz': round(float(sm[loc]), 4),
                'peak_fr_hz_raw': round(float(raw[loc]), 4),
                'field_raw_max_fr_hz': round(float(raw[members].max()), 4),
                'n_bins': len(members),
                'pct_of_field_area': round(100.0 * len(members) / n_field, 2),
                'max_saddle_hz': round(max_saddle, 4),
                'max_saddle_rel': round(max_saddle / peak_hz[k - 1], 4),
            }
            if circle:
                com = np.arctan2(np.sum(sm_m * np.sin(theta_c[members])),
                                 np.sum(sm_m * np.cos(theta_c[members]))) % (2.0 * np.pi)
                row.update({
                    'kde_peak_bin_theta': int(nodes[seed]),
                    'kde_peak_theta_deg': round(float(np.degrees(theta_c[seed])), 2),
                    'peak_bin_theta': int(nodes[loc]),
                    'peak_theta_deg': round(float(np.degrees(theta_c[loc])), 2),
                    'com_theta_deg': round(float(np.degrees(com)), 2),
                    'arc_length_cm': round(len(members) * MP_BIN_CM, 2),
                    'area_cm2': round(len(members) * MP_BIN_CM * cc['track_width_cm'], 2),
                    'bin_coords': ';'.join(bin_str[m] for m in members),
                })
            else:
                bxs, bys = idx[members, 0], idx[members, 1]
                com_x = float(np.sum(sm_m * bxs) / np.sum(sm_m))
                com_y = float(np.sum(sm_m * bys) / np.sum(sm_m))
                row.update({
                    'kde_peak_bin_x': nodes[seed][0], 'kde_peak_bin_y': nodes[seed][1],
                    'peak_bin_x': nodes[loc][0], 'peak_bin_y': nodes[loc][1],
                    'com_bin_x': round(com_x, 3), 'com_bin_y': round(com_y, 3),
                    'com_cm_x': round(com_x * MP_BIN_CM, 2),
                    'com_cm_y': round(com_y * MP_BIN_CM, 2),
                    'area_cm2': round(len(members) * MP_BIN_CM ** 2, 2),
                    'bin_coords': ';'.join(bin_str[m] for m in members),
                })
            sub_rows.append(row)

    return field_rows, sub_rows, plot_fields


# ── Multi-peak: per-cell figures ────────────────────────────────────────────

_MP_GRAY_CMAP = ListedColormap(['#bdbdbd'])     # background (visited, non-field) bins
_MP_TXT_FX    = [path_effects.withStroke(linewidth=2, foreground='black')]


def _mask_outline_segments(mask: np.ndarray) -> list:
    """Bin-edge segments outlining the True region of a 2-D mask (bin (i, j)
    spans [i, i+1] x [j, j+1])."""
    p = np.pad(mask, 1)
    segs = []
    for i, j in zip(*np.nonzero(p[1:, 1:-1] != p[:-1, 1:-1])):     # vertical edges at x = i
        segs.append([(i, j), (i, j + 1)])
    for i, j in zip(*np.nonzero(p[1:-1, 1:] != p[1:-1, :-1])):     # horizontal edges at y = j
        segs.append([(i, j), (i + 1, j)])
    return segs


def _label_cut_segments(sub_id: np.ndarray) -> list:
    """Bin-edge segments between neighbouring bins with different non-zero labels."""
    segs = []
    a, b = sub_id[:-1, :], sub_id[1:, :]
    for i, j in zip(*np.nonzero((a > 0) & (b > 0) & (a != b))):
        segs.append([(i + 1, j), (i + 1, j + 1)])
    a, b = sub_id[:, :-1], sub_id[:, 1:]
    for i, j in zip(*np.nonzero((a > 0) & (b > 0) & (a != b))):
        segs.append([(i, j + 1), (i + 1, j + 1)])
    return segs


def _field_summary_title(plot_fields: list) -> str:
    return '  '.join(f"F{pf['field_number']}: {len(pf['counted'])} peak"
                     f"{'s' if len(pf['counted']) != 1 else ''}" for pf in plot_fields)


def _mp_plot_cell_2d(ctx: dict, plot_fields: list, title: str, save_path: str) -> None:
    nx, ny     = ctx['n_bins_x'], ctx['n_bins_y']
    valid_mask = ctx['valid_mask']

    in_field = np.zeros((nx, ny), dtype=bool)
    sub_id   = np.zeros((nx, ny), dtype=int)      # unique id per (field, sub-field)
    kde_map  = np.full((nx, ny), np.nan)
    uid = 0
    for pf in plot_fields:
        for (bx, by), lab, val in zip(pf['nodes'], pf['labels'], pf['rate']):
            in_field[bx, by] = True
            kde_map[bx, by]  = val
            sub_id[bx, by]   = uid + max(int(lab), 1)
        uid += max(int(pf['labels'].max()), 1)

    bg       = np.where(valid_mask & ~in_field, 1.0, np.nan)
    outline  = _mask_outline_segments(in_field)
    cuts     = _label_cut_segments(sub_id)
    extent   = (0, nx, 0, ny)

    fig = Figure(figsize=(19, 5.8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    panels = ((axes[0], ctx['fr_smooth'], f'Gaussian-smoothed ratemap (σ = {MP_GAUSSIAN_SIGMA_CM} cm)'),
              (axes[1], ctx['kde_in'],    f'KDE input map: {KDE_RATEMAP_TYPE}'),
              (axes[2], kde_map,          f'Field KDE ({KDE_BW_METHOD}), peaks ≥ {PEAK_REL_HEIGHT_FRAC:.0%} of field max'))
    for ax, data, panel_title in panels:
        ax.imshow(bg.T, origin='lower', extent=extent, cmap=_MP_GRAY_CMAP,
                  vmin=0, vmax=1, interpolation='nearest')
        shown = np.ma.masked_where(~in_field, np.nan_to_num(data))
        finite = data[in_field & np.isfinite(data)]
        vmax  = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        im = ax.imshow(shown.T, origin='lower', extent=extent, cmap='jet',
                       vmin=0, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Firing rate (Hz)', fraction=0.046, pad=0.04)

        ax.add_collection(LineCollection(outline, colors='black', linewidths=2.0))
        if cuts:
            ax.add_collection(LineCollection(cuts, colors='white', linewidths=2.5))

        for pf in plot_fields:
            for i in pf['local_max']:
                if i not in pf['counted']:
                    bx, by = pf['nodes'][i]
                    ax.plot(bx + 0.5, by + 0.5, 'x', color='black', ms=6, mew=1.5)
            for k, i in enumerate(pf['counted'], start=1):
                bx, by = pf['nodes'][i]
                ax.plot(bx + 0.5, by + 0.5, '*', ms=13, mfc='white', mec='black', mew=1.0)
                label = f"{pf['field_number']}.{k}" if pf['multi'] else f"F{pf['field_number']}"
                ax.text(bx + 0.5, by + 1.6, label, color='white', fontsize=8,
                        ha='center', va='bottom', path_effects=_MP_TXT_FX)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect('equal')
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel('x bin')
        ax.set_ylabel('y bin')

    fig.suptitle(f'{title}\n{_field_summary_title(plot_fields)}   '
                 f'(star = counted peak, × = peak < {PEAK_REL_HEIGHT_FRAC:.0%}, white line = trough cut)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def _nan_at_wrap(deg: np.ndarray, *ys: np.ndarray):
    """Insert NaN where the angle wraps 360 -> 0, so a seam-straddling field
    is drawn as two pieces instead of a line across the whole axis."""
    breaks = np.where(np.diff(deg) < 0)[0] + 1
    return (np.insert(deg.astype(float), breaks, np.nan),
            *(np.insert(y.astype(float), breaks, np.nan) for y in ys))


def _mp_plot_cell_circular(cc: dict, plot_fields: list, title: str, save_path: str) -> None:
    n          = cc['n_bins_theta']
    bw         = cc['bin_width_rad']
    r_min      = cc['r_min']
    r_max      = cc['r_max']
    valid_mask = cc['valid_mask']
    fr_smooth  = cc['fr_smooth']
    r_mid      = 0.5 * (r_min + r_max)

    in_field = np.zeros(n, dtype=bool)
    for pf in plot_fields:
        in_field[pf['nodes']] = True

    fig = Figure(figsize=(14, 5.8))
    FigureCanvasAgg(fig)
    ax_ring = fig.add_subplot(1, 2, 1, projection='polar')
    ax_lin  = fig.add_subplot(1, 2, 2)

    # ── Ring heatmap ──────────────────────────────────────────────────────────
    theta_mesh, r_mesh = np.meshgrid(np.arange(n + 1) * bw, np.array([r_min, r_max]))
    bg = np.ma.masked_invalid(np.where(valid_mask & ~in_field, 1.0, np.nan).reshape(1, n))
    ax_ring.pcolormesh(theta_mesh, r_mesh, bg, cmap=_MP_GRAY_CMAP, vmin=0, vmax=1, shading='flat')
    vmax = float(fr_smooth[in_field].max()) if in_field.any() else 1.0
    mesh = ax_ring.pcolormesh(theta_mesh, r_mesh,
                              np.ma.masked_where(~in_field, fr_smooth).reshape(1, n),
                              cmap='jet', vmin=0, vmax=vmax, shading='flat')
    fig.colorbar(mesh, ax=ax_ring, label='Firing rate (Hz)', fraction=0.046, pad=0.08)

    for pf in plot_fields:
        L  = len(pf['nodes'])
        t0 = pf['nodes'][0] * bw
        t1 = t0 + L * bw
        th = np.linspace(t0, t1, max(20, 4 * L))
        for r in (r_min, r_max):
            ax_ring.plot(th, np.full_like(th, r), color='black', lw=2.0)
        for t in (t0, t1):
            ax_ring.plot([t, t], [r_min, r_max], color='black', lw=2.0)
        for p in range(1, L):
            if pf['labels'][p] != pf['labels'][p - 1]:
                ax_ring.plot([t0 + p * bw] * 2, [r_min, r_max], color='white', lw=2.5)
        for i in pf['local_max']:
            if i not in pf['counted']:
                ax_ring.plot(t0 + (i + 0.5) * bw, r_mid, 'x', color='black', ms=6, mew=1.5)
        for k, i in enumerate(pf['counted'], start=1):
            ax_ring.plot(t0 + (i + 0.5) * bw, r_mid, '*', ms=13, mfc='white', mec='black')
            label = f"{pf['field_number']}.{k}" if pf['multi'] else f"F{pf['field_number']}"
            ax_ring.text(t0 + (i + 0.5) * bw, r_max * 1.12, label, fontsize=8,
                         ha='center', va='center')

    ax_ring.set_theta_zero_location('E')
    ax_ring.set_theta_direction(1)
    ax_ring.set_ylim(0, r_max * 1.2)
    ax_ring.set_yticklabels([])
    ax_ring.set_title(f'Gaussian-smoothed angular ratemap (σ = {MP_GAUSSIAN_SIGMA_CM} cm)', fontsize=10)

    # ── Linearised profile: smoothed rate + field KDE vs track angle ──────────
    deg_per_bin = np.degrees(bw)
    all_deg = (np.arange(n) + 0.5) * deg_per_bin
    ax_lin.plot(all_deg, np.where(valid_mask, fr_smooth, np.nan), color='#bdbdbd', lw=1.2,
                label='smoothed ratemap')
    for n_pf, pf in enumerate(plot_fields):
        nodes = np.asarray(pf['nodes'])
        deg = (nodes + 0.5) * deg_per_bin
        x, y_sm, y_in, y_kde = _nan_at_wrap(deg, fr_smooth[nodes], cc['kde_in'][nodes], pf['rate'])
        ax_lin.plot(x, y_sm, color='tab:blue', lw=2.0, label='field (smoothed)' if n_pf == 0 else None)
        ax_lin.plot(x, y_in, color='tab:green', lw=1.2,
                    label=f'KDE input: {KDE_RATEMAP_TYPE}' if n_pf == 0 else None)
        ax_lin.plot(x, y_kde, color='tab:red', lw=1.5, ls='--', label='field KDE' if n_pf == 0 else None)
        x_thr, y_thr = _nan_at_wrap(deg, np.full(len(nodes), PEAK_REL_HEIGHT_FRAC * pf['ref_hz']))
        ax_lin.plot(x_thr, y_thr, color='tab:red', lw=0.8, ls=':',
                    label=f'{PEAK_REL_HEIGHT_FRAC:.0%} of KDE max' if n_pf == 0 else None)
        for p in range(1, len(nodes)):
            if pf['labels'][p] != pf['labels'][p - 1]:
                ax_lin.axvline(((nodes[0] + p) % n) * deg_per_bin, color='black', lw=1.2, ls='-.')
        for i in pf['local_max']:
            if i not in pf['counted']:
                ax_lin.plot(deg[i], pf['rate'][i], 'x', color='black', ms=6, mew=1.5)
        for k, i in enumerate(pf['counted'], start=1):
            ax_lin.plot(deg[i], pf['rate'][i], '*', ms=13, mfc='white', mec='black')
            label = f"{pf['field_number']}.{k}" if pf['multi'] else f"F{pf['field_number']}"
            ax_lin.annotate(label, (deg[i], pf['rate'][i]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8)
    ax_lin.set_xlim(0, 360)
    ax_lin.set_xlabel('Track angle (deg)')
    ax_lin.set_ylabel('Firing rate (Hz)')
    ax_lin.set_title('Field profiles (dash-dot line = trough cut)', fontsize=10)
    ax_lin.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'{title}\n{_field_summary_title(plot_fields)}   '
                 f'(star = counted peak, × = peak < {PEAK_REL_HEIGHT_FRAC:.0%}, white line = trough cut)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


# ── Multi-peak: per-session job (called from thread pool) ───────────────────

def _mp_png_name(session: str, unit: str) -> str:
    """File name of a cell's multi-peak figure (in MP_RATEMAP_DIR\\MultiPeak or SinglePeak)."""
    return session.replace('\\', '_').replace('/', '_') + '__' + os.path.splitext(unit)[0] + '.png'


def _mp_run_session(args):
    sess_idx, n_sessions, session, units = args
    dirpath = os.path.join(MP_PLACE_CELL_FOLDER, session)
    field_rows, sub_rows, cell_rows = [], [], []

    with _print_lock:
        print(f'[{sess_idx}/{n_sessions}]  {session}  ({len(units)} cell(s))')

    if not os.path.isdir(dirpath):
        with _print_lock:
            print(f'  [SKIP] session folder not found: {dirpath}')
        return field_rows, sub_rows, cell_rows

    names = os.listdir(dirpath)
    if MP_COORD_UNITS == 'cm':
        tracking = [f for f in names if f.lower().endswith('_cm.csv')]
    else:
        tracking = [f for f in names if f.lower().endswith(('.csv', '.xlsx'))
                    and not f.lower().endswith('_cm.csv')]
    if len(tracking) != 1:
        with _print_lock:
            print(f'  [SKIP] expected 1 tracking file in {dirpath}, found {len(tracking)}')
        return field_rows, sub_rows, cell_rows

    try:
        track = _mp_load_session_track(os.path.join(dirpath, tracking[0]))
    except Exception as e:
        with _print_lock:
            print(f'  TRACKING ERROR in {session}: {e}')
        return field_rows, sub_rows, cell_rows
    if track is None:
        with _print_lock:
            print(f'  [SKIP] no usable tracking samples in {session}')
        return field_rows, sub_rows, cell_rows

    for unit, fdf in units:
        ntt_path = os.path.join(dirpath, unit)
        if not os.path.isfile(ntt_path):
            with _print_lock:
                print(f'  [SKIP] .ntt not found: {ntt_path}')
            continue
        arena_type = str(fdf['arena_type'].iloc[0])
        try:
            ctx = _mp_build_unit_ratemap(track, ntt_path)
            ctx['kde_in'] = _mp_kde_input_map_2d(ctx)
            cc  = _mp_build_ratemap_circular(ctx) if arena_type == 'Circle' else None
            if cc is not None:
                cc['kde_in'] = _mp_kde_input_map_circular(cc)
            f_rows, s_rows, plot_fields = _analyse_cell(ctx, cc, fdf, arena_type, session, unit)
        except Exception as e:
            with _print_lock:
                print(f'  ERROR in {session} | {unit}: {e}')
            continue

        n_multi = sum(r['multi_peak'] for r in f_rows)
        check   = all(r['ratemap_check'] for r in f_rows)
        if not check:
            with _print_lock:
                print(f'  [WARN] {session} | {unit}: recomputed field peak != Excel peak_fr_hz '
                      f'(check the ratemap settings match the PlaceCell_Main run)')

        sub_dir   = os.path.join(MP_RATEMAP_DIR, 'MultiPeak' if n_multi else 'SinglePeak')
        save_path = os.path.join(sub_dir, _mp_png_name(session, unit))
        try:
            title = f'{session}  |  {unit}'
            if cc is not None:
                _mp_plot_cell_circular(cc, plot_fields, title, save_path)
            else:
                _mp_plot_cell_2d(ctx, plot_fields, title, save_path)
        except Exception as e:
            with _print_lock:
                print(f'  PLOT ERROR in {session} | {unit}: {e}')
            save_path = ''

        field_rows += f_rows
        sub_rows   += s_rows
        cell_rows.append({
            'session': session, 'unit': unit, 'arena_type': arena_type,
            'n_fields': len(f_rows),
            'n_multi_peak_fields': int(n_multi),
            'has_multi_peak_field': bool(n_multi),
            'max_n_peaks': max(r['n_peaks'] for r in f_rows),
            'ratemap_check': check,
            'ratemap_png': save_path,
        })

    return field_rows, sub_rows, cell_rows


# ── Multi-peak: batch run + comparison tables ───────────────────────────────

MP_FIELD_COLUMNS = ['session', 'unit', 'arena_type', 'field_number',
                    'multi_peak', 'n_peaks', 'n_kde_local_maxima', 'peak_fallback',
                    'kde_peak_hz', 'peak_threshold_hz', 'peak_kde_hz', 'peak_bins',
                    'local_max_rel_heights', 'local_max_trough_rel',
                    'shallowest_trough_rel', 'troughs_rel', 'lowest_counted_peak_rel',
                    'peak_separation_cm', 'peak_separations_cm',
                    'kde_n_points', 'kde_n_eff', 'kde_factor', 'kde_axes', 'kde_bw_cm', 'kde_note',
                    'n_bins', 'area_cm2', 'pct_of_occupied_area',
                    'peak_fr_hz', 'recomputed_peak_fr_hz', 'ratemap_check', 'bin_coords']

MP_SUBFIELD_COLUMNS = ['session', 'unit', 'arena_type', 'field_number',
                       'sub_field_number', 'sub_field_id',
                       'kde_peak_hz', 'kde_peak_rel', 'max_saddle_hz', 'max_saddle_rel',
                       'peak_fr_hz', 'peak_fr_hz_raw', 'field_raw_max_fr_hz',
                       'n_bins', 'area_cm2', 'pct_of_field_area',
                       # 2-D grid (Open field / Linear track)
                       'kde_peak_bin_x', 'kde_peak_bin_y', 'peak_bin_x', 'peak_bin_y',
                       'com_bin_x', 'com_bin_y', 'com_cm_x', 'com_cm_y',
                       # Angular (Circle track)
                       'kde_peak_bin_theta', 'kde_peak_theta_deg', 'peak_bin_theta',
                       'peak_theta_deg', 'com_theta_deg', 'arc_length_cm',
                       'bin_coords']

MP_CELL_COLUMNS = ['session', 'unit', 'arena_type', 'n_fields', 'n_multi_peak_fields',
                   'has_multi_peak_field', 'max_n_peaks', 'ratemap_check', 'ratemap_png']


def run_multipeak_kde(full: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    """KDE multi-peak analysis of every field of every place cell (place_cell
    == True in `full`, fields from `fields` = INPUT_EXCEL 'PlaceFields');
    writes the per-cell figures to MP_RATEMAP_DIR and MULTIPEAK_EXCEL
    (sheets Fields, SubFields, Cells, Settings). Returns the Fields table."""
    os.makedirs(os.path.join(MP_RATEMAP_DIR, 'MultiPeak'),  exist_ok=True)
    os.makedirs(os.path.join(MP_RATEMAP_DIR, 'SinglePeak'), exist_ok=True)

    # PlaceFields also lists cells that were overruled AFTER field detection
    # (per-field / total area caps) -- keep only final place_cell == True cells.
    is_pc = full['place_cell'].apply(_to_bool)
    place_keys = set(zip(full.loc[is_pc, 'session'], full.loc[is_pc, 'unit']))
    keep = np.array([(s, u) in place_keys for s, u in zip(fields['session'], fields['unit'])],
                    dtype=bool)
    df_fields = fields[keep & fields['bin_coords'].notna().to_numpy()]

    sessions: dict[str, list] = {}
    for (session, unit), fdf in df_fields.groupby(['session', 'unit'], sort=True):
        sessions.setdefault(session, []).append((unit, fdf))
    n_cells = sum(len(u) for u in sessions.values())
    print(f'\nMulti-peak KDE: {len(place_keys)} place cell(s) in {INPUT_EXCEL}; '
          f'{n_cells} with fields, across {len(sessions)} session(s).\n')

    job_args = [(i, len(sessions), session, units)
                for i, (session, units) in enumerate(sessions.items(), start=1)]

    all_fields, all_subs, all_cells = [], [], []
    # The figures keep the full axes box (the stats-plot style hides the top
    # and right spines).
    with matplotlib.rc_context({'axes.spines.top': True, 'axes.spines.right': True}), \
            concurrent.futures.ThreadPoolExecutor(max_workers=MP_MAX_WORKERS) as executor:
        for f_rows, s_rows, c_rows in executor.map(_mp_run_session, job_args):
            all_fields += f_rows
            all_subs   += s_rows
            all_cells  += c_rows

    out_fields = pd.DataFrame(all_fields, columns=MP_FIELD_COLUMNS)
    out_subs   = pd.DataFrame(all_subs,   columns=MP_SUBFIELD_COLUMNS)
    out_cells  = pd.DataFrame(all_cells,  columns=MP_CELL_COLUMNS)
    out_settings = pd.DataFrame(
        [{'parameter': 'script_name',   'value': os.path.basename(__file__)},
         {'parameter': 'run_timestamp', 'value': datetime.now().isoformat(timespec='seconds')}]
        + [{'parameter': name, 'value': str(globals()[name])} for name in MP_RUN_CONFIG_VARS])

    with pd.ExcelWriter(MULTIPEAK_EXCEL, engine='openpyxl') as writer:
        out_fields.to_excel(writer,   sheet_name='Fields',    index=False)
        out_subs.to_excel(writer,     sheet_name='SubFields', index=False)
        out_cells.to_excel(writer,    sheet_name='Cells',     index=False)
        out_settings.to_excel(writer, sheet_name='Settings',  index=False)

    n_bad_check = int((~out_cells['ratemap_check'].astype(bool)).sum())
    print(f'\nDone. Multi-peak results saved to {MULTIPEAK_EXCEL}')
    print(f'Multi-peak ratemaps saved to        : {MP_RATEMAP_DIR}')
    print(f'Cells analysed                      : {len(out_cells)} / {n_cells}')
    print(f'Fields analysed                     : {len(out_fields)}')
    print(f'Multi-peak fields (>= 2 peaks at '
          f'{PEAK_REL_HEIGHT_FRAC:.0%}) : {int(out_fields["multi_peak"].sum())}  '
          f'({len(out_subs)} sub-fields)')
    print(f'Cells with >= 1 multi-peak field    : {int(out_cells["has_multi_peak_field"].sum())}')
    if n_bad_check:
        print(f'[WARN] {n_bad_check} cell(s) failed ratemap_check -- their recomputed '
              f'smoothed ratemap does not match the Excel; check the MP_ ratemap settings.')
    return out_fields


def build_multipeak_dfs(full: pd.DataFrame, fields: pd.DataFrame) -> dict:
    """Multi-peak tables to compare, {'Cell': one row per place cell,
    'Field': one row per field, 'MultiField': multi-peak fields only}, tagged
    with arena type, session type, animal and cell_id like the other
    dataframes. Fields come from run_multipeak_kde if RUN_MULTIPEAK_KDE, else
    from the 'Fields' sheet of MULTIPEAK_EXCEL; EXCLUDE_NO_KDE_FIELDS /
    EXCLUDE_RATEMAP_MISMATCH fields are dropped before the per-cell values are
    computed (a cell with 3 fields, one dropped, counts as a 2-field cell)."""
    if RUN_MULTIPEAK_KDE:
        mp_fields = run_multipeak_kde(full, fields)
    else:
        print(f'Loading Fields sheet of {MULTIPEAK_EXCEL} ...')
        mp_fields = pd.read_excel(MULTIPEAK_EXCEL, sheet_name='Fields')
    n_all = len(mp_fields)
    if EXCLUDE_NO_KDE_FIELDS:
        mp_fields = mp_fields[mp_fields['kde_note'].fillna('').astype(str).str.strip() == '']
    if EXCLUDE_RATEMAP_MISMATCH:
        mp_fields = mp_fields[mp_fields['ratemap_check'].apply(_to_bool)]
    print(f'Multi-peak: {len(mp_fields)} of {n_all} fields kept '
          f'(EXCLUDE_NO_KDE_FIELDS={EXCLUDE_NO_KDE_FIELDS}, '
          f'EXCLUDE_RATEMAP_MISMATCH={EXCLUDE_RATEMAP_MISMATCH}).')

    # 'arena_type' in the Fields sheet is the field geometry ('2D'/'Circle');
    # _tag_groups replaces it with the arena (Circle / Linear / Open).
    mp_fields = mp_fields.drop(columns=['arena_type'])
    mp_fields['multi_peak'] = mp_fields['multi_peak'].apply(_to_bool)

    cells = (mp_fields.groupby(['session', 'unit'])
                      .agg(n_fields=('field_number', 'count'),
                           n_multi_peak_fields=('multi_peak', 'sum'),
                           max_n_peaks=('n_peaks', 'max'))
                      .reset_index())
    cells['pct_multi_peak_fields'] = 100.0 * cells['n_multi_peak_fields'] / cells['n_fields']
    cells['has_multi_peak_field']  = cells['n_multi_peak_fields'] > 0

    fields_df = _tag_groups(mp_fields, 'place field(s) (multi-peak)')
    return {'Cell':       _tag_groups(cells, 'place cell(s) (multi-peak)'),
            'Field':      fields_df,
            'MultiField': fields_df[fields_df['multi_peak']]}


def _mp_map_path(session: str, unit: str) -> str:
    """Multi-peak figure written by _mp_run_session for this cell (MultiPeak or
    SinglePeak subfolder; the MultiPeak path if neither exists)."""
    paths = [os.path.join(MP_RATEMAP_DIR, sub, _mp_png_name(session, unit))
             for sub in ('MultiPeak', 'SinglePeak')]
    return next((p for p in paths if os.path.isfile(p)), paths[0])


# ── One full comparison (session types of one arena, or arena types) ────────

def _print_significant(tbl: pd.DataFrame, title: str, metric_width: int, test_width: int):
    sig = tbl[tbl['significant'] == True]  # noqa: E712
    if not len(sig):
        print(f'\nNo significant differences found ({title}).')
        return
    print(f'\nSignificant differences, {title} (p < 0.05):')
    for _, row in sig.iterrows():
        stat = row['statistic']
        stat_str = f'{stat:.3f}' if pd.notna(stat) else 'nan'
        print(f"  {row['metric']:{metric_width}s} {row['test']:{test_width}s} "
              f"stat={stat_str}  p={row['p_value']:.4g}")


def _concat_tables(tables: list) -> pd.DataFrame:
    tables = [t for t in tables if len(t)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _subset_mp(mp_dfs: dict, select) -> dict:
    """`select` applied to every multi-peak table (None stays None)."""
    return None if mp_dfs is None else {level: select(d) for level, d in mp_dfs.items()}


def run_comparison(grp: Grouping, full_df: pd.DataFrame, field_df: pd.DataFrame,
                   shape_df: pd.DataFrame, all_units_df: pd.DataFrame, plots_dir: str,
                   ts_df: pd.DataFrame = None, mp_dfs: dict = None) -> dict:
    """Compare every metric across the groups of `grp` -- statistics, box/bar
    plots in `plots_dir` and histograms in `plots_dir`/Histograms. The
    dataframes must already be restricted to the cells to compare; `ts_df`
    (temporal stability PerCell table) and `mp_dfs` (multi-peak tables,
    build_multipeak_dfs) are optional. Returns {table_name: table}."""
    name = grp.title or grp.noun.capitalize()
    print(f'\n{"=" * 70}\n{name} -- comparing {grp.noun}: '
          f'{[_label(g) for g in grp.order]}\n{"=" * 70}')
    print('Place cells per group (Full sheet):')
    print(full_df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int))
    if not len(full_df):
        print('No place cells; skipping.')
        return {}

    # Each cell's shade, from this comparison's place cells only.
    grp = replace(grp, shade=cell_shades(full_df))

    # ── Continuous metrics ──────────────────────────────────────────────────
    full_desc, full_omnibus, full_posthoc = run_all_comparisons(
        full_df, FULL_METRICS, 'Full', grp, plots_dir)
    field_desc, field_omnibus, field_posthoc = run_all_comparisons(
        field_df, FIELD_METRICS, 'Fields', grp, plots_dir)
    shape_desc, shape_omnibus, shape_posthoc = run_all_comparisons(
        shape_df, FIELD_SHAPE_METRICS, 'FieldShape', grp, plots_dir)

    desc_df    = pd.concat([full_desc, field_desc, shape_desc], ignore_index=True)
    omnibus_df = pd.concat([full_omnibus, field_omnibus, shape_omnibus], ignore_index=True)
    posthoc_df = pd.concat([full_posthoc, field_posthoc, shape_posthoc], ignore_index=True)

    extra_desc, extra_omnibus, extra_posthoc = run_all_comparisons(
        full_df, EXTRA_FULL_METRICS, 'Full', grp, plots_dir)

    # ── Stability score within each stability type ─────────────────────────
    stab_desc_all, stab_omni_all, stab_post_all = [], [], []
    for key, desc in STABILITY_TYPES.items():
        sub = full_df[full_df['stability_type'] == key]
        label = f'Stability score ({desc})'
        d, o, p = compare_groups(sub, 'stability_score', label, grp)
        stab_desc_all.append(d)
        stab_omni_all.append(o)
        if len(p):
            stab_post_all.append(p)
        plot_metric(sub, 'stability_score', label,
                    os.path.join(plots_dir, f'Full_stability_score_{key}.png'), grp,
                    stats_lines=_continuous_stats_lines(d, o, p, grp), posthoc_df=p)

    stab_desc_df    = pd.concat(stab_desc_all, ignore_index=True)
    stab_omnibus_df = pd.DataFrame(stab_omni_all)
    stab_posthoc_df = pd.concat(stab_post_all, ignore_index=True) if stab_post_all else pd.DataFrame()

    # ── Categorical (proportion) comparisons ────────────────────────────────
    cat_desc_all, cat_omni_all, cat_post_all = [], [], []
    for pop, metrics in ((all_units_df, CATEGORICAL_METRICS_ALL_UNITS),
                         (full_df, CATEGORICAL_METRICS_PLACE_CELLS)):
        for col, label in metrics.items():
            d, o, p = compare_categorical(pop, col, label, grp)
            cat_desc_all.append(d)
            cat_omni_all.append(o)
            if len(p):
                cat_post_all.append(p)
            plot_categorical(pop, col, label,
                             os.path.join(plots_dir, f'Categorical_{col}.png'), grp,
                             stats_lines=_categorical_stats_lines(d, o, p, grp),
                             posthoc_df=p)

    cat_desc_df    = pd.concat(cat_desc_all, ignore_index=True) if cat_desc_all else pd.DataFrame()
    cat_omnibus_df = pd.DataFrame(cat_omni_all)
    cat_posthoc_df = pd.concat(cat_post_all, ignore_index=True) if cat_post_all else pd.DataFrame()

    speed_dir_desc, speed_dir_omnibus, speed_dir_posthoc = compare_speed_direction(full_df, grp)
    plot_speed_direction(full_df, os.path.join(plots_dir, 'Categorical_speed_direction.png'), grp,
                         stats_lines=_speed_dir_stats_lines(speed_dir_desc, speed_dir_omnibus,
                                                            speed_dir_posthoc, grp),
                         posthoc_df=speed_dir_posthoc)
    speed_dir_omnibus_df = pd.DataFrame([speed_dir_omnibus])

    # ── Temporal stability (one value per cell; points shaded like full_df) ─
    ts_tables = {}
    if ts_df is not None:
        print('Place cells per group (temporal stability):')
        print(ts_df[grp.col].value_counts().reindex(grp.order).fillna(0).astype(int))
        ts_desc, ts_omnibus, ts_posthoc = run_all_comparisons(
            ts_df, TEMPORAL_STABILITY_METRICS, 'TemporalStability', grp, plots_dir)
        ts_tables = {'TS_Desc': ts_desc, 'TS_Omnibus': ts_omnibus, 'TS_PostHoc': ts_posthoc}

    # ── Multi-peak fields (per cell / field / multi-peak field; shaded like full_df) ─
    mp_tables = {}
    if mp_dfs is not None:
        print('Place cells / fields / multi-peak fields per group (multi-peak):')
        print(pd.DataFrame({level: d[grp.col].value_counts() for level, d in mp_dfs.items()})
              .reindex(grp.order).fillna(0).astype(int))
        mp_desc, mp_omni, mp_post = [], [], []
        for level, metrics in MULTIPEAK_METRICS.items():
            d, o, p = run_all_comparisons(mp_dfs[level], metrics, f'MultiPeak_{level}',
                                          grp, plots_dir)
            mp_desc.append(d)
            mp_omni.append(o)
            mp_post.append(p)
        mp_cat_desc, mp_cat_omni, mp_cat_post = [], [], []
        for level, metrics in MULTIPEAK_CATEGORICAL.items():
            for col, label in metrics.items():
                d, o, p = compare_categorical(mp_dfs[level], col, label, grp)
                mp_cat_desc.append(d)
                mp_cat_omni.append(o)
                mp_cat_post.append(p)
                plot_categorical(mp_dfs[level], col, label,
                                 os.path.join(plots_dir, f'MultiPeak_Categorical_{col}.png'), grp,
                                 stats_lines=_categorical_stats_lines(d, o, p, grp),
                                 posthoc_df=p,
                                 ylabel='% of place fields' if level == 'Field' else '% of cells')
        mp_tables = {'MP_Desc':       _concat_tables(mp_desc),
                     'MP_Omnibus':    _concat_tables(mp_omni),
                     'MP_PostHoc':    _concat_tables(mp_post),
                     'MP_CatDesc':    _concat_tables(mp_cat_desc),
                     'MP_CatOmnibus': pd.DataFrame(mp_cat_omni),
                     'MP_CatPostHoc': _concat_tables(mp_cat_post)}

    print(f'Box/bar plots saved to {plots_dir}')
    _print_significant(omnibus_df, f'{name}, continuous metrics', 45, 16)
    _print_significant(
        pd.concat([extra_omnibus, stab_omnibus_df, cat_omnibus_df, speed_dir_omnibus_df],
                  ignore_index=True),
        f'{name}, extra/categorical metrics', 65, 30)
    if ts_tables:
        _print_significant(ts_tables['TS_Omnibus'], f'{name}, temporal stability', 65, 16)
    if mp_tables:
        _print_significant(pd.concat([mp_tables['MP_Omnibus'], mp_tables['MP_CatOmnibus']],
                                     ignore_index=True),
                           f'{name}, multi-peak fields', 65, 30)

    # ── Histograms ──────────────────────────────────────────────────────────
    hist_dir = os.path.join(plots_dir, 'Histograms')
    os.makedirs(hist_dir, exist_ok=True)
    for col, label in HIST_FULL_METRICS.items():
        plot_histogram_by_group(full_df, col, label,
                                os.path.join(hist_dir, f'Hist_Full_{col}.png'), grp)
    for col, label in FIELD_METRICS.items():
        plot_histogram_by_group(field_df, col, label,
                                os.path.join(hist_dir, f'Hist_Fields_{col}.png'), grp)
    for col, label in FIELD_SHAPE_METRICS.items():
        plot_histogram_by_group(shape_df, col, label,
                                os.path.join(hist_dir, f'Hist_FieldShape_{col}.png'), grp)
    for col, label in SPEED_METRICS.items():
        plot_speed_histograms_by_group(full_df, col, label,
                                       os.path.join(hist_dir, f'Hist_Speed_{col}.png'), grp)
    if ts_df is not None:
        for col, label in TEMPORAL_STABILITY_METRICS.items():
            plot_histogram_by_group(ts_df, col, label,
                                    os.path.join(hist_dir, f'Hist_TemporalStability_{col}.png'), grp)
    if mp_dfs is not None:
        for level, metrics in MULTIPEAK_METRICS.items():
            for col, label in metrics.items():
                plot_histogram_by_group(mp_dfs[level], col, label,
                                        os.path.join(hist_dir, f'Hist_MultiPeak_{level}_{col}.png'),
                                        grp)
    print(f'Histogram plots saved to {hist_dir}')

    return {
        'Desc':             desc_df,
        'Omnibus':          omnibus_df,
        'PostHoc':          posthoc_df,
        'Desc_Extra':       extra_desc,
        'Omnibus_Extra':    extra_omnibus,
        'PostHoc_Extra':    extra_posthoc,
        'StabType_Desc':    stab_desc_df,
        'StabType_Omnibus': stab_omnibus_df,
        'StabType_PostHoc': stab_posthoc_df,
        'CatDesc':          cat_desc_df,
        'CatOmnibus':       cat_omnibus_df,
        'CatPostHoc':       cat_posthoc_df,
        'SpeedDir_Desc':    speed_dir_desc,
        'SpeedDir_Omnibus': speed_dir_omnibus_df,
        'SpeedDir_PostHoc': speed_dir_posthoc,
        **ts_tables,
        **mp_tables,
    }


def _stitch_side_by_side(paths: list, out_path: str):
    """Save the images `paths` side by side (top-aligned, PANEL_GAP_IN apart,
    on white) as one image. A missing image (None) leaves a blank space as
    wide as the widest one, so every comparison keeps its position."""
    imgs = []
    for p in paths:
        if p is None:
            imgs.append(None)
            continue
        with Image.open(p) as im:
            imgs.append(im.convert('RGB'))
    present = [im for im in imgs if im is not None]
    blank_w = max(im.width for im in present)
    widths = [im.width if im is not None else blank_w for im in imgs]
    gap = int(round(PANEL_GAP_IN * FIG_DPI))
    canvas = Image.new('RGB', (sum(widths) + gap * (len(imgs) - 1),
                               max(im.height for im in present)), 'white')
    x = 0
    for im, w in zip(imgs, widths):
        if im is not None:
            canvas.paste(im, (x, 0))
        x += w + gap
    canvas.save(out_path, dpi=(FIG_DPI, FIG_DPI))


def _combine_side_by_side(src_dirs: list, out_dir: str):
    """For every .png found in any of `src_dirs` (by path relative to it,
    e.g. 'Histograms\\Hist_Full_sir.png'), save the versions from all
    `src_dirs` side by side, in that order, to the same path in `out_dir`."""
    rel_paths = sorted({os.path.relpath(os.path.join(root, f), d)
                        for d in src_dirs for root, _dirs, files in os.walk(d)
                        for f in files if f.lower().endswith('.png')})
    for rel in rel_paths:
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        srcs = [os.path.join(d, rel) for d in src_dirs]
        _stitch_side_by_side([p if os.path.isfile(p) else None for p in srcs], out_path)
    print(f'\n{len(rel_paths)} side-by-side plots '
          f'({" | ".join(os.path.basename(d) for d in src_dirs)}) saved to {out_dir}')


def run_animal_controlled(full_df: pd.DataFrame, field_df: pd.DataFrame, shape_df: pd.DataFrame,
                          all_units_df: pd.DataFrame, out_dir: str,
                          ts_df: pd.DataFrame = None, mp_dfs: dict = None) -> dict:
    """Level 3: for each entry of ANIMAL_CONTROLLED_COMPARISONS, compare its
    arenas using only the cells of its animals (same tests/plots as the
    arena-type comparison). Each plot is saved to out_dir (histograms to
    out_dir\\Histograms) with the comparisons side by side, each with its own
    statistics -- the comparisons are not tested against each other. The
    tables are also saved to out_dir\\AnimalControlled_Stats.xlsx. With
    `ts_df` / `mp_dfs`, the temporal stability / multi-peak example maps of
    every comparison go to out_dir\\EXAMPLE_MAPS_SUBDIR. Returns
    {sheet_name: table}."""
    all_tables = {}
    ex_dir = os.path.join(out_dir, EXAMPLE_MAPS_SUBDIR)
    if (ts_df is not None or mp_dfs is not None) and os.path.isdir(ex_dir):
        # Old example maps are removed first so a re-run never leaves stale ones.
        for f in os.listdir(ex_dir):
            if f.lower().endswith('.png'):
                os.remove(os.path.join(ex_dir, f))
    # Each comparison's plots are first saved to its own temporary folder,
    # then combined side by side.
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, (animals, arenas) in ANIMAL_CONTROLLED_COMPARISONS.items():
            grp = animal_controlled_grouping(animals, arenas)
            wanted = {a.lower() for a in animals}

            def _select(d):
                if d is None:
                    return None
                return d[d['animal'].astype(str).str.lower().isin(wanted)
                         & d['arena_type'].isin(grp.order)]

            sel_full = _select(full_df)
            print(f'\n{name}: place cells per animal and arena (Full sheet):')
            print(pd.crosstab(sel_full['animal'], sel_full['arena_type'], margins=True))
            for arena in grp.order:
                missing = [a for a in animals
                           if not ((sel_full['animal'].astype(str).str.lower() == a.lower())
                                   & (sel_full['arena_type'] == arena)).any()]
                if missing:
                    print(f'WARNING: no place cells of {missing} in the {arena} arena.')

            sel_ts = _select(ts_df)
            if sel_ts is not None:
                print(f'\n{name}: place cells per animal and arena (temporal stability):')
                print(pd.crosstab(sel_ts['animal'], sel_ts['arena_type'], margins=True))

            sel_mp = _subset_mp(mp_dfs, _select)
            if sel_mp is not None:
                print(f'\n{name}: place cells per animal and arena (multi-peak):')
                print(pd.crosstab(sel_mp['Cell']['animal'], sel_mp['Cell']['arena_type'],
                                  margins=True))

            tables = run_comparison(grp, sel_full, _select(field_df), _select(shape_df),
                                    _select(all_units_df), os.path.join(tmp_dir, name),
                                    ts_df=sel_ts, mp_dfs=sel_mp)
            tag = 'AC' + ''.join(grp.order)   # e.g. 'ACCircleOpen'
            for tname, tbl in tables.items():
                if len(tbl):
                    tbl.insert(0, 'animals', ', '.join(animals))
                all_tables[f'{tag}_{tname}'] = tbl
            if sel_ts is not None and len(sel_ts):
                _save_example_maps(grp, sel_ts, ex_dir, name, EXAMPLE_METRIC, _map_path)
            if sel_mp is not None and len(sel_mp['Cell']):
                _save_example_maps(grp, sel_mp['Cell'], ex_dir, f'MultiPeak_{name}',
                                   MP_EXAMPLE_METRIC, _mp_map_path)

        _combine_side_by_side([os.path.join(tmp_dir, name) for name in ANIMAL_CONTROLLED_COMPARISONS],
                              out_dir)

    if all_tables:
        os.makedirs(out_dir, exist_ok=True)
        xlsx = os.path.join(out_dir, 'AnimalControlled_Stats.xlsx')
        with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
            for sheet_name, tbl in all_tables.items():
                tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f'\nAnimal-controlled statistics saved to {xlsx}')
    return all_tables


# ── Run snapshot (code + metadata, saved before the analysis) ───────────────

_SNAPSHOT_VALUE_TYPES = (str, int, float, bool, list, tuple, dict, type(None))
_SNAPSHOT_PACKAGES = ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels',
                      'openpyxl', 'pillow']


def _module_parameters(mod) -> list[dict]:
    """Every public plain-valued (str / number / bool / list / dict ...)
    module-level setting of `mod`, as {module, parameter, value} rows."""
    return [{'module': mod.__name__, 'parameter': k, 'value': str(v)}
            for k, v in vars(mod).items()
            if not k.startswith('_') and isinstance(v, _SNAPSHOT_VALUE_TYPES)]


def _git(*args) -> str:
    """Output of a git command run in this script's folder ('' if git fails)."""
    try:
        return subprocess.run(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def save_run_snapshot() -> str:
    """Copy this script and its local modules (PlaceCell_Main_v21,
    CellClusteredStats_Utils, PlaceFieldShape_Utils) to
    OUTPUT_DIR\\RUN_SNAPSHOT_SUBDIR\\<timestamp>\\code, and write RunMetadata.xlsx
    there: 'Run' (time, script, git commit / uncommitted changes, Python and
    package versions, machine) and 'Parameters' (every setting of this script
    and of those modules at the start of the run). Returns the folder."""
    now = datetime.now()
    snap_dir = os.path.join(OUTPUT_DIR, RUN_SNAPSHOT_SUBDIR, now.strftime('%Y%m%d_%H%M%S'))
    code_dir = os.path.join(snap_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)

    script = os.path.abspath(__file__)
    local_modules = [ccs, pfs, pc]
    for path in [script] + [m.__file__ for m in local_modules]:
        shutil.copy2(path, code_dir)

    git_status = _git('status', '--porcelain', '--', script,
                      *[m.__file__ for m in local_modules])
    run_rows = [
        ('run_timestamp',  now.isoformat(timespec='seconds')),
        ('script',         script),
        ('command',        ' '.join(sys.argv)),
        ('working_dir',    os.getcwd()),
        ('git_commit',     _git('rev-parse', 'HEAD') or 'n/a'),
        ('git_branch',     _git('rev-parse', '--abbrev-ref', 'HEAD') or 'n/a'),
        ('git_uncommitted_changes (script + local modules)', git_status or 'none'),
        ('python',         sys.version.replace('\n', ' ')),
        ('python_executable', sys.executable),
        ('platform',       platform.platform()),
        ('machine',        platform.node()),
    ]
    for pkg in _SNAPSHOT_PACKAGES:
        try:
            run_rows.append((f'{pkg}_version', importlib.metadata.version(pkg)))
        except importlib.metadata.PackageNotFoundError:
            run_rows.append((f'{pkg}_version', 'not installed'))
    run_rows.append(('code_snapshot', code_dir))

    params = [row for mod in [sys.modules[__name__]] + local_modules
              for row in _module_parameters(mod)]
    with pd.ExcelWriter(os.path.join(snap_dir, 'RunMetadata.xlsx'), engine='openpyxl') as writer:
        pd.DataFrame(run_rows, columns=['item', 'value']).to_excel(
            writer, sheet_name='Run', index=False)
        pd.DataFrame(params, columns=['module', 'parameter', 'value']).to_excel(
            writer, sheet_name='Parameters', index=False)
    print(f'Run snapshot (code + metadata) saved to {snap_dir}')
    return snap_dir


if __name__ == '__main__':
    save_run_snapshot()

    print(f'Loading {INPUT_EXCEL} ...')
    full, fields = load_sheets()

    full_df      = build_full_metrics_df(full)
    field_df     = build_field_metrics_df(full, fields)
    shape_df     = build_field_shape_df(full, fields)
    all_units_df = build_all_units_df(full)

    print('\nPlace cells by arena and session type (Full sheet):')
    print(pd.crosstab(full_df['arena_type'], full_df['session_type'].fillna('unrecognized'),
                      margins=True))

    ts_df = None
    if INCLUDE_TEMPORAL_STABILITY:
        ts_df = build_temporal_stability_df()
        print('\nPlace cells by arena and session type (temporal stability):')
        print(pd.crosstab(ts_df['arena_type'], ts_df['session_type'].fillna('unrecognized'),
                          margins=True))

    mp_dfs = None
    if INCLUDE_MULTIPEAK:
        mp_dfs = build_multipeak_dfs(full, fields)
        print('\nPlace cells by arena and session type (multi-peak):')
        print(pd.crosstab(mp_dfs['Cell']['arena_type'],
                          mp_dfs['Cell']['session_type'].fillna('unrecognized'), margins=True))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_tables = {}

    # ── 1. Session types within each arena ──────────────────────────────────
    for arena in ARENA_TYPES:
        grp = session_grouping(arena)

        def _in_arena(d):
            if d is None:
                return None
            return d[(d['arena_type'] == arena) & d['session_type'].isin(grp.order)]

        tables = run_comparison(grp, _in_arena(full_df), _in_arena(field_df),
                                _in_arena(shape_df), _in_arena(all_units_df),
                                os.path.join(OUTPUT_DIR, 'SessionType', arena),
                                ts_df=_in_arena(ts_df), mp_dfs=_subset_mp(mp_dfs, _in_arena))
        for name, tbl in tables.items():
            if len(tbl) and 'arena_type' not in tbl.columns:
                tbl.insert(0, 'arena_type', arena)
            all_tables[f'Ses{arena}_{name}'] = tbl

    # ── 2. Arena types as a whole (all session types pooled) ────────────────
    tables = run_comparison(ARENA_GROUPING, full_df, field_df, shape_df, all_units_df,
                            os.path.join(OUTPUT_DIR, 'ArenaType'), ts_df=ts_df, mp_dfs=mp_dfs)
    for name, tbl in tables.items():
        all_tables[f'Arena_{name}'] = tbl

    # ── 3. Arena types, only animals recorded in all compared arenas ────────
    all_tables.update(run_animal_controlled(
        full_df, field_df, shape_df, all_units_df,
        os.path.join(OUTPUT_DIR, ANIMAL_CONTROLLED_DIR), ts_df=ts_df, mp_dfs=mp_dfs))

    # ── Save all statistics (new workbook + one long-format CSV) ────────────
    # Sheet names are capped at 31 characters by Excel.
    with pd.ExcelWriter(STATS_XLSX, engine='openpyxl') as writer:
        for sheet_name, tbl in all_tables.items():
            tbl.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f'\nAll statistics saved to {STATS_XLSX}')

    # 'table' names the source sheet; the tables have different columns, so
    # cells that don't apply to a table are left blank.
    stats_csv_parts = []
    for sheet_name, tbl in all_tables.items():
        if tbl is None or not len(tbl):
            continue
        tbl = tbl.copy()
        tbl.insert(0, 'table', sheet_name)
        stats_csv_parts.append(tbl)
    if stats_csv_parts:
        pd.concat(stats_csv_parts, ignore_index=True).to_csv(
            STATS_CSV, index=False, encoding='utf-8-sig')
        print(f'All statistics also saved to {STATS_CSV}')

    print(f'\nContinuous-metric test used: '
          f'{"Kruskal-Wallis + Mann-Whitney U (Holm), independence assumed" if CONTINUOUS_TEST == "KW" else "nested LMM"}')
    print(f'\nDone. Plots saved under {OUTPUT_DIR}')
