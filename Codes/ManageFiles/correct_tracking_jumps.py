# -*- coding: utf-8 -*-
"""
Remove tracking jumps from every *_cm.csv under ROOT_DIR and interpolate the
removed frames from the neighbouring reliable frames. The files are OVERWRITTEN
in place, after the originals are copied to WORK_DIR\\OriginalFiles (outside
ROOT_DIR, because the analysis scripts scan ROOT_DIR for *_cm.csv files).

A frame's position is unreliable if (see check_tracking_geometry.py for the scale
and boundary):
    jump            reached at a speed above MAX_SPEED_CMS from the last reliable
                    frame, re-tested against the remaining reliable frames until no
                    new ones appear (same rule as PlaceCell_Main_v21's
                    _smooth_tracking_position, so runs of bad frames are caught)
    out_of_bounds   further than TOL_CM beyond the nominal arena shape fitted to the
                    concentration of points; tested first, so that a jump test
                    never starts from such a frame
    invalid         NaN, or the tracker's +-1 'no position' value
Speeds and distances are measured at cm_per_px_used, so a file on a wrong scale gets
the same physical thresholds. THE FILE'S SCALE IS NOT CHANGED: this script only
replaces unreliable frames, so a session on a wrong cm/pixel stays on it.

Unreliable frames get x_cm, y_cm linearly interpolated in time between the nearest
reliable frames (held at the nearest reliable value at the session's start / end).
x_px, y_px are interpolated the same way, for those frames and for frames whose
pixels do not match their cm position (conversion's jump cleaning changed only the cm
columns). Reliable frames, timestamps and the number of frames are left exactly
as they were. 'was_interpolated' is added, or updated if the file already has it:
True if the frame's cm position is interpolated.

Re-running is safe: a file with a copy in OriginalFiles is always processed from
that copy, and the copy is never overwritten.

Output (WORK_DIR):
    OriginalFiles\\<session>\\<file>    the original tracking files
    TrackingCorrection_Log.xlsx        one row per session
    TrackingCorrection_Frames.csv      every changed frame: reason, old and new position
"""

import hashlib
import os
import shutil

import numpy as np
import pandas as pd

import check_tracking_geometry as cg

# ── USER INPUT ──────────────────────────────────────────────────────────────
ROOT_DIR = (r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt'
            r'\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final')
WORK_DIR = ROOT_DIR + '_TrackingCorrection'

DRY_RUN = False                           # True: log only; no file is copied or changed
MAX_SPEED_CMS = cg.pc.POS_JUMP_THRESH_CMS  # same artifact speed as the place-cell analyses
TOL_CM = cg.TOL_CM                        # leeway around the nominal arena shape
PX_MATCH_CM = 1.0                         # pixels further than this from the cm position don't match
# ─────────────────────────────────────────────────────────────────────────────


def find_tracking_files():
    """(relative session folder, arena, csv path, relative csv path) of every session."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(ROOT_DIR):
        tracking = [f for f in filenames if f.lower().endswith('_cm.csv')]
        if not tracking:
            continue
        rel = os.path.relpath(dirpath, ROOT_DIR)
        parts = rel.split(os.sep)
        arena = next((a for a in cg.ARENAS if any(p.lower() == a.lower() for p in parts)), None)
        if arena is None or len(tracking) > 1:
            print(f'WARNING: skipped {dirpath} (arena {arena}, {len(tracking)} *_cm.csv files).')
            continue
        files.append((rel, arena, os.path.join(dirpath, tracking[0]), os.path.join(rel, tracking[0])))
    return sorted(files)


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def source_path(csv_path, rel_csv):
    """The original file: its copy in OriginalFiles if there is one."""
    copy = os.path.join(WORK_DIR, 'OriginalFiles', rel_csv)
    return copy if os.path.isfile(copy) else csv_path


def back_up(csv_path, rel_csv):
    """Copy to OriginalFiles (never over an existing copy) and verify the copy."""
    copy = os.path.join(WORK_DIR, 'OriginalFiles', rel_csv)
    if os.path.isfile(copy):
        return
    os.makedirs(os.path.dirname(copy), exist_ok=True)
    shutil.copy2(csv_path, copy)
    if md5(copy) != md5(csv_path):
        os.remove(copy)
        raise IOError(f'Backup does not match the original: {csv_path}')


def find_jump_frames(x, y, t, bad):
    """`bad` plus every frame reached too fast from the last reliable frame, re-tested
    against the remaining reliable frames until none are added."""
    bad = bad.copy()
    for _ in range(len(x)):
        good = np.flatnonzero(~bad)
        if len(good) < 2:
            break
        dt = np.diff(t[good]) * 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            speed = np.hypot(np.diff(x[good]), np.diff(y[good])) / dt
        speed[dt <= 0] = 0.0
        too_fast = speed > MAX_SPEED_CMS
        if not too_fast.any():
            break
        bad[good[1:][too_fast]] = True
    return bad


def interpolate(t, values, bad):
    out = values.copy()
    if bad.any() and (~bad).any():
        out[bad] = np.interp(t[bad], t[~bad], values[~bad])
    return out


def longest_run(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def correct_session(rel, arena, source, scale, cm_per_px_ref):
    """Returns (corrected table, log row, changed-frames table)."""
    data = pd.read_csv(source, float_precision='round_trip')
    t  = data.iloc[:, 0].to_numpy(float)
    px = data.iloc[:, 1].to_numpy(float)
    py = data.iloc[:, 2].to_numpy(float)
    x  = data.iloc[:, 3].to_numpy(float)
    y  = data.iloc[:, 4].to_numpy(float)
    if np.any(np.diff(t) < 0):
        raise ValueError(f'{rel}: timestamps are not in order')

    invalid = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(t) | np.isin(x, [1, -1])
    used = cm_per_px_ref if abs(scale / cm_per_px_ref - 1) > cg.SCALE_FRAC else scale
    ratio = used / scale
    xr, yr = x * ratio, y * ratio

    angle, cu, cv = cg.fit_boundary(xr[~invalid], yr[~invalid], arena)
    uu, vv = cg.rotate(np.nan_to_num(xr), np.nan_to_num(yr), angle)
    out_of_bounds = (cg.distance_outside(uu - cu, vv - cv, arena) > TOL_CM) & ~invalid
    jump = find_jump_frames(xr, yr, t, invalid | out_of_bounds) & ~invalid & ~out_of_bounds
    bad = invalid | out_of_bounds | jump

    x_new, y_new = interpolate(t, x, bad), interpolate(t, y, bad)

    # Pixels: same slope as the cm columns, offset from the typical frame.
    ok = ~invalid & np.isfinite(px) & np.isfinite(py)
    x_px = scale * px + np.median((x - scale * px)[ok])
    y_px = scale * py + np.median((y - scale * py)[ok])
    px_mismatch = ~(np.hypot(x_px - x, y_px - y) * ratio <= PX_MATCH_CM)   # NaN counts as mismatch
    px_bad = bad | px_mismatch
    px_new, py_new = interpolate(t, px, px_bad), interpolate(t, py, px_bad)

    out = data.copy()
    was_corrected = 'was_interpolated' in out.columns
    earlier = out['was_interpolated'].astype(bool).to_numpy() if was_corrected else np.zeros(len(out), dtype=bool)
    for col, values in zip(out.columns[1:5], (px_new, py_new, x_new, y_new)):
        out[col] = values
    out['was_interpolated'] = earlier | bad

    dt = np.diff(t) * 1e-6
    frame_s = np.median(dt) if len(dt) else np.nan
    longest = longest_run(bad)
    parts = rel.split(os.sep)
    log = dict(session=rel, animal=parts[0], arena=arena, n_frames=len(x),
               cm_per_px=scale, cm_per_px_used=used, already_had_was_interpolated=was_corrected,
               n_invalid=int(invalid.sum()), n_out_of_bounds=int(out_of_bounds.sum()), n_jump=int(jump.sum()),
               n_cm_interpolated=int(bad.sum()), pct_cm_interpolated=100.0 * bad.mean(),
               longest_interpolated_run_frames=longest, longest_interpolated_run_s=longest * frame_s,
               max_shift_cm=float(np.nanmax(np.hypot(x_new - x, y_new - y) * ratio)) if bad.any() else 0.0,
               n_px_interpolated=int(px_bad.sum()), pct_px_interpolated=100.0 * px_bad.mean())
    reason = np.where(invalid, 'invalid', np.where(out_of_bounds, 'out_of_bounds', 'jump'))
    frames = pd.DataFrame(dict(session=rel, frame_index=np.flatnonzero(bad), time=t[bad], reason=reason[bad],
                               x_cm_old=x[bad], y_cm_old=y[bad], x_cm_new=x_new[bad], y_cm_new=y_new[bad]))
    return out, log, frames


def write_and_verify(out, csv_path, source):
    """Write through a temporary file, then check that only interpolated values changed."""
    tmp = csv_path + '.tmp'
    out.to_csv(tmp, index=False)
    new = pd.read_csv(tmp, float_precision='round_trip')
    old = pd.read_csv(source, float_precision='round_trip')
    if len(new) != len(old) or list(new.columns[:5]) != list(old.columns[:5]):
        os.remove(tmp)
        raise IOError(f'Rows or columns changed: {csv_path}')
    if not np.array_equal(new.iloc[:, 0].to_numpy(), old.iloc[:, 0].to_numpy()):
        os.remove(tmp)
        raise IOError(f'Timestamps changed: {csv_path}')
    interpolated = new['was_interpolated'].to_numpy(bool)
    for i in (3, 4):
        if not np.array_equal(new.iloc[:, i].to_numpy()[~interpolated], old.iloc[:, i].to_numpy()[~interpolated]):
            os.remove(tmp)
            raise IOError(f'A reliable frame changed: {csv_path}')
    os.replace(tmp, csv_path)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    files = find_tracking_files()
    print(f'Found {len(files)} tracking file(s) under {ROOT_DIR}.')

    # Reference scale of each animal + arena, always from the original files.
    scales = {}
    for rel, arena, csv_path, rel_csv in files:
        s = cg.load_session(rel, arena, source_path(csv_path, rel_csv), ROOT_DIR)
        scales[rel] = (s['animal'], arena, s['cm_per_px'])
    table = pd.DataFrame(scales.values(), columns=['animal', 'arena', 'cm_per_px'])
    refs = table.groupby(['animal', 'arena'])['cm_per_px'].median()

    logs, changed = [], []
    for i, (rel, arena, csv_path, rel_csv) in enumerate(files, start=1):
        print(f'  [{i}/{len(files)}] {rel}')
        source = source_path(csv_path, rel_csv)
        animal, _arena, scale = scales[rel]
        out, log, frames = correct_session(rel, arena, source, scale, refs[(animal, arena)])
        logs.append(log)
        changed.append(frames)
        if not DRY_RUN:
            back_up(csv_path, rel_csv)
            write_and_verify(out, csv_path, os.path.join(WORK_DIR, 'OriginalFiles', rel_csv))

    log_df = pd.DataFrame(logs)
    log_df.to_excel(os.path.join(WORK_DIR, 'TrackingCorrection_Log.xlsx'), index=False)
    pd.concat(changed, ignore_index=True).to_csv(os.path.join(WORK_DIR, 'TrackingCorrection_Frames.csv'), index=False)
    print(f'\n{"Dry run: no tracking file changed." if DRY_RUN else "Tracking files corrected."} '
          f'{int(log_df["n_cm_interpolated"].sum())} frames interpolated in {int((log_df["n_cm_interpolated"] > 0).sum())} '
          f'of {len(log_df)} sessions. Log: {WORK_DIR}')


if __name__ == '__main__':
    main()
