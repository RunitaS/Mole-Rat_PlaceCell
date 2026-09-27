# -*- coding: utf-8 -*-
r"""
Check that every session of a recording day has the same set of .ntt files.

Expected layout (same as SessionType_StatsComparison_v3.py):
    ROOT_DIR\<AnimalID>\<Arena>\Day<N>\<k>_<SessionType>\*.ntt
e.g. ...\Fa1059\Linear\Day10\1_0\TT5_SS_16.ntt

Every folder that directly or indirectly holds .ntt files is treated as a
session; its parent folder is the recording day. Within a day, the reference
set is the union of .ntt filenames over all sessions, and each session is
checked against it. Arena and session type are detected from the folder path
with the same rules as SessionType_StatsComparison_v3.py:

    Open   : Cntrl, Rotate, Zero          (e.g. '1_Cntrl', '2Rotate')
    Linear : 0, 90, 180, 270              (digits at the end, e.g. '3_180')
    Circle : Stnd, NoRot, Rot             (e.g. '2NoRot', '4Stnd')

Folders whose session type isn't recognized (e.g. a 'Full' concatenated
folder, or sessions with no Day<N> level) and days with only one session are
not compared; they are listed in the 'SkippedFolders' sheet.

If any day has sessions with differing .ntt files, an Excel report is written
with four sheets:
    Summary        : one row per recording day, number of .ntt files per session
    Mismatched     : one row per session that is missing files (folder + list)
    MissingFiles   : one row per missing file (long format, easy to filter)
    SkippedFolders : folders that were not compared, and why
"""

import os
import re
import sys
from collections import defaultdict

import pandas as pd

# ── Parameters ──────────────────────────────────────────────────────────────

ROOT_DIR    = r"X:\NMR_group_data\Runita\Data\Ephys_Data\AllSortedData\Tetrode"
OUTPUT_XLSX = os.path.join(ROOT_DIR, "Ntt_SessionConsistency_report.xlsx")

ARENA_TYPES = ['Open', 'Linear', 'Circle']
ARENA_SESSION_TYPES = {
    'Open':   ['Cntrl', 'Rotate', 'Zero'],
    'Linear': ['0', '90', '180', '270'],
    'Circle': ['Stnd', 'NoRot', 'Rot'],
}

# ── Session-type detection (from SessionType_StatsComparison_v3.py) ─────────

def extract_arena_type(session_path: str):
    """Return 'Open', 'Linear', or 'Circle' if one of those appears as a
    path component of `session_path`, else None."""
    parts = re.split(r'[\\/]+', str(session_path))
    for part in parts:
        for arena in ARENA_TYPES:
            if part.strip().lower() == arena.lower():
                return arena
    return None


def extract_session_type(session_path: str, arena):
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

# ── Scan ────────────────────────────────────────────────────────────────────

DAY_PATTERN = re.compile(r'^Day\d+$', re.IGNORECASE)


def _session_folder(dirpath, root_dir):
    """The folder directly under a Day<N> folder that contains `dirpath`, so
    .ntt files in sub-folders of a session still count towards that session.
    Falls back to `dirpath` itself if there is no Day<N> folder above it."""
    root = os.path.normcase(os.path.abspath(root_dir))
    path = os.path.abspath(dirpath)
    while os.path.normcase(path) != root and os.path.dirname(path) != path:
        if DAY_PATTERN.match(os.path.basename(os.path.dirname(path))):
            return path
        path = os.path.dirname(path)
    return dirpath

def find_sessions(root_dir):
    """{day_folder: {session_folder: set of .ntt paths relative to the
    session folder}}. A session is the top-level folder under a day that
    contains .ntt files (in itself or any sub-folder)."""
    days = defaultdict(lambda: defaultdict(set))
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        ntt = [f for f in filenames if f.lower().endswith('.ntt')]
        if not ntt:
            continue
        session = _session_folder(dirpath, root_dir)
        day = os.path.dirname(session)
        for f in ntt:
            days[day][session].add(os.path.relpath(os.path.join(dirpath, f), session))
    return days


def session_order_key(path):
    m = re.match(r'^(\d+)', os.path.basename(path))
    return (int(m.group(1)) if m else float('inf'), os.path.basename(path))


def check_days(days):
    summary_rows, mismatch_rows, missing_rows, skipped_rows = [], [], [], []

    for day in sorted(days):
        arena = extract_arena_type(day)
        sessions = {}
        for s, files in days[day].items():
            if extract_session_type(s, arena) is None:
                # e.g. a 'Full' concatenated folder, or no Day<N> level at all.
                skipped_rows.append({'folder': s, 'arena_type': arena,
                                     'n_ntt': len(files),
                                     'reason': 'session type not recognized'})
            else:
                sessions[s] = files
        if len(sessions) < 2:
            for s, files in sessions.items():
                skipped_rows.append({'folder': s, 'arena_type': arena,
                                     'n_ntt': len(files),
                                     'reason': 'only session of its day'})
            continue

        order = sorted(sessions, key=session_order_key)
        all_files = set().union(*sessions.values())
        counts = {s: len(sessions[s]) for s in order}
        consistent = all(sessions[s] == all_files for s in order)

        summary_rows.append({
            'day_folder': day,
            'arena_type': arena,
            'n_sessions': len(order),
            'n_ntt_union': len(all_files),
            'ntt_per_session': '; '.join(
                f'{os.path.basename(s)}={counts[s]}' for s in order),
            'same_count': len(set(counts.values())) == 1,
            'consistent': consistent,
        })
        if consistent:
            continue

        for s in order:
            missing = sorted(all_files - sessions[s])
            if not missing:
                continue
            sess_type = extract_session_type(s, arena)
            mismatch_rows.append({
                'day_folder': day,
                'session_folder': s,
                'arena_type': arena,
                'session_type': sess_type,
                'n_ntt_present': len(sessions[s]),
                'n_ntt_expected': len(all_files),
                'n_missing': len(missing),
                'missing_files': ', '.join(missing),
            })
            for f in missing:
                # Which other sessions of this day do have the file.
                present_in = [os.path.basename(o) for o in order if f in sessions[o]]
                missing_rows.append({
                    'day_folder': day,
                    'session_folder': s,
                    'arena_type': arena,
                    'session_type': sess_type,
                    'missing_file': f,
                    'present_in_sessions': ', '.join(present_in),
                })

    return (pd.DataFrame(summary_rows), pd.DataFrame(mismatch_rows),
            pd.DataFrame(missing_rows), pd.DataFrame(skipped_rows))


def main(root_dir=ROOT_DIR, output_xlsx=OUTPUT_XLSX):
    if not os.path.isdir(root_dir):
        sys.exit(f'Not a valid directory: {root_dir}')

    print(f'Scanning {root_dir} ...')
    days = find_sessions(root_dir)
    summary_df, mismatch_df, missing_df, skipped_df = check_days(days)

    if len(skipped_df):
        print(f'\nSkipped {len(skipped_df)} folder(s) (not compared):')
        for _, r in skipped_df.iterrows():
            print(f"   {r['folder']}  ({r['reason']})")

    n_bad = int((~summary_df['consistent']).sum()) if len(summary_df) else 0
    print(f'\nChecked {len(summary_df)} recording day(s); {n_bad} with differing .ntt files.')

    if n_bad == 0:
        print('All sessions within each day have the same .ntt files. No report written.')
        return

    for _, r in mismatch_df.iterrows():
        print(f"\n{r['session_folder']}  [{r['arena_type']} / {r['session_type']}]")
        print(f"   {r['n_ntt_present']}/{r['n_ntt_expected']} .ntt present; missing: {r['missing_files']}")

    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        mismatch_df.to_excel(writer, sheet_name='Mismatched', index=False)
        missing_df.to_excel(writer, sheet_name='MissingFiles', index=False)
        skipped_df.to_excel(writer, sheet_name='SkippedFolders', index=False)
    print(f'\nReport saved: {output_xlsx}')


if __name__ == '__main__':
    main()
