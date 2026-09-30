# -*- coding: utf-8 -*-
"""
Copy the recorded data into a second folder tree grouped by SESSION type
instead of by recording day. The source data are left untouched.

Source layout (current):
    SRC_ROOT \\ Animal ID \\ Arena type \\ Recording day \\ Session \\ <neural + tracking files>
    e.g.  Fa1059\\Linear\\Day10\\3_180\\TT1_SS_01.ntt, ..._cm.csv

Destination layout (new):
    DST_ROOT \\ Arena type \\ Session type \\ <Animal>_<Day>_<Session> \\ <same files>
    e.g.  Linear\\180\\Fa1059_Day10_3_180\\TT1_SS_01.ntt, ..._cm.csv

The session folder name carries the animal ID first, the recording day in the
middle and the original session folder name (which ends in the session type)
last, so the paths still parse in SessionType_StatsComparison /
CellClusteredStats_Utils (animal and day are matched by name, the session type
from the last folder).

A session folder is any folder that contains at least one .ntt file. Animal,
arena, day and session type are recognized by name wherever they sit in the
path. Folders where any of them can't be recognized are skipped and listed.

Parameters to edit (below):
    SRC_ROOT        = root of the current (animal-first) data tree
    DST_ROOT        = root of the new (session-type-first) tree; must not be
                      inside SRC_ROOT
    DRY_RUN         = True: only print what would be copied. Set to False to copy.
    COPY_EXTENSIONS = None copies every file of the session folder (including
                      sub-folders); or a tuple of extensions to copy only those
    OVERWRITE       = False: files already present at the destination with the
                      same size are skipped (so the script can be re-run)
"""

import csv
import os
import re
import shutil

# ── Parameters ──────────────────────────────────────────────────────────────

SRC_ROOT = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\PC_True\RecDaySorted'
DST_ROOT = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\PC_True\SessionTypeSorted'

DRY_RUN         = True
COPY_EXTENSIONS = None   # e.g. ('.ntt', '.csv', '.xlsx', '.nvt')
OVERWRITE       = True

ARENA_TYPES = ['Open', 'Linear', 'Circle']
ARENA_SESSION_TYPES = {
    'Open':   ['Cntrl', 'Rotate', 'Zero'],
    'Linear': ['0', '90', '180', '270'],
    'Circle': ['Stnd', 'NoRot', 'Rot'],
}

# Same patterns as CellClusteredStats_Utils.py.
ANIMAL_PATTERN = r'^Fa[0-9A-Za-z]+$'
DAY_PATTERN    = r'^(?:Exp)?Day\d+$'

MANIFEST_NAME = 'copy_manifest.csv'

FAILED = []   # (source, size, error) of files that could not be copied

# ── Path parsing ────────────────────────────────────────────────────────────

def _path_token(rel_path: str, pattern: str):
    for tok in re.split(r'[\\/_]+', rel_path):
        if re.match(pattern, tok.strip(), flags=re.IGNORECASE):
            return tok.strip()
    return None


def extract_arena_type(rel_path: str):
    for part in re.split(r'[\\/]+', rel_path):
        for arena in ARENA_TYPES:
            if part.strip().lower() == arena.lower():
                return arena
    return None


def extract_session_type(session_folder: str, arena: str):
    """Session type from the session folder's name, as in
    SessionType_StatsComparison_v6.extract_session_type."""
    label = session_folder.strip().lower()
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


# ── Copying ─────────────────────────────────────────────────────────────────

def _wanted(filename: str) -> bool:
    return COPY_EXTENSIONS is None or filename.lower().endswith(
        tuple(e.lower() for e in COPY_EXTENSIONS))


def _copy_file(src: str, dst: str) -> str:
    """Copy one file; returns 'copied', 'skipped' or 'would copy'."""
    if (not OVERWRITE and os.path.exists(dst)
            and os.path.getsize(dst) == os.path.getsize(src)):
        return 'skipped'
    if DRY_RUN:
        return 'would copy'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except OSError as e:
        # Don't leave a partial file behind (it would look half-copied).
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        size_gb = os.path.getsize(src) / 1024**3
        FAILED.append((src, f'{size_gb:.2f} GB', str(e)))
        return 'failed'
    return 'copied'


def find_session_folders(root: str):
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith('.ntt') for f in filenames):
            yield dirpath


def main():
    src_root = os.path.abspath(SRC_ROOT)
    dst_root = os.path.abspath(DST_ROOT)
    if os.path.normcase(dst_root).startswith(os.path.normcase(src_root) + os.sep) \
            or os.path.normcase(dst_root) == os.path.normcase(src_root):
        raise ValueError('DST_ROOT must not be SRC_ROOT or inside it.')
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f'SRC_ROOT not found: {src_root}')

    print(f"{'DRY RUN -- nothing is copied' if DRY_RUN else 'COPYING'}\n"
          f'  from: {src_root}\n  to:   {dst_root}\n')

    sessions = sorted(find_session_folders(src_root))
    rows, skipped_sessions, dst_seen = [], [], {}
    counts = {'copied': 0, 'skipped': 0, 'would copy': 0, 'failed': 0}

    for sess_dir in sessions:
        rel = os.path.relpath(sess_dir, src_root)
        sess_name = os.path.basename(sess_dir)
        animal  = _path_token(rel, ANIMAL_PATTERN)
        arena   = extract_arena_type(rel)
        day     = _path_token(rel, DAY_PATTERN)
        stype   = extract_session_type(sess_name, arena) if arena else None

        missing = [n for n, v in (('animal', animal), ('arena', arena),
                                  ('day', day), ('session type', stype)) if v is None]
        if missing:
            skipped_sessions.append((rel, ', '.join(missing)))
            continue

        new_name = f'{animal}_{day}_{sess_name}'
        dst_dir = os.path.join(dst_root, arena, stype, new_name)
        if dst_dir in dst_seen:
            skipped_sessions.append((rel, f'same destination as {dst_seen[dst_dir]}'))
            continue
        dst_seen[dst_dir] = rel

        n_files = 0
        for dirpath, _, filenames in os.walk(sess_dir):
            for f in sorted(filenames):
                if not _wanted(f):
                    continue
                src = os.path.join(dirpath, f)
                dst = os.path.join(dst_dir, os.path.relpath(src, sess_dir))
                status = _copy_file(src, dst)
                counts[status] += 1
                n_files += 1
                rows.append({'animal': animal, 'arena': arena, 'day': day,
                             'session_type': stype, 'source': src,
                             'destination': dst, 'status': status})
        print(f'  {rel}  ->  {os.path.relpath(dst_dir, dst_root)}  ({n_files} files)')

    print(f'\n{len(sessions)} session folder(s) found, '
          f'{len(sessions) - len(skipped_sessions)} mapped.')
    print(f"Files: {counts['copied']} copied, {counts['would copy']} would be copied, "
          f"{counts['skipped']} already present (skipped), {counts['failed']} failed.")

    if FAILED:
        print(f'\nWARNING: {len(FAILED)} file(s) could NOT be copied:')
        for src, size, err in FAILED:
            print(f'    {src}  ({size})  {err}')
        print('  Errno 22 on a file > 4 GB usually means the destination drive is '
              'FAT32 (4 GB file limit) -- use an NTFS/exFAT drive.')

    if skipped_sessions:
        print(f'\nWARNING: {len(skipped_sessions)} session folder(s) NOT copied:')
        for rel, why in skipped_sessions:
            print(f'    {rel}   (unrecognized/conflict: {why})')

    if not DRY_RUN and rows:
        os.makedirs(dst_root, exist_ok=True)
        manifest = os.path.join(dst_root, MANIFEST_NAME)
        with open(manifest, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nManifest written: {manifest}')


if __name__ == '__main__':
    main()
