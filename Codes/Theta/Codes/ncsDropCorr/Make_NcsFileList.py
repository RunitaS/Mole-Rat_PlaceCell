# -*- coding: utf-8 -*-
"""
Make the list of .ncs files an LFP analysis should run on: every .ncs file
under ROOT_DIR, with copied files (bit-identical copies of one LFP signal under
another channel name, see NcsFileList_Utils.py) left out. Run this ONCE per data
tree (and again whenever .ncs files are added, removed or changed); the analysis
scripts then read the list (NCS_FILE_LIST_CSV) instead of walking the tree and
comparing files on every run.

Output (OUTPUT_DIR, default ROOT_DIR/NcsFileList):
  Ncs_files_to_analyse.csv : the files to analyse -- the first file (natural name
                             order, e.g. CSC1 of CSC1 = CSC4) of each set of
                             copies; n_copies / copies name the files it stands for.
  Ncs_files_all.csv        : every .ncs file: use_for_analysis, copy_of,
                             signature, size, modification time, read errors.

How copies are found (quickly)
  * Only files in the SAME folder are compared (copies were placed next to the
    original in the session folder).
  * A file whose byte size no other file in its folder shares cannot be a copy
    and is never read.
  * Same-size files are compared by signature (NcsFileList_Utils.
    ncs_signal_signature): CHECK_MODE 'full' hashes every record (exact),
    'sampled' only ~1 MB at the start, middle and end of each file (much faster;
    practically certain, but not byte-by-byte).
  * Files are read in N_WORKERS parallel threads (network drive throughput).
  * Re-runs reuse the signature of every file whose size and modification time
    are unchanged since the previous Ncs_files_all.csv (same CHECK_MODE), so only
    new or changed files are read again.

Usage: set ROOT_DIR below and run (cell by cell or as a script), or
    python Make_NcsFileList.py "<root dir>" [--out <dir>] [--workers N] [--mode full|sampled]
"""

import os
import sys
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else \
    r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import NcsFileList_Utils as nfl  # noqa: E402


# %% ==================== Configuration ===========================================

ROOT_DIR = r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final'

# Where the two CSVs go. None: ROOT_DIR/NcsFileList.
OUTPUT_DIR = None

# 'full' (exact, reads every same-size file completely) or 'sampled' (fast).
CHECK_MODE = 'full'

# Parallel file reads.
N_WORKERS = 6

# Folders (by name, at any depth) not searched for .ncs files: analysis outputs.
SKIP_DIR_NAMES = ('Results', nfl.FILE_LIST_SUBDIR)


# %% ==================== Steps ===================================================

def scan_ncs(root_dir, skip_dir_names=SKIP_DIR_NAMES):
    """Every .ncs under root_dir (os.scandir walk: size and mtime come with the
    directory listing, no extra request per file)."""
    skip = {s.lower() for s in skip_dir_names}
    rows, stack, n_dirs = [], [root_dir], 0
    t0 = time.time()
    while stack:
        d = stack.pop()
        n_dirs += 1
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.lower() not in skip:
                            stack.append(e.path)
                    elif e.name.lower().endswith('.ncs') and e.is_file():
                        st = e.stat()
                        rel = os.path.relpath(e.path, root_dir)
                        rows.append({'rel_path': rel, 'folder': os.path.dirname(rel),
                                     'file_name': e.name, 'size_bytes': st.st_size,
                                     'mtime_ns': st.st_mtime_ns})
        except OSError as err:
            print(f'  cannot list {d}: {err}')
    print(f'Scanned {n_dirs} folders in {time.time() - t0:.0f} s: {len(rows)} .ncs files '
          f'({sum(r["size_bytes"] for r in rows) / 1e9:.1f} GB).')
    return pd.DataFrame(rows, columns=['rel_path', 'folder', 'file_name', 'size_bytes', 'mtime_ns'])


def load_cache(all_csv, root_dir, mode):
    """{rel_path: (size, mtime_ns, signature)} from a previous run's table."""
    if not os.path.isfile(all_csv):
        return {}
    try:
        old = pd.read_csv(all_csv, dtype={'rel_path': str, 'signature': str, 'root_dir': str})
    except Exception as err:  # noqa: BLE001
        print(f'Previous {all_csv} not readable ({err}); every file is checked again.')
        return {}
    old = old[(old['root_dir'].map(nfl.norm_path) == nfl.norm_path(root_dir))
              & (old['check_mode'] == mode) & old['signature'].notna()
              & (old['signature'].astype(str).str.len() == 40)]
    return {r: (int(s), int(m), g) for r, s, m, g in
            zip(old['rel_path'], old['size_bytes'], old['mtime_ns'], old['signature'])}


def compute_signatures(files, root_dir, mode, n_workers, cache):
    """Fill 'signature' for files that share their size with another file of
    the same folder; '' for size-unique files (never read)."""
    files = files.copy()
    files['signature'] = ''
    files['read_error'] = ''
    n_same = files.groupby(['folder', 'size_bytes'])['rel_path'].transform('size')
    need = files.index[n_same > 1]

    todo = []
    for i in need:
        c = cache.get(files.at[i, 'rel_path'])
        if c and c[0] == files.at[i, 'size_bytes'] and c[1] == files.at[i, 'mtime_ns']:
            files.at[i, 'signature'] = c[2]
        else:
            todo.append(i)
    gb = files.loc[todo, 'size_bytes'].sum() / 1e9
    print(f'{len(files) - len(need)} files have a size no other file in their folder has '
          f'(not read); {len(need) - len(todo)} signatures reused from the previous list; '
          f'{len(todo)} files to read ({gb:.1f} GB, CHECK_MODE={mode!r}, {n_workers} threads).')

    t0, done, done_bytes = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(nfl.ncs_signal_signature,
                            os.path.join(root_dir, files.at[i, 'rel_path']), mode): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                files.at[i, 'signature'] = fut.result()
            except Exception as err:  # noqa: BLE001
                files.at[i, 'read_error'] = str(err)
                print(f'  READ ERROR {files.at[i, "rel_path"]}: {err}')
            done += 1
            done_bytes += files.at[i, 'size_bytes']
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print(f'  {done}/{len(todo)} files, {done_bytes / 1e9:.1f} GB in {el:.0f} s '
                      f'({done_bytes / 1e6 / max(el, 1e-9):.0f} MB/s)', flush=True)
    return files


def assign_copies(files):
    """use_for_analysis / copy_of / set sizes. Files with equal (folder,
    signature) are one signal; the first in natural name order is kept.
    Size-unique and unreadable files are their own set."""
    files = files.copy()
    files['_order'] = [(f.lower(), nfl.natural_key(n)) for f, n in zip(files['folder'], files['file_name'])]
    files = files.iloc[sorted(range(len(files)), key=lambda i: files['_order'].iat[i])].drop(columns='_order')
    own = (files['signature'] == '') | (files['read_error'] != '')
    files['_set'] = files['folder'].str.lower() + '|' + files['signature']
    files.loc[own, '_set'] = 'own|' + files.loc[own, 'rel_path']
    first = files.groupby('_set', sort=False)['file_name'].transform('first')
    files['use_for_analysis'] = files['file_name'] == first
    files['copy_of'] = first.where(~files['use_for_analysis'], '')
    files['n_files_in_set'] = files.groupby('_set', sort=False)['rel_path'].transform('size')
    copies = (files[~files['use_for_analysis']].groupby('_set')['file_name']
              .agg(lambda s: ';'.join(s)))
    files['copies'] = files['_set'].map(copies).fillna('').where(files['use_for_analysis'], '')
    return files.drop(columns='_set').reset_index(drop=True)


def _write_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def make_ncs_file_list(root_dir=ROOT_DIR, output_dir=OUTPUT_DIR, mode=CHECK_MODE,
                       n_workers=N_WORKERS):
    root_dir = os.path.normpath(root_dir)
    output_dir = output_dir or nfl.default_list_dir(root_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_csv = os.path.join(output_dir, nfl.ALL_FILES_CSV_NAME)
    use_csv = os.path.join(output_dir, nfl.TO_ANALYSE_CSV_NAME)
    t_start = time.time()
    print(f'Data tree: {root_dir}')

    files = scan_ncs(root_dir)
    if files.empty:
        raise ValueError(f'No .ncs files under {root_dir}')
    files = compute_signatures(files, root_dir, mode, n_workers, load_cache(all_csv, root_dir, mode))
    files = assign_copies(files)

    made = datetime.now().isoformat(sep=' ', timespec='seconds')
    files.insert(0, 'root_dir', root_dir)
    files['n_copies'] = files['n_files_in_set'] - 1
    files['modified'] = pd.to_datetime(files['mtime_ns'], unit='ns').dt.strftime('%Y-%m-%d %H:%M:%S')
    files['check_mode'] = mode
    files['list_made'] = made

    _write_csv(files[['root_dir', 'rel_path', 'folder', 'file_name', 'use_for_analysis', 'copy_of',
                      'n_files_in_set', 'size_bytes', 'modified', 'mtime_ns', 'signature',
                      'check_mode', 'read_error', 'list_made']], all_csv)
    use = files[files['use_for_analysis']]
    _write_csv(use[['root_dir', 'rel_path', 'folder', 'file_name', 'n_copies', 'copies',
                    'size_bytes', 'modified', 'check_mode', 'read_error', 'list_made']], use_csv)

    n_err = int((files['read_error'] != '').sum())
    n_dup_folders = files.loc[~files['use_for_analysis'], 'folder'].nunique()
    print(f'\n{len(files)} .ncs files -> {len(use)} to analyse; {len(files) - len(use)} copies '
          f'in {n_dup_folders} of {files["folder"].nunique()} folders left out.'
          + (f' {n_err} unreadable file(s) kept in the list (see read_error).' if n_err else ''))
    print(f'Written ({time.time() - t_start:.0f} s):\n  {use_csv}\n  {all_csv}')
    return use_csv


# %% ==================== Run =====================================================

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('root_dir', nargs='?', default=ROOT_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--workers', type=int, default=N_WORKERS)
    ap.add_argument('--mode', choices=('full', 'sampled'), default=CHECK_MODE)
    args, _unknown = ap.parse_known_args()   # tolerate IDE-injected arguments
    make_ncs_file_list(args.root_dir, args.out, args.mode, args.workers)
