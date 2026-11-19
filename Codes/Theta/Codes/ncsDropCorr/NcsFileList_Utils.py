# -*- coding: utf-8 -*-
"""
The .ncs files an LFP analysis should run on (copied LFP files left out).

Some recording folders hold bit-identical copies of one LFP signal under several
.ncs names (placed there so that every tetrode had a matched LFP file, e.g.
CSC1 = CSC4, CSC2 = CSC3, CSC5 = CSC6 = CSC7 = CSC8). Make_NcsFileList.py finds
the copies ONCE per data tree and writes two CSVs (default <root>/NcsFileList/):

  Ncs_files_to_analyse.csv : one row per distinct signal = the first file
                             (natural name order) of each set of copies.
                             THIS is the list the analysis scripts read.
  Ncs_files_all.csv        : every .ncs file found, with use_for_analysis,
                             copy_of and the signature that was compared.

Analysis scripts no longer walk the data tree or compare files themselves:

    from NcsFileList_Utils import listed_ncs_files
    ncs_files = listed_ncs_files(NCS_FILE_LIST_CSV, under=root_dir)

returns the absolute paths of the listed files under `root_dir` (folder, then
natural name order). It raises if a listed file is missing or its size changed
-- then re-run Make_NcsFileList.py. New .ncs files added to the tree after the
list was made are NOT seen until the list is re-made.

Signature of a file (same definition as find_duplicate_ncs in
ThetaVsSpeed_v8_Ferret_Mobility.py / ThetaVsAcc.py): SHA-1 of the header's
ADBitVolts and of every record's timestamp, valid-sample count and samples.
The rest of the header text and the channel-number / sampling-rate fields are
left out, so a copy saved under another channel name still matches.
"""

import os
import re
import sys
import hashlib

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from ncs_repair import ncs_dtype, NCS_HEADER_BYTES  # noqa: E402

FILE_LIST_SUBDIR = 'NcsFileList'
TO_ANALYSE_CSV_NAME = 'Ncs_files_to_analyse.csv'
ALL_FILES_CSV_NAME = 'Ncs_files_all.csv'

# Records read per chunk when hashing (1044 bytes each -> ~4 MB).
HASH_CHUNK_RECORDS = 4096
# 'sampled' check: records hashed at each of the start / middle / end of a file.
SAMPLED_RECORDS = 1024


# %% ==================== Naming / paths ==========================================

def natural_key(path):
    """'CSC10.ncs' after 'CSC2.ncs' (numbers compared as numbers)."""
    return tuple((0, int(t), '') if t.isdigit() else (1, 0, t.lower())
                 for t in re.split(r'(\d+)', os.path.basename(str(path))) if t)


def norm_path(path):
    """Case- and separator-normalised absolute path, for matching."""
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def default_list_dir(root_dir):
    return os.path.join(root_dir, FILE_LIST_SUBDIR)


# %% ==================== Signatures ==============================================

def _adbitvolts(header_bytes):
    m = re.search(rb'-ADBitVolts\s+(\S+)', header_bytes)
    return repr(float(m.group(1))) if m else 'none'


def _hash_records(h, buf):
    """Add the timestamp / n_valid / samples fields of the whole records in
    `buf` to hash `h`; trailing bytes of an incomplete record are hashed raw."""
    n = len(buf) // ncs_dtype.itemsize
    if n:
        rec = np.frombuffer(buf, dtype=ncs_dtype, count=n)
        for field in ('timestamp', 'n_valid', 'samples'):
            h.update(np.ascontiguousarray(rec[field]).tobytes())
    tail = buf[n * ncs_dtype.itemsize:]
    if tail:
        h.update(tail)


def ncs_signal_signature(fpath, mode='full'):
    """Signature of one .ncs file (see module docstring).

    mode='full'    : every record is read (exact; reads the whole file).
    mode='sampled' : only SAMPLED_RECORDS records at the start, middle and end
                     of the file plus the record count (fast; two different
                     electrodes cannot plausibly agree on ~3 MB of samples, but
                     this is not a byte-by-byte proof)."""
    h = hashlib.sha1()
    rec_size = ncs_dtype.itemsize
    with open(fpath, 'rb') as f:
        h.update(_adbitvolts(f.read(NCS_HEADER_BYTES)).encode())
        if mode == 'full':
            chunk = HASH_CHUNK_RECORDS * rec_size
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                _hash_records(h, buf)
        elif mode == 'sampled':
            data_bytes = os.fstat(f.fileno()).st_size - NCS_HEADER_BYTES
            n_rec = max(data_bytes, 0) // rec_size
            h.update(f'n_records={n_rec};tail={data_bytes - n_rec * rec_size}'.encode())
            starts = sorted({0, max(n_rec // 2 - SAMPLED_RECORDS // 2, 0),
                             max(n_rec - SAMPLED_RECORDS, 0)})
            for s in starts:
                f.seek(NCS_HEADER_BYTES + s * rec_size)
                _hash_records(h, f.read(min(SAMPLED_RECORDS, n_rec - s) * rec_size))
        else:
            raise ValueError(f"mode must be 'full' or 'sampled', not {mode!r}")
    return h.hexdigest()


# %% ==================== Reading the list ========================================

def _as_list(csv_paths):
    if csv_paths is None:
        raise ValueError('No .ncs file list given (NCS_FILE_LIST_CSV is None).')
    return [csv_paths] if isinstance(csv_paths, (str, os.PathLike)) else list(csv_paths)


def _not_found_msg(csv_path):
    return (f'.ncs file list not found: {csv_path}\n'
            f'Make it first: run Make_NcsFileList.py with ROOT_DIR set to the data '
            f'tree (it writes {FILE_LIST_SUBDIR}\\{TO_ANALYSE_CSV_NAME} there).')


def read_file_list(csv_paths):
    """The to-analyse table(s), concatenated, with an 'abs_path' column
    (root_dir + rel_path) and a normalised 'key' column."""
    tables = []
    for p in _as_list(csv_paths):
        if not os.path.isfile(p):
            raise FileNotFoundError(_not_found_msg(p))
        t = pd.read_csv(p, dtype={'rel_path': str, 'root_dir': str, 'folder': str})
        t['list_csv'] = str(p)
        tables.append(t)
    t = pd.concat(tables, ignore_index=True)
    t['abs_path'] = [os.path.normpath(os.path.join(r, rp)) for r, rp in zip(t['root_dir'], t['rel_path'])]
    t['key'] = t['abs_path'].map(norm_path)
    return t


def read_all_files_table(csv_paths):
    """Ncs_files_all.csv next to each given to-analyse CSV (every file, copies
    included), concatenated; empty if none exists."""
    tables = []
    for p in _as_list(csv_paths):
        a = os.path.join(os.path.dirname(str(p)), ALL_FILES_CSV_NAME)
        if os.path.isfile(a):
            tables.append(pd.read_csv(a, dtype={'rel_path': str, 'root_dir': str, 'folder': str,
                                                'copy_of': str}))
    if not tables:
        return pd.DataFrame()
    t = pd.concat(tables, ignore_index=True)
    t['abs_path'] = [os.path.normpath(os.path.join(r, rp)) for r, rp in zip(t['root_dir'], t['rel_path'])]
    t['key'] = t['abs_path'].map(norm_path)
    return t


def _under(keys, folder):
    if folder is None:
        return np.ones(len(keys), dtype=bool)
    f = norm_path(folder).rstrip(os.sep) + os.sep
    return np.array([k.startswith(f) for k in keys], dtype=bool)


def listed_ncs_files(csv_paths, under=None, check_files=True, verbose=True):
    """Absolute paths of the .ncs files to analyse, from the list(s) made by
    Make_NcsFileList.py -- no directory walk, no file comparison.

    csv_paths   : path of Ncs_files_to_analyse.csv, or a list of them (several
                  data trees).
    under       : keep only listed files inside this folder (e.g. one animal's
                  folder). None: every listed file.
    check_files : raise if a listed file is missing or its size differs from
                  the size recorded in the list (the list is then out of date).
    Returns the paths sorted by folder, then natural file-name order."""
    t = read_file_list(csv_paths)
    t = t[_under(t['key'].tolist(), under)]
    if under is not None and t.empty:
        raise ValueError(f'No listed .ncs file under {under}\n(list: {_as_list(csv_paths)}). '
                         f'Is the list made for this data tree? Run Make_NcsFileList.py on it.')
    paths = sorted(t['abs_path'], key=lambda p: (os.path.dirname(p).lower(), natural_key(p)))
    if check_files:
        size = dict(zip(t['abs_path'], t['size_bytes']))
        missing, changed = [], []
        for p in paths:
            try:
                if os.path.getsize(p) != int(size[p]):
                    changed.append(p)
            except OSError:
                missing.append(p)
        if missing or changed:
            lines = ([f'  missing: {p}' for p in missing[:10]]
                     + [f'  size changed: {p}' for p in changed[:10]])
            raise RuntimeError(
                f'The .ncs file list is out of date ({len(missing)} missing, {len(changed)} '
                f'changed files):\n' + '\n'.join(lines)
                + '\nRe-run Make_NcsFileList.py for this data tree.')
    if verbose:
        n_copies = ''
        allt = read_all_files_table(csv_paths)
        if not allt.empty:
            allt = allt[_under(allt['key'].tolist(), under)]
            n_copies = f' ({int((~allt["use_for_analysis"].astype(bool)).sum())} copied files left out)'
        print(f'{len(paths)} .ncs files to analyse from the file list'
              f'{" under " + str(under) if under is not None else ""}{n_copies}.')
    return paths


def listed_copies(csv_paths, under=None):
    """Table of the copied .ncs files the list leaves out (Recording, file_name,
    copy_of, abs_path, copy_of_path), from Ncs_files_all.csv."""
    allt = read_all_files_table(csv_paths)
    if allt.empty:
        return pd.DataFrame(columns=['Recording', 'file_name', 'copy_of', 'abs_path', 'copy_of_path'])
    allt = allt[_under(allt['key'].tolist(), under)]
    c = allt[~allt['use_for_analysis'].astype(bool)]
    return pd.DataFrame({
        'Recording':    c['folder'].str.replace('\\', '/', regex=False).values,
        'file_name':    c['file_name'].values,
        'copy_of':      c['copy_of'].values,
        'abs_path':     c['abs_path'].values,
        'copy_of_path': [os.path.normpath(os.path.join(os.path.dirname(p), n))
                         for p, n in zip(c['abs_path'], c['copy_of'])],
    }).reset_index(drop=True)


def split_by_list(files, csv_paths):
    """For .ncs paths that were analysed already (e.g. the 'file' column of a
    results table): (set of paths to keep, table of the dropped copies).
    Raises if a path is neither listed nor a known copy (list out of date, or
    made for another data tree)."""
    keep_keys = set(read_file_list(csv_paths)['key'])
    allt = read_all_files_table(csv_paths)
    copy_of = {} if allt.empty else dict(
        zip(allt.loc[~allt['use_for_analysis'].astype(bool), 'key'],
            allt.loc[~allt['use_for_analysis'].astype(bool), 'copy_of']))
    keep, dropped, unknown = set(), [], []
    for f in pd.unique(pd.Series(list(files), dtype=object)):
        k = norm_path(f)
        if k in keep_keys:
            keep.add(f)
        elif k in copy_of:
            dropped.append({'file': f, 'file_name': os.path.basename(str(f)),
                            'copy_of': copy_of[k],
                            'kept_file': os.path.join(os.path.dirname(str(f)), copy_of[k])})
        else:
            unknown.append(f)
    if unknown:
        raise ValueError(
            f'{len(unknown)} .ncs file(s) are not in the file list {_as_list(csv_paths)}, e.g.\n  '
            + '\n  '.join(map(str, unknown[:10]))
            + '\nRe-run Make_NcsFileList.py for this data tree, or point NCS_FILE_LIST_CSV '
              'at the list of the tree these files come from.')
    return keep, pd.DataFrame(dropped, columns=['file', 'file_name', 'copy_of', 'kept_file'])
