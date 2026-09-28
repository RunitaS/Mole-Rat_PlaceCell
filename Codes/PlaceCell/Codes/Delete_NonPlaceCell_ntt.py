# -*- coding: utf-8 -*-
"""
Delete the .ntt files of every unit classified as NOT a place cell.

Reads the 'Full' sheet of the PlaceCell_Main_v10.py output workbook. Each row
gives:
    column A  'session'    – session folder, relative to the root_folder the
                             analysis was run on (os.path.relpath, '.' = root)
    column B  'unit'       – .ntt filename
    'place_cell'           – final place-cell verdict

For every row with place_cell == False, the file
    <data_root>/<session>/<unit>
is deleted. Rows with place_cell == True or blank (None/NaN, i.e. the cell
errored out and was never classified) are left untouched.

Parameters to edit:
input_excel – PlaceCell_Main_v10.py results workbook
data_root   – root directory whose sub-folders match column A exactly
DRY_RUN     – True: only list what would be deleted. Set to False to delete.
"""

import os
from datetime import datetime

import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

input_excel = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\SessionType_Sorted\CorrectedData\All_TT_PlaceChar_VisitCrit.xlsx'
data_root   = r'X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\SessionType_Sorted\CorrectedData'

SHEET_NAME = 'Full'   # place_cell verdict (incl. field-detection overrides) lives here
DRY_RUN    = True     # True: report only, nothing is deleted


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_false(val) -> bool:
    """True only for an explicit False verdict (bool, 0/1 or 'FALSE' text).
    Blank / NaN / None (unclassified) is NOT treated as False."""
    if isinstance(val, bool):
        return val is False
    if isinstance(val, str):
        return val.strip().upper() == 'FALSE'
    if pd.isna(val):
        return False
    try:
        return float(val) == 0.0
    except (TypeError, ValueError):
        return False


def _resolve_ntt_path(session: str, unit: str) -> str:
    session = str(session).strip()
    dirpath = data_root if session in ('.', '') else os.path.join(data_root, session)
    return os.path.normpath(os.path.join(dirpath, str(unit).strip()))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = pd.read_excel(input_excel, sheet_name=SHEET_NAME)
    session_col, unit_col = df.columns[0], df.columns[1]   # column A, column B
    if 'place_cell' not in df.columns:
        raise KeyError(f"'place_cell' column not found in sheet '{SHEET_NAME}' of {input_excel}")

    root_norm = os.path.normcase(os.path.normpath(data_root))
    to_delete = df[df['place_cell'].apply(_is_false)]

    print(f'{"DRY RUN – " if DRY_RUN else ""}Units in sheet: {len(df)}  |  '
          f'place_cell == False: {len(to_delete)}\n')

    log_rows = []
    n_deleted = n_missing = n_skipped = 0

    for _, row in to_delete.iterrows():
        session, unit = row[session_col], row[unit_col]
        ntt_path = _resolve_ntt_path(session, unit)

        # Safety: only ever touch .ntt files that sit inside data_root.
        if (not ntt_path.lower().endswith('.ntt')
                or not os.path.normcase(ntt_path).startswith(root_norm + os.sep)):
            status = 'skipped (not an .ntt inside data_root)'
            n_skipped += 1
        elif not os.path.isfile(ntt_path):
            status = 'not found'
            n_missing += 1
        elif DRY_RUN:
            status = 'would delete'
            n_deleted += 1
        else:
            try:
                os.remove(ntt_path)
                status = 'deleted'
                n_deleted += 1
            except OSError as e:
                status = f'error: {e}'
                n_skipped += 1

        print(f'  [{status.upper()}] {ntt_path}')
        log_rows.append({'session': session, 'unit': unit, 'path': ntt_path, 'status': status})

    log_path = os.path.splitext(input_excel)[0] + (
        f'_ntt_deletion_{"dryrun_" if DRY_RUN else ""}{datetime.now():%Y%m%d_%H%M%S}.csv')
    pd.DataFrame(log_rows, columns=['session', 'unit', 'path', 'status']).to_csv(log_path, index=False)

    print(f'\n{"Would delete" if DRY_RUN else "Deleted"} : {n_deleted}')
    print(f'Not found    : {n_missing}')
    print(f'Skipped/error: {n_skipped}')
    print(f'Log saved to : {log_path}')
    if DRY_RUN:
        print('\nDRY_RUN is True – nothing was deleted. Set DRY_RUN = False to delete.')
