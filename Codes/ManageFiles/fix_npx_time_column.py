"""
Recursively goes through every .xlsx and .csv file under a hardcoded root
directory (all folders and subfolders) and, for every sheet (or the single
table in a .csv), checks whether cell A1 already holds "time" or 0/"0".

If neither is present, column A is assumed to be missing its header, so the
script:
    1. Shifts all of column A's data down by one cell.
    2. Writes the text "time" into A1.
    3. Drops the original last entry of column A (so the column length is
       unchanged instead of growing by one row).

Requires: openpyxl  (pip install openpyxl)
"""

import csv
import os
from openpyxl import load_workbook

# ---- hardcode your root directory here ----
ROOT_DIRECTORY = r"X:\NMR_group_data\Runita\AllData_Backup\AllSortedData\Neuropixel"


def needs_time_header(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() != "time" and value.strip() != "0"
    if isinstance(value, (int, float)):
        return value != 0
    return True


def fix_sheet(ws):
    a1 = ws["A1"].value

    if not needs_time_header(a1):
        return False

    max_row = ws.max_row
    col_a_values = [ws.cell(row=r, column=1).value for r in range(1, max_row + 1)]

    # Drop the last entry, then shift everything down by one row.
    shifted_values = col_a_values[:-1]

    ws.cell(row=1, column=1, value="time")
    for i, val in enumerate(shifted_values):
        ws.cell(row=i + 2, column=1, value=val)

    return True


def fix_xlsx(filepath):
    wb = load_workbook(filepath)

    modified = False
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if fix_sheet(ws):
            modified = True
            print(f"Fixed '{sheet_name}' in {filepath}")

    if modified:
        wb.save(filepath)


def fix_csv(filepath):
    # Preserve a UTF-8 BOM and the original line endings if present.
    with open(filepath, "rb") as f:
        raw = f.read()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    line_terminator = "\r\n" if b"\r\n" in raw else "\n"

    with open(filepath, "r", newline="", encoding=encoding) as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    a1 = rows[0][0] if rows[0] else None
    if not needs_time_header(a1):
        return

    col_a_values = [row[0] if row else "" for row in rows]

    # Drop the last entry, then shift everything down by one row.
    shifted_values = ["time"] + col_a_values[:-1]
    for row, val in zip(rows, shifted_values):
        if row:
            row[0] = val
        else:
            row.append(val)

    with open(filepath, "w", newline="", encoding=encoding) as f:
        csv.writer(f, lineterminator=line_terminator).writerows(rows)

    print(f"Fixed {filepath}")


def main():
    for dirpath, _, filenames in os.walk(ROOT_DIRECTORY):
        for filename in filenames:
            # Skip Excel's temporary lock files (e.g. "~$data.xlsx").
            if filename.startswith("~$"):
                continue

            filepath = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == ".xlsx":
                    fix_xlsx(filepath)
                elif ext == ".csv":
                    fix_csv(filepath)
            except Exception as e:
                print(f"ERROR processing {filepath}: {e}")


if __name__ == "__main__":
    main()
