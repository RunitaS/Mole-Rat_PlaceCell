"""
Copy folders (with all their contents) listed in column I of a Google Sheet
into a destination directory, renaming each copy using the 3 path components
that precede the folder itself.

Example:
    Source folder:
        Z:\\NMR_group_data\\_Scratch_folder\\AdrianLabelledData\\SingleSessions\\
        1680378B\\Day1_CntrlCntrlCntrlZero_Shank1BankA\\Cntrl1\\
        2024-07-28_18-36-45_20_KS4-ADR-curated

    Copied to:
        <DEST_DIR>\\1680378B_Day1_CntrlCntrlCntrlZero_Shank1BankA_Cntrl1

Requires the Google Sheet to be shared as "Anyone with the link can view"
(the CSV export endpoint is used, so no Google auth/API key is needed).
If the sheet is private, share it or switch to an authenticated method
(e.g. gspread with a service account).
"""

import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHEET_ID = "1Vtzaw2A1nBL43KXtJBgeFVLy2L5iO7pIvgScwa9L0Vc"
GID = "0"  # tab (gid) that contains column I with the folder paths; change if needed
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Column I = 9th column -> zero-based index 8
FOLDER_PATH_COLUMN_INDEX = 8

# Where the renamed copies should go. <-- EDIT THIS
DEST_DIR = Path(r"Z:\path\to\destination")


def derive_label(folder: Path) -> str:
    """3 path components immediately before the folder itself, joined with '_'."""
    parts = folder.parts
    if len(parts) < 4:
        raise ValueError(f"path too short to derive a label: {folder}")
    return "_".join(parts[-4:-1])


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_URL, header=None, dtype=str)
    raw_paths = df.iloc[:, FOLDER_PATH_COLUMN_INDEX].dropna()

    for raw_path in raw_paths:
        raw_path = raw_path.strip()
        if not raw_path or raw_path.lower() in ("folder path", "path"):
            continue  # skip blanks / header text

        src = Path(raw_path)
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        if not src.is_dir():
            print(f"SKIP (not a folder): {src}")
            continue

        try:
            label = derive_label(src)
        except ValueError as e:
            print(f"SKIP ({e})")
            continue

        dest = DEST_DIR / label
        if dest.exists():
            print(f"SKIP (destination already exists): {dest}")
            continue

        print(f"Copying:\n  {src}\n  -> {dest}")
        try:
            shutil.copytree(src, dest)
        except Exception as e:
            print(f"ERROR copying {src}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
