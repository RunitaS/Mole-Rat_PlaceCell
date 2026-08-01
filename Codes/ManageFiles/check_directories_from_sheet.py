"""
Check whether each directory listed in column I of a Google Sheet actually
exists on disk, and report which ones are missing.

Requires the Google Sheet to be shared as "Anyone with the link can view"
(the CSV export endpoint is used, so no Google auth/API key is needed).
If the sheet is private, share it or switch to an authenticated method
(e.g. gspread with a service account).
"""

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

# Optional: write a CSV report of the results here. Set to None to skip.
REPORT_PATH = Path(r"check_directories_report.csv")


def main() -> None:
    df = pd.read_csv(CSV_URL, header=None, dtype=str)
    raw_paths = df.iloc[:, FOLDER_PATH_COLUMN_INDEX].dropna()

    results = []
    for row_idx, raw_path in raw_paths.items():
        raw_path = raw_path.strip()
        if not raw_path or raw_path.lower() in ("folder path", "path"):
            continue  # skip blanks / header text

        path = Path(raw_path)
        exists = path.is_dir()
        results.append({"row": row_idx + 1, "path": raw_path, "exists": exists})
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {raw_path}")

    total = len(results)
    missing = [r for r in results if not r["exists"]]

    print("\n--- Summary ---")
    print(f"Checked: {total}")
    print(f"Missing: {len(missing)}")
    for r in missing:
        print(f"  row {r['row']}: {r['path']}")

    if REPORT_PATH is not None and results:
        pd.DataFrame(results).to_csv(REPORT_PATH, index=False)
        print(f"\nReport written to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
