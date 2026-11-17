"""
Standalone re-plot of the "both significant" figure from
ThetaVsSpeed_v8_Ferret_Mobility.py, without re-running the LFP pipeline.

Reads the per-file classes that ThetaVsSpeed_v8_Ferret_Mobility.py saved in
<OUTPUT_DIR>/IndividualCorrel_per_file.xlsx (sheet per_file: one row per file
and measure, with beta1, BH-FDR q-value and Category), keeps only the files
whose slope is significant for both theta frequency and relative theta power,
and counts them by the signs of the two slopes:
  * Both positive  -- red  (the "Positive, significant" colour)
  * Both negative  -- blue (the "Negative, significant" colour)
  * Opposite signs -- violet, the 50/50 RGB mix of that red and blue
overall and per animal.

Output (same as the main script, overwritten):
  <OUTPUT_DIR>/IndividualCorrel_BothSignificant.png
  <OUTPUT_DIR>/IndividualCorrel_BothSignificant.xlsx  (per_file, counts)

The plotting/counting code is imported from ThetaVsSpeed_v8_Ferret_Mobility.py
(save_both_significant), so both scripts always make the identical figure.
Run the main script first if the per-file workbook does not exist yet.
"""

import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import ThetaVsSpeed_v8_Ferret_Mobility as tvs

# Results folder of the main script (Mobility_v3); change to re-plot another run.
OUTPUT_DIR = tvs.OUTPUT_DIR
PER_FILE_XLSX = Path(OUTPUT_DIR) / "IndividualCorrel_per_file.xlsx"


if __name__ == "__main__":
    if not PER_FILE_XLSX.is_file():
        sys.exit(f"{PER_FILE_XLSX} not found -- run ThetaVsSpeed_v8_Ferret_Mobility.py first.")
    cls_long = pd.read_excel(PER_FILE_XLSX, sheet_name="per_file")
    print(f"Read {cls_long['Session'].nunique()} files from {PER_FILE_XLSX}")
    tvs.save_both_significant(cls_long, OUTPUT_DIR)
    print(f"Saved -> {Path(OUTPUT_DIR) / 'IndividualCorrel_BothSignificant.png'}")
