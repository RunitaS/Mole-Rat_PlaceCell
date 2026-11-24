"""
STANDALONE (temporary -- delete once ThetaVsAcc.py has been re-run in full).

Runs only the acceleration / deceleration split of ThetaVsAcc.py
(run_acc_decel_split) on the per-time-bin rows already saved by the last full
run (ThetaVsAcc_out.parquet), so no .ncs file is re-read:

  * Acc   : 0.25 s time bins with a >= 2 cm/s^2  -> Results/ThetaVsAcc/Acc
  * Decel : 0.25 s time bins with a < -2 cm/s^2  -> Results/ThetaVsAcc/Decel
  (the near-zero bin [-2, 2) cm/s^2 is in neither; predictor = |a| in both)

In each folder: mixed model (Frequency, Power, PowerdB, PowerRel), the
speed-controlled model, the per-session OLS (Fig. 3) with group tests, and the
individual-file fits (one PNG per file + % positive + both-significant), with
every table as .csv and the split settings in Split_settings.csv.

All code lives in ThetaVsAcc.py (imported, its pipeline is behind __main__),
so this script and a full ThetaVsAcc.py run give the same Acc / Decel output.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr')
try:
    HERE = Path(__file__).resolve().parent
except NameError:      # run cell-by-cell
    pass
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ThetaVsAcc as tva   # noqa: E402


if __name__ == "__main__":
    parquet = Path(tva.OUTPUT_PARQUET)
    if not parquet.is_file():
        raise FileNotFoundError(f'{parquet} not found -- run ThetaVsAcc.py in full first.')
    df = pd.read_parquet(parquet)
    needed = {"Session", "Recording", "Animal", "Mobile", "Speed", "Acceleration",
              "Frequency", "Power", "PowerdB", "PowerRel"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{parquet.name} lacks columns {sorted(missing)} -- re-run ThetaVsAcc.py in full.')
    print(f'Loaded {len(df)} time bins, {df["Session"].nunique()} sessions from {parquet}')

    tva.run_acc_decel_split(df, out_root=tva.OUTPUT_DIR, source=str(parquet))
    print(f'\nDone -> {tva.OUTPUT_DIR / "Acc"}  and  {tva.OUTPUT_DIR / "Decel"}')
