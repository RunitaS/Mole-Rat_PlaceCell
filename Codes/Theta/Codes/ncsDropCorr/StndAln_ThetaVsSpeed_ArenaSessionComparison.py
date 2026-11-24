"""
STANDALONE (temporary -- delete once ThetaVsSpeed_v8_Ferret_Mobility.py has been
re-run in full).

Runs only the arena / session-type comparisons of ThetaVsSpeed_v8_Ferret_Mobility.py
(run_arena_session_comparisons) on the per-time-bin rows already saved by the
last full run (ThetaVsSpeed_Mobility_out.parquet), so no .ncs file is re-read:

  * SessionType/<arena>        session types within each arena
  * ArenaType                  Circle / Linear / Open
  * AnimalControlledArenaType  Circle vs Open (Fa8477, Fa5834) | Linear vs Open
                               (Fa1059, Fa23BD), side by side

Each folder (under Mobility_v3/Comparisons) holds the mixed-model figures, the
per-recording slope figures, the slope-class %, the proportion of speed-modulated
files and the both-significant figure, with the group tests;
every table -> Comparisons/ThetaVsSpeed_Comparison_Stats.xlsx (+ _AllStats.csv).

All code lives in ThetaVsSpeed_v8_Ferret_Mobility.py (imported, its pipeline is
behind __main__), so this script and a full run give the same Comparisons output.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd  # noqa: E402

HERE = Path(r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr')
try:
    HERE = Path(__file__).resolve().parent
except NameError:      # run cell-by-cell
    pass
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ThetaVsSpeed_v8_Ferret_Mobility as tvs   # noqa: E402

# None: tvs.COMPARISON_DIR (OUTPUT_DIR/Comparisons); set a path to write elsewhere.
OUT_DIR = None


if __name__ == "__main__":
    parquet = Path(tvs.OUTPUT_PARQUET)
    if not parquet.is_file():
        raise FileNotFoundError(f'{parquet} not found -- run ThetaVsSpeed_v8_Ferret_Mobility.py in full first.')
    df = pd.read_parquet(parquet)
    needed = {"Session", "Recording", "Animal", "Mobile", "Speed",
              "Frequency", "Power", "PowerdB", "PowerRel"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{parquet.name} lacks columns {sorted(missing)} -- '
                         f're-run ThetaVsSpeed_v8_Ferret_Mobility.py in full.')
    print(f'Loaded {len(df)} time bins, {df["Session"].nunique()} files from {parquet}')

    tvs.run_arena_session_comparisons(df, out_dir=OUT_DIR)
    print(f'\nDone -> {OUT_DIR or tvs.COMPARISON_DIR}')
