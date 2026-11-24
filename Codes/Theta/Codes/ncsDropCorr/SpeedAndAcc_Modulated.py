"""
Standalone: proportion of LFP files whose theta is modulated by BOTH running
speed and acceleration. Nothing is recomputed from the .ncs data.

Reads the per-file slope classes saved by the two main scripts:
  * ThetaVsSpeed_v8_Ferret_Mobility.py -> <SPEED_DIR>/IndividualCorrel_per_file.xlsx
  * ThetaVsAcc.py                       -> <ACC_DIR>/IndividualCorrel_per_file.xlsx
(sheet per_file: one row per file and measure, with beta1, BH-FDR q-value and
Category). A file is "modulated" by a variable for a measure if its slope class
is "Positive, significant" or "Negative, significant" (BH-FDR q < 0.05 over the
files, per measure, as in the main scripts). Unit = LFP file (one channel of
one recording, 219 files), the same unit as the IndividualCorrel plots.

Each file is put in one group per measure (theta frequency; relative theta power):
  Both (speed AND acceleration) / Speed only / Acceleration only / Neither
and files in "Both" are further split by the signs of their two slopes.

Figure (one row per measure):
  left  -- 100 % stacked bars of the four groups: all files, then each animal
  right -- the "Both" files by sign combination (speed slope, acceleration
           slope), stacked by animal

Output (in ACC_DIR):
  SpeedAndAcc_Modulated.png
  SpeedAndAcc_Modulated.xlsx  (per_file: one row per file and measure;
                               counts: group counts per measure and animal;
                               direction: sign combinations of the Both files)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = r'C:\Runita\NMR\Mole-Rat_PlaceCell\Codes\Theta\Codes\ncsDropCorr'
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import ThetaVsAcc as tva   # shared helpers only; its pipeline is behind __main__

RESULTS = Path(r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt\SpikeQualityFilt'
               r'\SpikeQualityFilt_v1_PlaceCellData\PC_True_SiSkaggsAdptBin_Final\Results')
SPEED_DIR = RESULTS / 'ThetaVsSpeed' / 'Mobility_v3'
ACC_DIR = RESULTS / 'ThetaVsAcc'
OUT_DIR = ACC_DIR
OUT_NAME = 'SpeedAndAcc_Modulated'

MEASURES = [(col, title) for col, title, _ in tva.INDIV_MEASURES]   # Frequency, PowerRel
SIG_SIGN = {"Positive, significant": 1, "Negative, significant": -1}

# (group, colour), stacked from the baseline up. Slots 1-3 of the validated
# categorical palette, then the "Not fitted" gray of the main scripts.
GROUPS = [
    ("Both",              "#2a78d6"),
    ("Speed only",        "#eb6834"),
    ("Acceleration only", "#1baf7a"),
    ("Neither",           "#d9d8d4"),
]
_DARK_GROUPS = {"Both", "Speed only", "Acceleration only"}
# Sign combinations of the Both files, labelled (speed slope, acceleration slope).
DIRECTIONS = ["Speed +, Acc +", "Speed −, Acc +", "Speed +, Acc −", "Speed −, Acc −"]


def load_per_file():
    """One row per file and measure, with speed and acceleration slope, q, class."""
    keys = ["Measure", "Session", "Animal", "Recording"]
    cols = keys + ["beta1", "qvalue", "Category"]
    tabs = {}
    for name, d in (("Speed", SPEED_DIR), ("Acc", ACC_DIR)):
        p = d / "IndividualCorrel_per_file.xlsx"
        if not p.is_file():
            sys.exit(f"{p} not found -- run the main script for {name} first.")
        tabs[name] = pd.read_excel(p, sheet_name="per_file")[cols]
    m = tabs["Speed"].merge(tabs["Acc"], on=keys, how="outer", suffixes=("_Speed", "_Acc"),
                            indicator=True)
    if (m["_merge"] != "both").any():
        miss = m.loc[m["_merge"] != "both", ["Measure", "Session", "_merge"]]
        print(f"WARNING: {len(miss)} file x measure rows are in only one workbook (dropped):")
        print(miss.to_string(index=False))
        m = m[m["_merge"] == "both"]
    m = m.drop(columns="_merge")
    m = m[m["Measure"].isin([c for c, _ in MEASURES])].copy()

    s_sign = m["Category_Speed"].map(SIG_SIGN)
    a_sign = m["Category_Acc"].map(SIG_SIGN)
    m["SpeedModulated"] = s_sign.notna()
    m["AccModulated"] = a_sign.notna()
    m["Group"] = np.select(
        [m.SpeedModulated & m.AccModulated, m.SpeedModulated, m.AccModulated],
        ["Both", "Speed only", "Acceleration only"], default="Neither")
    sym = {1: "+", -1: "−"}
    both = m["Group"] == "Both"
    m["Direction"] = ""
    m.loc[both, "Direction"] = [f"Speed {sym[s]}, Acc {sym[a]}"
                                for s, a in zip(s_sign[both], a_sign[both])]
    return m.sort_values(["Measure", "Session"]).reset_index(drop=True)


def stacked_group_bars(ax, bars):
    """`bars`: list of (x label, Series of group counts). One 100 % bar each."""
    cats = [g for g, _ in GROUPS]
    for i, (_label, counts) in enumerate(bars):
        counts = counts.reindex(cats, fill_value=0)
        total = int(counts.sum())
        bottom = 0.0
        for cat, color in GROUPS:
            n = int(counts[cat])
            pct = 100 * n / total if total else 0
            if pct > 0:
                ax.bar(i, pct, bottom=bottom, width=0.7, color=color, edgecolor='white', linewidth=1.5)
                if pct >= 4:
                    label = f"{pct:.0f}%\n({n})" if pct >= 9 else f"{pct:.0f}% ({n})"
                    ax.text(i, bottom + pct / 2, label, ha='center', va='center', fontsize=8,
                            color='white' if cat in _DARK_GROUPS else '0.15')
                else:   # too thin for text inside: % and count right of the segment
                    ax.text(i + 0.37, bottom + pct / 2, f"{pct:.1f}% ({n})", ha='left',
                            va='center', fontsize=6.5, color='0.15')
            bottom += pct
        ax.text(i, 101.5, f"n = {total}", ha='center', va='bottom', fontsize=8, color='0.3')
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0, 108)
    ax.spines[['top', 'right']].set_visible(False)


def direction_bars(ax, both_rows, animals, ymax):
    """Count of Both files per sign combination, stacked by animal."""
    tab = (both_rows.groupby(["Direction", "Animal"]).size().unstack(fill_value=0)
                    .reindex(index=DIRECTIONS, columns=animals, fill_value=0))
    bottom = np.zeros(len(DIRECTIONS))
    for a in animals:
        v = tab[a].to_numpy()
        ax.bar(range(len(DIRECTIONS)), v, bottom=bottom, width=0.65,
               color=tva.animal_color(a), edgecolor='0.25', linewidth=0.6)
        for i, (b, n) in enumerate(zip(bottom, v)):
            if n and n >= 0.06 * ymax:
                ax.text(i, b + n / 2, str(n), ha='center', va='center', fontsize=8, color='0.15')
        bottom += v
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 0.02 * ymax, f"{int(tot)}", ha='center', va='bottom',
                fontsize=10, weight='bold', color='0.15')
    ax.set_xticks(range(len(DIRECTIONS)))
    ax.set_xticklabels([d.replace(", ", "\n") for d in DIRECTIONS], fontsize=9)
    ax.set_ylim(0, ymax * 1.15 + 0.5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Number of LFP files", fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)


def plot(m, out_path):
    animals = tva.ordered_animals(m["Animal"])
    both_all = m[m["Group"] == "Both"]
    ymax = max(1, int(both_all.groupby(["Measure", "Direction"]).size().max()) if len(both_all) else 1)

    fig = plt.figure(figsize=(4.0 + 1.3 * len(animals) + 5.0, 4.4 * len(MEASURES) + 1.8))
    gs = fig.add_gridspec(len(MEASURES) + 1, 2, height_ratios=[4] * len(MEASURES) + [1.5],
                          width_ratios=[1 + len(animals), 4], wspace=0.25, hspace=0.45, top=0.92)
    lines = []
    for r, (col, title) in enumerate(MEASURES):
        sub = m[m["Measure"] == col]
        ax = fig.add_subplot(gs[r, 0])
        stacked_group_bars(ax, [("All files", sub["Group"].value_counts())] +
                           [(a, sub.loc[sub["Animal"] == a, "Group"].value_counts()) for a in animals])
        ax.axvline(0.5, color='0.75', lw=0.8, ls=':')
        ax.set_ylabel("% of LFP files", fontsize=10)
        ax.set_title(f"{title}: speed vs. acceleration modulation", fontsize=11, weight='bold')

        b = sub[sub["Group"] == "Both"]
        axd = fig.add_subplot(gs[r, 1])
        direction_bars(axd, b, animals, ymax)
        axd.set_title(f"{title}: direction of the {len(b)} 'Both' file{'' if len(b) == 1 else 's'}",
                      fontsize=11, weight='bold')

        c = sub["Group"].value_counts().reindex([g for g, _ in GROUPS], fill_value=0)
        n = len(sub)
        n_s, n_a = int(sub.SpeedModulated.sum()), int(sub.AccModulated.sum())
        lines.append(f"{title}: speed-modulated {n_s}/{n} ({100 * n_s / n:.1f}%), "
                     f"acceleration-modulated {n_a}/{n} ({100 * n_a / n:.1f}%), "
                     f"both {c['Both']}/{n} ({100 * c['Both'] / n:.1f}%); "
                     f"of the speed-modulated files, {100 * c['Both'] / n_s if n_s else 0:.0f}% "
                     f"are also acceleration-modulated.")
    lines.append(f"Modulated = per-file OLS slope significant at BH-FDR q < {tva.INDIV_FDR_Q:g} "
                 "(FDR over files, per measure and per variable), from the saved "
                 "IndividualCorrel_per_file.xlsx of ThetaVsSpeed Mobility_v3 and ThetaVsAcc.")

    axl = fig.add_subplot(gs[-1, :])
    axl.axis('off')
    h1 = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in GROUPS]
    h2 = [plt.Rectangle((0, 0), 1, 1, facecolor=tva.animal_color(a), edgecolor='0.25', linewidth=0.6)
          for a in animals]
    leg = axl.legend(h1, [g for g, _ in GROUPS], loc='upper center', bbox_to_anchor=(0.27, 1.0),
                     ncol=len(GROUPS), frameon=False, fontsize=9, title="Group (left)",
                     title_fontsize=9)
    axl.add_artist(leg)
    axl.legend(h2, animals, loc='upper center', bbox_to_anchor=(0.76, 1.0), ncol=len(animals),
               frameon=False, fontsize=9, title="Animal (right)", title_fontsize=9)
    axl.text(0.5, 0.42, "\n".join(lines), transform=axl.transAxes, ha='center', va='top',
             fontsize=9, linespacing=1.5)
    fig.suptitle("Theta modulated by both running speed and acceleration (per LFP file)",
                 fontsize=13, weight='bold', y=0.965)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def counts_tables(m):
    animals = tva.ordered_animals(m["Animal"])
    cats = [g for g, _ in GROUPS]
    rows, drows = [], []
    for col, _ in MEASURES:
        sub = m[m["Measure"] == col]
        for a in animals + ["All files"]:
            s = sub if a == "All files" else sub[sub["Animal"] == a]
            c = s["Group"].value_counts().reindex(cats, fill_value=0)
            row = {"Measure": col, "Animal": a, **c.to_dict(), "n_files": len(s)}
            for g in cats:
                row[f"pct_{g}"] = 100 * c[g] / len(s) if len(s) else np.nan
            rows.append(row)
            d = s.loc[s["Group"] == "Both", "Direction"].value_counts().reindex(DIRECTIONS, fill_value=0)
            drows.append({"Measure": col, "Animal": a, **d.to_dict(), "n_both": int(d.sum())})
    return pd.DataFrame(rows), pd.DataFrame(drows)


if __name__ == "__main__":
    m = load_per_file()
    print(f"{m['Session'].nunique()} files in both workbooks")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(m, OUT_DIR / f"{OUT_NAME}.png")
    counts, direction = counts_tables(m)
    with pd.ExcelWriter(OUT_DIR / f"{OUT_NAME}.xlsx") as xw:
        m.to_excel(xw, sheet_name="per_file", index=False)
        counts.to_excel(xw, sheet_name="counts", index=False)
        direction.to_excel(xw, sheet_name="direction", index=False)
    print(counts.to_string(index=False))
    print(direction.to_string(index=False))
    print(f"Saved -> {OUT_DIR / (OUT_NAME + '.png')}")
