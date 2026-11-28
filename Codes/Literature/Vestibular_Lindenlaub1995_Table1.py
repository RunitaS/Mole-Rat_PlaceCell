"""
Plot Table 1 of Lindenlaub, Burda & Nevo (1995), J. Morphol. 224:303-311.
"Morphometrical parameters and estimated mechanical sensitivity of the cupula
of semicircular ducts": Rattus norvegicus vs the Zambian common mole-rat
(reported as Cryptomys sp., earlier C. hottentotus; this Lusaka population is
now Fukomys anselli). The paper also measured Spalax ehrenbergi; its values are
kept in PARAMS for completeness but are not plotted or tested (PLOT_SPECIES).

Values are means +/- SD over 8 organs from 8 animals per species, transcribed
from the table. The paper gives no per-animal data, so true box plots
(median / quartiles) cannot be drawn: each box spans mean - SD .. mean + SD,
with a line at the mean.

Figures written to OUT_DIR:
  1. Vestibular_Lindenlaub1995_Table1                - data only, all 7 parameters
  1b. Vestibular_Lindenlaub1995_DuctSize_AsPublished - L, R, P as published, no
       correction or tests (same layout as the size-corrected figure)
  2. Vestibular_Lindenlaub1995_Table1_PaperStats     - p-values as quoted in the paper
  3. Vestibular_Lindenlaub1995_Table1_WelchT         - our tests from mean/SD/n
  4. Vestibular_Lindenlaub1995_SizeCorrected_WelchT
       - L, R, P after the paper's body-size (allometric) correction, our tests
  + Vestibular_Lindenlaub1995_OurStats.csv (all Welch t-test results)

Our tests (not the paper's): per parameter and duct, Welch two-sample t-test
(rat vs Fukomys) from the published mean, SD and n = 8. p-values are
Holm-corrected within each figure: 21 tests (7 parameters x 3 ducts) for the
Table 1 figure, 9 tests (L, R, P x 3 ducts) for the size-corrected figure.
Stars show Holm-adjusted p; the CSV holds both raw and adjusted p. Ducts are not compared with each other: they come from the
same animals (paired design) and the raw data are not published.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy import stats

OUT_DIR = r"C:\Runita\NMR\Thesis\Figs\Sensory\Vestib"
FILE_PREFIX = "Vestibular_Lindenlaub1995"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Species, colours, ducts
# ---------------------------------------------------------------------------
PLOT_SPECIES = ["Rattus", "Fukomys"]          # Spalax deliberately excluded
REF, TEST = "Rattus", "Fukomys"
SPECIES_LABEL = {
    "Rattus":  "Rattus norvegicus",
    "Fukomys": "Fukomys anselli\n(reported as Cryptomys sp.)",
}
COLORS = {
    "Rattus":  "#000000",   # black
    "Fukomys": "#CC9A1F",   # ochre yellow
}
MEAN_LINE_COLOR = {"Rattus": "#FFFFFF", "Fukomys": "#000000"}
DUCTS = ["Anterior\n(DSA)", "Posterior\n(DSP)", "Lateral\n(DSL)"]
DUCT_SHORT = ["DSA", "DSP", "DSL"]
N_PER_SPECIES = 8

X_OFFSET = {"Rattus": -0.16, "Fukomys": 0.16}
BOX_W = 0.28

# Allometric correction factors from the paper (Methods, Statistics, p. 306),
# after Jones & Spells (1963). Rattus is the reference (factor 1).
CORR_LINEAR = {"Rattus": 1.0, "Spalax": 1.0550, "Fukomys": 1.1355}
CORR_SQUARED = {"Rattus": 1.0, "Spalax": 1.0686, "Fukomys": 1.1705}

# ---------------------------------------------------------------------------
# Data: Table 1. Per species, the three values are [DSA, DSP, DSL].
# 'paper' = (between ducts, Friedman; between species, Kruskal-Wallis) as
# quoted in Results pp. 306-307. Both tests included Spalax.
# '†' = pooled statement on duct size.
# ---------------------------------------------------------------------------
PARAMS = [
    {
        "key": "L", "title": "Streamline length (L)", "unit": "mm",
        "mean": {"Rattus": [8.84, 7.58, 7.25], "Spalax": [8.80, 7.42, 8.24], "Fukomys": [7.14, 6.03, 6.61]},
        "sd":   {"Rattus": [0.32, 0.28, 0.29], "Spalax": [0.34, 0.28, 0.30], "Fukomys": [0.28, 0.25, 0.26]},
        "paper": ("P < 0.0003", "P < 0.0005†"),
    },
    {
        "key": "R", "title": "Curvature radius (R)", "unit": "mm",
        "mean": {"Rattus": [1.52, 1.20, 1.14], "Spalax": [1.48, 1.18, 1.31], "Fukomys": [1.21, 0.95, 1.05]},
        "sd":   {"Rattus": [0.05, 0.04, 0.04], "Spalax": [0.06, 0.04, 0.05], "Fukomys": [0.05, 0.04, 0.04]},
        "paper": ("P < 0.0003", "P < 0.0005†"),
    },
    {
        "key": "P", "title": "Plane area (P)", "unit": "mm²",
        "mean": {"Rattus": [6.41, 4.34, 3.81], "Spalax": [5.80, 4.05, 6.07], "Fukomys": [4.23, 2.83, 3.48]},
        "sd":   {"Rattus": [0.23, 0.15, 0.15], "Spalax": [0.23, 0.16, 0.22], "Fukomys": [0.19, 0.13, 0.20]},
        "paper": ("not reported", "P < 0.0005†"),
    },
    {
        "key": "A", "title": "Lumen cross-sectional area (A)", "unit": "mm²",
        "mean": {"Rattus": [0.0286, 0.0283, 0.0279], "Spalax": [0.0377, 0.0374, 0.0383], "Fukomys": [0.0374, 0.0364, 0.0367]},
        "sd":   {"Rattus": [0.0013, 0.0011, 0.0011], "Spalax": [0.0016, 0.0013, 0.0018], "Fukomys": [0.0014, 0.0013, 0.0014]},
        "paper": ("P = 0.037", "P < 0.0001"),
    },
    {
        "key": "Ac", "title": "Cupula cross-sectional area (A$_c$)", "unit": "mm²",
        "mean": {"Rattus": [0.206, 0.217, 0.208], "Spalax": [0.275, 0.281, 0.286], "Fukomys": [0.242, 0.246, 0.214]},
        "sd":   {"Rattus": [0.013, 0.018, 0.018], "Spalax": [0.014, 0.016, 0.019], "Fukomys": [0.008, 0.012, 0.009]},
        "paper": ("P = 0.037", "P < 0.0001"),
    },
    {
        "key": "S_A2", "title": "Shape-weighted inverse area (S/A²)", "unit": "mm$^{-4}$",
        "mean": {"Rattus": [796, 873, 825], "Spalax": [430, 467, 497], "Fukomys": [480, 472, 510]},
        "sd":   {"Rattus": [34, 44, 48], "Spalax": [22, 22, 23], "Fukomys": [33, 28, 26]},
        "paper": ("P = 0.037", "P < 0.0001"),
    },
    {
        "key": "xc_gamma", "title": "Mechanical sensitivity (x$_c$/γ)", "unit": "nm / (deg/s)",
        "mean": {"Rattus": [7.38, 5.04, 5.11], "Spalax": [9.30, 6.94, 8.66], "Fukomys": [8.52, 6.75, 8.05]},
        "sd":   {"Rattus": [0.71, 0.59, 0.57], "Spalax": [0.89, 0.72, 0.88], "Fukomys": [0.80, 0.69, 0.76]},
        "paper": ("P < 0.00005", "P = 0.0002"),
    },
]


def size_corrected_params():
    """L and R x linear factor, P x squared factor (P: our extension)."""
    out = []
    for p in PARAMS:
        if p["key"] not in ("L", "R", "P"):
            continue
        f = CORR_LINEAR if p["key"] in ("L", "R") else CORR_SQUARED
        q = dict(p)
        q["title"] = p["title"] + ", size-corrected"
        q["mean"] = {sp: [v * f[sp] for v in p["mean"][sp]] for sp in PLOT_SPECIES}
        q["sd"] = {sp: [v * f[sp] for v in p["sd"][sp]] for sp in PLOT_SPECIES}
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Statistics from summary data: Welch t-test, rat vs Fukomys, per duct
# ---------------------------------------------------------------------------
def run_tests(p):
    """Returns per duct: (t, df, p)."""
    res = []
    for d in range(len(DUCTS)):
        m1, s1 = p["mean"][REF][d], p["sd"][REF][d]
        m2, s2 = p["mean"][TEST][d], p["sd"][TEST][d]
        t, pv = stats.ttest_ind_from_stats(m2, s2, N_PER_SPECIES, m1, s1, N_PER_SPECIES, equal_var=False)
        v1, v2 = s1 ** 2 / N_PER_SPECIES, s2 ** 2 / N_PER_SPECIES
        df = (v1 + v2) ** 2 / (v1 ** 2 / (N_PER_SPECIES - 1) + v2 ** 2 / (N_PER_SPECIES - 1))
        res.append((t, df, pv))
    return res


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def add_holm(all_tests):
    """One Holm family over every test in all_tests -> (t, df, p, p_holm)."""
    flat = [pv for tests in all_tests for (_, _, pv) in tests]
    adj = iter(holm(flat))
    return [[(t, df, pv, next(adj)) for (t, df, pv) in tests] for tests in all_tests]


def stars(pv):
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    if pv < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def value_range(p):
    lo = min(p["mean"][sp][d] - p["sd"][sp][d] for sp in PLOT_SPECIES for d in range(3))
    hi = max(p["mean"][sp][d] + p["sd"][sp][d] for sp in PLOT_SPECIES for d in range(3))
    return lo, hi, hi - lo


def draw_boxes(ax, p):
    for sp in PLOT_SPECIES:
        for d in range(len(DUCTS)):
            x = d + X_OFFSET[sp]
            m, s = p["mean"][sp][d], p["sd"][sp][d]
            ax.add_patch(Rectangle((x - BOX_W / 2, m - s), BOX_W, 2 * s,
                                   facecolor=COLORS[sp], edgecolor="none", zorder=2))
            ax.plot([x - BOX_W / 2, x + BOX_W / 2], [m, m], color=MEAN_LINE_COLOR[sp],
                    lw=1.6, solid_capstyle="butt", zorder=3)
    ax.set_xlim(-0.55, len(DUCTS) - 0.45)
    ax.set_xticks(range(len(DUCTS)))
    ax.set_xticklabels(DUCTS)
    ax.tick_params(axis="x", length=0)
    ax.set_title(p["title"], fontsize=9.5, fontweight="bold", loc="left")
    ax.set_ylabel(p["unit"])
    ax.yaxis.grid(True, color="#E3E3E3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def draw_brackets(ax, p, tests, span):
    tick = 0.025 * span
    for d, (_, _, _, pv) in enumerate(tests):          # pv = Holm-adjusted
        y = max(p["mean"][sp][d] + p["sd"][sp][d] for sp in PLOT_SPECIES) + 0.06 * span
        x1, x2 = d + X_OFFSET[REF], d + X_OFFSET[TEST]
        ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#000000", lw=0.7, zorder=4)
        lab = stars(pv)
        ax.text((x1 + x2) / 2, y + (0.0 if lab != "ns" else 0.012 * span), lab,
                ha="center", va="bottom", fontsize=8 if lab != "ns" else 6.5, zorder=4)


def legend_panel(ax, notes, notes_y=0.52):
    ax.axis("off")
    handles = [Patch(facecolor=COLORS[s], edgecolor="none", label=SPECIES_LABEL[s]) for s in PLOT_SPECIES]
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.6,
              handleheight=1.2, labelspacing=0.9, prop={"style": "italic", "size": 9})
    ax.text(0.0, notes_y, notes, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.3, color="#404040", linespacing=1.35)


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f"{FILE_PREFIX}_{name}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=300)
    print(f"Saved: {base}.png / .pdf / .svg")


SOURCE = "Data: Lindenlaub, Burda & Nevo (1995),\nJ. Morphol. 224:303–311, Table 1."
BOX_NOTE = (f"Box = mean ± SD (not quartiles; per-animal data\n"
            f"are not published); line = mean. n = {N_PER_SPECIES} per species.")
TEST_NOTE = ("Our tests from published mean, SD, n (not the\n"
             "paper's): Welch t-test, rat vs Fukomys, per duct.\n"
             "Holm-corrected over all 21 tests in this figure.\n"
             "Holm p: * <0.05  ** <0.01  *** <0.001  ns not significant.")


def figure_table(params, mode, all_tests=None):
    """mode: 'plain', 'paper' or 'ours'."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8 if mode != "plain" else 6.2),
                             constrained_layout=True)
    axes = axes.ravel()
    for i, (ax, p) in enumerate(zip(axes, params)):
        draw_boxes(ax, p)
        lo, hi, span = value_range(p)
        if mode == "plain":
            ax.set_ylim(lo - 0.08 * span, hi + 0.08 * span)
        elif mode == "paper":
            ax.set_ylim(lo - 0.08 * span, hi + 0.42 * span)
            ducts_txt, species_txt = p["paper"]
            ax.text(0.02, 0.98, f"Ducts (Friedman): {ducts_txt}\nSpecies (Kruskal–Wallis): {species_txt}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="#202020")
        else:
            ax.set_ylim(lo - 0.08 * span, hi + 0.25 * span)
            draw_brackets(ax, p, all_tests[i], span)

    if mode == "plain":
        notes = BOX_NOTE + "\n\nx$_c$/γ from the Oman et al. (1987) model:\n" \
                "x$_c$/γ = P / (4π L A$_c$ ν · S/A²)\n\n" + SOURCE
        title = "Semicircular duct morphometry and cupula mechanical sensitivity"
    elif mode == "paper":
        notes = (BOX_NOTE + "\n\n"
                 "p-values as quoted in the paper (Results,\npp. 306–307); not recomputed.\n"
                 "Both tests also included Spalax ehrenbergi\n(not shown); no rat vs Fukomys test reported.\n"
                 "Ducts: Friedman ANOVA within each species.\n"
                 "Species: Kruskal–Wallis among 3 species.\n"
                 "† 'All the differences were significant (P < 0.0005)',\n"
                 "   said of duct size; which of L, R, P is not stated.\n"
                 "Only the lateral duct was larger relative to log\nbody mass (Discussion, no P given).\n\n"
                 + SOURCE)
        title = "Semicircular duct morphometry: statistics reported by Lindenlaub et al. (1995)"
    else:
        notes = (BOX_NOTE + "\n\n" + TEST_NOTE + "\n"
                 "Ducts not compared (paired; raw data unpublished).\n"
                 "Source means are rounded, so p are approximate.\n\n"
                 + SOURCE)
        title = "Semicircular duct morphometry, rat vs Fukomys: Welch t-tests (computed from summary data)"
    legend_panel(axes[-1], notes, notes_y=0.62 if mode != "plain" else 0.52)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig


def figure_size_corrected(params, all_tests):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), constrained_layout=True)
    for i, (ax, p) in enumerate(zip(axes, params)):
        draw_boxes(ax, p)
        lo, hi, span = value_range(p)
        ax.set_ylim(lo - 0.08 * span, hi + 0.25 * span)
        draw_brackets(ax, p, all_tests[i], span)
    notes = ("Fukomys values × the paper's allometric factors\n"
             "(Jones & Spells 1963; rat = 1):\n"
             "L, R × 1.1355 (linear).\n"
             "P × 1.1705 (squared); our extension, the paper\n"
             "lists it only for A, A$_c$, S/A².\n"
             "Assumes Table 1 holds uncorrected values.\n"
             "Box = mean ± SD; line = mean; n = 8.\n"
             "Welch t-test (ours), Holm-corrected over the\n"
             "9 tests in this figure.\n"
             "Holm p: * <0.05  ** <0.01  *** <0.001  ns")
    legend_panel(axes[-1], notes, notes_y=0.68)
    fig.suptitle("Semicircular duct size after body-size (allometric) correction",
                 fontsize=11, fontweight="bold")
    return fig


def figure_size_as_published():
    """L, R, P exactly as in Table 1: no correction, no tests."""
    params = [p for p in PARAMS if p["key"] in ("L", "R", "P")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), constrained_layout=True)
    for ax, p in zip(axes, params):
        draw_boxes(ax, p)
        lo, hi, span = value_range(p)
        ax.set_ylim(lo - 0.08 * span, hi + 0.08 * span)
    notes = ("Values as published in Table 1: mean ± SD,\n"
             f"n = {N_PER_SPECIES} organs from {N_PER_SPECIES} animals per species.\n"
             "No body-size correction, no statistical tests.\n"
             "Box = mean ± SD (not quartiles); line = mean.\n\n"
             + SOURCE)
    legend_panel(axes[-1], notes, notes_y=0.62)
    fig.suptitle("Semicircular duct size as published (uncorrected)", fontsize=11, fontweight="bold")
    return fig


def write_stats_csv(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{FILE_PREFIX}_OurStats.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "parameter", "duct", f"mean_{REF}", f"mean_{TEST}",
                    f"diff_{TEST}_minus_{REF}", "welch_t", "welch_df", "welch_p", "p_holm"])
        w.writerows(rows)
    print(f"Saved: {path}")


def collect_rows(dataset, params, all_tests):
    rows = []
    for p, tests in zip(params, all_tests):
        for d, (t, df, pv, ph) in enumerate(tests):
            m1, m2 = p["mean"][REF][d], p["mean"][TEST][d]
            rows.append([dataset, p["key"], DUCT_SHORT[d], f"{m1:.5g}", f"{m2:.5g}", f"{m2 - m1:.5g}",
                         f"{t:.3f}", f"{df:.2f}", f"{pv:.3g}", f"{ph:.3g}"])
    return rows


def main():
    # 1. Data only
    save(figure_table(PARAMS, "plain"), "Table1")
    # 1b. Duct size (L, R, P) as published, no correction or tests
    save(figure_size_as_published(), "DuctSize_AsPublished")
    # 2. Paper-quoted statistics
    save(figure_table(PARAMS, "paper"), "Table1_PaperStats")
    # 3. Our tests on Table 1
    tests_raw = add_holm([run_tests(p) for p in PARAMS])
    save(figure_table(PARAMS, "ours", tests_raw), "Table1_WelchT")
    # 4. Size-corrected L, R, P with our tests
    corr = size_corrected_params()
    tests_corr = add_holm([run_tests(p) for p in corr])
    save(figure_size_corrected(corr, tests_corr), "SizeCorrected_WelchT")

    write_stats_csv(collect_rows("Table1", PARAMS, tests_raw)
                    + collect_rows("SizeCorrected", corr, tests_corr))
    plt.show()


if __name__ == "__main__":
    main()
