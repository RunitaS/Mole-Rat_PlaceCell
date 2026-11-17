"""
ExtractPlotStatsText_v1.py

Reads the stats text printed at the bottom of every plot (.png) in a folder
(recursively) and writes the values to an Excel workbook in the same folder.

The plots are PNG images, so the text is read with the OCR engine that ships
with Windows 10/11 (Windows.Media.Ocr, called through PowerShell; nothing to
install). The text is then parsed into numbers.

Text formats recognised (as written by ArenaSessionType_StatsComparison_v20.py):
    <group> (n=N): mean M ± SD SD | median Md [Q1 a, Q3 b]          continuous
    <group> (n=N): Y yes / Z no = P %                               categorical
    <group> (n=N): p-Speed a (x %), n-Speed b (y %), non-speed c (z %)
    Ringed: <group> k/N
    Red-/Green-ringed points only:   -> the entries after it get subset
                                        'Red-ringed only' / 'Green-ringed only'

Output workbook  <PLOT_DIR>\\PlotStatsText.xlsx
    Values    one row per parsed entry, with consistency checks
    RawText   the full OCR text of every plot footer
    Problems  plots / entries that failed a check (empty if all is well)

Checks (to catch OCR misreads):
    - every plot gave at least one parsed entry
    - Q1 <= median <= Q3, SD >= 0
    - categorical: yes + no == n and pct == 100*yes/n (to 0.1 %)
    - speed direction: counts sum to n and each % matches its count
    - histogram 'Hist_<metric>.png' values equal those of box plot '<metric>.png'
    - a plot with ringed sub-set values also gave its main (All points) values
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

# ── Settings ────────────────────────────────────────────────────────────────
PLOT_DIR = (r'X:\NMR_group_data\Runita\Analysis\Thesis\Corr_Data_SpkQltyFilt'
            r'\SpikeQualityFilt\SpikeQualityFilt_v1_PlaceCellData'
            r'\PC_True_SiSkaggsAdptBin_Final\Results\PC_Char\v4\AllCells')
OUT_NAME = 'PlotStatsText.xlsx'

INK_THRESHOLD = 128   # grey level below which a pixel counts as text / line
BLOCK_GAP_PX  = 60    # white rows separating two blocks of content
UPSCALE       = 2     # enlarge the footer crop before OCR (better accuracy)
RETRY_SCALES  = [3, 1.5, 4, 1]   # zoom levels tried for plots that fail at UPSCALE

if len(sys.argv) > 1:
    PLOT_DIR = sys.argv[1]

# ── Footer crop ─────────────────────────────────────────────────────────────

def crop_footer(png_path: str, out_path: str, scale: float = UPSCALE):
    """Save everything below the tallest content block (the axes) to out_path."""
    im = Image.open(png_path).convert('L')
    rows = np.flatnonzero((np.asarray(im) < INK_THRESHOLD).any(axis=1))
    blocks, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > BLOCK_GAP_PX:
            blocks.append((start, prev))
            start = r
        prev = r
    blocks.append((start, prev))
    tallest = max(range(len(blocks)), key=lambda i: blocks[i][1] - blocks[i][0])
    top = blocks[tallest][1] + 1
    crop = im.crop((0, top, im.width, im.height))
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    crop.save(out_path)


# ── Windows OCR (batch, one PowerShell process) ─────────────────────────────

PS_OCR = r'''
param([string]$ListFile, [string]$OutFile)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) {
    $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(); $task.Result }
$eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$out = @{}
foreach ($p in (Get-Content -Encoding UTF8 $ListFile)) {
    if (-not $p) { continue }
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile])
    $st   = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $dec  = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($st)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bmp  = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $res  = Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
    $out[$p] = (($res.Lines | ForEach-Object { $_.Text }) -join "`n")
    $st.Dispose()
}
$out | ConvertTo-Json -Depth 3 | Out-File -Encoding utf8 $OutFile
'''


def ocr_images(paths: list) -> dict:
    with tempfile.TemporaryDirectory() as td:
        ps, lst, res = (os.path.join(td, n) for n in ('ocr.ps1', 'list.txt', 'out.json'))
        with open(ps, 'w', encoding='utf-8-sig') as f:
            f.write(PS_OCR)
        with open(lst, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(paths))
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                        '-File', ps, lst, res], check=True)
        with open(res, encoding='utf-8-sig') as f:
            return json.load(f)


# ── Parsing ─────────────────────────────────────────────────────────────────

NUM = r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?'


def clean_text(t: str) -> str:
    """Join wrapped lines and fix common OCR slips."""
    t = ' '.join(t.split())
    t = t.replace('−', '-').replace('–', '-').replace('—', '-')
    t = re.sub(r'\bQ[Il|]\b', 'Q1', t)            # QI / Ql -> Q1
    t = re.sub(r'\bQ[Il|](?=\d)', 'Q1 ', t)
    t = re.sub(r'\s[Il]\s(?=median)', ' | ', t)    # 'I median' -> '| median'
    t = re.sub(r'SD\s*[Il]\b', 'SD |', t)
    t = re.sub(r'(?<=\d)[Oo](?=[\d.])|(?<=[\d.])[Oo](?=\d)', '0', t)
    t = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', t)    # '0 .95' -> '0.95'
    t = re.sub(r'\(?\s*\b[n0O]\s*[=-]\s*(?=[\dlI|]+\s*\)\s*:)', '(n=', t)  # '0=523):', '(n-523):'
    t = re.sub(r'\(n=([\dlI|]+)\)', lambda m: '(n=' + re.sub(r'[lI|]', '1', m[1]) + ')', t)
    # a lone 'O' where a number belongs: 'median O', 'Q1 O,', 'mean O ±'
    t = re.sub(r'(median|mean|Q1|Q3|±) [Oo](?=[\s,\]]|$)', r'\1 0', t)
    return t


PAT_CONT = re.compile(
    rf'(?P<group>[^:|]*?)\s*\(n=(?P<n>\d+)\)\s*:\s*mean\s+(?P<mean>{NUM})\s*[±+t]\s*'
    rf'(?P<sd>{NUM}|n/a)\s*SD\s*\|?\s*median\s+(?P<median>{NUM})\s*\[\s*Q1\s+(?P<q1>{NUM})'
    rf'\s*,\s*Q3\s+(?P<q3>{NUM})\s*\]?', re.I)
PAT_CAT = re.compile(
    rf'(?P<group>[^:|]*?)\s*\(n=(?P<n>\d+)\)\s*:\s*(?P<yes>\d+)\s*yes\s*/\s*(?P<no>\d+)\s*no\s*=\s*'
    rf'(?P<pct>{NUM})\s*%?', re.I)
PAT_SPD = re.compile(
    rf'(?P<group>[^:|]*?)\s*\(n=(?P<n>\d+)\)\s*:\s*p-Speed\s+(?P<np>\d+)\s*\(\s*(?P<pp>{NUM})\s*%?\s*\)?\s*,'
    rf'\s*n-Speed\s+(?P<nn>\d+)\s*\(\s*(?P<pn>{NUM})\s*%?\s*\)?\s*,'
    rf'\s*non-speed\s+(?P<nx>\d+)\s*\(\s*(?P<px>{NUM})', re.I)
PAT_RING = re.compile(r'Ringed:\s*(?P<group>.+?)\s+(?P<k>\d+)\s*/\s*(?P<N>\d+)')
SUBSET_MARK = re.compile(r'(?P<colour>Red|Green)-ringed points only\s*:?', re.I)


def _group_name(g: str) -> str:
    # The text before '(n=' can also hold the x-axis label (histograms) or the end of
    # the previous entry; the group label starts at 'All cells' when that is present.
    g = g.strip()
    i = g.lower().rfind('all cells')
    if i >= 0:
        g = g[i:]
    else:
        g = re.sub(r'.*(?:higher|\.png\)\.?|only:|%\)?|\])\s*', '', g)
    return g.strip(' .,:;') or 'All cells'


def parse_text(text: str) -> list:
    t = clean_text(text)
    # split into the main part and the 'Red-ringed points only' part
    parts, last, subset = [], 0, 'All points'
    for m in SUBSET_MARK.finditer(t):
        parts.append((subset, t[last:m.start()]))
        last, subset = m.end(), f'{m["colour"].capitalize()}-ringed only'
    parts.append((subset, t[last:]))

    rows = []
    for subset, seg in parts:
        for m in PAT_CONT.finditer(seg):
            sd = m['sd']
            rows.append(dict(subset=subset, kind='continuous', group=_group_name(m['group']),
                             n=int(m['n']), mean=float(m['mean']),
                             sd=np.nan if sd.lower() == 'n/a' else float(sd),
                             median=float(m['median']), q1=float(m['q1']), q3=float(m['q3'])))
        for m in PAT_CAT.finditer(seg):
            rows.append(dict(subset=subset, kind='categorical', group=_group_name(m['group']),
                             n=int(m['n']), n_yes=int(m['yes']), n_no=int(m['no']),
                             pct_yes=float(m['pct'])))
        for m in PAT_SPD.finditer(seg):
            rows.append(dict(subset=subset, kind='speed_direction', group=_group_name(m['group']),
                             n=int(m['n']),
                             n_p_speed=int(m['np']), pct_p_speed=float(m['pp']),
                             n_n_speed=int(m['nn']), pct_n_speed=float(m['pn']),
                             n_non_speed=int(m['nx']), pct_non_speed=float(m['px'])))
        for m in PAT_RING.finditer(seg):
            rows.append(dict(subset=subset, kind='ringed_count', group=_group_name(m['group']),
                             n=int(m['N']), n_ringed=int(m['k']),
                             pct_ringed=round(100 * int(m['k']) / int(m['N']), 1)))
    return rows


# ── Checks ──────────────────────────────────────────────────────────────────

def row_check(r) -> str:
    bad = []
    if r['kind'] == 'continuous':
        if not (r['q1'] <= r['median'] <= r['q3']):
            bad.append('Q1<=median<=Q3 fails')
        if pd.notna(r['sd']) and r['sd'] < 0:
            bad.append('SD<0')
    elif r['kind'] == 'categorical':
        if r['n_yes'] + r['n_no'] != r['n']:
            bad.append('yes+no != n')
        if r['n'] and abs(100 * r['n_yes'] / r['n'] - r['pct_yes']) > 0.051:
            bad.append('pct != yes/n')
    elif r['kind'] == 'speed_direction':
        if r['n_p_speed'] + r['n_n_speed'] + r['n_non_speed'] != r['n']:
            bad.append('counts do not sum to n')
        for k in ('p_speed', 'n_speed', 'non_speed'):
            if r['n'] and abs(100 * r[f'n_{k}'] / r['n'] - r[f'pct_{k}']) > 0.051:
                bad.append(f'pct_{k} != n_{k}/n')
    return '; '.join(bad) if bad else 'OK'


VALUE_COLS = ['n', 'mean', 'sd', 'median', 'q1', 'q3', 'n_yes', 'n_no', 'pct_yes',
              'n_p_speed', 'pct_p_speed', 'n_n_speed', 'pct_n_speed',
              'n_non_speed', 'pct_non_speed', 'n_ringed', 'pct_ringed']


def hist_vs_box(df: pd.DataFrame) -> pd.Series:
    """For histogram rows: compare with the box-plot row of the same metric."""
    out = pd.Series('', index=df.index, dtype=object)
    key = ['folder', 'metric', 'subset', 'kind', 'group']
    box = df[df['plot_type'] != 'histogram'].set_index(key)
    for i, r in df[df['plot_type'] == 'histogram'].iterrows():
        k = (r['folder'].replace('\\Histograms', '').replace('Histograms', ''),
             r['metric'], r['subset'], r['kind'], r['group'])
        k = (k[0].rstrip('\\') or '.',) + k[1:]
        if k not in box.index:
            out[i] = 'no matching box plot'
            continue
        b = box.loc[[k]].iloc[0]
        diff = [c for c in VALUE_COLS
                if not (pd.isna(r.get(c)) and pd.isna(b.get(c))) and r.get(c) != b.get(c)]
        out[i] = 'matches box plot' if not diff else 'DIFFERS from box plot: ' + ', '.join(diff)
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    pngs = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(PLOT_DIR)
                  for f in fs if f.lower().endswith('.png'))
    print(f'{len(pngs)} plots in {PLOT_DIR}')

    # OCR at UPSCALE; plots whose text does not parse, or fails a check, are read
    # again at the other zoom levels and the first reading that passes is kept.
    texts, scale_used, pending = {}, {}, list(pngs)
    with tempfile.TemporaryDirectory() as td:
        for scale in [UPSCALE] + [s for s in RETRY_SCALES if s != UPSCALE]:
            if not pending:
                break
            crops = {}
            for i, p in enumerate(pending):
                c = os.path.join(td, f'crop_{scale}_{i:04d}.png')
                crop_footer(p, c, scale)
                crops[p] = c
            print(f'Windows OCR at x{scale} on {len(pending)} plots ...')
            ocr = ocr_images(list(crops.values()))
            still = []
            for p, c in crops.items():
                t = ocr.get(c, '') or ''
                parsed = parse_text(t)
                ok = bool(parsed) and all(row_check(pd.Series(r)) == 'OK' for r in parsed)
                if p not in texts or ok:
                    texts[p], scale_used[p] = t, scale
                if not ok:
                    still.append(p)
            pending = still

    rows, raw = [], []
    for p in pngs:
        rel = os.path.relpath(p, PLOT_DIR)
        folder = os.path.dirname(rel) or '.'
        fname = os.path.basename(rel)
        stem = os.path.splitext(fname)[0]
        is_hist = stem.startswith('Hist_')
        metric = stem[5:] if is_hist else stem
        plot_type = ('histogram' if is_hist else
                     'categorical' if re.match(r'(MultiPeak_)?Categorical_', stem) else 'box')
        parsed = parse_text(texts[p])
        raw.append(dict(folder=folder, file=fname, ocr_zoom=scale_used[p], ocr_text=texts[p],
                        cleaned_text=clean_text(texts[p]), n_entries_parsed=len(parsed)))
        for r in parsed:
            rows.append(dict(folder=folder, file=fname, metric=metric, plot_type=plot_type, **r))

    df = pd.DataFrame(rows)
    cols = (['folder', 'file', 'metric', 'plot_type', 'subset', 'kind', 'group']
            + [c for c in VALUE_COLS if c in df.columns])
    df = df.reindex(columns=cols)
    df['check'] = df.apply(row_check, axis=1)
    df['hist_vs_box'] = hist_vs_box(df)

    raw_df = pd.DataFrame(raw)
    # plots that gave sub-set values but no main ('All points') value line, e.g. when
    # the legend is drawn over the first stats line
    has_main = df[(df['subset'] == 'All points') & (df['kind'] != 'ringed_count')]['file']
    no_main = df.loc[~df['file'].isin(has_main), ['folder', 'file']].drop_duplicates()
    probs = pd.concat([
        no_main.assign(problem='main (All points) stats line not read; only sub-set values parsed'),
        raw_df.loc[raw_df['n_entries_parsed'] == 0, ['folder', 'file']]
              .assign(problem='no values parsed from the plot text'),
        df.loc[df['check'] != 'OK', ['folder', 'file']]
          .assign(problem=df.loc[df['check'] != 'OK', 'check']),
        df.loc[df['hist_vs_box'].str.startswith(('DIFFERS', 'no matching')), ['folder', 'file']]
          .assign(problem=df.loc[df['hist_vs_box'].str.startswith(('DIFFERS', 'no matching')),
                                 'hist_vs_box']),
    ], ignore_index=True)

    out = os.path.join(PLOT_DIR, OUT_NAME)
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        df.to_excel(xw, sheet_name='Values', index=False)
        raw_df.to_excel(xw, sheet_name='RawText', index=False)
        probs.to_excel(xw, sheet_name='Problems', index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = 'A2'
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(8, w + 2), 60)
    print(f'{len(df)} entries from {len(pngs)} plots -> {out}')
    print(f'{len(probs)} problem rows (see sheet "Problems")')


if __name__ == '__main__':
    main()
