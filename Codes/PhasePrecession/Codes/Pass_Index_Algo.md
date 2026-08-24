# Pass Index Phase Precession Algorithm

PassIndexPhasePrecession.py, method from Climer, Newman & Hasselmo (2013) with Kempter et al. (2012) for the circular statistics. The core idea: instead of measuring phase precession against linear position (which only works for one-dimensional, single-pass trajectories), the "pass index" is a local, trajectory-relative coordinate — roughly "how far through the current pass across the firing field am I?" — built directly from the spatial rate map. This generalizes precession analysis to 2D open-field environments and irregular running paths.

## Step 0 — Load raw data
load_ncs: reads the Neuralynx continuous LFP file (theta reference channel), decoding the binary record format and converting to µV using the ADBitVolts header field. Returns the raw voltage trace, per-sample timestamps, and sampling rate.
load_ntt_spike_times: reads spike timestamps per sorted unit from the tetrode file (cell 0 = unsorted noise, dropped if real clusters exist).
load_tracking: reads the (timestamp, x, y) position samples, cleans NaNs/duplicate timestamps, sorts by time.
Everything downstream operates on: pos_ts/pos_xy (behavior), spk_ts (one unit's spikes), lfp_ts/lfp_sig (theta LFP).

## Step 1 — Assign each spike a position: spk_pos
For every spike time, find the nearest-in-time tracked (x, y) sample via searchsorted + left/right distance comparison. This gives spk_xy, the animal's location at each spike.

## Step 2 — Build the occupancy-normalized rate map: rate_map
This is a standard place-field rate map, needed because the pass index is defined relative to the firing field's shape, not relative to raw position:

Bin the environment into a 2D grid (binside, default 2 * n_dims = 4 position units).
Occupancy: histogram of time spent in each bin (occupancy = counts * dt, where dt is the mean tracking sample interval).
Spike counts: histogram of spk_xy into the same bins.
Rate = spike count / occupancy time, per bin → raw rate map.
Bins with zero occupancy become NaN, then filled by nearest-neighbor extrapolation (_fill_nan_nearest, via distance_transform_edt) so smoothing doesn't get corrupted by holes.
Gaussian-smooth the map (smth_width, default 3 * binside), then re-zero the never-visited bins.

## Step 3 — Convert rate map to a "field index" map: field_index_map
This normalizes the smoothed rate map to a 0–1 scale representing "how deep into the field" each spatial bin is:
method='place' (default): min-max normalize the rate map so the field's peak = 1, its lowest occupied-rate bin = 0. This is appropriate for a single, roughly unimodal place field.
method='grid': rank-normalize (percentile transform) instead — appropriate for grid cells with many repeating fields of different peak rates, since it treats each field's local peak comparably.
Then field_index_per_position looks up this per-bin field-index value at every tracked position sample (not just spike positions), giving a continuous time series of "field index at the animal's current location."

## Step 4 — Resample onto arc length: sample_along_arc
This is a key trick from the original toolbox. Behavioral tracking is unevenly sampled in distance traveled (the animal pauses, moves at variable speed, etc.), but the spatial bandpass filter in Step 5 needs to operate on a variable that's uniform in space, not time:

Compute cumulative arc length (Euclidean path distance) traveled: arc = cumsum(||Δxy||).
Drop non-moving samples (zero displacement).
Resample to a uniform grid in arc length (cc = linspace(0, arc.max(), N)), interpolating both timestamps (ts2) and the field-index trace (resampled) onto this grid.
Now resampled is the field-index value as a function of distance traveled, evenly sampled — a proper spatial signal.

## Step 5 — Spatial bandpass filter: bandpass_filter + auto_filter_band
The field-index trace along the path rises and falls once per traversal of the field ("pass"), roughly like a spatial oscillation whose period equals the field's spatial extent. To isolate that oscillation (and reject slow drift and fast noise from residual jitter):

Butterworth bandpass filter (filtfilt, zero-phase) applied to resampled, in units of cycles per unit distance.
Auto band selection:
place: estimate field diameter from the area where rate > 10% of peak (volume), convert to an effective radius r, then set passband (1/(6r), 3/r) — i.e., allow spatial frequencies from about 1 cycle per 6 field-radii up to 3 cycles per field-radius, centered around "roughly one cycle per field traversal."
grid: fixed band tuned to typical grid spacing.
Falls back to sosfilt if filtfilt produces NaNs (can happen with short/edge-heavy segments).

## Step 6 — Extract the pass index via Hilbert transform: compute_pass_index

pass_index_trace = angle(hilbert(filtered_field_index)) / π
The Hilbert transform gives the analytic signal of the filtered spatial oscillation, whose instantaneous phase angle tracks progress through each field crossing:

Phase −π → entering the field
Phase 0 → field center
Phase +π → leaving the field
Dividing by π rescales this to the pass index range [−1, 1]. This is the spatial analogue of theta phase — a normalized "where am I within this pass through the field" coordinate that works regardless of running speed or path curvature.

The phase is then unwrapped (np.unwrap) so it's continuous across many consecutive passes, and interpolated to each spike's exact time (_interp_nearest_extrap), then re-wrapped into [−1, 1] to give spk_pass_index — the pass index at each spike.

## Step 7 — Extract LFP theta phase at spike times

filtered_lfp = bandpass_filter(lfp_sig, 3-7 Hz, lfp_fs)
lfp_phase = angle(hilbert(filtered_lfp))
Standard theta-phase extraction: bandpass the raw LFP to the theta band (LFP_FILTER_BAND, default 3–7 Hz here but named 6–10 Hz in the function default — the module-level config wins), Hilbert-transform to get instantaneous phase, unwrap, then linearly interpolate onto spike times and re-wrap to (−π, π] → spk_theta_phase.

## Step 8 — Circular-linear regression: anglereg / kempter_lincirc
This is where "does pass index predict theta phase, and with what slope" gets quantified, following Kempter et al. (2012):

anglereg finds the best-fit slope s (cycles of theta phase per unit of pass index) and intercept b:
First does a rough initial slope estimate by finding a phase offset φ that best linearizes the circular variable (minimizing squared residual of a normal linear fit after "unrolling" the phase — phi_cost).
Then refines by directly maximizing the mean resultant vector length of θ − 2πs·x over candidate slopes s (this is the proper circular objective — it's insensitive to the arbitrary 2π wrapping ambiguity that a naive least-squares fit would fall prey to). Two starting points (slope0 and its reciprocal-negative) are tried via Nelder-Mead to avoid local minima, unless SLOPE_BNDS constrains the search to one bounded region.
Final intercept b = circular mean of residuals.
kempter_lincirc computes the circular-linear correlation coefficient ρ and its p-value:
Converts x (pass index) to a phase-like variable φ = s·x mod 2π using the fitted slope.
Computes circular means φ̄, θ̄ of both variables.
ρ = correlation between sin(θ−θ̄) and sin(φ−φ̄) (the circular analogue of Pearson's r), signed by the slope's direction.
Significance via an approximate z-test using circular moments λ20, λ02, λ22, converted to a p-value via the error function.
Applied here as: x = spk_pass_index, theta = spk_theta_phase → returns (rho, p, s, b).

## Step 9 — Classify precession

slope_deg_per_pass = rad2deg(2π · s)
is_precessing = p < 0.05 AND -1440 < slope_deg_per_pass < -22
Convert slope from cycles/pass-index-unit to degrees per full pass (pass index spans 2 units, −1 to 1, so 2π·s cycles → degrees over one full traversal).
A cell is called "precessing" only if the fit is statistically significant and the slope is negative (phase decreases as the animal moves through the field — the hallmark of precession) and within a plausible physiological range (excludes near-zero slopes and implausibly steep multi-cycle slopes).

## Step 10 — Density map (visualization/QC)
Builds a 2D occupancy-normalized histogram of spike density over (pass index × LFP phase):

occ_density: how much time the animal spent at each (pass-index, LFP-phase) combination (from continuous LFP-phase and interpolated pass-index traces).
spk_density: spike counts binned the same way.
density = spk_density / occ_density, Gaussian-smoothed — this is the 2D analogue of the classic phase-vs-position precession scatter/heatmap, letting you see the precession band visually independent of the regression fit.

## Step 11 — Output: plot_unit_summary
Six-panel figure per unit: trajectory colored by pass index, rate map, field-index map, phase-vs-pass-index scatter with fitted regression line (doubled to 720° for visibility), the density heatmap, and a text summary (ρ, p, slope, precessing flag). main loops this over every .ntt file/unit in the session folder and writes a CSV summary.

**why pass index instead of raw position: Ordinary phase precession analysis regresses theta phase against linear position within a single, well-defined 1D field crossing. That breaks down in 2D open fields with curved, variable-speed trajectories and multiple field passes at different entry angles. The pass index sidesteps this by deriving a local, field-shape-relative progress variable (via spatial filtering + Hilbert phase of the field-index trace) that behaves consistently across arbitrarily-shaped passes, then applies the same Kempter circular-linear regression machinery to test for phase-position coupling against that variable instead of raw x/y.**





### Step 3 and 4 splanation:

sample_along_arc — PassIndexPhasePrecession.py:283-293

def sample_along_arc(pos_ts, pos_xy, field_index):
    arc = np.concatenate(([0.0], np.cumsum(np.sqrt(np.sum(np.diff(pos_xy, axis=0) ** 2, axis=1)))))
    moving = np.concatenate(([True], np.diff(arc) > 0))
    arc_m, ts_m = arc[moving], pos_ts[moving]

    cc = np.linspace(0, arc_m.max(), len(ts_m))
    ts2 = np.interp(cc, arc_m, ts_m)
    resampled = np.interp(ts2, pos_ts, field_index)
    return cc, ts2, resampled
Inputs: pos_ts (timestamps of position samples), pos_xy (x,y at each timestamp), field_index (the place-field-index value already computed at each position sample, from field_index_per_position). All three arrays share the same original time base.

Line 286 — cumulative arc length. np.diff(pos_xy, axis=0) gives the per-step (x,y) displacement; its Euclidean norm summed cumulatively gives distance traveled up to each sample, prepended with 0 for the first sample. So arc[i] = total path distance from the start up through sample i. This is monotonically non-decreasing — it stays flat whenever the animal is still.

Line 287 — drop non-moving samples. moving is True wherever arc actually increased from the previous sample (plus the first sample by convention). This filters out timestamps where the animal was stationary, because those would give repeated arc values — and np.interp requires its x-coordinates (arc_m here) to be strictly increasing to behave sensibly. Physically, this also means "pauses" contribute nothing to the spatial signal — you're building a signal indexed by distance traveled, not time, so time spent standing still shouldn't stretch or distort it.

Line 290 — a uniform grid in space. cc = linspace(0, arc_m.max(), len(ts_m)) creates evenly spaced arc-length values covering the full path, with as many points as there were moving samples (roughly preserving the original sampling density, just now evenly spaced in distance instead of time).

Line 291 — arc → time. ts2 = np.interp(cc, arc_m, ts_m) answers: "at this evenly-spaced distance-traveled value, what time was it?" Since the animal moves at variable speed, equal steps in cc correspond to unequal steps in real time — ts2 is the (non-uniform) time warp that makes the signal uniform in space.

Line 292 — time → field index. resampled = np.interp(ts2, pos_ts, field_index) then looks up the field-index value at those warped times, interpolating against the original (full, including stationary) pos_ts/field_index series. This two-hop interpolation (arc→time via the moving-only subset, then time→field_index via the full original series) is why the function does two np.interp calls instead of one direct arc → field_index interpolation.

Output: cc (uniform distance grid), ts2 (corresponding real times), resampled (field-index trace, now evenly sampled per unit distance rather than per unit time).

Why this matters downstream (compute_pass_index):

cc, ts2, resampled = sample_along_arc(pos_ts, pos_xy, field_index)
fs_arc = 1.0 / np.mean(np.diff(cc))
filtered_field_index = bandpass_filter(resampled, filter_band[0], filter_band[1], fs_arc)
fs_arc is a sampling rate in samples-per-unit-distance (not samples-per-second). The Butterworth bandpass in Step 5 is then defined in cycles-per-distance, isolating the spatial oscillation of the field-index trace as the animal crosses the place field — which is only a well-posed operation because resampled is uniformly sampled in space. Doing this bandpass on the raw time-sampled field_index would conflate running-speed variations with genuine spatial frequency content.

Also, explain how the field_index is created. I don't understand how binned data from a rate map  is retransformed into a time series again
Read c:\Runita\NMR\Codes\myCodes\PhasePRecession\Main\PassIndexPhasePrecession.py (lines 181-284)
How the (static, binned) rate map becomes a (per-sample) time series
The trick is simple: the rate map is a spatial lookup table, and at every moment in time the animal is sitting in exactly one bin of that table. So you don't need interpolation to convert map → time series — you just ask "which bin is the animal in right now?" and read off that bin's value. That's a table lookup, not a re-binning.

Concretely, three steps:

1. Build the rate map (time series → 2D binned map) — rate_map

Bin the arena into a grid (x_edges, y_edges) of size binside.
occupancy: time spent in each bin (histogram of pos_xy, scaled by dt).
spk_counts: spike count per bin (histogram of spk_xy, the position at each spike time).
rmap = spk_counts / occupancy → firing rate per bin, then Gaussian-smoothed. This is the classic 2D place-field map, and yes, all the time-ordering is thrown away here — it's just "how much did the cell fire, per unit time, wherever the animal was."
2. Normalize it into a 0–1 index — field_index_map

Same shape as rmap, but rescaled per-bin to [0,1]: either min-max normalized (method='place') or rank-normalized (method='grid').
Still purely spatial — fi_map[i,j] is a property of bin (i,j), with no notion of time.
3. Reproject the map back onto the original time samples — the actual "retransform" — field_index_per_position


xi = np.clip(np.digitize(pos_xy[:, 0], x_edges) - 1, 0, len(x_edges) - 2)
yi = np.clip(np.digitize(pos_xy[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
field_index = fi_map[xi, yi]
For every original position sample pos_xy[t] (there are len(pos_ts) of these, at full temporal resolution), np.digitize finds which (x_edges, y_edges) bin that (x,y) falls into, giving bin indices (xi[t], yi[t]). Then field_index[t] = fi_map[xi[t], yi[t]] — you just index into the map with those coordinates. Since pos_xy still has one row per original timestamp, field_index comes out as a proper time series (same length and time base as pos_ts), even though its values only ever come from the small set of bin values in fi_map.

So the result looks like a staircase: as the animal moves smoothly through the arena, field_index[t] jumps between the discrete bin values of the (smoothed) map, occupied bin by occupied bin, moment by moment — reflecting where in the place field the animal currently is (edge, near-peak, etc.), sampled at the original tracking rate.

_fill_nan_nearest (line 280) then patches any samples that landed in a bin with zero occupancy elsewhere (fi_map was NaN there) by nearest-neighbor fill, so there are no gaps.

This is exactly the input sample_along_arc then takes and re-samples onto a uniform arc-length grid (previous explanation) — so the full pipeline is: time series of positions → binned spatial rate map → binned 0-1 field-index map → re-expanded to a field-index time series by lookup → re-expressed as a field-index distance series by arc-length resampling.