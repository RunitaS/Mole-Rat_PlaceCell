# Adaptive Bin Algo


The bin size on the plotted grid is uniform and fixed — every pixel of the rate map is target_bin_cm × target_bin_cm (2.0 cm by default). What varies from bin to bin is not the size of the bin drawn on the plot, but the radius of the neighborhood used to estimate that bin's firing rate — a sparsely-visited bin borrows occupancy/spikes from a larger surrounding radius; a densely-visited bin needs almost no borrowing. So "adaptive binning" here really means adaptive smoothing radius on a fixed-size grid, not variable-size bins/pixels.

# Algorithm

## Jim's Code
1. Load & clean tracking (PlaceCellChar_Main_v2_AdaptiveBin.py:1496)
Pixel (or cm) tracking is loaded and converted to cm using arena_width_cm.
2. Smooth position (_smooth_tracking_position, :346)
Iteratively flags frame-to-frame jumps implying speed > POS_JUMP_THRESH_CMS as tracking artifacts, linearly interpolates over them, then Gaussian-smooths x/y with sigma = POS_SMOOTH_SIGMA_SMP samples.
3. Build the fixed spatial grid (:1505-1509)
n_bins_x = ceil(x_cm.max() / target_bin_cm)
n_bins_y = ceil(y_cm.max() / target_bin_cm)
beh_bx   = clip(floor(x_cm / target_bin_cm), 0, n_bins_x-1)   # per-frame bin index
beh_by   = clip(floor(y_cm / target_bin_cm), 0, n_bins_y-1)
This grid — and its bin size, target_bin_cm — is fixed for the whole session/plot.
4. Assign spikes to position frames (:1521-1532)
Each spike timestamp is matched to its nearest tracking frame; a spike is discarded if that gap exceeds MAX_GAP_US (50 ms) — this prevents spikes recorded during tracking dropouts from being assigned to the wrong location.
5. Build the raw occupancy and spike-count histograms (:1538-1551)
dt_frames: duration credited to each position sample (capped at 2/fps so a tracking gap doesn't inflate one bin's occupancy).
occ_map[bx,by] += dt_frames and spike_map[bx,by] += 1 for every frame/spike — these are still on the fixed target_bin_cm grid.
6. Adaptive-binned firing-rate estimate (_adaptive_binned_ratemap, :216-283) — this is the actual "adaptive" step, applied per occupied bin (i0, j0):
Compute squared distance (in bin units, not cm) from (i0,j0) to every other bin: *d2 = (I-i0)² + (J-j0)²*.
Sort all bins by d2 and take cumulative sums of occupancy and spike count as the radius grows outward — this is equivalent to growing a circular neighborhood one bin-shell at a time, but vectorized.
For each candidate squared radius rsq = 0, 1, 2, … , max_rsq-1, test the Skaggs & McNaughton (1998) adaptive-smoothing criterion:
*alpha² · occ_at_rsq² · rsq · nspikes2 > 1*
where *alpha = ADAPTIVE_ALPHA_BASE * fps*, *occ_at_rsq is cumulative occupancy enclosed so far*, and *nspikes2 is the cumulative spike count enclosed* (or 1 if zero, to avoid the criterion being trivially satisfied at rsq=0).
Take the smallest rsq that satisfies the criterion (or max_rsq-1 if it's never satisfied).
The bin's rate = spikes enclosed at that radius / occupancy enclosed at that radius.
The bin is marked valid only if the enclosed occupancy at that final radius is ≥ min_occ_s; otherwise it's left blank (masked), even though it was visited.
Net effect: low-occupancy/low-spike bins pull in a wide neighborhood (large radius) to get a stable rate estimate; high-occupancy/high-firing bins settle with a small radius (often just themselves). The bin grid itself never changes size — only how far each bin "reaches" to compute its own rate.
7. Mask invalid bins (:1344)
np.ma.masked_where(~valid_mask, fr_map) — bins that never reached min_occ_s even after expanding to max_rsq stay blank in the plot rather than showing a rate computed from insufficient data.
8. Plot (_plot_and_save_ratemap, :1334-1382)
Left panel: raw trajectory (x_cm, y_cm) with spike locations overlaid in red.
Right panel: ax_rate.imshow(display_map.T, origin='lower', extent=[0, n_bins_x*target_bin_cm, 0, n_bins_y*target_bin_cm], cmap='jet', interpolation='nearest') — one solid-colored pixel per fixed-size bin, colorbar labeled "firing rate (Hz)".
Title shows cell name, peak/mean firing rate, and SIR (spatial information rate).
Saved as <cellname>_ratemap.png in a ratemaps subfolder next to the .ntt file.

## Roddy's code
Skaggs & McNaughton (1998) criterion from Roddy Grieves' rate_mapper.m 'adaptive' method, rather than the ad-hoc version that was there before. Summary of what changed and why, directly against the bugs your own AdaptiveBinFixes.md had flagged:
1. Criterion now uses raw occupancy sample counts, not time. The old code approximated n (Skaggs' formula wants a sample count) by scaling occupancy-in-seconds by fps. I added a genuine occ_count_map (raw per-bin visit counts, built alongside the existing time-based occ_map) and the search criterion r >= a/(n·√s) now runs on that — exact rather than approximated, and no longer entangled with the capped/irregular dt_frames used for the time map.
2. Removed the "substitute 1 spike for 0 spikes" hack. That was letting bins with zero enclosed spikes satisfy the criterion prematurely (settle on a tiny radius and report rate 0 too early) instead of continuing to expand until a spike is actually found or maxdist is hit — which is what rate_mapper.m does naturally (division by sqrt(0) → infinite threshold, never satisfied). This was your item 2 in the TODO doc.
3. Added the mindist checkpoint gate. rate_mapper.m invalidates a bin if the occupancy within a small fixed radius (mindist, default ratio 2 bins) doesn't reach mindwell — independent of whatever radius the adaptive search eventually settles on. Your old code instead checked occupancy at the final (adaptively chosen) radius, which is nearly always satisfiable by expanding far enough — so it was barely masking anything, including bins with no real local coverage (e.g. outside the arena the animal actually explored).
4. Every reachable bin in the grid is now evaluated, not just bins with nonzero occupancy of their own — matching rate_mapper.m, which loops over every bin center. A distance-transform prefilter skips bins with no visited bin within maxdist (they'd fail the mindist gate anyway), so this doesn't tank performance.
One thing to be aware of: I replaced ADAPTIVE_MAX_RSQ = 200 (≈28 cm radius cap at your 2 cm bins) with ADAPTIVE_MINDIST_CM = 4.0 / ADAPTIVE_MAXDIST_CM = 64.0, matching Grieves' default ratio (mindist/binsize=2, maxdist/binsize=32). That's a materially larger max search radius than before — worth re-checking rate maps against a few known cells before trusting new outputs, and retuning ADAPTIVE_MAXDIST_CM down if it over-smooths.


# Parameters that control the plot:
Parameter	Line	Role
target_bin_cm (2.0)	:145	The fixed spatial resolution of every plotted pixel/bin and of the underlying occupancy/spike grid. This is the same for every bin in a given map.
arena_width_cm (80.0)	:146	Physical arena size, used to convert raw tracking pixels to cm before binning.
fps (30)	:144	Tracking frame rate; scales alpha (alpha = ADAPTIVE_ALPHA_BASE * fps) and caps per-frame occupancy duration.
min_occ_s (1.0)	:147	Minimum occupancy (s) a bin's final expanded neighborhood must reach to be plotted at all; otherwise masked/blank.
MAX_GAP_US (50,000 µs)	:148	Max allowed spike–position timestamp gap; spikes beyond this are discarded before they ever reach the map.
ADAPTIVE_ALPHA_BASE (0.0001)	:154	The core adaptive-smoothing constant (Skaggs & McNaughton 1998 eq. 11). Smaller → radius must grow larger before the criterion is satisfied → more smoothing/larger neighborhoods everywhere. This is the single biggest knob for how "smoothed" the adaptive map looks.
ADAPTIVE_MAX_RSQ (200)	:157	Cap on squared search radius (bin units); stops runaway expansion in extremely sparse regions — those bins then use whatever they've accumulated at the cap, and may still fail the min_occ_s test and end up masked.
cmap='jet'	:1363	Color scale for firing rate (low=blue → high=red).
interpolation='nearest'	:1363	Tells matplotlib not to blur/interpolate between pixels when rendering — each bin is drawn as one flat-colored square at its true target_bin_cm size. Any smoothing you see is entirely from step 6 (the adaptive radius), not from the rendering.
origin='lower'	:1362	Puts (0,0) at bottom-left, matching standard x/y cm coordinates.
extent	:1361	Maps the pixel grid to physical cm axes: [0, n_bins_x*target_bin_cm, 0, n_bins_y*target_bin_cm].
Note: peak_fr and sir reported in the plot title are computed from the raw fixed-bin map (fr_raw, :1571), not the adaptively-binned one — only mean_fr, sparsity, and coherence use the adaptive map — while the map actually drawn is the adaptive one (fr_adaptive).


# Fixed Kernel vs Adaptive Binning

What each method does:
Fixed-kernel smoothing (_triangular_smooth in the pasted script — a 3×3 triangular/bilinear kernel, functionally the same idea as a small Gaussian): every bin gets smoothed by borrowing from the same fixed neighborhood — its immediate 3×3 neighbors (6×6 cm at your 2 cm bin size) — regardless of how much occupancy or spike data is actually in that neighborhood. Weights are renormalized by the convolved valid-mask to correct edge effects, but the neighborhood radius itself never changes.

Adaptive binning (_adaptive_binned_ratemap, per your AdaptiveBinAlgo.md): the grid is still fixed 2 cm bins, but each bin's search radius grows outward until the Skaggs & McNaughton (1998) criterion α²·occ²·r²·nspikes² > 1 is satisfied. Sparse bins keep expanding (up to ADAPTIVE_MAX_RSQ) until they've pooled enough occupancy/spikes for a stable estimate; dense, well-sampled bins (e.g. near a place field peak) settle at radius ≈ 0 and barely change.

Why they look so different
This is a fixed-bandwidth vs. variable-bandwidth smoothing problem:

Fixed kernel applies uniform smoothing bandwidth everywhere, blind to local sample density. In sparsely-visited regions (edges, corners, less-explored parts of the arena), a 3×3 neighborhood often doesn't contain enough valid neighbors either — so those bins stay noisy, patchy, or get masked out entirely (more NaNs). In densely-sampled regions it can also slightly blur a sharp peak it didn't need to blur.
Adaptive binning explicitly equalizes estimation reliability: undersampled bins keep pulling in data from farther away until they have "enough" (per min_occ_s and the Skaggs criterion), which is exactly why you see fewer NaNs and a cleaner-looking map — sparse regions get actively rescued rather than left thin. Meanwhile well-sampled peaks keep a small radius, so they aren't necessarily over-blurred the way uniform smoothing would.

Pros / cons
Fixed-kernel:
✅ Simple, deterministic, cheap; smoothing bandwidth has a direct, interpretable physical size (cm) that's identical across bins/cells/sessions — good for comparing spatial resolution apples-to-apples.
✅ Never smears information in from far away — a bin's rate reflects only its true local vicinity.
❌ Blind to occupancy — sparse bins stay noisy or get masked, producing the patchy/NaN-heavy look you're seeing.
❌ Same bandwidth is applied to a well-sampled peak and a barely-visited corner alike — not really the "correct" amount of smoothing for either.

Adaptive binning:
✅ Smoothing bandwidth matches local data density — sparse bins get more denoising, dense bins get less, which is the statistically principled reason this method exists in the literature.
✅ Far fewer masked bins, since sparse regions actively search outward instead of being at the mercy of a small fixed window.
❌ Effective spatial resolution now varies across the map — a "hot pixel" might represent a true 2 cm estimate in one place and an effectively 20+ cm-radius pooled estimate elsewhere. The map's apparent smoothness can be mistaken for spatial precision it doesn't have.
❌ More sensitive to tuning (ADAPTIVE_ALPHA_BASE, ADAPTIVE_MAX_RSQ) that lacks the intuitive cm-based interpretation a fixed kernel has.
❌ Worth flagging: your own AdaptiveBinFixes.md has open TODOs that peak FR/SIS look "way off" for this method, and a suspicion arena_width_cm may be inflating values — so the adaptive map looking cleaner isn't yet confirmation it's correct. Also note per your own algo doc: peak_fr and SIR in the plot title are computed from the raw fixed-bin map either way, not the adaptive one — only mean_fr, sparsity, and coherence actually use the adaptive map, so don't assume all reported stats moved together with the visual change.
Bottom line: the adaptive map looking "cleaner" is expected behavior, not a bug — but given the unresolved TODOs on peak FR/SIS, I'd hold off treating it as more quantitatively trustworthy until those are checked.