# Mean RM analysis


## Quadrant division

(The earlier, simpler diagnostics are still there too — QuadrantFold_Diagnostic_*.png.)

Each new figure walks the full cut → register → average protocol in 3 rows:

Row 1 — overview: whole arena split into 4 raw quadrants/arcs, the fold colour-code over the whole grid, and the final folded reference region.
Row 2 — each of the 4 quadrants/arcs individually, as physically cut (no transform applied) — you can see they're mirror images of each other (Q1/Q2 flip left-right, Q1/Q3 flip up-down, Q1/Q4 flip both; for the ring, each arc keeps its true 0–90/90–180/180–270/270–360° position).
Row 3 — the same 4 pieces after registration (the actual transform handler._quad_idx_flat applies), plus their bin-by-bin average = the real fold output.
What to look for: in Row 3, all 4 registered panels should look pixel-identical to each other (same colour in the same spot = same bin matched). That's true for all three arenas here — open field and linear track show clean mirror-registration (Row 2 mirrored → Row 3 identical), and the circular track's 4 arcs roll onto the reference arc with no rotation artifact. So for the current default geometry configs, the matching itself is internally consistent — no axis-swap or transpose bug.

One caveat that these plots can't surface (since they only test index-space symmetry, not real geometry): _build_reflect_quadrant_fold centers the fold on the bin-grid center (nx*bin_cm/2), not the arena's physical center (diameter_cm/2). Those coincide exactly for the current configs (60 cm / 2 cm bins, 80×8 cm / 2 cm bins — all exact), so there's no offset today. But if you ever run this with a target_bin_cm or arena dimension that doesn't divide evenly, the circular/rectangular boundary would sit slightly off the fold axis, and quadrant partners near the edge could mismatch — worth checking if that's where your "wrong result" is coming from.


## Population vector analysis on mean RM


Use also from Fenton paper:

Unmasking the CA1 Ensemble Place Code by Exposures to Small and Large Environments: More Place Cells and Multiple, Irregularly Arranged, and Expanded Place Fields in the Larger Space
André A. Fenton, Hsin-Yi Kao, Samuel A. Neymotin, Andrey Olypher, Yevgeniy Vayntrub, William W. Lytton and Nandor Ludvig
Journal of Neuroscience 29 October 2008, 28 (44) 11250-11262; https://doi.org/10.1523/JNEUROSCI.2862-08.2008 

Population vector analyses.

CA1 population activity characterized as a firing rate vector was used to estimate the rat's location during a short period of time (Wilson and McNaughton, 1993; Fenton and Muller, 1998). The activity of simultaneously recorded cells was characterized as a population vector at each time step. The duration of time steps was varied from 0.17 to 5 s. The rat's position was decoded from the activity of ensembles comprised of 5–25 cells using a simple template-matching method (Wilson and McNaughton, 1993; Fenton and Muller, 1998). At each location, the average firing rate of each cell in the ensemble was used to construct a location-specific template firing rate vector. The decoded position was the location that maximized the projection of the current firing rate vector onto one of the location-specific template vectors. If there was no activity during a time step, the current vector was null, and no attempt to decode position was made for the time step. This method was chosen because it is an explicit test of how well location-specific firing rate itself predicts the rat's location (Wilson and McNaughton, 1993; Fenton and Muller, 1998). Note that the method makes no assumptions about the importance of previous or subsequent discharge and therefore does not attempt to optimize the decoding of position from spike trains (Brown et al., 1998).

We used simulated place cell spike trains to estimate how well position could be decoded if we had recorded more cells simultaneously. A simulated Poisson spike train for each place cell was derived from the average firing rate map of the cell by generating a spike if a random number exceeded the probability of observing a spike during the time step at the rat's current location (Fenton and Muller, 1998). This common but oversimplified inhomogeneous Poisson model was used because our goal was to reproduce place cell positional firing patterns in a straightforward manner rather than estimate the complex temporal dynamics of these spike trains (Barbieri et al., 2001). The positional firing patterns of the 322 place cells in the chamber were used to generate 322 location-specific simulated spike trains for a single position time series taken from a real recording. This simulated a 322 place cell ensemble recording in which temporal coordination beyond location specificity was ignored. The same was done for the cylinder, in which case a randomly selected subset of 322 place cell recordings from the two cylinder sessions was used. The rat's position was then decoded from the simulated 322 cell ensemble spike trains.