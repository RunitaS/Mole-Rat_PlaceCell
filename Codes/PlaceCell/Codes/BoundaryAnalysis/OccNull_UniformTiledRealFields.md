# Occupancy Null: Method 1 
## Unifromly tiled real fields

Method 1: Uniformly tiled fields
Extract the field index transformed place fields from each cell/.ntt file from a given recording session.
Distribute them such that they are unfiformly spread out across the arena with minimal overlap and minimal cutting of the real place field around the boundaries. The centroids of the place fields should be as equidistance as possible.
Now get the jump corrected trajectory for this session and assuming that the animal traversed through each place field that was uniformly tiled, what would the eventual firing rate map look like for that particular field? Extract the occupncy controlled rate maps for each field, retain the new firing rate that may have increased or decreased due to over or undersampling of the place field area.
Take these occupany controlled rate maps and plot the occupancy null mean rate maps of all 3 types (overall, field only, peak roportion).
Apply KDE analysis extactly like the one used on real data. 
Compare the two kernels using Duong's test and highlight the areas where real data is higher than occupancy null in green and those where real data is significantly lower than occupancy null in magneta outlines. 

## Execution:

Extract the fields. Every place field of every cell becomes a shape that can be moved around, with its own rate values.
Tile them. Slot centres come from a tessellation that splits the arena into equal shares, so the centres are as evenly spaced as the shape allows.
Fields are matched to slots so that as little as possible gets cut off by the wall: small fields go to wall slots.
Each field can then shift up to 2 cm to reduce cutting and overlap.
Each layout is a randomly rotated or reflected copy of one of 8 tessellations, so no fixed pattern shows up in the null.
Run the real trajectory through the tiled map. I used each session's jump-corrected, speed-filtered occupancy. Spikes per bin are drawn at the field's rate times the time spent there. The null cells then go through exactly the same code as the real cells (1 s dwell minimum, smoothing, field index, field detection, peak), so a field placed where the animal rarely went gets noisier, loses bins and its measured rate changes.
Pool and test. The null cells are pooled into all 3 map types, fitted with the same KDE, and compared with the Duong test. Green outlines mean real is higher than null; magenta means lower.
Settings I chose after testing

Field template 'raw': taking fields from the smoothed map smooths them twice. As a check I left every field at its own location: the raw template reproduced the real cells (peak 1.85 vs 1.84 Hz, field size 73 vs 75 bins), while the smoothed one lost 16% of the peak.
Slots 'cover': there is one field-sized tile per share of the arena, and each layout fills a random subset of them. Your literal spec (one slot per field) is available as 'fields'. The two gave nearly the same result here; 'cover' has less overlap (19% vs 25%) and won't push fields inward in sessions with few fields.
First results (open field, 'cover', 100 layouts)

Map type	Significant bins after correction	Clusters
Overall	none	none
Peak proportion	none	none
Field-only	10	one magenta cluster (real lower than null), about 14 cm from the wall
No map shows real data significantly above the null near the walls.
Occupancy is about 3× higher at the wall, but the null's overall field-index profile against distance to the wall is flat, and so is the real one.
Null cells fire about 45% fewer spikes than the real cells. Only about 231 of 716 bins per session pass the 1 s dwell criterion, so many tiled fields land in poorly sampled areas.
Limits to keep in mind

Few, large fields per session: a session with only a few large fields can't evenly cover the wall band or the centre. The dashed "tiling coverage" line in TiledNull_Diagnostics.png shows how flat the null really was. It stayed between about 0.8 and 1.15 on your data, but please check it each run. Nudging fields further from the wall (TILE_MAX_NUDGE_CM) cuts fewer fields but makes the null emptier at the walls, which would favour your hypothesis.
Null cells are not independent: in the default Duong setting, real bins from the overall and field-only maps are treated as independent, which makes the test liberal. DUONG_SAMPLE_SIZE_MODE = 'cells' is the conservative check.
No bursts: null spikes don't reproduce bursting.
New outputs: TiledNull_* mean maps and KDEs, TiledNull_Diagnostics.png, and TiledNull_Summary.xlsx (tiling quality per session, and real vs null spike count and rates per cell, i.e. the rate change from over- or under-sampling). The existing DuongTest_* files now compare against this null.