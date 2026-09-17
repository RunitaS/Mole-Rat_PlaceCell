# Place Cell criteria setting

Key finding confirmed: Gauss detects only 12/43 place cells vs Speed's 38/43. Gauss adds coherence bootstrap significance, stability testing, and extra speed columns. Now let me reverse-engineer the exact classification rule for each file to find the limiting factor.

No simple AND-rule on the Full sheet alone matches — place-cell classification here likely spans the half-session sheets (stability across halves). Since both files contain the same 43 units, the cleanest approach is to align them cell-by-cell and see exactly which cells flipped from place-cell in Speed to non-place-cell in Gauss, and which criterion caused each flip.

The exact deterministic rule involves a continuous stability/sparsity component (2 low-sparsity cells pass despite failing my p<0.05 stability threshold, 5 higher-sparsity cells fail despite passing it), but the attribution is model-independent and robust. Let me do one final clean verification of the key drivers.

I've traced this all the way through. Here's what's happening.

The two pipelines use different place-cell definitions

The reason isn't the data — both files contain the same 43 units. It's that Gaussv2 applies a much stricter classification rule.

Speedv2 gates a place cell essentially on one criterion: spatial-information bootstrap significance (bootstrap_sig). That flag alone reproduces its place_cell labels with 93% agreement and misses none. → 38/43 place cells.
Gaussv2 adds two extra conjunctive gates that a cell must also pass — coherence bootstrap significance (coherence_bootstrap_sig) and within-session stability (stability_p_value). Speedv2 doesn't even contain these columns. → 12/43 place cells.

So the drop from 38 to 12 comes entirely from the two extra "AND" requirements layered on top of the spatial-information test.

Which factor is the major contributor: stability

Taking the 36 units that pass the spatial-information test (the pool Speedv2 would keep) and asking what knocks them out in Gaussv2:

Why spatially-informative cells are rejected in Gaussv2 (of 36)

The within-session stability requirement is the single biggest contributor.

Stability rejects 18 of 36 spatially-informative cells (50%); coherence rejects 12 (33%).
On unique attribution: 9 cells are lost to stability alone, only 3 to coherence alone, and 9 fail both.
Counterfactual — if Gaussv2 dropped just one extra gate: dropping the stability requirement recovers the count to 24, while dropping coherence only recovers it to 18. Stability removes roughly twice as many cells as coherence.
One thing to rule out

A naive comparison of the flipped cells would suggest "speed modulation" is the culprit (73% of flipped cells aren't speed-modulated). That's a red herring — speed_modulated isn't a required gate in either pipeline. Of the 12 Gaussv2 place cells, only 6 are speed-modulated, so it can't be part of the AND condition. The co-occurrence is incidental.

Bottom line

Gaussv2 detects far fewer place cells because it demands cells be spatially informative and spatially coherent and temporally stable, whereas Speedv2 requires only spatial information. The stability criterion (stability_p_value) is the dominant filter driving the low count, with the coherence bootstrap test a secondary contributor.

One caveat worth noting: the exact Gaussv2 rule isn't a clean p < 0.05 cutoff on stability — a couple of highly spatially-selective (low-sparsity) cells pass despite a non-significant stability p-value, so the pipeline appears to blend a continuous selectivity/stability measure with the significance flags. If you can point me to the classification code or the intended thresholds, I can confirm the precise rule and tell you exactly how many cells each threshold choice would recover.