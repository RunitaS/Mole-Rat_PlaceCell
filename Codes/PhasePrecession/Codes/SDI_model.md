# SDI model

The core problem they're solving
O'Keefe & Recce (1993) found that place cells fire at progressively earlier phases of the theta cycle as the animal crosses the cell's place field — sometimes advancing by more than 360°. Nobody had a clear cellular mechanism for this. This paper proposes one based on competition between two things happening at opposite phases of theta.

The two competing signals (SDI = Somatic-Dendritic Interaction)
Somatic inhibition (theta-locked): During theta, the soma of the pyramidal cell gets rhythmically hyperpolarized by GABAergic basket/chandelier cells (driven by the septum). This inhibition also shunts the cell — input resistance drops ~39% — making the soma hard to excite. Somatic firing normally clusters near the negative peak of the extracellular theta wave (the point of least inhibition).

Dendritic depolarization (theta-locked, but ~180° out of phase with the soma): At the same time, the distal apical dendrites get rhythmically depolarized, driven by entorhinal (perforant path) input. This dendritic depolarization peaks at the positive phase of field theta — i.e., exactly when the soma is most inhibited.

So on every theta cycle, there's a built-in tug-of-war: the dendrite wants to fire the cell at one phase, the soma is actively vetoing it at that same phase via inhibition/shunting.

How a place field crossing shifts the balance
As the rat enters, crosses, and exits the place field, the strength of the dendritic excitatory drive ramps up and back down (this is the "ramp" depolarization onto the theta oscillation — modeled as I_dendrite = A + B·sin(2π f t), where the constant term A grows as the animal approaches field center).

At the edge of the field: dendritic drive is weak, so it can only produce a spike burst near the peak of the dendritic depolarization cycle — which is fixed at a late/positive phase relative to somatic theta. Only a few spikes occur, at a late phase.
As the animal moves toward the center: dendritic drive gets stronger. Because of the slow K⁺ current (I_KS) that dendrites/soma use to counterbalance the persistent Na⁺ current (I_NaP), the model neuron's threshold-crossing point shifts earlier and earlier within the cycle — stronger drive lets it reach threshold on the rising phase of the dendritic input rather than waiting for the peak. This produces a systematic phase advance as excitation increases, plus more spikes per cycle (a burst) because the higher drive keeps the cell above threshold longer.
Because the dendrite is ~180° out of phase with the soma, the amount of phase-advance possible is huge — the model shows advances of up to 360°, matching the experimentally observed >180° precession that a soma-only depolarization model could never produce (soma-only depolarization in their experiments and model topped out around 120–180°).
In short: the phase of firing within the theta cycle directly encodes how strongly the dendrite is being driven. Weak dendritic drive → late phase, few spikes, near field edge. Strong dendritic drive (field center) → early phase (wrapping through the entire cycle), more spikes, bursts. As the rat exits the field and drive weakens again, phase retreats back to late phase — this reproduces the classic precession pattern (increasing spike count toward center, peak firing at center, then progressively earlier phase across the whole crossing).

Why the out-of-phase relationship matters so much
If the soma and dendrite were driven in phase, added depolarization would just make the cell fire more but wouldn't force much of a phase shift — it would still fire near the same somatic-theta phase, just more so. It's specifically because the dendrite's preferred phase is opposite the soma's (i.e., dendritic drive is highest exactly when the soma is most hyperpolarized/shunted) that increasing dendritic strength can "pull" firing continuously earlier and earlier — the cell has to fight through decreasing inhibition/increasing excitability as it goes, and as dendritic drive gets strong enough, it starts winning that fight earlier and earlier in the cycle, eventually able to trigger spikes even during the rising phase of dendritic depolarization rather than only near its peak.

The slow K⁺ current (I_KS) — why it's asymmetric, not symmetric
Without I_KS, the model just fires symmetrically around the peak of dendritic input (no precession). **I_KS is a slowly-activating potassium current that builds up during the rising phase of depolarization and suppresses excitability for the rest of that cycle.** This is what breaks the symmetry: it biases spiking toward the early/rising portion of each dendritic depolarization wave, and as overall drive strengthens, that biased "early window" opens earlier and earlier — producing the progressive, ramp-dependent phase advance rather than a simple more-spikes-at-fixed-phase response.

The bottom line, in one sentence
Phase precession emerges because the pyramidal cell's spike timing is a readout of a competition between theta-locked somatic inhibition (vetoing spikes at most phases) and theta-locked dendritic excitation (which is 180° out of phase and whose strength tracks how deep into the place field the animal is) — as the dendrite gets progressively more excited near field center, it progressively "wins" the competition earlier and earlier in each cycle, dragging the spike phase backward through the whole theta cycle and, via the slow K⁺ current, doing so asymmetrically rather than just symmetrically clustering around a fixed peak.

What would be the condition for phase procession i.e starting at an earlier phase and moving towards a later phase in succeeding cycles? 
Is this biological plausible in case of any mutation? For example KCC2 mutation that causes chloride and potassium ion concentrations to be inbalanced.
Show less
Condition for the reverse pattern (early → late across cycles)
In the SDI model, the direction of phase shift is set by two ingredients working together:

The 180° phase-antagonism between somatic inhibition and dendritic excitation — dendritic drive peaks exactly when somatic inhibition is strongest.
The slow K⁺ current, I_KS, which activates progressively during the rising phase of depolarization and suppresses excitability for the rest of that cycle — this is what biases the "permissive" window toward the early part of each depolarizing cycle rather than symmetrically around its peak.
Given those two fixed ingredients, the direction of precession is set entirely by whether the net drive is increasing or decreasing from cycle to cycle (Fig. 12: onset goes from −20° at A=0.8 up to −350° at A=4 — more drive → more negative/earlier phase). So:

Rising drive (as when a rat approaches the center of a place field) → threshold is crossed progressively earlier in each successive rising phase → classic precession (late → early).
Falling drive (mirror image — e.g., as the cell's dendritic excitation wanes) → threshold-crossing retreats back toward the peak/later part of the cycle → phase shifts early → late.
So the simplest condition for what you're calling "phase procession" is: the same 180°-antiphase/I_KS machinery, but with the excitatory drive amplitude (A) declining across successive cycles rather than growing. This is mechanistically just precession run in reverse-drive-direction, not a different mechanism — which is also why, in real data, you don't usually see a clean early→late sweep within one field crossing: firing probability collapses as drive falls off, so there are too few spikes left on the "declining" side to resolve a mirror-image sweep.

A second, independent route to the same outcome is to change the phase relationship itself rather than the drive trajectory: if dendritic excitation and somatic inhibition were shifted away from strict antiphase (say toward in-phase), the direction the I_KS-biased window sweeps as drive increases could flip, since it's specifically the antiphase arrangement that makes stronger dendritic drive "eat into" the somatic-inhibition-dominated part of the cycle from one particular side.

Is this plausible with a KCC2 mutation?
Yes — and interestingly, KCC2 dysfunction attacks exactly the second ingredient (the antiphase relationship), not the drive-ramp ingredient, so it's a genuinely different route to disrupted/altered precession, not just a relabeling of the model's existing knob.

Mechanism: KCC2 extrudes intracellular Cl⁻ (with K⁺) to keep E_GABA hyperpolarized relative to rest. This is what lets the perisomatic basket/chandelier cell input in this paper's model act as hyperpolarizing inhibition that vetoes somatic spikes for most of the theta cycle (Fig. 2, Fig. 3). Loss-of-function KCC2 (as in SLC12A5 mutations linked to human epilepsies, or KCC2 downregulation after injury/seizures) raises intracellular Cl⁻ and depolarizes E_GABA — GABA-A input becomes weaker, purely shunting, or even net depolarizing.

Consequence for the model's two ingredients:

The conductance/shunting component of somatic inhibition (the input-resistance drop the paper measured, ~39%) can largely survive KCC2 loss, since that depends on Cl⁻ channels opening, not on their reversal potential.
The hyperpolarizing, spike-vetoing component is what degrades or reverses. Since it's precisely this hyperpolarization that keeps the soma "out of the competition" during the phase when the dendrite is depolarized, weakening it blurs the sharp phase-confinement of somatic firing to the negative theta peak (Fig. 2, Fig. 10).
In the more severe case (E_GABA fully depolarized), the GABAergic input at that same phase stops being a veto and becomes an additional depolarizing, theta-phase-locked input — i.e., it now acts more like a second excitatory drive that happens to be in phase with basket-cell firing rather than antiphase with dendritic excitation. That's exactly the "collapse the antiphase relationship" scenario above, which is one of the two levers that can reverse or scramble the direction of the phase-code ratchet.
How to frame this honestly: the paper doesn't test this, so anything here is an extrapolation from its own stated logic, not a reported result. The safest, best-supported prediction is that KCC2 loss-of-function would degrade phase precession's fidelity — noisier phase-locking, reduced or variable phase advance per cycle, broader spike-phase distributions — because the clean somatic veto that the whole ratchet depends on is compromised. A clean, literal reversal (spikes reliably sweeping early→late) is a more specific and stronger claim; it would require the GABAergic signal to become not just weaker but actually excitatory and reasonably well phase-locked in a way that systematically opposes the dendritic ratchet, which is plausible in principle under substantial Cl⁻ dysregulation but isn't something this paper's data can confirm.