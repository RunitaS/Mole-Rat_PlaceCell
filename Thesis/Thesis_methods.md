# Thesis methods

## Animals
- Tetrode
    Fa8477  ♀ 
    Fa5834  ♀ 
    Fa1059 ♂
    Fa23BD  ♀ 
- Neuropixels
    Fa0156  ♀ 
    Fa1680378B ♂
<Get 'DOBs'>
6 *Fukomys anselli* 12 to 24 months old at time of implant

## Recordings arena and behavior
NOTE: We recorded sleep session only for the first tetrode implant animal but stopped recording them later because the point of recording these rest periods was to get a readout of bot silent and active cells which fire spontaneously during period of sleep. However, due tot he hyperactive nature of the animals, recoridng these sessions during periods dedicated to rest only aooyed them due to the tether weight while they remained fully active during this period making these recordings quiet useless. Sleep session recordings was discontinued later.

## Recording system

## Data analysis

### Animal tracking

### Single unit criteria

- Firing rate estimate using Kernel smoothed density estimate https://www.science.org/doi/10.1126/science.1114037 *see supp methods*

### Place cell criteria

#### Place field criteria
Peaks of individual fields detected followed by obtaining neighbouring contiguous bins within 20% of thir respective peak firing rate, with atleast 5 contiguous bins in Open and Linear arenas and 3 contiguous bins in Circlular track.

### LFP cleaning and theta detection
- IIR bandstop Notch filter at 50 Hz and it's harmonics
- MAD robust outlier filter
- Savitzky-Golay smoothing 

#### PSD plot using FOOOF ananlysis
- Post LFP signal cleaning procedure
- FOOOF (fitting oscillations and one over f) estimated the 1/f aperiodic component using knee methd

#### Autocorrelogram method to detect theta epochs and theta continuity

#### fBOSC and Bycycle analysis to characterize theta

### Spike theta coupling using pass index method




# Paper methods:

## Ferret ACG; Dunn et.al 2022

**Autocorrelation metric of oscillatory activity**: The LFP was first high-pass filtered using IIR (Infinite Impulse Response) filters (rat: 6th-order high-pass Chebyshev Type II filter with 80 dB of stopband attenuation and a passband edge frequency of 2 Hz; ferret: 4th-order high-pass Chebyshev Type II filter with 80 dB of stopband attenuation and a passband edge frequency of 1 Hz) and 50 Hz notch filtered (14th-order band-stop Chebyshev Type II filter with 60 dB of stopband attenuation, pass-band edge frequencies of 48 and 52 Hz and stop-band frequencies of 49 and 51 Hz). The signal was then segmented into one-second epochs and autocorrelograms were calculated for each epoch. For each autocorrelogram, the Euclidean distance (ED) to the autocorrelations of sine waves of varying frequency (4-14 Hz for the rat, 2–14 Hz for the ferret, 0.1 Hz increments) was calculated. The ED between the data and the sine autocorrelograms was normalised by the ED of the individual ED of each sine autocorrelogram. The ‘matched’ sine autocorrelogram (i.e. with the minimum Euclidean distance from the data epoch autocorrelogram) was used to identify the first autocorrelogram peak. The range of this peak was measured from the maxima of the first peak to the average of the two surrounding troughs. This peak range measurement was then normalised by the peak range of the matched sine, as the peak range was found to vary as a function of frequency. The frequency of the signal in the data epoch was estimated as the frequency of the sine wave that was used to calculate the matched sine autocorrelogram.
The head speed signal was also segmented into corresponding 1 s epochs, for which the mean speed was found for each epoch. For comparison of moving vs. immobile epochs (Figs. 4, 5) conservative speed thresholds were chosen to ensure good separation of locomotor contingencies (moving vs. immobile) given that speeds were averaged over a one second window.
*Also see fig 5 of supp data for detailed explanation of the method.*

## Pass Index; Climer et.al. 2013

**Field index**: Calculation of a measure of how in field the animal was, or the ‘field index’, started by calculating the occupancy-normalized firing rate for 1 × 1-cm bins of position data. Data were then smoothed by a two-dimensional (2D) convolution with a pseudo-Gaussian kernel with a five-pixel (5 cm) standard deviation (Fig. 3E). The value at each bin was then percentile normalized between 0 and 1, and this was called the field index map (Fig. 3F). Then, the trajectory of the animal was sampled evenly along the arc-length of the trajectory at as many points as there were position tracking samples (50 Hz). The nearest bins were then found by minimizing the difference between the x and y positions and the center of the bins via the MATLAB function: bsxfun (Fig. 3G, blue). The smoothing and small bin size contributed to a more continuous estimation of the field index.

**Omnidirectional pass index**: To compute the omnidirectional pass index, the field index along the trajectory was first band pass filtered to include frequencies between twice the largest spacing of grid cells we can observe in a 100-cm enclosure (1.7 per m) and one-eighth of the smallest spacing of grid cells reported by Hafting et al. (2005; 26.7 per m) using a zero phase shift Butterworth filter (Fig. 3G, green). The phase of this signal was then calculated by finding the argument of the complex analytic signal produced by the hilbert function in MATLAB, and normalized to −1 to 1 so that −1 represents the beginning of a pass, 0 represents the center and +1 represents the end. This signal was then sampled back into the video frequency of sampling using MATLAB's interp1 nearest-neighbor interpolation (

## Bycycle; Cole Voytek 2017 et.al.

**Theta cycle analysis**:  The presence and features of hippocampal theta oscillations were analyzed using our previously described cycle-by-cycle analysis approach ​(Cole and Voytek, 2018)​. Briefly, a broad bandpass filter (1-25 Hz) was applied and then peaks and troughs were localized (Figure 1A, dots) in order to segment the signal into theta (4-10 Hz) cycles. Note this broad bandpass filter did not substantially affect the theta oscillation asymmetry of interest (compare gray and black traces in Figure 1A). A peak-to-peak segmentation was chosen because spiking was concentrated around the trough (Figure 4D,E) and so bursts of spiking around the trough would be analyzed in a single cycle (rather than 2 cycles if a trough-to-trough segmentation was used). For each cycle, four features were computed as shown in Figure 1B: amplitude, period, rise-decay symmetry, and peak-trough symmetry. Rise and decay midpoints were defined as the time points at which the voltage was halfway between the adjacent peak and trough voltages. These midpoints were used to represent the boundaries between peak and trough segments. Rise-decay symmetry is defined as the fraction of the period that is comprised of the rise phase. Peak-trough symmetry is similarly defined as the fraction of the period comprised of the peak phase, but the period in this case is bounded by consecutive rise midpoints instead of consecutive peaks.  It is important to appreciate that the neural oscillations are not present during the entire recording ​(Feingold et al., 2015; Jones, 2016; Lundqvist et al., 2016)​. Therefore, it is useful to determine the segments of the signal in which the oscillation is present because measuring theta features of a signal segment without a prominent theta oscillation will add noise to the analysis ​(Cole and Voytek, 2018)​. Therefore, only cycles that are determined to be part of a theta oscillatory burst were analyzed. However, the task of identifying the segments of the signal with oscillatory components is challenging and currently unsolved ​(Kosciessa et al., 2018)​. It is unclear if there are discrete times in which an oscillator is on and off, so perhaps there is no objective solution.  The approach for burst detection has been thoroughly described previously ​(Cole and Voytek, 2018)​, but briefly, a segment (cycle) of the signal was determined to be part of an oscillatory burst if its amplitude and period were comparable to adjacent cycles, and if its rise and decay flanks were mainly monotonic. Like with any burst detection algorithm, it relies on thresholds that must be semi-arbitrarily defined ​(Feingold et al., 2015; Hughes et al., 2012)​. In order to address this limitation, we ran our analysis with a range of burst detection parameters to assure that results were not simply dependent on one specific choice of settings. For the results shown in the main paper, the parameters were chosen as those that optimized the F1 score (equally weighted precision and recall) of a simulated signal with a signal-to-noise ratio that appears roughly similar to the hippocampal theta rhythm ​(Cole and Voytek, 2018)​. Thresholds were set such that adjacent cycles’ amplitudes and periods could be no more than 60% and 45% different, respectively, and the cycle flanks must be at least 80% monotonic. With these settings, theta oscillations were detected to be present 50-85% of the time across recordings.