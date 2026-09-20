"""Generate 10 simulated LFP traces for the pseudo-animal 'FaSimul'.

Purpose: null test for the PSD -> FOOOF theta pipeline (Fig4a_PSD_v2_Simul.py).
If the pipeline reports theta peaks in these traces, it is finding "theta" in
signals whose oscillation frequencies were assigned completely at random.

Each trace (5-10 min, 32 kHz, same as the real acquisition):
  * N random sinusoidal components, N ~ U{3..12}
  * every frequency ~ U(1, 100) Hz, random phase, random (log-uniform)
    amplitude, and its own slow random amplitude modulation -- nothing is
    placed at or excluded from the theta band
  * Poisson shot noise: Poisson spike counts per sample convolved with an
    exponential synaptic-like kernel, scaled relative to the oscillations
Traces are written as Neuralynx .ncs files (int16, 512-sample records,
16 kB header) so the real pipeline loads them unchanged. No tracking .csv is
written, so the running-speed filter is skipped for these files.

The ground-truth frequencies of every trace go to ground_truth.json.
"""
import os
import json
import numpy as np

ROOT_DIR   = r'C:/Runita/NMR/analysis/AllSort_Results/LFP'
ANIMAL_ID  = 'FaSimul'
N_TRACES   = 10
FS         = 32000
ADBitVolts = 0.000003051757812500000169
TARGET_STD_UV = 300.0          # overall trace std (uV), typical LFP scale
MASTER_SEED   = 20260919

ncs_dtype = np.dtype([
    ('timestamp'  , '<u8'),
    ('sc_number'  , '<u4'),
    ('cell_number', '<u4'),
    ('params'     , '<u4'),
    ('samples'    , '<i2', (512,)),
])


def simulate_trace(rng, dur_s):
    n = int(dur_s * FS)
    t = np.arange(n, dtype=np.float64) / FS

    n_comp = int(rng.integers(3, 13))
    freqs  = rng.uniform(1.0, 100.0, n_comp)
    amps   = np.exp(rng.uniform(np.log(0.2), np.log(1.0), n_comp))
    phases = rng.uniform(0, 2 * np.pi, n_comp)

    osc = np.zeros(n, dtype=np.float64)
    for f, a, ph in zip(freqs, amps, phases):
        am_f  = rng.uniform(0.01, 0.2)                  # slow amplitude modulation (Hz)
        am_ph = rng.uniform(0, 2 * np.pi)
        env   = 1.0 + 0.5 * np.sin(2 * np.pi * am_f * t + am_ph)
        osc  += a * env * np.sin(2 * np.pi * f * t + ph)
    osc /= osc.std()

    # Poisson shot noise: Poisson counts convolved with an exponential kernel
    lam    = rng.uniform(0.02, 0.3)                     # mean events / sample
    counts = rng.poisson(lam, n).astype(np.float64)
    tau    = rng.uniform(0.001, 0.005)                  # kernel decay (s)
    kt     = np.arange(int(6 * tau * FS)) / FS
    kernel = np.exp(-kt / tau)
    from scipy.signal import fftconvolve
    noise  = fftconvolve(counts - counts.mean(), kernel, mode='same')
    noise /= noise.std()

    noise_ratio = rng.uniform(0.3, 1.0)                 # noise std relative to oscillations
    x = osc + noise_ratio * noise
    x *= TARGET_STD_UV / x.std()
    return x, dict(freqs_hz=np.sort(freqs).round(3).tolist(),
                   amps=amps.round(3).tolist(), poisson_lambda=float(lam),
                   kernel_tau_s=float(tau), noise_ratio=float(noise_ratio))


def write_ncs(path, x_uv):
    counts = np.clip(np.round(x_uv / (ADBitVolts * 1e6)), -32768, 32767).astype('<i2')
    n_rec  = len(counts) // 512
    rec    = np.zeros(n_rec, dtype=ncs_dtype)
    rec['timestamp'] = 1_700_000_000_000_000 + np.arange(n_rec, dtype=np.uint64) * (512 * 1_000_000 // FS)
    rec['samples']   = counts[:n_rec * 512].reshape(n_rec, 512)
    with open(path, 'wb') as fh:
        fh.write(b'\x00' * 16 * 1024)
        rec.tofile(fh)


if __name__ == '__main__':
    rng = np.random.default_rng(MASTER_SEED)
    truth = {}
    for i in range(1, N_TRACES + 1):
        dur = float(rng.uniform(300, 600))              # 5-10 min
        x, info = simulate_trace(rng, dur)
        info['duration_s'] = round(dur, 1)
        folder = os.path.join(ROOT_DIR, ANIMAL_ID, f'Sim{i:02d}')
        os.makedirs(folder, exist_ok=True)
        write_ncs(os.path.join(folder, 'CSC1ch1.ncs'), x)
        truth[f'Sim{i:02d}'] = info
        print(f"Sim{i:02d}: {dur/60:.1f} min, {len(info['freqs_hz'])} components: {info['freqs_hz']}")
    with open(os.path.join(ROOT_DIR, ANIMAL_ID, 'ground_truth.json'), 'w') as fh:
        json.dump(truth, fh, indent=1)
