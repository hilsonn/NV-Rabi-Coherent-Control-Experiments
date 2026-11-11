"""Coloured-noise numerics: PSD model, IFFT generator, Welch ASD. No matplotlib, no physics.

Conventions: PSDs are one-sided. Units of the series are whatever units the ASD is given in;
time in s, frequency in Hz here (the config's f_ref_hz is in Hz).

Known limits: noise below 1/(n*dt) is not represented because the DC bin is dropped; sources are
independent zero-mean Gaussian processes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

SOURCES = ("baseline", "omega", "detuning", "thermal", "vibration", "shot", "ensemble")


def psd_model(f, asd_at_ref: float, slope: float, f_ref: float) -> np.ndarray:
    """One-sided power-law PSD S(f) = A_ref^2 (f/f_ref)^-slope, for f > 0.

    f, f_ref in Hz; asd_at_ref in units/sqrt(Hz); returns units^2/Hz.
    """
    return asd_at_ref**2 * (np.asarray(f, dtype=float) / f_ref) ** (-slope)


def generate_noise_from_psd(n: int, dt: float, psd_fn, rng: np.random.Generator) -> np.ndarray:
    """Real zero-mean Gaussian series of length n (interval dt, s) with one-sided PSD psd_fn(f).

    psd_fn(f) takes positive frequencies (Hz) and returns units^2/Hz. Built by IFFT with
    X_k = sqrt(S(f_k) n / (2 dt)) g_k, g_k complex normal (real at an even-n Nyquist bin).
    The DC bin is zero, so noise below 1/(n dt) is not represented.
    """
    f = np.fft.rfftfreq(n, dt)
    g = (rng.standard_normal(f.size - 1) + 1j * rng.standard_normal(f.size - 1)) / np.sqrt(2.0)
    if n % 2 == 0:
        g[-1] = rng.standard_normal()
    spectrum = np.zeros(f.size, dtype=complex)
    spectrum[1:] = np.sqrt(psd_fn(f[1:]) * n / (2.0 * dt)) * g
    return np.fft.irfft(spectrum, n)


def generate_noise(n: int, dt: float, asd_at_ref: float, slope: float, f_ref: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Power-law noise series (units of asd_at_ref), PSD A_ref^2 (f/f_ref)^-slope; dt in s, f_ref in Hz."""
    return generate_noise_from_psd(n, dt, lambda f: psd_model(f, asd_at_ref, slope, f_ref), rng)


def welch_asd(x, dt: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (Hz) and one-sided ASD sqrt(S(f)) (units/sqrt(Hz)) of x (1-D, or realisations x samples).

    For 2-D input the PSD is averaged over realisations before the square root.
    """
    x = np.asarray(x, dtype=float)
    f, pxx = welch(x, fs=1.0 / dt, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return f, np.sqrt(pxx.mean(axis=0) if pxx.ndim == 2 else pxx)


def spawn_rngs(seed: int, n_realisations: int, stream: int = 0) -> dict[str, list[np.random.Generator]]:
    """Independent generators per source and per realisation from SeedSequence(seed).spawn.

    `stream` separates Path A (0) from Path B (1); adding a source name never shifts the others.
    """
    kids = np.random.SeedSequence(seed, spawn_key=(stream,)).spawn(len(SOURCES))
    return {
        name: [np.random.default_rng(s) for s in kid.spawn(n_realisations)]
        for name, kid in zip(SOURCES, kids)
    }
