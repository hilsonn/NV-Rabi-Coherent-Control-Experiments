"""Plotting for the NV Rabi simulation. No physics in this file: it only draws arrays it is given."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path, dpi: int) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_rabi(tau_ns, single, ensemble, noisy, fit_curve, out: Path, dpi: int, title: str,
              band=None) -> Path:
    """Ideal single NV vs noise-free ensemble vs 'measured' (one realisation) with fit.

    band: optional (mean, std) arrays across noise realisations, drawn as a shaded mean +/- 1 std.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tau_ns, single, color="0.6", lw=1.2, label="ideal single NV")
    ax.plot(tau_ns, ensemble, color="C0", lw=1.8, label="ensemble (noise-free)")
    if band is not None:
        mean, std = band
        ax.fill_between(tau_ns, mean - std, mean + std, color="C1", alpha=0.25,
                        label="mean $\\pm$ 1 std over realisations")
    ax.plot(tau_ns, noisy, ".", color="C1", ms=3, label="one realisation")
    if fit_curve is not None:
        ax.plot(tau_ns, fit_curve, "--", color="k", lw=1.2, label="damped-cosine fit")
    ax.set_xlabel("microwave pulse length $\\tau$ (ns)")
    ax.set_ylabel("normalised fluorescence")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return _save(fig, out, dpi)


def plot_sensitivity(freq_hz, total_t, sources_t: dict, eta_shot_t, corner_hz, dy_db_per_t: float,
                     title: str, out: Path, dpi: int) -> Path:
    """Square-root PSD in T/sqrt(Hz) (log-log): total, each source, dashed shot-noise floor.

    The right axis shows the same data as fractional fluorescence ASD (1/sqrt(Hz)) = T/sqrt(Hz) * |dy/dB|.
    corner_hz (or None) marks where the total crosses 1.5x the floor.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(freq_hz, total_t, color="k", lw=2, label="total")
    for name, asd in sources_t.items():
        ax.loglog(freq_hz, asd, lw=1.1, label=name)
    ax.axhline(eta_shot_t, color="0.4", ls="--", label="shot-noise floor")
    if corner_hz is not None:
        ax.axvline(corner_hz, color="C3", ls=":", label=f"1.5x floor at {corner_hz:.3g} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("sensitivity (T/$\\sqrt{\\mathrm{Hz}}$)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    k = abs(dy_db_per_t)
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * k, lambda v: v / k))
    sec.set_ylabel("fractional fluorescence ASD (1/$\\sqrt{\\mathrm{Hz}}$)")
    return _save(fig, out, dpi)


def plot_noise_traces(traces: dict, spectra: dict, out: Path, dpi: int) -> Path:
    """Top: one realisation per source vs time. Bottom: Welch ASD of generated noise vs target.

    traces: name -> (t_s, x). spectra: name -> (f_hz, asd_measured, asd_target).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    for i, (name, (t, x)) in enumerate(traces.items()):
        ax1.plot(t, x, lw=0.7, color=f"C{i}", label=name)
        f, meas, target = spectra[name]
        ax2.loglog(f, meas, lw=0.9, color=f"C{i}", label=f"{name} (generated)")
        ax2.loglog(f, target, "--", lw=1.2, color="k")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("source amplitude (own units)")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("ASD (own units/$\\sqrt{\\mathrm{Hz}}$)")
    ax2.set_title("Welch ASD vs target (dashed)", fontsize=10)
    ax2.legend(fontsize=8)
    return _save(fig, out, dpi)


def plot_jitter(tau_ns, no_jitter, jittered, quantised, sigma_ns: float, out: Path, dpi: int) -> Path:
    """Ensemble Rabi curve with sigma = 0 vs the configured jitter, optionally with quantised tau."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tau_ns, no_jitter, color="C0", lw=1.6, label="no jitter")
    ax.plot(tau_ns, jittered, color="C3", lw=1.6, label=f"jitter $\\sigma_\\tau$ = {sigma_ns:g} ns")
    if quantised is not None:
        ax.plot(tau_ns, quantised, ":", color="k", lw=1.4, label="quantised $\\tau$ + jitter")
    ax.set_xlabel("microwave pulse length $\\tau$ (ns)")
    ax.set_ylabel("normalised fluorescence")
    ax.legend(fontsize=8)
    return _save(fig, out, dpi)


def plot_chevron(tau_ns, detunings_mhz, fluor_map, out: Path, dpi: int) -> Path:
    """Fluorescence vs pulse length and microwave detuning."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(tau_ns, detunings_mhz, fluor_map, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="normalised fluorescence")
    ax.set_xlabel("microwave pulse length $\\tau$ (ns)")
    ax.set_ylabel("microwave detuning (MHz)")
    ax.set_title("Ensemble Rabi chevron")
    return _save(fig, out, dpi)


def plot_power(tau_ns, scales, fluor_curves, fitted_mhz, nominal_mhz, out: Path, dpi: int) -> Path:
    """Rabi curves for several microwave powers, plus fitted Rabi frequency vs sqrt(power)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for s, curve in zip(scales, fluor_curves):
        ax1.plot(tau_ns, curve, label=f"power x{s:g}")
    ax1.set_xlabel("microwave pulse length $\\tau$ (ns)")
    ax1.set_ylabel("normalised fluorescence")
    ax1.set_title("Rabi curves vs microwave power")
    ax1.legend(fontsize=8)

    root = np.sqrt(np.asarray(scales, dtype=float))
    ax2.plot(root, nominal_mhz * root, "-", color="0.6", label="$\\Omega \\propto \\sqrt{P}$ (input)")
    ok = [i for i, f in enumerate(fitted_mhz) if f is not None]
    ax2.plot(root[ok], np.array(fitted_mhz, dtype=float)[ok], "o", color="C3", label="fitted")
    ax2.set_xlabel("$\\sqrt{\\mathrm{power\\ scale}}$")
    ax2.set_ylabel("Rabi frequency (MHz)")
    ax2.set_title("Fitted Rabi frequency")
    ax2.legend(fontsize=8)
    return _save(fig, out, dpi)
