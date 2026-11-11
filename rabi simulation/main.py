"""Run the ensemble NV Rabi simulation (Path A noisy Rabi sweep + Path B sensitivity).

Usage:  python main.py [path/to/config.json]
Edit config.json to change experimental parameters; no code changes needed.

Path A: full nonlinear model, one noise sample per sweep point (interval dt = n_repeats * sequence length).
Path B: small-signal linear response about a fixed working point, long time series at sensitivity.dt_s.
Units: time us (s for acquisition), frequency MHz (Hz for noise PSDs), pulse length ns, field T.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import noise
import physics
import plotting

# source -> channel plotted/compared in noise_traces.png
PRIMARY_CHANNEL = {"baseline": "baseline", "omega": "omega", "detuning": "detuning",
                   "thermal": "detuning", "vibration": "detuning"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def sweep_dt(cfg: dict) -> float:
    """Time per Rabi-sweep point in s: n_repeats * sequence_length_us * 1e-6."""
    return cfg["readout"]["n_repeats"] * cfg["acquisition"]["sequence_length_us"] * 1e-6


def _channel_sum(chans: dict, channel: str):
    arrays = [v[channel] for v in chans.values() if channel in v]
    return np.sum(arrays, axis=0) if arrays else None


def _stats(values: list) -> tuple:
    return (float(np.mean(values)), float(np.std(values))) if values else (None, None)


def simulate_path_a(cfg: dict, rng: np.random.Generator) -> dict:
    """Noisy Rabi sweep. Realisation 0 draws its shot noise from `rng` (as the original code did), so
    with every new source off its data equal the previous version's. Returns arrays and fits."""
    drive, ens, ro, sens = cfg["drive"], cfg["ensemble"], cfg["readout"], cfg["sensitivity"]
    jit, vib = cfg["jitter"], cfg["vibration"]
    omega, det = drive["rabi_mhz"], drive["mw_detuning_mhz"]
    tau_us = np.linspace(0.0, drive["tau_max_ns"] / 1000.0, drive["n_points"])
    dt, n_real = sweep_dt(cfg), int(sens["n_realisations"])

    sigma_vib = (physics.vibration_sigma_mhz(vib, cfg["noise"]["f_ref_hz"], dt,
                                             cfg["acquisition"]["sequence_length_us"])
                 if vib["enabled"] else 0.0)
    offsets, amp = physics.sample_ensemble(ens, rng, sigma_vib)

    def jittered(om, de, quantise_ns):
        return physics.ensemble_population_jittered(
            tau_us, om, de, offsets, amp, jit["sigma_ns"], jit["n_quadrature"], quantise_ns)

    p_single = physics.single_spin_population(tau_us, omega, det)
    p_ens = (jittered(omega, det, jit["quantise_ns"]) if jit["enabled"]
             else physics.ensemble_population(tau_us, omega, det, offsets, amp))
    f_ens = physics.fluorescence(p_ens, ro["contrast"])

    rngs = noise.spawn_rngs(cfg["run"]["seed"], n_real, stream=0)
    curves, fits = [], []
    for r in range(n_real):
        chans = physics.realisation_noise(cfg, tau_us.size, dt, {k: v[r] for k, v in rngs.items()}, "A")
        eps_om, delta = _channel_sum(chans, "omega"), _channel_sum(chans, "detuning")
        if eps_om is None and delta is None:
            f = f_ens.copy()
        else:
            om_k = omega if eps_om is None else omega * (1.0 + eps_om)
            de_k = det if delta is None else det + delta
            p = (jittered(om_k, de_k, jit["quantise_ns"]) if jit["enabled"]
                 else physics.ensemble_population_pointwise(tau_us, om_k, de_k, offsets, amp))
            f = physics.fluorescence(p, ro["contrast"])
        base = _channel_sum(chans, "baseline")
        if base is not None:
            f = f * (1.0 + base)
        if ro["add_shot_noise"]:
            f = physics.add_shot_noise(f, ro, rng if r == 0 else rngs["shot"][r])
        curves.append(f)
        fits.append(physics.fit_damped_rabi(tau_us, f, omega))

    curves = np.array(curves)
    return {
        "tau_us": tau_us, "offsets": offsets, "amp": amp, "dt": dt, "sigma_vib_mhz": sigma_vib,
        "f_single": physics.fluorescence(p_single, ro["contrast"]), "f_ens": f_ens,
        "f_meas": curves, "fits": fits,
        "f_nojitter": (physics.fluorescence(
            physics.ensemble_population(tau_us, omega, det, offsets, amp), ro["contrast"])
            if jit["enabled"] else None),
        "f_quantised": (physics.fluorescence(jittered(omega, det, jit["quantise_ns"]), ro["contrast"])
                        if jit["enabled"] and jit["quantise_ns"] > 0 else None),
    }


def simulate_path_b(cfg: dict) -> dict:
    """Small-signal sensitivity simulation. Returns ASDs (T/sqrt(Hz)), working point and summary values."""
    drive, ens, ro, sw = cfg["drive"], cfg["ensemble"], cfg["readout"], cfg["sweeps"]
    sens, seq_us = cfg["sensitivity"], cfg["acquisition"]["sequence_length_us"]
    dt, n, n_real = sens["dt_s"], int(sens["n_samples"]), int(sens["n_realisations"])
    omega = drive["rabi_mhz"]
    rngs = noise.spawn_rngs(cfg["run"]["seed"], n_real, stream=1)

    offsets, amp = physics.sample_ensemble(dict(ens, n_spins=sens["n_spins_slope"]), rngs["ensemble"][0])
    wp = sens["working_point"]
    if wp["mode"] == "auto":
        found = physics.find_working_point(
            np.linspace(0.0, drive["tau_max_ns"] / 1000.0, drive["n_points"]),
            np.linspace(*sw["detuning_range_mhz"], sw["n_detuning"]), omega, offsets, amp, ro["contrast"])
        tau0, det0 = found["tau_ns"], found["detuning_mhz"]
    else:
        tau0, det0 = wp["tau_ns"], wp["detuning_mhz"]
    lr = physics.linear_response(tau0, det0, omega, offsets, amp, ro["contrast"])
    y0, dy_dd, dy_do = lr["y0"], lr["dy_dDelta"], lr["dy_dOmega"]
    dy_db = dy_dd * physics.GAMMA_E_MHZ_PER_T
    n_counts = ro["photons_per_shot"] * dt / (seq_us * 1e-6)

    contrib = {}
    primary = {}
    for r in range(n_real):
        chans = physics.realisation_noise(cfg, n, dt, {k: v[r] for k, v in rngs.items()}, "B")
        for src, ch in chans.items():
            c = contrib.setdefault(src, np.zeros((n_real, n)))
            c[r] = (y0 * ch.get("baseline", 0.0) + dy_do * omega * ch.get("omega", 0.0)
                    + dy_dd * ch.get("detuning", 0.0))
            primary.setdefault(src, np.zeros((n_real, n)))[r] = ch[PRIMARY_CHANNEL[src]]
    shot = np.array([rngs["shot"][r].normal(0.0, np.sqrt(y0 / n_counts), n) for r in range(n_real)])
    total = shot + sum(contrib.values())

    nperseg = sens["welch_nperseg"]
    freq, asd_total_y = (a[1:] for a in noise.welch_asd(total, dt, nperseg))
    to_t = 1.0 / abs(dy_db)
    sources_t = {src: noise.welch_asd(c, dt, nperseg)[1][1:] * to_t for src, c in contrib.items()}
    total_t = asd_total_y * to_t
    eta_shot = float(np.sqrt(2.0 * dt * y0 / n_counts) * to_t)

    spectra = {}
    for src, series in primary.items():
        f_s, asd = (a[1:] for a in noise.welch_asd(series, dt, nperseg))
        spectra[src] = (f_s, asd, physics.source_target_asd(cfg, src, f_s, dt))
    return {
        "dt": dt, "freq": freq, "total_t": total_t, "sources_t": sources_t, "eta_shot": eta_shot,
        "tau0_ns": tau0, "det0_mhz": det0, "dy_dB": dy_db, "y0": y0, "n_counts": n_counts,
        "corner_hz": physics.corner_frequency(freq, total_t, eta_shot),
        "eta_at": {f"{fq:g}": float(np.interp(fq, freq, total_t))
                   for fq in (0.1, 1.0, float(freq[-1])) if freq[0] <= fq <= freq[-1]},
        "traces": {src: (np.arange(n) * dt, series[0]) for src, series in primary.items()},
        "spectra": spectra, "asd_total_y": asd_total_y,
    }


def main(config_path: Path) -> dict:
    cfg = load_config(config_path)
    drive, ens, ro, sw, run = cfg["drive"], cfg["ensemble"], cfg["readout"], cfg["sweeps"], cfg["run"]

    out_dir = Path(run["output_dir"])
    if not out_dir.is_absolute():
        out_dir = config_path.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = run["dpi"]

    rng = np.random.default_rng(run["seed"])
    omega, det = drive["rabi_mhz"], drive["mw_detuning_mhz"]

    # --- Path A: Rabi sweep, realisation 0 plotted, all realisations give fit statistics ---
    a = simulate_path_a(cfg, rng)
    tau_us, offsets, amp = a["tau_us"], a["offsets"], a["amp"]
    tau_ns = tau_us * 1000.0
    fit = a["fits"][0]
    mean_curve, std_curve = a["f_meas"].mean(axis=0), a["f_meas"].std(axis=0)

    title = (
        f"Rabi: $\\Omega/2\\pi$={omega:g} MHz, $T_2^*$={ens['t2star_us']:g} us, "
        f"amp. spread {100 * ens['amplitude_spread_frac']:g}%"
    )
    plotting.plot_rabi(
        tau_ns, a["f_single"], a["f_ens"], a["f_meas"][0], None if fit is None else fit["curve"],
        out_dir / "rabi.png", dpi, title, band=(mean_curve, std_curve),
    )
    if cfg["jitter"]["enabled"]:
        plotting.plot_jitter(tau_ns, a["f_nojitter"], a["f_ens"], a["f_quantised"],
                             cfg["jitter"]["sigma_ns"], out_dir / "jitter.png", dpi)

    # --- detuning chevron ---
    detunings = np.linspace(*sw["detuning_range_mhz"], sw["n_detuning"])
    p_map = physics.detuning_sweep(tau_us, omega, detunings, offsets, amp)
    plotting.plot_chevron(tau_ns, detunings, physics.fluorescence(p_map, ro["contrast"]),
                          out_dir / "chevron.png", dpi)

    # --- power sweep ---
    scales = sw["power_scales"]
    p_pow = physics.power_sweep(tau_us, omega, scales, det, offsets, amp)
    f_pow = physics.fluorescence(p_pow, ro["contrast"])
    f_pow_meas = [physics.add_shot_noise(c, ro, rng) if ro["add_shot_noise"] else c for c in f_pow]
    fits = [physics.fit_damped_rabi(tau_us, c, omega * np.sqrt(s)) for s, c in zip(scales, f_pow_meas)]
    fitted_mhz = [None if f is None else f["rabi_mhz"] for f in fits]
    plotting.plot_power(tau_ns, scales, f_pow, fitted_mhz, omega, out_dir / "power_sweep.png", dpi)

    # --- Path B: sensitivity ---
    b = simulate_path_b(cfg)
    plotting.plot_sensitivity(
        b["freq"], b["total_t"], b["sources_t"], b["eta_shot"], b["corner_hz"], b["dy_dB"],
        f"Working point $\\tau_0$ = {b['tau0_ns']:.0f} ns, $\\Delta_0$ = {b['det0_mhz']:.2f} MHz, "
        f"dt_s = {b['dt']:g} s", out_dir / "sensitivity.png", dpi)
    if b["traces"]:
        plotting.plot_noise_traces(b["traces"], b["spectra"], out_dir / "noise_traces.png", dpi)

    rabi_mean, rabi_std = _stats([f["rabi_mhz"] for f in a["fits"] if f is not None])
    decay_mean, decay_std = _stats([f["decay_us"] for f in a["fits"] if f is not None])
    summary = {
        "input_rabi_mhz": omega,
        "pi_pulse_ns_ideal": 1000.0 / (2.0 * omega),
        "fit_rabi_mhz": None if fit is None else fit["rabi_mhz"],
        "fit_decay_us": None if fit is None else fit["decay_us"],
        "shot_noise_snr_per_point": physics.snr_estimate(ro),
        "power_sweep": [
            {"scale": s, "input_mhz": omega * float(np.sqrt(s)), "fitted_mhz": f}
            for s, f in zip(scales, fitted_mhz)
        ],
        "dt_s": a["dt"],
        "vibration_sigma_mhz_path_a": a["sigma_vib_mhz"],
        "working_point": {"tau_ns": b["tau0_ns"], "detuning_mhz": b["det0_mhz"],
                          "dy_dB_per_T": b["dy_dB"]},
        "eta_shot_t_per_rthz": b["eta_shot"],
        "eta_total_at_f_t_per_rthz": b["eta_at"],
        "corner_frequency_hz": b["corner_hz"],
        "fit_rabi_mhz_mean": rabi_mean, "fit_rabi_mhz_std": rabi_std,
        "fit_decay_us_mean": decay_mean, "fit_decay_us_std": decay_std,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Figures and summary.json written to {out_dir}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    here = Path(__file__).resolve().parent
    arg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here / "config.json"
    main(arg)
