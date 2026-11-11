"""Physics for ensemble Rabi oscillations of NV centres. No plotting in this file.

Model: each NV is a driven two-level system {m_s = 0, m_s = -1} in the rotating frame
with the rotating-wave approximation. Spin i has

    Omega_i = Omega * amp_i              (microwave amplitude inhomogeneity)
    Delta_i = mw_detuning - offset_i     (detuning spread: T2* dephasing + 14N hyperfine)

and the excited-state (m_s = -1) population after a pulse of length t is

    P_i(t) = Omega_i^2 / (Omega_i^2 + Delta_i^2) * sin^2( sqrt(Omega_i^2 + Delta_i^2) * t / 2 )

The ensemble signal is the average over spins. Readout is fluorescence, bright for m_s = 0.

Units: time in us, ordinary frequency in MHz, angular rates in rad/us.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

import noise

TWO_PI = 2.0 * np.pi
GAMMA_E_MHZ_PER_T = 28000.0  # electron gyromagnetic ratio, MHz/T (only converts detuning <-> field)


def dephasing_sigma_mhz(t2star_us: float) -> float:
    """Std-dev of the detuning spread (MHz) for a Gaussian free-induction decay exp(-(t/T2*)^2)."""
    return np.sqrt(2.0) / (TWO_PI * t2star_us)


def sample_ensemble(ens_cfg: dict, rng: np.random.Generator,
                    extra_sigma_mhz: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Draw per-spin resonance offsets (MHz) and relative microwave amplitudes.

    extra_sigma_mhz (MHz): additional Gaussian detuning spread (e.g. sub-point vibration),
    added in quadrature to the T2* spread. 0 leaves the draw unchanged.
    """
    n = int(ens_cfg["n_spins"])
    sigma = dephasing_sigma_mhz(ens_cfg["t2star_us"])
    if extra_sigma_mhz > 0.0:
        sigma = np.hypot(sigma, extra_sigma_mhz)
    offsets = rng.normal(0.0, sigma, n)

    hf = ens_cfg["hyperfine"]
    if hf["enabled"]:
        # 14N nuclear spin-1: three lines, equal weight
        offsets = offsets + rng.choice([-1, 0, 1], size=n) * hf["splitting_mhz"]

    amp = 1.0 + ens_cfg["amplitude_spread_frac"] * rng.standard_normal(n)
    return offsets, np.clip(amp, 0.0, None)


def rabi_population(tau_us, omega_mhz, detuning_mhz) -> np.ndarray:
    """Population in m_s = -1 for each spin and pulse length. Returns shape (n_spins, n_tau).

    omega_mhz, detuning_mhz: scalar or per-spin arrays, ordinary frequency (MHz).
    """
    omega = TWO_PI * np.atleast_1d(omega_mhz)[:, None]
    delta = TWO_PI * np.atleast_1d(detuning_mhz)[:, None]
    t = np.asarray(tau_us)[None, :]
    w = np.sqrt(omega**2 + delta**2)
    w = np.where(w == 0.0, 1e-12, w)
    return (omega / w) ** 2 * np.sin(w * t / 2.0) ** 2


def ensemble_population(tau_us, omega_mhz, mw_detuning_mhz, offsets_mhz, amp) -> np.ndarray:
    """Spin-averaged population versus pulse length."""
    return rabi_population(
        tau_us, omega_mhz * amp, mw_detuning_mhz - offsets_mhz
    ).mean(axis=0)


def single_spin_population(tau_us, omega_mhz, mw_detuning_mhz) -> np.ndarray:
    """Ideal single NV: no spread in detuning or amplitude."""
    return rabi_population(tau_us, omega_mhz, mw_detuning_mhz)[0]


def fluorescence(population: np.ndarray, contrast: float) -> np.ndarray:
    """Normalised fluorescence (baseline 1 for m_s = 0). Population in m_s = -1 lowers it."""
    return 1.0 - contrast * population


def add_shot_noise(signal: np.ndarray, readout_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Poisson photon-counting noise on a normalised signal; returns normalised noisy signal."""
    total = readout_cfg["photons_per_shot"] * readout_cfg["n_repeats"]
    return rng.poisson(np.clip(signal * total, 0.0, None)) / total


def snr_estimate(readout_cfg: dict) -> float:
    """Shot-noise-limited SNR of the contrast measurement, per point (baseline ~ 1)."""
    total = readout_cfg["photons_per_shot"] * readout_cfg["n_repeats"]
    return readout_cfg["contrast"] * np.sqrt(total)


def _damped_cos(t, c, a, f_mhz, t_decay_us):
    return c + a * np.exp(-t / t_decay_us) * np.cos(TWO_PI * f_mhz * t)


def fit_damped_rabi(tau_us, y, f_guess_mhz: float) -> dict | None:
    """Fit y = c + a exp(-t/Td) cos(2 pi f t). Returns fitted f (MHz), Td (us), a, c, or None on failure."""
    y = np.asarray(y)
    p0 = [y.mean(), 0.5 * (y.max() - y.min()), f_guess_mhz, 1.0]
    try:
        popt, _ = curve_fit(_damped_cos, tau_us, y, p0=p0, maxfev=20000)
    except RuntimeError:
        return None
    c, a, f, td = popt
    return {
        "rabi_mhz": abs(float(f)),
        "decay_us": abs(float(td)),
        "amplitude": float(a),
        "offset": float(c),
        "curve": _damped_cos(np.asarray(tau_us), *popt),
    }


def detuning_sweep(tau_us, omega_mhz, detunings_mhz, offsets, amp) -> np.ndarray:
    """Ensemble population map, shape (n_detuning, n_tau). Looped to keep memory small."""
    return np.array(
        [ensemble_population(tau_us, omega_mhz, d, offsets, amp) for d in detunings_mhz]
    )


def power_sweep(tau_us, omega_mhz, scales, mw_detuning_mhz, offsets, amp) -> np.ndarray:
    """Ensemble population for microwave power scales; Omega scales as sqrt(power).

    Returns shape (n_scales, n_tau).
    """
    return np.array(
        [
            ensemble_population(tau_us, omega_mhz * np.sqrt(s), mw_detuning_mhz, offsets, amp)
            for s in scales
        ]
    )


# ----------------------------------------------------------------------------------------
# Pulse-length jitter
# ----------------------------------------------------------------------------------------
def ensemble_population_pointwise(tau_us, omega_mhz, mw_detuning_mhz, offsets_mhz, amp) -> np.ndarray:
    """Spin-averaged population versus pulse length, with per-point drive parameters.

    tau_us: (n_points,) us. omega_mhz, mw_detuning_mhz: scalar or (n_points,) MHz (ordinary frequency).
    offsets_mhz, amp: per-spin arrays. Returns (n_points,).
    """
    omega = TWO_PI * np.atleast_1d(omega_mhz)[None, :] * np.asarray(amp)[:, None]
    delta = TWO_PI * (np.atleast_1d(mw_detuning_mhz)[None, :] - np.asarray(offsets_mhz)[:, None])
    t = np.asarray(tau_us, dtype=float)[None, :]
    w = np.sqrt(omega**2 + delta**2)
    w = np.where(w == 0.0, 1e-12, w)
    return ((omega / w) ** 2 * np.sin(w * t / 2.0) ** 2).mean(axis=0)


def ensemble_population_jittered(tau_us, omega_mhz, mw_detuning_mhz, offsets_mhz, amp,
                                 sigma_ns: float, n_quadrature: int, quantise_ns: float = 0.0) -> np.ndarray:
    """Ensemble population averaged over Gaussian shot-to-shot pulse-length jitter.

    Gauss-Hermite expectation of P(tau') under N(tau, sigma^2); tau' clipped at 0.
    tau_us in us, sigma_ns and quantise_ns in ns (quantise_ns > 0 rounds tau to that grid first).
    The expectation is computed, not sampled, so it adds no random scatter.
    """
    tau = np.asarray(tau_us, dtype=float)
    if quantise_ns > 0.0:
        step = quantise_ns / 1000.0
        tau = np.round(tau / step) * step
    nodes, weights = np.polynomial.hermite.hermgauss(int(n_quadrature))
    sigma_us = sigma_ns / 1000.0
    total = np.zeros_like(tau)
    for x, w in zip(nodes, weights):
        shifted = np.clip(tau + np.sqrt(2.0) * sigma_us * x, 0.0, None)
        total += w * ensemble_population_pointwise(shifted, omega_mhz, mw_detuning_mhz, offsets_mhz, amp)
    return total / np.sqrt(np.pi)


# ----------------------------------------------------------------------------------------
# Thermal and vibration noise (feeds baseline / Omega / detuning channels)
# ----------------------------------------------------------------------------------------
def thermal_detuning_coeff_mhz_per_k(th_cfg: dict) -> float:
    """c_D + gamma_e * B_bias * c_M in MHz/K (bias field converted from mT to T)."""
    return th_cfg["diamond_d_mhz_per_k"] + (
        GAMMA_E_MHZ_PER_T * th_cfg["bias_field_mt"] * 1e-3 * th_cfg["magnet_tempco_per_k"]
    )


def vibration_psd_m2(f, vib_cfg: dict, f_ref_hz: float, f_max_hz: float = np.inf) -> np.ndarray:
    """One-sided displacement PSD S_x(f) in m^2/Hz: power-law floor + second-order resonances.

    f in Hz (> 0). Resonances with f0 > f_max_hz are left out (not representable at that Nyquist).
    Each term peaks at asd_peak_m^2 at f = f0.
    """
    f = np.asarray(f, dtype=float)
    fl = vib_cfg["floor"]
    s = noise.psd_model(f, fl["asd_at_ref_m"], fl["slope"], f_ref_hz)
    for r in vib_cfg["resonances"]:
        if r["f0_hz"] <= f_max_hz:
            x, q = f / r["f0_hz"], r["q"]
            s = s + r["asd_peak_m"] ** 2 / q**2 / ((1.0 - x**2) ** 2 + (x / q) ** 2)
    return s


def vibration_sigma_mhz(vib_cfg: dict, f_ref_hz: float, dt_s: float, sequence_length_us: float,
                        n_grid: int = 4000) -> float:
    """Detuning spread (MHz) from vibration between the Path A Nyquist and 1/(2 * sequence length).

    sigma^2 = integral (gamma_e g)^2 S_x(f) df on a log grid from f_N = 1/(2 dt_s) to
    f_hi = 1/(2 sequence_length_us 1e-6). Returns 0 if f_hi <= f_N (zero-width grid).
    """
    f_n, f_hi = 0.5 / dt_s, 0.5 / (sequence_length_us * 1e-6)
    f = np.geomspace(f_n, max(f_hi, f_n), n_grid)
    gk = GAMMA_E_MHZ_PER_T * vib_cfg["bias_gradient_t_per_m"]
    return float(np.sqrt(gk**2 * np.trapezoid(vibration_psd_m2(f, vib_cfg, f_ref_hz), f)))


def realisation_noise(cfg: dict, n: int, dt: float, rngs: dict, path: str) -> dict:
    """One realisation of every enabled source as channel series, length n, interval dt (s).

    Returns {source: {channel: array}} with channel in {"baseline", "omega", "detuning"}:
    baseline and omega are fractional, detuning is MHz. `rngs` maps source -> Generator.
    path "A" adds deterministic microwave heating; path "B" (stationary) does not, and leaves out
    vibration resonances above its Nyquist frequency (printing a warning).
    Thermal feeds detuning and Omega from ONE temperature series, so they are fully correlated.
    """
    nz, out = cfg["noise"], {}
    f_ref = nz["f_ref_hz"]
    for name in ("baseline", "omega", "detuning"):
        s = nz[name]
        if s["enabled"]:
            out[name] = {name: noise.generate_noise(n, dt, s["asd_at_ref"], s["slope"], f_ref, rngs[name])}

    th = cfg["thermal"]
    if th["enabled"]:
        t = th["temperature"]
        d_t = noise.generate_noise(n, dt, t["asd_at_ref_k"], t["slope"], f_ref, rngs["thermal"])
        heat = th["mw_heating"]
        if path == "A" and heat["enabled"]:
            d_t = d_t + heat["delta_t_final_k"] * (1.0 - np.exp(-np.arange(n) * dt / heat["time_constant_s"]))
        out["thermal"] = {
            "detuning": thermal_detuning_coeff_mhz_per_k(th) * d_t,
            "omega": th["mw_gain_tempco_per_k"] * d_t,
        }

    vib = cfg["vibration"]
    if vib["enabled"]:
        nyq = 0.5 / dt
        if path == "B":
            dropped = [r["f0_hz"] for r in vib["resonances"] if r["f0_hz"] > nyq]
            if dropped:
                print(f"warning: vibration resonances {dropped} Hz exceed the Nyquist frequency "
                      f"{nyq:g} Hz and are left out")
        limit = nyq if path == "B" else np.inf
        x = noise.generate_noise_from_psd(
            n, dt, lambda f: vibration_psd_m2(f, vib, f_ref, limit), rngs["vibration"])
        out["vibration"] = {"detuning": GAMMA_E_MHZ_PER_T * vib["bias_gradient_t_per_m"] * x}
    return out


def source_target_asd(cfg: dict, source: str, f, dt: float) -> np.ndarray:
    """Target ASD of a source's primary channel series at f (Hz), Path B conventions.

    Units: fractional/sqrt(Hz) for baseline and omega, MHz/sqrt(Hz) for detuning, thermal, vibration.
    """
    nz = cfg["noise"]
    if source in ("baseline", "omega", "detuning"):
        s = nz[source]
        return np.sqrt(noise.psd_model(f, s["asd_at_ref"], s["slope"], nz["f_ref_hz"]))
    if source == "thermal":
        t = cfg["thermal"]["temperature"]
        return abs(thermal_detuning_coeff_mhz_per_k(cfg["thermal"])) * np.sqrt(
            noise.psd_model(f, t["asd_at_ref_k"], t["slope"], nz["f_ref_hz"]))
    vib = cfg["vibration"]
    return GAMMA_E_MHZ_PER_T * vib["bias_gradient_t_per_m"] * np.sqrt(
        vibration_psd_m2(f, vib, nz["f_ref_hz"], 0.5 / dt))


# ----------------------------------------------------------------------------------------
# Linear response and working point (Path B)
# ----------------------------------------------------------------------------------------
def linear_response(tau_ns: float, detuning_mhz: float, omega_mhz: float, offsets, amp,
                    contrast: float, step_rel: float = 1e-3) -> dict:
    """Fluorescence y0 and central-difference slopes at a working point (one ensemble draw).

    tau_ns in ns; detuning_mhz, omega_mhz in MHz. Returns y0 (normalised), dy_dDelta (1/MHz),
    dy_dOmega (1/MHz). Linear response only holds for noise small vs the linewidth and the contrast.
    """
    tau = np.array([tau_ns / 1000.0])

    def y(om, de):
        return fluorescence(ensemble_population(tau, om, de, offsets, amp), contrast)[0]

    h_d, h_o = step_rel * max(abs(omega_mhz), 1.0), step_rel * omega_mhz
    return {
        "y0": float(y(omega_mhz, detuning_mhz)),
        "dy_dDelta": float((y(omega_mhz, detuning_mhz + h_d) - y(omega_mhz, detuning_mhz - h_d)) / (2 * h_d)),
        "dy_dOmega": float((y(omega_mhz + h_o, detuning_mhz) - y(omega_mhz - h_o, detuning_mhz)) / (2 * h_o)),
    }


def find_working_point(tau_us, detunings_mhz, omega_mhz, offsets, amp, contrast: float) -> dict:
    """Grid point maximising |dy/dDelta| (fixed-tau, fixed-detuning readout, not Ramsey/echo).

    tau_us in us, detunings_mhz in MHz. Returns tau_ns, detuning_mhz and the |slope| map
    (n_detuning, n_tau) in 1/MHz so the choice can be verified.
    """
    y = fluorescence(detuning_sweep(tau_us, omega_mhz, detunings_mhz, offsets, amp), contrast)
    slope = np.abs(np.gradient(y, np.asarray(detunings_mhz, dtype=float), axis=0))
    i, j = np.unravel_index(np.argmax(slope), slope.shape)
    return {"tau_ns": float(tau_us[j] * 1000.0), "detuning_mhz": float(detunings_mhz[i]), "slope_map": slope}


def corner_frequency(f, asd, floor: float, factor: float = 1.5):
    """Highest frequency (Hz) at which asd >= factor * floor; None if never or always above."""
    above = np.asarray(asd) >= factor * floor
    if not above.any() or above.all():
        return None
    return float(np.asarray(f)[np.nonzero(above)[0][-1]])
