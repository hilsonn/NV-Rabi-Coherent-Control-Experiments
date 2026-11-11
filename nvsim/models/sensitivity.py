import numpy as np

from models import detection, optics, spin, thermal
from models.common import GAMMA, Q_E, override

MODES = ("cw", "po", "sq", "sq_alt", "dq")
DRIFT_KEY = {"cw": "sq", "po": "sq", "sq": "sq", "sq_alt": "sq_alt", "dq": "dq"}
NAMES = {"cw": "CW ODMR (FM lock-in)", "po": "pulsed ODMR", "sq": "SQ Ramsey", "sq_alt": "SQ Ramsey +-1 alt", "dq": "DQ Ramsey"}


def asd_at(m, f):
    a = m["noise_asd_a_rthz"]
    return np.exp(np.interp(np.log(f), np.log(a["f_hz"]), np.log(a["asd"])))


def excess(m, f, scale=1.0):
    return np.maximum(1.0, scale * asd_at(m, f) / np.sqrt(2 * Q_E * m["photocurrent_a"]))


def spin_cfg(cfg, m):
    """Config whose spin section reproduces the metric values (spread, DQ/SQ T2* ratio via the common-mode fraction)."""
    ratio = m["t2star_dq_us"] / m["t2star_sq_us"]
    cm = float(np.clip(1 - (1 / (2 * ratio)) ** 2, 0.0, 0.999))
    return override(cfg, {"spin.amp_spread_frac": m["rabi_spread_frac"], "spin.t2star_sq_us": m["t2star_sq_us"], "spin.common_mode_fraction": cm})


def pulse_omega(cfg, m):
    return min(cfg["protocol"]["rabi_mhz"], m["rabi_max_mhz"])


def cycle_us(cfg, m, mode, tau):
    pr = cfg["protocol"]
    if mode == "po":
        tp, npulse = pr["po_t_pi_us"], 1
    else:
        tp, npulse = spin.pulse_len("dq" if mode == "dq" else "sq", pulse_omega(cfg, m), m["dq_tone_gain"]), 2
    return pr["init_factor"] * m["pol_time_us"] + npulse * (tp + pr["pi_overhead_us"]) + tau + m["readout_window_ns"] * 1e-3 + pr["dead_us"]


def eta_pulsed(cfg, m, mode, tau, slope_t, noise_scale=1.0):
    """White-noise sensitivity (T/rtHz, one-sided ASD) for pulsed modes; slope_t is dy/dB in 1/T."""
    tc = cycle_us(cfg, m, mode, tau) * 1e-6
    sigma = np.sqrt(2.0 / m["photons_per_shot"]) * excess(m, 1 / tc, noise_scale)
    return sigma * np.sqrt(2 * tc) / np.maximum(slope_t, 1e-300)


def cw_shape(cfg, m):
    """FM lock-in first-harmonic slope (per MHz) of the CW lineshape at the operating point, maximised over depth."""
    d, c, _ = optics.cw_spectrum(cfg, m["laser_cw_mw"], m["omega_cw_mhz"])
    dc, k, ph = np.gradient(c, d), int(np.argmax(c)), np.linspace(0, 2 * np.pi, 181)
    best = (0.0, 0.0)
    for depth in cfg["protocol"]["cw_mod_depths_mhz"]:
        s = abs(np.trapezoid(np.interp(d[k] + depth * np.cos(ph), d, dc) * np.cos(ph), ph) / np.pi)
        best = max(best, (s, depth))
    return best[0] * optics.fwhm(d, c) / c.max(), best[1]


def eta_cw(cfg, m, sf, noise_scale=1.0):
    slope = sf * m["contrast_cw"] / m["linewidth_cw_mhz"] * GAMMA
    return np.sqrt(2) * noise_scale * asd_at(m, cfg["protocol"]["cw_f_mod_hz"]) / (m["photocurrent_a"] * slope)


def tau_grid(cfg, t2):
    pr = cfg["protocol"]
    return np.linspace(0.02 * t2, pr["tau_max_t2"] * t2, pr["n_tau"])


def evaluate(cfg, m, n_spins=1500):
    """Numerical evaluation of every mode with the spin model. Returns (results, calibration for the fast model)."""
    sc = spin_cfg(cfg, m)
    ens = spin.sample(sc, np.random.default_rng(cfg["seed"]), n=n_spins, t2star_us=m["t2star_sq_us"])
    om, hf, res, cal = pulse_omega(cfg, m), cfg["spin"]["hyperfine_mhz"], {}, {}
    sf, depth = cw_shape(cfg, m)
    res["cw"] = {"eta_white": eta_cw(cfg, m, sf), "tau_opt": np.nan, "f_eff_hz": cfg["protocol"]["cw_f_mod_hz"], "depth": depth}
    cal["cw"] = sf
    t_pi = cfg["protocol"]["po_t_pi_us"]
    om_po = min(0.5 / t_pi, m["rabi_max_mhz"])
    fr = np.linspace(-4, 4, 81)
    p = spin.odmr(fr, om_po, t_pi, ens)
    s_po = m["contrast_pulsed"] * np.abs(np.gradient(p, fr)).max() * GAMMA
    e = float(eta_pulsed(cfg, m, "po", 0.0, s_po))
    res["po"] = {"eta_white": e, "tau_opt": np.nan, "f_eff_hz": 1e6 / cycle_us(cfg, m, "po", 0.0), "slope_t": s_po, "odmr": (fr, p)}
    cal["po"] = {"slope": s_po, "contrast": m["contrast_pulsed"], "t2": m["t2star_sq_us"], "t_pi": t_pi}
    for mode in ("sq", "dq"):
        t2 = m["t2star_dq_us"] if mode == "dq" else m["t2star_sq_us"]
        tau = tau_grid(cfg, t2)
        s, ph = spin.ramsey_slope(tau, om, ens, mode, 0.0, None, m["dq_tone_gain"])
        st = m["contrast_pulsed"] * np.abs(s)
        eta = eta_pulsed(cfg, m, mode, tau, st)
        k = int(np.argmin(eta))
        cal[mode] = {"tau": tau, "s": np.abs(s), "t2": t2}
        res[mode] = {"eta_white": float(eta[k]), "tau_opt": float(tau[k]), "tau": tau, "eta_tau": eta, "slope_t": st, "phase": ph,
                     "f_eff_hz": 1e6 / cycle_us(cfg, m, mode, tau[k])}
    res["sq_alt"] = {**res["sq"], "eta_white": res["sq"]["eta_white"] * np.sqrt(cfg["protocol"]["alt_time_factor"])}
    for k_ in res:
        res[k_]["eta_white"] = float(res[k_]["eta_white"])
    return res, cal


def fast(cfg, m, cal, noise_scale=1.0):
    """Cheap evaluation of white sensitivity per mode (T/rtHz): rescales the spin-model slope curves in tau/T2*."""
    out = {"cw": float(eta_cw(cfg, m, cal["cw"], noise_scale))}
    c = cal["po"]
    w = lambda t2: np.hypot(0.5 / c["t_pi"], 1 / (np.pi * t2))
    s_po = c["slope"] * m["contrast_pulsed"] / c["contrast"] * w(c["t2"]) / w(m["t2star_sq_us"])
    out["po"] = float(eta_pulsed(cfg, m, "po", 0.0, s_po, noise_scale))
    for mode in ("sq", "dq"):
        t2 = m["t2star_dq_us"] if mode == "dq" else m["t2star_sq_us"]
        tau = tau_grid(cfg, t2)
        c = cal[mode]
        r = c["t2"] / t2
        st = m["contrast_pulsed"] * np.interp(tau * r, c["tau"], c["s"], right=0.0) / r
        out[mode] = float(np.min(eta_pulsed(cfg, m, mode, tau, np.abs(st), noise_scale)))
    out["sq_alt"] = out["sq"] * np.sqrt(cfg["protocol"]["alt_time_factor"])
    return out


def total_asd(cfg, m, res, mode, f, heating=0.0):
    """Field-equivalent ASD (T/rtHz) including temperature drift terms at frequency f (array)."""
    dr = thermal.drift_asd(cfg, DRIFT_KEY[mode], f)
    return np.sqrt(res[mode]["eta_white"] ** 2 + dr["diamond"] ** 2 + dr["magnet"] ** 2)


def eta_band(cfg, m, res, mode):
    lo, hi = cfg["sensitivity"]["band_hz"]
    f = np.geomspace(lo, hi, 50)
    return float(np.sqrt(np.mean(total_asd(cfg, m, res, mode, f) ** 2)))


def budget(cfg, m, res, mode, f=1.0):
    """Variance fractions at f (Hz): shot, electronics, residual RIN, diamond and magnet temperature drift."""
    comp = detection.asd(cfg["detection"], m["photocurrent_a"], f=[res[mode]["f_eff_hz"]])
    parts = np.array([comp["shot"][0] ** 2, comp["electronics"][0] ** 2, comp["rin"][0] ** 2])
    white = res[mode]["eta_white"] ** 2 * parts / parts.sum()
    dr = thermal.drift_asd(cfg, DRIFT_KEY[mode], f)
    v = np.concatenate([white, [float(dr["diamond"]) ** 2, float(dr["magnet"]) ** 2]])
    names = ["shot", "electronics", "RIN", "diamond T", "magnet T"]
    return dict(zip(names, v / v.sum()))


def monte_carlo(cfg, m, cal, rng, n=None):
    n = n or cfg["sensitivity"]["mc_draws"]
    unc = cfg["unc"]
    out = {k: np.zeros(n) for k in MODES}
    for i in range(n):
        mm = dict(m)
        for k, u in unc.items():
            if k in m:
                mm[k] = m[k] * np.exp(rng.uniform(-np.log(u), np.log(u)))
        ns = np.exp(rng.uniform(-np.log(unc["noise_scale"]), np.log(unc["noise_scale"])))
        for k, v in fast(cfg, mm, cal, ns).items():
            out[k][i] = v
    return out


def scale_metric(m, key, factor):
    mm = dict(m)
    if key == "photon_rate":
        for k in ("photon_rate_hz", "photocurrent_a", "photons_per_shot"):
            mm[k] = m[k] * factor
    elif key == "t2star":
        mm["t2star_sq_us"], mm["t2star_dq_us"] = m["t2star_sq_us"] * factor, m["t2star_dq_us"] * factor
    elif key != "noise_scale":
        mm[key] = m[key] * factor
    return mm
