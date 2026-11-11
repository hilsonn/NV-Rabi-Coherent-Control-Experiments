import numpy as np

from models import noise
from models.common import GAMMA

MODES = ("sq", "sq_alt", "dq")


def coefficients(cfg, bias_t=None):
    """Apparent-field drift per kelvin (T/K) of each readout mode, split into diamond and magnet temperature.

    Lines: nu_pm = D +- gamma B. SQ on the -1 line, SQ on +1, alternation and DQ use (nu+ - nu-)/2gamma.
    """
    th = cfg["thermal"]
    d_t = th["d_tempco_khz_per_k"] * 1e-3 / GAMMA  # T/K equivalent of the D shift
    b = cfg["bias"]["field_mt"] * 1e-3 if bias_t is None else bias_t
    db = b * th["magnet_tempco_per_k"]
    return {"sq_m1": {"diamond": -d_t, "magnet": db}, "sq_p1": {"diamond": d_t, "magnet": db},
            "sq": {"diamond": -d_t, "magnet": db}, "sq_alt": {"diamond": 0.0, "magnet": db},
            "dq": {"diamond": 0.0, "magnet": db}}


def khz_per_k(cfg, bias_t=None):
    c = coefficients(cfg, bias_t)
    d = cfg["thermal"]["d_tempco_khz_per_k"]
    gb = GAMMA * 1e3 * (cfg["bias"]["field_mt"] * 1e-3 if bias_t is None else bias_t) * cfg["thermal"]["magnet_tempco_per_k"]
    return {"nu_minus": d - gb, "nu_plus": d + gb, "dq_sum": 2 * gb, "d_only": d}


def heating_rise_k(cfg, heating_k_per_w):
    mw = cfg["microwave"]
    return heating_k_per_w * mw["drive_power_w"] * mw["rf_duty"]


def drift_asd(cfg, mode, f, heating_k_per_w=0.0):
    """Analytic apparent-field ASD (T/rtHz) of the temperature noise in a mode at frequency f (Hz); components dict."""
    th, c = cfg["thermal"], coefficients(cfg)[mode]
    f = np.asarray(f, float)
    td = th["diamond_t_asd_k"] * f ** (-th["diamond_t_slope"] / 2)
    tm = th["magnet_t_asd_k"] * f ** (-th["magnet_t_slope"] / 2)
    return {"diamond": abs(c["diamond"]) * td, "magnet": abs(c["magnet"]) * np.hypot(tm, th["magnet_coupling_frac"] * td)}


def series(cfg, heating_k_per_w, rng):
    """Apparent-field series (T) per mode and component at the configured rate and duration."""
    th = cfg["thermal"]
    dt = 1.0 / th["sample_hz"]
    n = int(th["duration_s"] * th["sample_hz"])
    td = noise.generate_noise(n, dt, th["diamond_t_asd_k"], th["diamond_t_slope"], 1.0, rng)
    tm = noise.generate_noise(n, dt, th["magnet_t_asd_k"], th["magnet_t_slope"], 1.0, rng)
    t = np.arange(n) * dt
    heat = heating_rise_k(cfg, heating_k_per_w) * (1 - np.exp(-t / th["heat_tau_s"]))
    c = coefficients(cfg)
    out = {"t": t}
    for mode in MODES:
        out[f"{mode}_diamond"] = c[mode]["diamond"] * td
        out[f"{mode}_magnet"] = c[mode]["magnet"] * (tm + th["magnet_coupling_frac"] * td)
        out[f"{mode}_heating"] = c[mode]["diamond"] * heat
    return out
