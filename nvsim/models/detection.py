import numpy as np

from models.common import K_B, Q_E, override


def variants(cfg):
    d = cfg["detection"]
    return {k: {**d, **v} for k, v in d["variants"].items()}


def freqs(d, n=200):
    return np.geomspace(d["f_min_hz"], d["f_max_hz"], n)


def asd(d, i_a, f=None, balanced=None):
    """Current-noise ASD (A/rtHz) components versus f for photocurrent i_a. d is a detection config dict."""
    f = freqs(d) if f is None else np.asarray(f, float)
    bal = d["balanced"] if balanced is None else balanced
    rf = d["rf_ohm"]
    shot_ideal = np.sqrt(2 * Q_E * i_a) * np.ones_like(f)
    shot = shot_ideal * np.sqrt(1 + d["ref_current_ratio"]) if bal else shot_ideal
    johnson = np.sqrt(4 * K_B * d["temp_k"] / rf) * np.ones_like(f)
    en = d["en_nv_rthz"] * 1e-9 * np.sqrt(1 + d["en_corner_hz"] / f)
    amp = np.sqrt((en / rf) ** 2 + (d["in_fa_rthz"] * 1e-15) ** 2 + (en * 2 * np.pi * f * d["pd_capacitance_pf"] * 1e-12) ** 2)
    lsb = d["adc_range_v"] / 2 ** d["adc_bits"]
    adc = max(d["adc_nv_rthz"] * 1e-9, lsb / np.sqrt(12) / np.sqrt(d["adc_fs_mhz"] * 1e6 / 2)) / rf * np.ones_like(f)
    rin = np.sqrt(d["rin_floor"] ** 2 + d["rin_at_1hz"] ** 2 * f ** (-d["rin_slope"])) * i_a
    if bal:
        rin = rin * 10 ** (-d["balance_rejection_db"] / 20)
    comp = {"shot": shot, "johnson": johnson, "amp": amp, "adc": adc, "rin": rin}
    total = np.sqrt(sum(v**2 for v in comp.values()))
    return {"f": f, "total": total, "shot_ideal": shot_ideal, **comp, "electronics": np.sqrt(johnson**2 + amp**2 + adc**2)}


def summary(d, i_a, band=(1.0, 100.0), balanced=None):
    r = asd(d, i_a, balanced=balanced)
    f = r["f"]
    sel = (f >= band[0]) & (f <= band[1])
    ratio = float(np.sqrt(np.mean(r["total"][sel] ** 2) / np.mean(r["shot_ideal"][sel] ** 2)))
    above = np.nonzero(r["total"] >= 2 * r["shot_ideal"])[0]
    f_exc = float(f[above[-1]]) if above.size else 0.0
    el2 = np.mean(r["electronics"][sel] ** 2)
    return {"shot_ratio": ratio, "f_excess_hz": f_exc, "i_shot_dominant_a": float(el2 / (2 * Q_E)), "curves": r}


def rejection_needed_db(d, i_a, band=(1.0, 100.0), target=2.0):
    """Smallest balancing rejection (dB) giving shot_ratio <= target over the band (None if never reached)."""
    for db in np.arange(0, 100.5, 0.5):
        if summary({**d, "balance_rejection_db": db}, i_a, band, True)["shot_ratio"] <= target:
            return float(db)
    return None
