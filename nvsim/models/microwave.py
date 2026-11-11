import numpy as np

from models import spin
from models.common import GAMMA

MU0 = 4e-7 * np.pi


def s11(f, f0, q0, beta):
    x = (np.asarray(f, float) - f0) / f0
    j = 2j * q0 * x
    return (1 - beta + j) / (1 + beta + j)


def absorbed(cfg, f):
    m = cfg["microwave"]
    return 1 - np.abs(s11(f, m["f0_mhz"], m["q_unloaded"], m["coupling_beta"])) ** 2


def bandwidth_mhz(cfg):
    m = cfg["microwave"]
    return m["f0_mhz"] * (1 + m["coupling_beta"]) / m["q_unloaded"]


def compress(cfg, p_cmd_w):
    m = cfg["microwave"]
    p = np.asarray(p_cmd_w, float)
    n = m["comp_sharpness"]
    return p / (1 + (p / m["psat_w"]) ** n) ** (1 / n)


def loop_field(pos_mm, a_mm, current_a, nseg=120):
    """Biot-Savart field (T) of a circular loop in the z=0 plane at points pos_mm (n,3)."""
    ph = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    src = a_mm * 1e-3 * np.stack([np.cos(ph), np.sin(ph), np.zeros(nseg)], 1)
    dl = a_mm * 1e-3 * (2 * np.pi / nseg) * np.stack([-np.sin(ph), np.cos(ph), np.zeros(nseg)], 1)
    r = pos_mm[:, None, :] * 1e-3 - src[None]
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    return MU0 * current_a / (4 * np.pi) * np.sum(np.cross(dl[None], r) / rn**3, axis=1)


def positions(cfg, rng):
    m, o = cfg["microwave"], cfg["optics"]
    n = m["n_positions"]
    w = 0.5 * o["spot_diameter_um"] * 1e-3
    x = rng.normal(m["sample_offset_mm"], w / 2, n)
    y = rng.normal(0, w / 2, n)
    z = m["sample_height_mm"] + rng.uniform(-0.5, 0.5, n) * o["depth_um"] * 1e-3
    return np.stack([x, y, z], 1)


def rabi_array(cfg, rng, p_abs_w=1.0, axis=None):
    """Per-NV Rabi frequency (MHz) for the driven orientation at absorbed power p_abs_w (peak amplitude fields)."""
    m = cfg["microwave"]
    i_pk = np.sqrt(2 * p_abs_w / m["z0_ohm"]) * m["field_gain"]
    b = loop_field(positions(cfg, rng), m["loop_radius_mm"], i_pk, m["n_seg"])
    n = spin.axes(cfg)[cfg["spin"]["driven_orientation"]] if axis is None else axis
    bperp = np.linalg.norm(b - (b @ n)[:, None] * n[None], axis=1)
    return GAMMA * bperp / np.sqrt(2)


def dq_tones(cfg):
    sp, b = cfg["spin"], cfg["bias"]
    bd = np.array(b["direction"], float)
    bd /= np.linalg.norm(bd)
    cos_t = float(bd @ spin.axes(cfg)[sp["driven_orientation"]])
    shift = GAMMA * b["field_mt"] * 1e-3 * cos_t
    return sp["d_mhz"] - shift, sp["d_mhz"] + shift, cos_t


def model(cfg, rng=None):
    rng = rng or np.random.default_rng(cfg["seed"])
    m, sp = cfg["microwave"], cfg["spin"]
    arr = rabi_array(cfg, rng)
    mean = float(arr.mean())
    f_lo, f_hi, cos_t = dq_tones(cfg)
    a = np.sqrt(absorbed(cfg, np.array([f_lo, f_hi])))
    span = f_hi - f_lo
    bw = bandwidth_mhz(cfg)
    p_sat_out = float(compress(cfg, 100 * m["psat_w"]))
    return {
        "resonator_f0_mhz": m["f0_mhz"], "resonator_bw_mhz": bw,
        "rabi_mhz_per_sqrtw": mean * np.sqrt(absorbed(cfg, sp["d_mhz"])),
        "rabi_max_mhz": mean * np.sqrt(p_sat_out * absorbed(cfg, sp["d_mhz"])),
        "rabi_spread_frac": float(arr.std() / mean), "amp_array": arr / mean,
        "dq_tone_gain": float(a.min() / a.max()), "dq_span_mhz": span, "dq_bw_ok": bool(bw >= span),
        "switch_leak_amp": 10 ** (-m["switch_isolation_db"] / 20),
        "heating_k_per_w": m["r_th_k_per_w"] * m["diss_fraction"] * float(absorbed(cfg, sp["d_mhz"])),
        "tone_hz_mhz": (f_lo, f_hi),
    }


def omega_vs_power(cfg, p_cmd_w, k_per_sqrtw):
    """Rabi frequency (MHz) after amplifier compression for commanded power(s) (W)."""
    return k_per_sqrtw * np.sqrt(compress(cfg, p_cmd_w))
