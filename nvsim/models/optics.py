import numpy as np

from models.common import C_L, H_P, Q_E

PPM = 1.76e17  # NV per cm^3 per ppm
# state order: g0, g1, e0, e1, singlet


def geometry(cfg):
    o = cfg["optics"]
    th = np.arcsin(min(o["na"], 1.0) / o["n_diamond"])
    fres = 1 - ((o["n_diamond"] - 1) / (o["n_diamond"] + 1)) ** 2
    eta = min(1.0, 0.5 * (1 - np.cos(th)) * fres * o["filter_transmission"] * o["collection_multiplier"])
    area_cm2 = np.pi * (o["spot_diameter_um"] * 0.5e-4) ** 2
    vol = area_cm2 * o["depth_um"] * 1e-4
    return {"eta_coll": eta, "area_cm2": area_cm2, "n_nv": o["nv_ppm"] * PPM * vol, "volume_cm3": vol}


def pump_rate_per_us(cfg, p_mw):
    o = cfg["optics"]
    flux = np.asarray(p_mw) * 1e-3 / geometry(cfg)["area_cm2"] / (H_P * C_L / (o["wavelength_nm"] * 1e-9))
    return flux * o["sigma_abs_cm2"] * 1e-6


def _k(cfg, kp, r):
    """Rate matrix (..., 5, 5) with d rho/dt = K rho, rates per us. kp, r broadcast to a common shape."""
    k = cfg["optics"]["rates_per_us"]
    kp, r = np.broadcast_arrays(np.asarray(kp, float), np.asarray(r, float))
    m = np.zeros(kp.shape + (5, 5))

    def flow(src, dst, rate):
        m[..., dst, src] += rate
        m[..., src, src] -= rate

    flow(0, 2, kp)
    flow(1, 3, kp)
    flow(2, 0, k["k_r"])
    flow(3, 1, k["k_r"])
    flow(2, 4, k["k_isc0"])
    flow(3, 4, k["k_isc1"])
    flow(4, 0, k["k_s0"])
    flow(4, 1, k["k_s1"])
    flow(0, 1, r)
    flow(1, 0, r)
    return m


def steady(cfg, kp, r):
    m = _k(cfg, kp, r)
    m[..., -1, :] = 1.0
    rhs = np.zeros(m.shape[:-1])
    rhs[..., -1] = 1.0
    return np.linalg.solve(m, rhs[..., None])[..., 0]


def fluor(cfg, rho):
    return cfg["optics"]["rates_per_us"]["k_r"] * (rho[..., 2] + rho[..., 3])


def mw_rate(cfg, omega_mhz, delta_mhz, kp):
    """Equivalent incoherent rate (per us) averaged over the three hyperfine lines."""
    sp = cfg["spin"]
    g2 = 1.0 / sp["t2star_sq_us"] + np.asarray(kp) / 2
    om = 2 * np.pi * np.asarray(omega_mhz)
    tot = 0.0
    for m in (-1, 0, 1):
        d = 2 * np.pi * (np.asarray(delta_mhz) - m * sp["hyperfine_mhz"])
        tot = tot + (om**2 / 2) * g2 / (g2**2 + d**2) / 3
    return tot


def photon_scale(cfg):
    g = geometry(cfg)
    return g["n_nv"] * g["eta_coll"] * 1e6  # NV count x collection x (1/us -> 1/s)


def photocurrent(cfg, rate_hz, p_mw):
    o, d = cfg["optics"], cfg["detection"]
    qe_a_per_ph = d["responsivity_a_per_w"] * H_P * C_L / (o["emission_nm"] * 1e-9)
    return rate_hz * qe_a_per_ph, o["bg_leak"] * p_mw * 1e-3 * o["bg_responsivity_a_per_w"]


def cw_spectrum(cfg, p_mw, omega_mhz, delta=None):
    """CW ODMR at one laser and microwave power: returns delta (MHz), contrast of the dip (total), detected photon rate (off resonance)."""
    g = cfg["optics"]["grid"]
    delta = np.linspace(-g["detuning_mhz"], g["detuning_mhz"], g["n_detuning"]) if delta is None else delta
    kp = pump_rate_per_us(cfg, p_mw)
    f_on = fluor(cfg, steady(cfg, kp, mw_rate(cfg, omega_mhz, delta, kp)))
    f_off = fluor(cfg, steady(cfg, kp, 0.0))
    rate = f_off * photon_scale(cfg)
    i_sig, i_bg = photocurrent(cfg, rate, p_mw)
    c = cfg["optics"]["driven_fraction"] * (1 - f_on / f_off) * i_sig / (i_sig + i_bg)
    return delta, c, rate


def fwhm(delta, c):
    """FWHM (MHz) of the dip around the maximum (single hyperfine line if resolved)."""
    k = int(np.argmax(c))
    half = c[k] / 2
    lo = hi = k
    while lo > 0 and c[lo - 1] >= half:
        lo -= 1
    while hi < c.size - 1 and c[hi + 1] >= half:
        hi += 1
    return float(delta[hi] - delta[lo])


def cw_maps(cfg):
    g = cfg["optics"]["grid"]
    pw = np.array(g["laser_fracs"]) * cfg["optics"]["laser_power_mw"]
    om = np.array(g["omega_mhz"])
    c, lw, rate, slope = (np.zeros((pw.size, om.size)) for _ in range(4))
    for i, p in enumerate(pw):
        for j, o in enumerate(om):
            d, cc, r = cw_spectrum(cfg, p, o)
            c[i, j], lw[i, j], rate[i, j] = cc.max(), fwhm(d, cc), r
            slope[i, j] = np.abs(np.gradient(cc, d)).max()
    fom = slope * np.sqrt(rate)
    i, j = np.unravel_index(np.nanargmax(fom), fom.shape)
    return {"laser_mw": pw, "omega_mhz": om, "contrast": c, "linewidth_mhz": lw, "rate_hz": rate, "fom": fom, "best": (i, j)}


def pol_time(cfg, p_mw):
    """Time (us) for the m_s=0 fraction to reach pol_fraction of its steady value, from a thermal start."""
    kp = pump_rate_per_us(cfg, p_mw)
    k = _k(cfg, kp, 0.0)
    w, v = np.linalg.eig(k)
    vinv = np.linalg.inv(v)
    rho0 = np.array([1 / 3, 2 / 3, 0, 0, 0.0])
    t = np.geomspace(1e-3, 500, 1500)
    rho = np.einsum("ik,kt,k->ti", v, np.exp(w[:, None] * t[None]), vinv @ rho0).real
    f0 = (rho[:, 0] + rho[:, 2]) / (1 - rho[:, 4])
    tgt = cfg["optics"]["pol_fraction"] * f0[-1]
    idx = np.nonzero(f0 >= tgt)[0]
    return float(t[idx[0]]) if idx.size else np.inf


def pol_curve(cfg, p_mw, t):
    kp = pump_rate_per_us(cfg, p_mw)
    w, v = np.linalg.eig(_k(cfg, kp, 0.0))
    rho = np.einsum("ik,kt,k->ti", v, np.exp(w[:, None] * np.asarray(t)[None]), np.linalg.inv(v) @ np.array([1 / 3, 2 / 3, 0, 0, 0.0])).real
    return (rho[:, 0] + rho[:, 2]) / (1 - rho[:, 4])


def _evolve(cfg, kp, rho0, t):
    w, v = np.linalg.eig(_k(cfg, kp, 0.0))
    return np.einsum("ik,kt,k->ti", v, np.exp(w[:, None] * np.asarray(t)[None]), np.linalg.inv(v) @ rho0).real


def readout(cfg, p_mw=None, dead_us=0.2, t_max=None, laser_tail=None):
    """Fluorescence after a pi pulse versus after no pulse: per-NV counts in a window [0, w].

    Returns t (us), bright and dark cumulative detected photons (whole ensemble), total contrast C(w), the window w*
    maximising C(w) sqrt(N(w)). laser_tail: optional fraction of laser on during the pulse (finite rise): ignored if None.
    """
    p = cfg["optics"]["laser_power_mw"] if p_mw is None else p_mw
    kp = pump_rate_per_us(cfg, p)
    rho_ss = steady(cfg, kp, 0.0)
    rho = _evolve(cfg, 0.0, rho_ss, [dead_us])[0]
    swap = rho[[1, 0, 3, 2, 4]]
    t = np.linspace(0, t_max or cfg["optics"]["readout_max_us"], 600)
    fb = fluor(cfg, _evolve(cfg, kp, rho, t))
    fd = fluor(cfg, _evolve(cfg, kp, swap, t))
    dt = t[1] - t[0]
    nb, nd = np.cumsum(fb) * dt * photon_scale(cfg) * 1e-6, np.cumsum(fd) * dt * photon_scale(cfg) * 1e-6
    f = cfg["optics"]["driven_fraction"]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = f * (nb - nd) / nb
    score = np.nan_to_num(c * np.sqrt(nb))
    i = int(np.argmax(score))
    return {"t_us": t, "n_bright": nb, "n_dark": nd, "contrast": c, "i_opt": i, "window_ns": t[i] * 1e3,
            "contrast_opt": c[i], "photons_opt": nb[i], "f_bright": fb, "f_dark": fd}
