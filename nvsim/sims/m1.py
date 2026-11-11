import numpy as np

import plotting
from models import detection, optics
from models.common import load_cfg, metrics, table, verdict

BAND = (10.0, 1e5)


def tia_bw(d, rf):
    return float(np.sqrt(d["amp_gbw_mhz"] * 1e6 / (2 * np.pi * rf * d["pd_capacitance_pf"] * 1e-12)))


def filtered_snr(ro, f3db_hz):
    """Readout SNR (contrast x sqrt counts) in the optimal window relative to an infinitely fast chain; single-pole filter."""
    t = ro["t_us"]
    dt = t[1] - t[0]
    tau = 1e6 / (2 * np.pi * f3db_hz)
    k = np.exp(-np.arange(t.size) * dt / tau)
    k /= k.sum()
    nb, nd = np.cumsum(np.convolve(ro["f_bright"], k)[: t.size]), np.cumsum(np.convolve(ro["f_dark"], k)[: t.size])
    j = ro["i_opt"]
    nb0, nd0 = np.cumsum(ro["f_bright"]), np.cumsum(ro["f_dark"])
    return ((nb[j] - nd[j]) / np.sqrt(nb[j])) / ((nb0[j] - nd0[j]) / np.sqrt(nb0[j]))


def main():
    cfg = load_cfg()
    m = metrics()
    t, z, d = cfg["targets"], cfg["m1"], cfg["detection"]
    o = cfg["optics"]
    i_w = m["photocurrent_a"]

    pw = np.linspace(1, 1.5 * o["laser_power_mw"], 40)
    rate = np.array([optics.fluor(cfg, optics.steady(cfg, optics.pump_rate_per_us(cfg, p), 0.0)) for p in pw]) * optics.photon_scale(cfg)
    i_sig = np.array([optics.photocurrent(cfg, r, p)[0] for r, p in zip(rate, pw)])
    i_bg = np.array([optics.photocurrent(cfg, 0, p)[1] for p in pw])
    mid = len(pw) // 2
    lin = i_sig[-1] / (i_sig[mid] * pw[-1] / pw[mid])
    rows = [("photocurrent at operating point (uA)", i_w * 1e6, "", "INFO"),
            ("background / signal (diamond removed)", float(i_bg[-1] / i_sig[-1]), "", "INFO"),
            ("linearity of signal at top power (1 = unsaturated)", lin, "", "INFO")]
    plotting.curves("m1_saturation", [dict(curves=[(pw, (i_sig + i_bg) * 1e6, "signal + background"), (pw, i_bg * 1e6, "background (diamond out)")], xlabel="laser (mW)", ylabel="uA", title="saturation curve")])

    var = detection.variants(cfg)
    panels, need = [], {}
    for name, dv in var.items():
        for bal in (False, True):
            s = detection.summary(dv, i_w, BAND, bal)
            c = s["curves"]
            tag = f"{name} {'balanced' if bal else 'single-ended'}"
            panels.append((c["f"], c["total"] * 1e12, tag))
            rows.append((f"shot ratio 10 Hz-100 kHz, {tag}", s["shot_ratio"], f"<= {t['shot_ratio_max']}", verdict(s["shot_ratio"] <= t["shot_ratio_max"])))
        need[name] = detection.rejection_needed_db(dv, i_w, BAND, t["shot_ratio_max"])
        rows.append((f"rejection needed (dB), {name}", need[name] if need[name] is not None else "unreachable", f"cfg {dv['balance_rejection_db']}", "INFO"))
    c = detection.asd(d, i_w)
    panels.append((c["f"], c["shot_ideal"] * 1e12, "ideal shot"))
    plotting.curves("m1_noise", [dict(curves=panels, xlabel="Hz", ylabel="pA/rtHz", logx=True, logy=True, title="detection noise by variant", vlines=list(BAND))])

    cur = np.geomspace(1e-8, 1e-3, 60)
    ok = np.array([detection.summary(d, i, BAND, True)["shot_ratio"] for i in cur])
    good = cur[ok <= t["shot_ratio_max"]]
    rows.append(("photocurrent window for shot ratio<=2 (uA)", f"{good.min() * 1e6:.3g}-{good.max() * 1e6:.3g}" if good.size else "none", f"working {i_w * 1e6:.3g}", verdict(good.size > 0 and good.min() <= i_w <= good.max())))
    plotting.curves("m1_ratio_vs_current", [dict(curves=[(cur * 1e6, ok, "balanced, config")], xlabel="photocurrent (uA)", ylabel="shot ratio (10 Hz-100 kHz)", logx=True, logy=True, hlines=[t["shot_ratio_max"]])])

    ro = optics.readout(cfg)
    w_ns = m["readout_window_ns"]
    rf_rows, curves = [], []
    for rf in z["rf_options_ohm"]:
        bw = tia_bw(d, rf)
        settle = 4.6 / (2 * np.pi * bw) * 1e9
        snr = filtered_snr(ro, bw)
        sr = detection.summary({**d, "rf_ohm": rf}, i_w, BAND, True)["shot_ratio"]
        rf_rows.append((rf, bw, settle, snr))
        rows.append((f"Rf {rf:.0e}: BW {bw / 1e3:.0f} kHz, settle {settle:.0f} ns, shot ratio {sr:.2f}", snr * m["contrast_pulsed"], f"window {w_ns:.0f} ns", verdict(settle <= w_ns) + f" (readout SNR x{snr:.3f})"))
    table("M1: detection and optics", rows)
    plotting.curves("m1_tia", [dict(curves=[([r[0] for r in rf_rows], [r[1] / 1e3 for r in rf_rows], "BW (kHz)")], xlabel="Rf (ohm)", ylabel="kHz", logx=True, logy=True, title="TIA bandwidth"),
                               dict(curves=[([r[0] for r in rf_rows], [r[3] for r in rf_rows], "readout SNR factor")], xlabel="Rf (ohm)", ylabel="relative readout SNR", logx=True, title="effect of settling")])


if __name__ == "__main__":
    main()
