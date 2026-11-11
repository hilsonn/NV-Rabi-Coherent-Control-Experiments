import numpy as np

from models import detection, microwave, optics, spin, thermal
from models.common import GAMMA, apply_measured, bom_totals, load_cfg, out, save_metrics, table, verdict


def run_subsystems(cfg, plot=False, save=False, apply_meas=False):
    """All sub-system models -> metrics dict (plus extras prefixed with an underscore-free name)."""
    rng = np.random.default_rng(cfg["seed"])
    sp, pr = cfg["spin"], cfg["protocol"]
    mw = microwave.model(cfg, rng)
    maps = optics.cw_maps(cfg)
    i, j = maps["best"]
    p_cw, om_cw = maps["laser_mw"][i], maps["omega_mhz"][j]
    d, c_cw, rate = optics.cw_spectrum(cfg, p_cw, om_cw)
    i_sig, i_bg = optics.photocurrent(cfg, rate, p_cw)
    i_cw = i_sig + i_bg
    ro = optics.readout(cfg)
    det = detection.summary(cfg["detection"], i_cw)
    cur = det["curves"]
    k = ro["i_opt"]

    ens = spin.sample(cfg, rng, n=3000, amp=mw["amp_array"])
    om_pulse = min(pr["rabi_mhz"], mw["rabi_max_mhz"])
    tr = np.linspace(0, 3, 300)
    fr = spin.fit_rabi(tr, spin.rabi(tr, om_pulse, ens))
    fits = {}
    cm = sp["common_mode_fraction"]
    for mode, sc in (("sq", 1.0), ("dq", 1 / (2 * np.sqrt(1 - cm)))):
        tau = np.linspace(0, 4 * sp["t2star_sq_us"] * sc, 400)
        y = spin.ramsey(tau, pr["hard_rabi_mhz"], ens, mode, det=0.8, gain2=mw["dq_tone_gain"])
        fits[mode] = spin.fit_ramsey(tau, y, mode, sp["hyperfine_mhz"], 0.8 * (2 if mode == "dq" else 1), sp["t2star_sq_us"] * sc)

    tk = thermal.khz_per_k(cfg)
    heat = mw["heating_k_per_w"]
    tc = thermal.coefficients(cfg)
    bt = bom_totals()
    m = {
        "collection_eff": optics.geometry(cfg)["eta_coll"], "photon_rate_hz": rate, "photocurrent_a": i_cw,
        "pol_time_us": optics.pol_time(cfg, cfg["optics"]["laser_power_mw"]),
        "contrast_cw": float(c_cw.max()), "linewidth_cw_mhz": optics.fwhm(d, c_cw),
        "contrast_pulsed": float(ro["contrast_opt"]), "readout_window_ns": float(ro["window_ns"]), "photons_per_shot": float(ro["photons_opt"]),
        "noise_asd_a_rthz": {"f_hz": cur["f"], "asd": cur["total"]}, "shot_ratio": det["shot_ratio"],
        **{kk: mw[kk] for kk in ("resonator_f0_mhz", "resonator_bw_mhz", "rabi_mhz_per_sqrtw", "rabi_max_mhz", "rabi_spread_frac", "dq_tone_gain", "heating_k_per_w")},
        "t2star_sq_us": fits["sq"]["t2star_us"], "t2star_dq_us": fits["dq"]["t2star_us"], "rabi_decay_periods": fr["periods"],
        "d_tempco_khz_per_k": cfg["thermal"]["d_tempco_khz_per_k"], "magnet_tempco_per_k": cfg["thermal"]["magnet_tempco_per_k"],
        "bias_field_t": cfg["bias"]["field_mt"] * 1e-3, "temp_asd_k_rthz": cfg["thermal"]["diamond_t_asd_k"],
        "power_w": bt["power_w"], "cost_usd": bt["cost_mid"],
        # extras
        "laser_cw_mw": float(p_cw), "omega_cw_mhz": float(om_cw), "mw_cw_w": float((om_cw / mw["rabi_mhz_per_sqrtw"]) ** 2),
        "rabi_pulse_mhz": float(om_pulse), "bg_current_a": float(i_bg), "dq_span_mhz": mw["dq_span_mhz"], "dq_bw_ok": mw["dq_bw_ok"],
        "fringe_contrast_sq": fits["sq"]["contrast"], "fringe_contrast_dq": fits["dq"]["contrast"],
        "n_nv": optics.geometry(cfg)["n_nv"],
    }
    series = thermal.series(cfg, heat, rng)
    if save:
        save_metrics(m)
        np.savez(out("drift_series.npz"), **series)
    if apply_meas:
        m, used = apply_measured(m)
    if plot:
        _plots(cfg, m, maps, mw, ro, det, tc, tk, rng)
    return m


def _plots(cfg, m, maps, mw, ro, det, tc, tk, rng):
    import plotting
    o = cfg["optics"]
    pw = np.linspace(1, o["laser_power_mw"] * 1.5, 40)
    rates = np.array([optics.fluor(cfg, optics.steady(cfg, optics.pump_rate_per_us(cfg, p), 0.0)) for p in pw]) * optics.photon_scale(cfg)
    i_s = np.array([sum(optics.photocurrent(cfg, r, p)) for r, p in zip(rates, pw)])
    plotting.curves("s2_saturation", [dict(curves=[(pw, i_s * 1e6, "photocurrent")], xlabel="laser power (mW)", ylabel="photocurrent (uA)", title="photocurrent vs laser power"),
                                      dict(curves=[(pw, rates, "fluorescence")], xlabel="laser power (mW)", ylabel="detected photons/s", title="fluorescence")])
    mk = maps["best"]
    plotting.heat("s2_cw_maps", [dict(x=maps["omega_mhz"], y=maps["laser_mw"], z=maps["contrast"] * 100, xlabel="Rabi frequency (MHz)", ylabel="laser (mW)", title="contrast (%)", mark=(maps["omega_mhz"][mk[1]], maps["laser_mw"][mk[0]])),
                                 dict(x=maps["omega_mhz"], y=maps["laser_mw"], z=maps["linewidth_mhz"], xlabel="Rabi frequency (MHz)", ylabel="laser (mW)", title="linewidth (MHz)")])
    t = np.geomspace(0.01, 50, 200)
    plotting.curves("s2_pol_time", [dict(curves=[(t, optics.pol_curve(cfg, p, t), f"{p:.0f} mW") for p in maps["laser_mw"]], xlabel="time (us)", ylabel="m_s=0 fraction", logx=True, title="polarisation", hlines=[0.9 * optics.pol_curve(cfg, o["laser_power_mw"], [500.0])[0]]),
                                    dict(curves=[(ro["t_us"], ro["f_bright"], "bright"), (ro["t_us"], ro["f_dark"], "after pi")], xlabel="time after laser on (us)", ylabel="fluorescence per NV (1/us)", title="readout transient", vlines=[ro["window_ns"] * 1e-3])])
    c = det["curves"]
    plotting.curves("s2_detection", [dict(curves=[(c["f"], c["total"] * 1e12, "total"), (c["f"], c["shot_ideal"] * 1e12, "shot (ideal)"), (c["f"], c["electronics"] * 1e12, "electronics"), (c["f"], c["rin"] * 1e12, "RIN residual")],
                                          xlabel="frequency (Hz)", ylabel="pA/rtHz", logx=True, logy=True, title=f"detection noise, shot ratio {m['shot_ratio']:.2f}")])
    f = np.linspace(2600, 3150, 600)
    ar = mw["amp_array"] * m["rabi_mhz_per_sqrtw"]
    plotting.curves("s2_microwave", [dict(curves=[(f, 10 * np.log10(np.abs(microwave.s11(f, cfg["microwave"]["f0_mhz"], cfg["microwave"]["q_unloaded"], cfg["microwave"]["coupling_beta"])) ** 2 + 1e-9), "|S11|^2 (dB)")], xlabel="MHz", ylabel="dB", title="resonator S11",
                                          vlines=list(mw["tone_hz_mhz"]))])
    h, e = np.histogram(mw["amp_array"], 40)
    plotting.curves("s2_rabi_hist", [dict(curves=[(0.5 * (e[1:] + e[:-1]), h, "")], xlabel="relative Rabi amplitude", ylabel="count", title=f"spread {m['rabi_spread_frac']*100:.2f}%")])
    labs = ["SQ -1 line", "SQ +1 line", "SQ alternation", "DQ"]
    vals = [abs(tc[k]["diamond"] + tc[k]["magnet"]) * 1e9 for k in ("sq_m1", "sq_p1", "sq_alt", "dq")]
    plotting.bars("s2_drift_per_k", labs, vals, "apparent field drift (nT/K)", "drift per kelvin, diamond and magnet moving together")


def main():
    cfg = load_cfg()
    m = run_subsystems(cfg, plot=True, save=True, apply_meas=True)
    t = cfg["targets"]
    table("stage 2: sub-system metrics", [
        ("collection_eff", m["collection_eff"], "", "INFO"),
        ("photon_rate_hz", m["photon_rate_hz"], "", "INFO"),
        ("photocurrent_a", m["photocurrent_a"], "", "INFO"),
        ("pol_time_us", m["pol_time_us"], "", "INFO"),
        ("contrast_cw", m["contrast_cw"], f">= {t['contrast_min']}", verdict(m["contrast_cw"] >= t["contrast_min"])),
        ("linewidth_cw_mhz", m["linewidth_cw_mhz"], "", "INFO"),
        ("contrast_pulsed", m["contrast_pulsed"], "", "INFO"),
        ("readout_window_ns", m["readout_window_ns"], "", "INFO"),
        ("photons_per_shot", m["photons_per_shot"], "", "INFO"),
        ("shot_ratio", m["shot_ratio"], f"<= {t['shot_ratio_max']}", verdict(m["shot_ratio"] <= t["shot_ratio_max"])),
        ("resonator_bw_mhz", m["resonator_bw_mhz"], f">= {m['dq_span_mhz']:.0f} (2 gamma B cos)", verdict(m["resonator_bw_mhz"] >= m["dq_span_mhz"])),
        ("rabi_mhz_per_sqrtw", m["rabi_mhz_per_sqrtw"], "", "INFO"),
        ("rabi_max_mhz", m["rabi_max_mhz"], "", "INFO"),
        ("rabi_spread_frac", m["rabi_spread_frac"], "", "INFO"),
        ("rabi_decay_periods", m["rabi_decay_periods"], f">= {t['rabi_decay_periods']}", verdict(m["rabi_decay_periods"] >= t["rabi_decay_periods"])),
        ("dq_tone_gain", m["dq_tone_gain"], "", "INFO"),
        ("heating_k_per_w", m["heating_k_per_w"], "", "INFO"),
        ("t2star_sq_us", m["t2star_sq_us"], "", "INFO"),
        ("t2star_dq_us", m["t2star_dq_us"], "", "INFO"),
        ("d_tempco_khz_per_k", m["d_tempco_khz_per_k"], f"{t['sq_tempco_khz_per_k']}", verdict(abs(m["d_tempco_khz_per_k"] - t["sq_tempco_khz_per_k"]) < 10)),
        ("power_w", m["power_w"], "", "INFO"),
        ("cost_usd (mid)", m["cost_usd"], "", "INFO"),
    ])
    print(f"CW operating point: laser {m['laser_cw_mw']:.0f} mW, Rabi {m['omega_cw_mhz']:.2f} MHz ({m['mw_cw_w']:.3g} W absorbed-equivalent)")


if __name__ == "__main__":
    main()
