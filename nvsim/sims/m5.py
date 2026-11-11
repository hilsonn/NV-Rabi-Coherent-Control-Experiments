import numpy as np

import plotting
from models import noise, sensitivity as sv, spin, thermal
from models.common import GAMMA, load_cfg, metrics, override, table, verdict
from sims.robust import contrast_fidelity, ens_with, composite
from sims.system import band, time_series

NT = 1e9


def cm_study(cfg, m, cal):
    cms = cfg["m5"]["cm_fracs"]
    om = cfg["protocol"]["hard_rabi_mhz"]
    rows = []
    for cm in cms:
        ens = spin.sample(override(cfg, {"spin.t2star_sq_us": m["t2star_sq_us"]}), np.random.default_rng(cfg["seed"]), n=1500, t2star_us=m["t2star_sq_us"], cm_frac=cm)
        t = {}
        for mode, sc in (("sq", 1.0), ("dq", 1 / (2 * np.sqrt(1 - cm)))):
            tau = np.linspace(0, 4 * m["t2star_sq_us"] * sc, 300)
            y = spin.ramsey(tau, om, ens, mode, det=0.8)
            t[mode] = spin.fit_ramsey(tau, y, mode, cfg["spin"]["hyperfine_mhz"], 0.8 * (2 if mode == "dq" else 1), m["t2star_sq_us"] * sc)["t2star_us"]
        theory = 1 / (2 * np.sqrt(1 - cm))
        mm = {**m, "t2star_dq_us": theory * m["t2star_sq_us"]}
        f = sv.fast(cfg, mm, cal)
        rows.append((cm, t["dq"] / t["sq"], theory, band(cfg, "sq", f["sq"]) / band(cfg, "dq", f["dq"])))
    return np.array(rows)


def dq_errors(cfg, m):
    z = cfg["m5"]
    rng = np.random.default_rng(cfg["seed"])
    om, g = sv.pulse_omega(cfg, m), m["dq_tone_gain"]
    segs = composite("square", np.pi)
    ideal = ens_with(cfg, 0.0, 0.0, rng, 1)
    ideal["a"] = 0.0
    sig_sq = np.sqrt(2) / (2 * np.pi * m["t2star_sq_us"])
    cmf = sv.spin_cfg(cfg, m)["spin"]["common_mode_fraction"]
    sig_mag = sig_sq * np.sqrt(1 - cmf)
    eps = 10 ** (-cfg["microwave"]["iq_image_rejection_db"] / 20)
    g_iq = (1 - eps) / (1 + eps)

    def run(amp=0.0, det=0.0, hf=False, gain=1.0):
        e = ens_with(cfg, amp, det, rng, 800)
        e["a"] = cfg["spin"]["hyperfine_mhz"] if hf else 0.0
        return contrast_fidelity(e, "dq", om, gain, segs, ideal)[0]

    base = run()
    cases = {"amplitude spread": run(amp=m["rabi_spread_frac"]), "resonator roll-off (tone gain)": run(gain=g),
             "IQ image imbalance": run(gain=g_iq), "detuning spread (T2*)": run(det=sig_mag), "14N hyperfine (+-A on the two tones)": run(hf=True)}
    total = run(amp=m["rabi_spread_frac"], det=sig_mag, hf=True, gain=g * g_iq)
    scan = [(gg, run(gain=gg)) for gg in np.linspace(0.3, 1.0, 8)]
    om_c = om * 10 ** (-cfg["microwave"]["iq_carrier_leak_db"] / 20)
    shift_khz = 2 * om_c**2 / (4 * (cfg["bias"]["field_mt"] * GAMMA)) * 1e3
    return base, cases, total, scan, shift_khz, om_c


def temp_steps(cfg):
    z, th = cfg["m5"], cfg["thermal"]
    kap = th["magnet_coupling_frac"]
    c = thermal.coefficients(cfg)
    dT = z["temp_step_k"]
    gam_khz_per_nt = GAMMA * 1e-6
    out = {}
    for mode in ("sq", "sq_p1", "sq_alt", "dq"):
        out[mode] = {"diamond only": c[mode]["diamond"] * dT * NT, "with magnet coupling": (c[mode]["diamond"] + kap * c[mode]["magnet"]) * dT * NT}
    d_khz = th["d_tempco_khz_per_k"] * dT
    return out, d_khz


def drift_ratio(cfg, tempco, bias_t, kappa, a_d, a_m):
    d_t = cfg["thermal"]["d_tempco_khz_per_k"] * 1e-3 / GAMMA
    cm = bias_t * tempco
    sq = np.sqrt(((-d_t + kappa * cm) ** 2) * a_d**2 + cm**2 * a_m**2)
    dq = np.sqrt(cm**2 * (a_m**2 + kappa**2 * a_d**2))
    return sq / np.maximum(dq, 1e-30)


def main():
    cfg = load_cfg()
    m = metrics()
    t, z, th = cfg["targets"], cfg["m5"], cfg["thermal"]
    rng = np.random.default_rng(cfg["seed"])
    sc = sv.spin_cfg(cfg, m)
    res, cal = sv.evaluate(cfg, m)
    white = {k: r["eta_white"] for k, r in res.items()}

    ens = spin.sample(sc, rng, n=1500, t2star_us=m["t2star_sq_us"])
    om = cfg["protocol"]["hard_rabi_mhz"]
    fr = {}
    for mode, scl in (("sq", 1.0), ("dq", m["t2star_dq_us"] / m["t2star_sq_us"])):
        tau = np.linspace(0, 4 * m["t2star_sq_us"] * scl, 400)
        y = spin.ramsey(tau, om, ens, mode, det=0.8, gain2=m["dq_tone_gain"])
        fr[mode] = (tau, y, spin.fit_ramsey(tau, y, mode, cfg["spin"]["hyperfine_mhz"], 0.8 * (2 if mode == "dq" else 1), m["t2star_sq_us"] * scl))
    rate = fr["dq"][2]["f_mhz"] / fr["sq"][2]["f_mhz"]

    cmr = cm_study(cfg, m, cal)
    gain = cmr[:, 3]
    stop = np.interp(1.0, gain[::-1], cmr[::-1, 0]) if gain.min() < 1 < gain.max() else None
    base, cases, total, scan, shift_khz, om_c = dq_errors(cfg, m)
    loss = {k: base - v for k, v in cases.items()}
    dom = max(loss, key=loss.get)
    steps, d_khz = temp_steps(cfg)
    k_cfg = th["magnet_coupling_frac"]
    d_only_sq, d_only_dq = steps["sq"]["diamond only"], steps["dq"]["diamond only"]

    ser, dt = time_series(cfg, m, res, rng)
    taus = np.geomspace(1, 1700, 30)
    adev = {k: np.sqrt(np.mean([noise.allan(x, dt, taus)[1] ** 2 for x in ser[k]], axis=0)) for k in ("sq", "sq_alt", "dq")}
    i = int(np.argmin(abs(taus - 1000)))
    a_ratio = adev["sq"][i] / adev["dq"][i]

    tempcos, biases = np.array(z["magnet_tempcos"]), np.array(z["bias_mt"]) * 1e-3
    a_d, a_m = th["diamond_t_asd_k"], th["magnet_t_asd_k"]
    dmap = np.array([[drift_ratio(cfg, tc, b, k_cfg, a_d, a_m) for tc in tempcos] for b in biases])
    dq_drift_current = drift_ratio(cfg, th["magnet_tempco_per_k"], cfg["bias"]["field_mt"] * 1e-3, k_cfg, a_d, a_m)

    dec = []
    for k in ("sq", "sq_alt", "dq"):
        dc, dp = z["mode_cost_power"][k]
        tc = sv.cycle_us(cfg, m, k if k != "sq_alt" else "sq", res[k]["tau_opt"])
        dec.append((k, band(cfg, k, white[k]) * NT, adev[k][i] * NT, adev["sq"][i] / adev[k][i], tc, dc, dp))
    table(f"M5: SQ vs SQ alternation vs DQ (fringe fits at {om} MHz drive)", [
        ("DQ/SQ fringe-rate ratio (fit)", rate, "2", verdict(abs(rate - 2) < 0.2)),
        ("fringe contrast SQ / DQ (hard pulses, metrics)", f"{m['fringe_contrast_sq']:.2f} / {m['fringe_contrast_dq']:.2f}", "", "INFO"),
        ("T2* SQ / DQ (us), model", f"{fr['sq'][2]['t2star_us']:.3g} / {fr['dq'][2]['t2star_us']:.3g}", f"ratio {m['t2star_dq_us'] / m['t2star_sq_us']:.3g}", "INFO"),
        ("DQ eta gain over SQ at cm fraction cfg", float(np.interp(sv.spin_cfg(cfg, m)["spin"]["common_mode_fraction"], cmr[:, 0], gain)), "> 1", "INFO"),
        ("eta gain (SQ/DQ) vs common-mode fraction", " ".join(f"{a:.2f}:{b:.2f}" for a, b in zip(cmr[:, 0], gain)), "", "INFO"),
        ("common-mode fraction where DQ stops helping", f"{stop:.2f}" if stop else "no crossing in scan range", "", "INFO"),
        ("DQ pulse contrast, ideal", base, "", "INFO"),
        *[(f"  loss: {k}", v, "", "INFO") for k, v in loss.items()],
        ("  all errors combined, contrast", total, f"dominant: {dom}", "INFO"),
        ("  carrier-leak Stark shift (kHz) at drive", shift_khz, "static offset, calibrate out", "INFO"),
        ("diamond +2 K step: SQ shift (kHz)", d_khz, f"{t['sq_tempco_khz_per_k'] * z['temp_step_k']:.0f}", verdict(abs(d_khz - t["sq_tempco_khz_per_k"] * z["temp_step_k"]) < 10)),
        ("  field-equivalent shifts, diamond only: SQ / DQ (nT)", f"{d_only_sq:.0f} / {d_only_dq:.1f}", "DQ near zero", verdict(abs(d_only_dq) < 0.05 * abs(d_only_sq))),
        *[(f"  with {k_cfg:.0%} magnet coupling, {k} (nT)", steps[k]["with magnet coupling"], "", "INFO") for k in steps],
        ("Allan SQ/DQ at 1000 s", a_ratio, f">= {t['dq_drift_ratio']}", verdict(a_ratio >= t["dq_drift_ratio"])),
        ("analytic drift ratio SQ/DQ at configured magnet", dq_drift_current, f">= {t['dq_drift_ratio']}", verdict(dq_drift_current >= t["dq_drift_ratio"])),
    ])
    print("\ndrift ratio SQ/DQ at 1000 s (rows: bias mT, columns: magnet tempco 1/K = " + ", ".join(f"{x:g}" for x in tempcos) + ")")
    for b, row in zip(biases, dmap):
        print(f"  {b * 1e3:.1f} mT: " + "  ".join(f"{min(v, 999):7.2f}" for v in row))
    table("M5 decision table (eta 1-10 Hz, Allan at 1000 s, cost per added capability)", [
        (k, f"eta {e:.3g} nT/rtHz, Allan {a:.3g} nT", f"drift {r:.2g}x better, cycle {tc:.1f} us", f"+${dc:.0f}, +{dp:.1f} W") for k, e, a, r, tc, dc, dp in dec])

    plotting.curves("m5_fringes", [dict(curves=[(fr[k][0], fr[k][1], k.upper()) for k in fr], xlabel="tau (us)", ylabel="P(m_s=0)", title=f"fringes, rate ratio {rate:.2f}")])
    plotting.curves("m5_cm", [dict(curves=[(cmr[:, 0], cmr[:, 1], "fitted"), (cmr[:, 0], cmr[:, 2], "1/(2 sqrt(1-f))")], xlabel="common-mode fraction", ylabel="T2* DQ/SQ", title="T2* ratio"),
                              dict(curves=[(cmr[:, 0], gain, "eta SQ / eta DQ")], xlabel="common-mode fraction", ylabel="eta gain", hlines=[1.0], title="DQ gain")])
    plotting.curves("m5_dq_errors", [dict(curves=[([s[0] for s in scan], [s[1] for s in scan], "DQ contrast")], xlabel="weak-tone gain", ylabel="fringe contrast", vlines=[m["dq_tone_gain"]], title="contrast vs tone imbalance")])
    plotting.bars("m5_dq_loss", list(loss), list(loss.values()), "contrast loss", "DQ pulse errors, each alone")
    plotting.bars("m5_temp_steps", ["SQ -1 line", "SQ +1 line", "SQ alt", "DQ"], [[steps[k]["diamond only"] for k in ("sq", "sq_p1", "sq_alt", "dq")], [steps[k]["with magnet coupling"] - steps[k]["diamond only"] for k in ("sq", "sq_p1", "sq_alt", "dq")]], "shift (nT)", f"+{z['temp_step_k']} K diamond step", stacked=["diamond", f"magnet ({k_cfg:.0%} coupling)"])
    plotting.curves("m5_allan", [dict(curves=[(taus, adev[k] * NT, k) for k in adev], xlabel="tau (s)", ylabel="nT", logx=True, logy=True, title="Allan deviation")])
    plotting.heat("m5_drift_map", [dict(x=tempcos, y=biases * 1e3, z=np.minimum(dmap, 1e3), xlabel="magnet tempco (1/K)", ylabel="bias (mT)", title="SQ/DQ drift ratio at 1000 s", log=True, levels=[t["dq_drift_ratio"]])])


if __name__ == "__main__":
    main()
