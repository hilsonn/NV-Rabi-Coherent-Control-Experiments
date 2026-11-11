import numpy as np

import plotting
from models import microwave, noise, optics, sensitivity as sv, spin
from models.common import load_cfg, metrics, override, table, verdict
from sims.system import band, best_ramsey

NT = 1e9


def switching(cfg, m, tau_free):
    z, o = cfg["m3"], cfg["optics"]
    kp = optics.pump_rate_per_us(cfg, o["laser_power_mw"])
    steady = lambda r: optics.steady(cfg, kp, r)
    pol = lambda r: (lambda s: (s[0] + s[2]) / (1 - s[4]))(steady(r))
    p0 = pol(0.0)
    om = m["rabi_pulse_mhz"]
    iso = np.arange(20, 101, 2.0)
    loss = []
    for db in iso:
        om_leak = om * 10 ** (-db / 20)
        r = optics.mw_rate(cfg, om_leak, 0.0, kp)
        laser_on = max(0.0, 1 - pol(r) / p0)
        free = np.sin(np.pi * om_leak * tau_free) ** 2
        loss.append(laser_on + free)
    loss = np.array(loss)
    ok = np.nonzero(loss < 0.01)[0]
    iso_req = float(iso[ok[0]]) if ok.size else np.inf
    tf = z["laser_rise_ns"] / 2.2 * 1e-3
    tail = kp * tf * np.exp(-cfg["protocol"]["dead_us"] / tf)
    photon_loss = 0.5 * z["laser_rise_ns"] * 1e-3 / (m["readout_window_ns"] * 1e-3)
    return iso, loss, iso_req, tail, photon_loss


def rabi_study(cfg, m, sc):
    mw = microwave.model(cfg)
    ens = spin.sample(sc, np.random.default_rng(cfg["seed"]), n=1500)
    ens_mw = spin.sample(cfg, np.random.default_rng(cfg["seed"]), n=1500, amp=mw["amp_array"])
    tr = np.linspace(0, 2.0, 400)
    oms = [2, 4, 6, 8, min(10, m["rabi_max_mhz"])]
    curves, per = [], []
    for om in oms:
        y = spin.rabi(tr, om, ens_mw)
        f = spin.fit_rabi(tr, y)
        curves.append((tr, y, f"{om:.0f} MHz"))
        per.append(f["periods"])
    p = np.geomspace(0.01, 30, 100)
    omp = microwave.omega_vs_power(cfg, p, m["rabi_mhz_per_sqrtw"])
    lin = m["rabi_mhz_per_sqrtw"] * np.sqrt(p)
    p1 = p[np.argmax(omp < 0.9 * lin)] if np.any(omp < 0.9 * lin) else np.inf
    return oms, curves, per, p, omp, lin, p1


def recal(cfg, m, rng):
    z = cfg["m3"]
    n = 3600
    pen, rms, loss = [], [], []
    for tc in z["recal_s"]:
        ls, es = [], []
        for _ in range(8):
            x = np.cumsum(z["omega_walk_per_rt_s"] * rng.standard_normal(n)) + noise.generate_noise(n, 1.0, z["omega_flicker"], 1.0, 1.0, rng)
            hat, err = 0.0, np.zeros(n)
            for i in range(n):
                if i % int(tc) == 0:
                    hat = x[i] + z["cal_noise_frac"] * rng.standard_normal()
                err[i] = x[i] - hat
            ls.append(np.mean((np.pi * err / 2) ** 2))
            es.append(np.sqrt(np.mean(err**2)))
        l = float(np.mean(ls))
        loss.append(l)
        rms.append(float(np.mean(es)))
        pen.append(1 / ((1 - l) * (1 - z["cal_time_s"] / tc)))
    return np.array(rms), np.array(loss), np.array(pen)


def odmr_study(cfg, m, sc, res):
    z = cfg["m3"]
    ens = spin.sample(sc, np.random.default_rng(cfg["seed"]), n=1000)
    fr = np.linspace(-6, 6, 241)
    curves, lw, eta, resolved = [], [], [], []
    for tp in z["t_pi_us"]:
        om = min(0.5 / tp, m["rabi_max_mhz"])
        p = spin.odmr(fr, om, tp, ens)
        c = 1 - p
        k = int(np.argmax(c))
        lo = hi = k
        while lo > 0 and c[lo - 1] >= c[k] / 2:
            lo -= 1
        while hi < c.size - 1 and c[hi + 1] >= c[k] / 2:
            hi += 1
        lw.append(fr[hi] - fr[lo])
        pk = np.nonzero((c[1:-1] > c[:-2]) & (c[1:-1] > c[2:]) & (c[1:-1] > 0.3 * c.max()))[0]
        resolved.append(len(pk))
        s = m["contrast_pulsed"] * np.abs(np.gradient(p, fr)).max() * 28000
        eta.append(float(sv.eta_pulsed(override(cfg, {"protocol.po_t_pi_us": tp}), m, "po", 0.0, s)))
        curves.append((fr, p, f"t_pi {tp} us"))
    return curves, np.array(lw), np.array(eta), resolved


def main():
    cfg = load_cfg()
    m = metrics()
    t, z = cfg["targets"], cfg["m3"]
    sc = sv.spin_cfg(cfg, m)
    rng = np.random.default_rng(cfg["seed"])
    res, cal = sv.evaluate(cfg, m)
    white = {k: r["eta_white"] for k, r in res.items()}
    bm, _ = best_ramsey(cfg, {k: white[k] for k in ("sq", "sq_alt", "dq")})

    iso, loss, iso_req, tail, ploss = switching(cfg, m, res["sq"]["tau_opt"])
    oms, rcurves, per, p, omp, lin, p1 = rabi_study(cfg, m, sc)
    rms, rloss, pen = recal(cfg, m, rng)
    ocurves, lw, eta_po, resolved = odmr_study(cfg, m, sc, res)

    pw = np.linspace(5, 1.5 * cfg["optics"]["laser_power_mw"], 30)
    tpol = np.array([optics.pol_time(cfg, x) for x in pw])
    ti = np.array(z["init_us"])
    f0 = optics.pol_curve(cfg, cfg["optics"]["laser_power_mw"], np.append(ti, 500.0))
    cf = np.clip((2 * f0[:-1] - 1) / (2 * f0[-1] - 1), 1e-3, None)
    eta_i = np.array([band(cfg, bm, sv.fast(cfg, {**m, "pol_time_us": x / cfg["protocol"]["init_factor"], "contrast_pulsed": m["contrast_pulsed"] * c}, cal)[bm]) for x, c in zip(ti, cf)])
    eta_cw = band(cfg, "cw", white["cw"])

    table("M3: pulsed control", [
        ("RF isolation needed for <1% loss (dB)", iso_req, f"> {t['switch_isolation_db']}", verdict(iso_req < t["switch_isolation_db"] and cfg["microwave"]["switch_isolation_db"] > t["switch_isolation_db"])),
        ("  configured isolation (dB)", cfg["microwave"]["switch_isolation_db"], "", "INFO"),
        ("laser tail pumping during MW window", tail, "", "INFO"),
        ("readout photon loss from laser rise", ploss, "", "INFO"),
        ("pol_time_us (to 90%)", m["pol_time_us"], "", "INFO"),
        ("Rabi decay periods (model, MW field map)", m["rabi_decay_periods"], f">= {t['rabi_decay_periods']}", verdict(m["rabi_decay_periods"] >= t["rabi_decay_periods"])),
        ("  periods at 2/4/6/8/10 MHz", ", ".join(f"{x:.1f}" for x in per), "", "INFO"),
        ("amplifier 10% compression at (W)", p1, "", "INFO"),
        ("recal interval minimising penalty (s)", z["recal_s"][int(np.argmin(pen))], "", "INFO"),
        ("  rms pi-pulse amplitude error / contrast loss / penalty", f"{rms[np.argmin(pen)] * 100:.2f}% / {rloss[np.argmin(pen)] * 100:.3f}% / x{pen.min():.3f}", "", "INFO"),
        ("pulsed ODMR linewidth (MHz) vs t_pi", ", ".join(f"{x:.2f}" for x in lw), f"1/(pi T2*) = {1 / (np.pi * m['t2star_sq_us']):.2f}", "INFO"),
        ("  hyperfine triplet resolved (lines seen)", ", ".join(map(str, resolved)), "3 lines", verdict(max(resolved) >= 3)),
        ("eta pulsed ODMR (nT/rtHz), best t_pi", eta_po.min() * NT, f"< eta CW {eta_cw * NT:.3g}", verdict(eta_po.min() < eta_cw)),
        (f"eta {bm} Ramsey vs CW (nT/rtHz)", band(cfg, bm, white[bm]) * NT, f"< eta CW {eta_cw * NT:.3g}", verdict(band(cfg, bm, white[bm]) < eta_cw)),
    ])
    plotting.curves("m3_switching", [dict(curves=[(iso, loss * 100, "contrast loss (%)")], xlabel="RF switch isolation (dB)", ylabel="%", logy=True, hlines=[1.0], vlines=[t["switch_isolation_db"]], title="RF leakage effect")])
    plotting.curves("m3_polarisation", [dict(curves=[(pw, tpol, "")], xlabel="laser (mW)", ylabel="time to 90% (us)", logy=True, title="polarisation time"),
                                        dict(curves=[(ti, eta_i * NT, f"{bm} Ramsey")], xlabel="initialisation time (us)", ylabel="eta (nT/rtHz)", hlines=[eta_cw * NT], title="duty-cycle penalty (line = CW)")])
    plotting.curves("m3_rabi", [dict(curves=rcurves, xlabel="pulse length (us)", ylabel="P(m_s=0)", title="Rabi curves"),
                                dict(curves=[(oms, per, "periods")], xlabel="Rabi frequency (MHz)", ylabel="decay (periods)", hlines=[t["rabi_decay_periods"]], title="decay vs drive"),
                                dict(curves=[(p, omp, "with compression"), (p, lin, "sqrt(P)")], xlabel="commanded power (W)", ylabel="Rabi (MHz)", logx=True, logy=True, title="Rabi vs power")])
    plotting.curves("m3_recal", [dict(curves=[(z["recal_s"], rloss * 100, "contrast loss (%)")], xlabel="T_cal (s)", ylabel="%", logx=True, logy=True, title="calibration drift"),
                                 dict(curves=[(z["recal_s"], pen, "penalty")], xlabel="T_cal (s)", ylabel="eta penalty factor", logx=True, title="recalibration trade-off")])
    plotting.curves("m3_odmr", [dict(curves=ocurves, xlabel="detuning (MHz)", ylabel="P(m_s=0)", title="pulsed ODMR"),
                                dict(curves=[(z["t_pi_us"], lw, "linewidth (MHz)")], xlabel="t_pi (us)", ylabel="MHz", logx=True, title="linewidth vs t_pi"),
                                dict(curves=[(z["t_pi_us"], eta_po * NT, "pulsed ODMR")], xlabel="t_pi (us)", ylabel="eta (nT/rtHz)", hlines=[eta_cw * NT], logy=True, title="eta vs t_pi (line = CW)")])


if __name__ == "__main__":
    main()
