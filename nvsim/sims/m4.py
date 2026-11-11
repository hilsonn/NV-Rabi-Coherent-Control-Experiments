import numpy as np

import plotting
from models import loop, noise, sensitivity as sv, spin
from models.common import GAMMA, load_cfg, metrics, override, table, verdict
from sims.system import band, time_series

NT = 1e9


def detuning_study(cfg, m, ens):
    om, hf, t2 = sv.pulse_omega(cfg, m), cfg["spin"]["hyperfine_mhz"], m["t2star_sq_us"]
    dets = np.linspace(0.2, 3.0, 29)
    tau = np.linspace(0.1, 3 * t2, 30)
    s = np.array([np.abs(spin.ramsey_slope(tau, om, ens, "sq", d, 0.0)[0]) for d in dets])
    eta = np.array([sv.eta_pulsed(cfg, m, "sq", tau, m["contrast_pulsed"] * x) for x in s])
    i, j = np.unravel_index(np.argmin(eta), eta.shape)
    d0, t0 = dets[i], tau[j]
    lines = np.array([np.sin(2 * np.pi * (d0 + k * hf) * t0) for k in (-1, 0, 1)])
    depth = 1 - abs(lines.mean()) / np.abs(lines).mean()
    ideal = m["contrast_pulsed"] * 0.5 * np.exp(-((tau / t2) ** 2)) * 2 * np.pi * GAMMA * tau
    eta_ideal = sv.eta_pulsed(cfg, m, "sq", tau, ideal).min()
    return dets, tau, s, eta, (d0, t0, eta[i, j], depth, lines, eta_ideal)


def fit_study(cfg, m, ens, d0, rng):
    z = cfg["m4"]
    om, hf, t2 = sv.pulse_omega(cfg, m), cfg["spin"]["hyperfine_mhz"], m["t2star_sq_us"]
    sig = np.sqrt(2 / (m["photons_per_shot"] * z["n_rep"])) * sv.excess(m, 1e5) / m["contrast_pulsed"]
    rows = []
    for tmax in (1.5, 3.0, 4.5):
        for npt in (20, 40, 80):
            tau = np.linspace(0, tmax * t2, npt)
            y = spin.ramsey(tau, om, ens, "sq", det=d0)
            f0 = spin.fit_ramsey(tau, y, "sq", hf, d0, t2)
            errs = []
            for _ in range(z["n_fit_draws"]):
                fr = spin.fit_ramsey(tau, y + sig * rng.standard_normal(npt), "sq", hf, d0, t2)
                errs.append(fr["t2star_us"] / t2 - 1)
            rows.append((tmax, npt, f0["t2star_us"] / t2 - 1, float(np.mean(errs)), float(np.std(errs)), y, tau, f0))
    return rows


def main():
    cfg = load_cfg()
    m = metrics()
    t, z = cfg["targets"], cfg["m4"]
    rng = np.random.default_rng(cfg["seed"])
    sc = sv.spin_cfg(cfg, m)
    ens = spin.sample(sc, rng, n=1500, t2star_us=m["t2star_sq_us"])
    res, cal = sv.evaluate(cfg, m)
    dets, tau, smap, emap, (d0, t0, e0, depth, lines, e_ideal) = detuning_study(cfg, m, ens)
    fits = fit_study(cfg, m, ens, d0, rng)
    ok_fit = [r for r in fits if abs(r[3]) <= t["t2star_tol"] and r[4] <= t["t2star_tol"]]
    best_fit = min(ok_fit, key=lambda r: r[1]) if ok_fit else min(fits, key=lambda r: abs(r[3]))
    eta_sq = band(cfg, "sq", res["sq"]["eta_white"])

    tc = sv.cycle_us(cfg, m, "sq", t0) * 1e-6
    t_u = tc * z["n_avg"]
    kp, ki = loop.gains(z["loop_gain"], z["ki_ratio"])
    sig_d = loop.G_HZ * res["sq"]["eta_white"] / np.sqrt(2 * t_u)
    n_s, n_r, n_rep = len(z["step_ut"]), len(z["ramp_ut_per_s"]), z["n_real"]
    st = np.repeat(np.array(z["step_ut"]) * 1e-6, n_r * n_rep)
    rp = np.tile(np.repeat(np.array(z["ramp_ut_per_s"]) * 1e-6, n_rep), n_s)
    t_step = 0.25 * z["sim_s"]
    field = lambda tt: st * (tt >= t_step) + rp * tt
    lost_map = {}
    for fb in (False, True):
        r = loop.simulate(t0, t_u, kp, ki, z["synth_step_hz"], sig_d, z["sim_s"], field, st.size, rng, fb, z["coarse_tau_us"])
        lost_map[fb] = r["final_lost"].reshape(n_s, n_r, n_rep).mean(axis=2)
    cap = 1e6 / (4 * t0)
    big = np.array(z["step_ut"]) * 1e-6
    n_b = 2 * n_rep
    sgn = np.tile([1.0, -1.0], n_rep)
    fb_run = loop.simulate(t0, t_u, kp, ki, z["synth_step_hz"], sig_d, z["sim_s"], lambda tt: sgn * big.max() * (tt >= t_step), n_b, rng, True, z["coarse_tau_us"], record=True)
    nf_run = loop.simulate(t0, t_u, kp, ki, z["synth_step_hz"], sig_d, z["sim_s"], lambda tt: sgn * big.max() * (tt >= t_step), n_b, rng, False, z["coarse_tau_us"], record=True)
    holds = not fb_run["final_lost"].any()
    bw = loop.bandwidth_hz(kp, ki, t_u)

    navg = np.array([1, 2, 5, 10, 20, 50, 100, 200])
    bws = np.array([loop.bandwidth_hz(kp, ki, tc * n) for n in navg])
    nz = np.array([loop.tracking_noise_t(kp, ki, tc * n, res["sq"]["eta_white"]) for n in navg])

    amb = noise.generate_noise(int(z["allan_s"] / t_u), t_u, cfg["m0"]["ambient_ut_rthz"] * 1e-6, 1.0, 1.0, rng)
    tt = np.arange(amb.size) * t_u
    dist = z["dist_ut"] * 1e-6 * np.sin(2 * np.pi * z["dist_hz"] * tt)
    ftrue = amb + dist
    fn = lambda k: ftrue[min(int(round(k / t_u)), ftrue.size - 1)] * np.ones(1)
    cl = loop.simulate(t0, t_u, kp, ki, z["synth_step_hz"], sig_d, z["allan_s"] - t_u, fn, 1, rng, True, z["coarse_tau_us"], record=True)
    ol = loop.simulate(t0, t_u, 0.0, 0.0, 1.0, sig_d, z["allan_s"] - t_u, fn, 1, rng, False, z["coarse_tau_us"], record=True)
    n_c = cl["estimate_hz"].shape[0]
    err_cl = cl["estimate_hz"][:, 0] / loop.G_HZ - ftrue[:n_c]
    err_ol = ol["estimate_hz"][:, 0] / loop.G_HZ - ftrue[:n_c]
    taus = np.geomspace(10 * t_u, z["allan_s"] / 4, 20)
    a_cl, a_ol = noise.allan(err_cl, t_u, taus), noise.allan(err_ol, t_u, taus)

    cfg3 = override(cfg, {"thermal.n_real": 2})
    ser, dt = time_series(cfg3, m, res, rng)
    tl = np.geomspace(1, 1700, 30)
    al = {k: noise.allan(ser[k][0], dt, tl) for k in ("sq", "sq_alt")}
    mix = {k: np.sqrt(np.mean([noise.allan(x, dt, tl)[1] ** 2 for x in ser[k]], axis=0)) for k in ("sq", "sq_alt")}

    table("M4: Ramsey and closed loop", [
        ("recommended detuning (MHz) / tau (us)", f"{d0:.2f} / {t0:.2f}", f"1/(4 tau) = {0.25 / t0:.2f}", "INFO"),
        ("  hyperfine line contributions at (d, tau)", np.round(lines, 2).tolist(), f"beating depth {depth:.2f}", "INFO"),
        ("  eta at that point (nT/rtHz, white)", e0 * NT, "", "INFO"),
        ("  three-tone drive gain in eta", e0 / e_ideal, "> 1.5 to be worth it", "INFO"),
        ("T2* fit error, recommended sampling (%)", best_fit[3] * 100, f"within {t['t2star_tol'] * 100:.0f}% (n={best_fit[1]}, tmax={best_fit[0]} T2*)", verdict(abs(best_fit[3]) <= t["t2star_tol"])),
        ("  noise-free bias of the same fit (%)", best_fit[2] * 100, "", "INFO"),
        ("eta SQ Ramsey 1-10 Hz (nT/rtHz)", eta_sq * NT, f"<= {t['eta_ramsey_nt']}", verdict(eta_sq * NT <= t["eta_ramsey_nt"])),
        ("fine capture range (MHz)", 0.25 / t0, f"step {t['lock_range_ut']} uT = {t['lock_range_ut'] * 1e-6 * loop.G_HZ / 1e6:.2f} MHz", "INFO"),
        ("lock after +-50 uT step, no fallback", f"lost in {nf_run['final_lost'].mean() * 100:.0f}% of runs", "", "INFO"),
        ("lock after +-50 uT step, with coarse fallback", f"lost in {fb_run['final_lost'].mean() * 100:.0f}% of runs", "holds", verdict(holds)),
        ("closed-loop bandwidth (Hz)", bw, f">= {t['lock_bw_hz']}", verdict(bw >= t["lock_bw_hz"])),
        ("tracking noise at that bandwidth (nT rms)", loop.tracking_noise_t(kp, ki, t_u, res["sq"]["eta_white"]) * NT, "", "INFO"),
        ("Allan error at 0.1 s: locked / open loop (nT)", f"{a_cl[1][np.argmin(abs(a_cl[0] - 0.1))] * NT:.3g} / {a_ol[1][np.argmin(abs(a_ol[0] - 0.1))] * NT:.3g}", "", "INFO"),
        ("Allan SQ at 1000 s: plain / alternation (nT)", f"{mix['sq'][np.argmin(abs(tl - 1000))] * NT:.3g} / {mix['sq_alt'][np.argmin(abs(tl - 1000))] * NT:.3g}", "", "INFO"),
    ])
    plotting.curves("m4_fringes", [dict(curves=[(best_fit[6], best_fit[5], "model"), (best_fit[6], best_fit[7]["curve"], "fit")], xlabel="tau (us)", ylabel="P(m_s=0)", title=f"SQ Ramsey, detuning {d0:.2f} MHz"),
                                   dict(curves=[(r[6][-1] / m["t2star_sq_us"] * np.ones(1), [r[3] * 100], f"n={r[1]}", "o") for r in fits], xlabel="tau_max / T2*", ylabel="fit error (%)", hlines=[10, -10], title="T2* fit bias")])
    plotting.heat("m4_detuning", [dict(x=tau, y=dets, z=smap, xlabel="tau (us)", ylabel="detuning (MHz)", title="slope at quadrature (1/T)", mark=(t0, d0)),
                                  dict(x=tau, y=dets, z=emap * NT, xlabel="tau (us)", ylabel="detuning (MHz)", title="eta white (nT/rtHz)", log=True, mark=(t0, d0))])
    plotting.curves("m4_eta_tau", [dict(curves=[(res["sq"]["tau"], res["sq"]["eta_tau"] * NT, "optimal phase")] + [(tau, emap[np.argmin(abs(dets - d)), :] * NT, f"det {d:.1f} MHz") for d in (0.5, d0, 2.0)], xlabel="tau (us)", ylabel="eta (nT/rtHz)", logy=True, title="sensitivity vs tau")])
    plotting.heat("m4_lock_loss", [dict(x=np.arange(n_r), y=np.arange(n_s), z=lost_map[f], xlabel="ramp index (" + "/".join(str(x) for x in z["ramp_ut_per_s"]) + " uT/s)", ylabel="step index (" + "/".join(str(x) for x in z["step_ut"]) + " uT)", title=f"lock-loss probability, fallback {f}", vmin=0, vmax=1) for f in (False, True)])
    k = 0
    th = np.arange(fb_run["delta"].shape[0]) * t_u
    plotting.curves("m4_step", [dict(curves=[(th, fb_run["delta"][:, k] / 1e6, "with fallback"), (th, nf_run["delta"][:, k] / 1e6, "no fallback")], xlabel="time (s)", ylabel="detuning from lock (MHz)", title="50 uT step response", hlines=[0.25 / t0, -0.25 / t0])])
    plotting.curves("m4_bandwidth", [dict(curves=[(navg, bws, "bandwidth (Hz)")], xlabel="averages per update", ylabel="Hz", logx=True, logy=True, hlines=[t["lock_bw_hz"]], title="loop bandwidth"),
                                     dict(curves=[(bws, nz * NT, "tracking noise (nT rms)")], xlabel="bandwidth (Hz)", ylabel="nT", logx=True, logy=True, title="noise cost of bandwidth")])
    plotting.curves("m4_allan", [dict(curves=[(a_cl[0], a_cl[1] * NT, "locked"), (a_ol[0], a_ol[1] * NT, "open loop")], xlabel="tau (s)", ylabel="nT", logx=True, logy=True, title="locked vs open-loop estimate error"),
                                 dict(curves=[(tl, mix[k_] * NT, k_) for k_ in mix], xlabel="tau (s)", ylabel="nT", logx=True, logy=True, title="Allan, 1 h, SQ vs alternation")])


if __name__ == "__main__":
    main()
