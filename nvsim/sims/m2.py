import json

import numpy as np

import plotting
from models import detection, optics, spin
from models.common import GAMMA, load_cfg, metrics, out, table, verdict
from sims.system import band

NT = 1e9


def fm_response(d, c, depth, d0):
    """First-harmonic lock-in output (per unit contrast) versus line detuning d0 (MHz) for FM depth (MHz)."""
    ph = np.linspace(0, 2 * np.pi, 361)
    k = int(np.argmax(c))
    y = np.array([np.trapezoid(np.interp(d[k] + x + depth * np.cos(ph), d, c) * np.cos(ph), ph) / np.pi for x in np.atleast_1d(d0)])
    return -y


def lockin_stats(d, c, depth, tol=0.05):
    x = np.linspace(-4, 4, 321)
    y = fm_response(d, c, depth, x)
    k0 = len(x) // 2
    slope = (y[k0 + 1] - y[k0 - 1]) / (x[k0 + 1] - x[k0 - 1])
    dev = np.abs(y - slope * x) / np.maximum(np.abs(slope * x), 1e-12)
    bad = np.nonzero((dev > tol) & (np.abs(x) > 0.05))[0]
    right = [b for b in bad if b > k0]
    rng = x[right[0]] if right else x[-1]
    return x, y, abs(slope), rng


def eta_grid(cfg, m, maps):
    from models import sensitivity as sv
    f_mod = cfg["protocol"]["cw_f_mod_hz"]
    eta = np.zeros_like(maps["contrast"])
    for i, p in enumerate(maps["laser_mw"]):
        for j, om in enumerate(maps["omega_mhz"]):
            dl, c, rate = optics.cw_spectrum(cfg, p, om)
            i_a = sum(optics.photocurrent(cfg, rate, p))
            best = max(abs(np.trapezoid(np.interp(dl[np.argmax(c)] + dd * np.cos(np.linspace(0, 2 * np.pi, 91)), dl, np.gradient(c, dl)) * np.cos(np.linspace(0, 2 * np.pi, 91)), np.linspace(0, 2 * np.pi, 91)) / np.pi) for dd in cfg["protocol"]["cw_mod_depths_mhz"])
            a = detection.asd(cfg["detection"], i_a, f=[f_mod])["total"][0]
            eta[i, j] = np.sqrt(2) * a / (i_a * best * GAMMA)
    return eta


def bias_design(cfg, m):
    z, sp = cfg["m2"], cfg["spin"]
    ax = spin.axes(cfg)
    th, ph = np.meshgrid(np.linspace(0.02, np.pi / 2, z["n_dir"]), np.linspace(0, np.pi / 2, z["n_dir"]), indexing="ij")
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    need = 2 * sp["hyperfine_mhz"] + m["linewidth_cw_mhz"]
    best, maps = None, []
    for bmt in z["bias_b_mt"]:
        lm, lp = spin.lines(sp["d_mhz"], dirs * bmt * 1e-3, ax)
        allf = np.sort(np.concatenate([lm, lp], axis=1), axis=1)
        gap = np.diff(allf, axis=1).min(axis=1)
        bz = np.abs(dirs @ ax.T)
        span = 2 * GAMMA * bmt * 1e-3 * bz.max(axis=1)
        maps.append((bmt, gap.reshape(th.shape), span.reshape(th.shape)))
        ok = (gap >= need) & (span <= m["resonator_bw_mhz"])
        score = np.where(ok, gap, -1)
        k = int(np.argmax(score if ok.any() else gap))
        cand = (bool(ok[k]), gap[k], bmt, dirs[k], int(np.argmax(bz[k])), span[k], allf[k])
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best, maps, th, ph, need


def main():
    cfg = load_cfg()
    m = metrics()
    t, z = cfg["targets"], cfg["m2"]
    maps = optics.cw_maps(cfg)
    eta = eta_grid(cfg, m, maps)
    i, j = np.unravel_index(np.argmin(eta), eta.shape)
    d, c, rate = optics.cw_spectrum(cfg, maps["laser_mw"][i], maps["omega_mhz"][j])
    lw = optics.fwhm(d, c)
    plotting.heat("m2_cw_maps", [dict(x=maps["omega_mhz"], y=maps["laser_mw"], z=maps["contrast"] * 100, xlabel="Rabi frequency (MHz)", ylabel="laser (mW)", title="contrast (%)"),
                                 dict(x=maps["omega_mhz"], y=maps["laser_mw"], z=maps["linewidth_mhz"], xlabel="Rabi frequency (MHz)", ylabel="laser (mW)", title="linewidth (MHz)"),
                                 dict(x=maps["omega_mhz"], y=maps["laser_mw"], z=eta * NT, xlabel="Rabi frequency (MHz)", ylabel="laser (mW)", title="eta CW white (nT/rtHz)", log=True, mark=(maps["omega_mhz"][j], maps["laser_mw"][i]))])
    depths = np.array(cfg["protocol"]["cw_mod_depths_mhz"])
    stats = [lockin_stats(d, c, dp) for dp in depths]
    k = int(np.argmax([s[2] for s in stats]))
    x, y, slope, lin = stats[k]
    plotting.curves("m2_lockin", [dict(curves=[(s[0], s[1], f"depth {dp} MHz") for s, dp in zip(stats, depths)], xlabel="line detuning (MHz)", ylabel="error signal (per unit contrast)", title="FM lock-in error signal"),
                                  dict(curves=[(depths, [s[3] for s in stats], "linear range (MHz)"), (depths, [s[2] for s in stats], "slope (1/MHz)")], xlabel="modulation depth (MHz)", ylabel="", logx=True, title="slope and linear range")])

    i_coil = z["current_ma"] * 1e-3
    b_coil = z["coil_const_ut_per_a"] * 1e-6 * i_coil
    shift = GAMMA * b_coil
    resp = fm_response(d, c, depths[k], np.array([shift]))[0]
    nonlin = abs(resp - slope * shift) / (slope * shift)
    res_err = z["current_step_ma"] / z["current_ma"]
    scale_err = float(np.hypot(nonlin, res_err))
    sh = np.linspace(0.02, 4, 200)
    nl = np.abs(fm_response(d, c, depths[k], sh) - slope * sh) / (slope * sh)
    over = np.nonzero(nl > t["scale_factor_tol"] / 2)[0]
    b_max = (sh[over[0]] if over.size else sh[-1]) / GAMMA * 1e6

    (ok, gap, bmt, bdir, kd, span, lines), bmaps, th, ph, need = bias_design(cfg, m)
    json.dump({"field_mt": bmt, "direction": bdir.tolist(), "driven_orientation": kd, "min_gap_mhz": float(gap), "dq_span_mhz": float(span), "lines_mhz": lines.tolist()}, open(out("bias.json"), "w"), indent=1)
    plotting.heat("m2_bias", [dict(x=np.degrees(ph[0]), y=np.degrees(th[:, 0]), z=g, xlabel="azimuth (deg)", ylabel="polar (deg)", title=f"min line gap (MHz), B={b} mT", clabel="MHz", levels=[need]) for b, g, _ in bmaps])
    f = np.linspace(lines.min() - 15, lines.max() + 15, 3000)
    spec = sum(1 / (1 + ((f - l - mm * cfg["spin"]["hyperfine_mhz"]) / (lw / 2)) ** 2) for l in lines for mm in (-1, 0, 1)) / 3
    plotting.curves("m2_eight_lines", [dict(curves=[(f, spec, "")], xlabel="MHz", ylabel="relative dip", title=f"eight lines, B={bmt} mT")])

    eta_b = band(cfg, "cw", eta[i, j])
    table("M2: CW ODMR", [
        ("contrast_cw (%)", c.max() * 100, ">= 1", verdict(c.max() >= t["contrast_min"])),
        ("linewidth (MHz)", lw, "", "INFO"),
        ("eta CW white (nT/rtHz)", eta[i, j] * NT, "", "INFO"),
        ("eta CW 1-10 Hz band (nT/rtHz)", eta_b * NT, f"<= {t['eta_cw_nt']}", verdict(eta_b * NT <= t["eta_cw_nt"])),
        ("laser power (mW)", maps["laser_mw"][i], "", "INFO"),
        ("Rabi frequency (MHz) / absorbed power (W)", f"{maps['omega_mhz'][j]:.2f} / {(maps['omega_mhz'][j] / m['rabi_mhz_per_sqrtw']) ** 2:.3g}", "", "INFO"),
        ("FM depth (MHz): slope, linear range", f"{depths[k]}: {slope:.3g}, +-{lin:.2f}", "", "INFO"),
        ("scale factor error (%)", scale_err * 100, f"<= {t['scale_factor_tol'] * 100}", verdict(scale_err <= t["scale_factor_tol"])),
        ("  max coil field for half the tolerance (uT)", b_max, f"cfg {b_coil * 1e6:.3g}", "INFO"),
        ("  of which nonlinearity / current step (%)", f"{nonlin * 100:.3f} / {res_err * 100:.3f}", "", "INFO"),
        ("bias B (mT), direction", f"{bmt}, {np.round(bdir, 3).tolist()}", "", "INFO"),
        ("min gap between 8 lines (MHz)", gap, f">= {need:.2f}", verdict(ok)),
        ("DQ span 2 gamma B cos (MHz)", span, f"<= {m['resonator_bw_mhz']:.0f} bw", verdict(span <= m["resonator_bw_mhz"])),
    ])


if __name__ == "__main__":
    main()
