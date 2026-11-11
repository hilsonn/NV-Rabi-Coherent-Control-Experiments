import csv
import json

import numpy as np

import plotting
from models import sensitivity as sv, spin, vector
from models.common import GAMMA, bom, bom_totals, load_cfg, metrics, out, override, table, verdict
from sims.subsystems import run_subsystems
from sims.system import band_all

NT = 1e9


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def perturb_axes(axes, deg, rng):
    out_ = []
    for a in axes:
        v = rng.standard_normal(3)
        v -= (v @ a) * a
        v /= np.linalg.norm(v)
        t = np.radians(deg) * abs(rng.standard_normal())
        out_.append(np.cos(t) * a + np.sin(t) * v)
    return np.array(out_)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def scan(cfg, true_axes, nom_axes, gscale, b0, sigma_mhz, rng, t0, n=72, tilt=0.0):
    z = cfg["m6"]
    d = cfg["spin"]["d_mhz"]
    b_e = z["field_ut"] * 1e-6 * np.array([np.cos(np.radians(z["dip_deg"])), 0, -np.sin(np.radians(z["dip_deg"]))])
    u = np.array([0.3, -0.5, 0.8])
    u /= np.linalg.norm(u)
    psi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    truth, meas = [], []
    for p in psi:
        bs = (rot_z(p) @ rot_x(np.radians(tilt))).T @ b_e
        drift = z["bias_drift_ppm"] * 1e-6 * np.linalg.norm(b0) * (t0 + p / (2 * np.pi)) * u
        lines = vector.forward(b0 + drift + bs, d, true_axes, gscale) + sigma_mhz * rng.standard_normal(8)
        truth.append(bs)
        meas.append(vector.recover(lines, d, nom_axes, b0) - b0)
    return psi, np.array(truth), np.array(meas)


def heading_err(truth, meas):
    e = np.degrees(np.arctan2(meas[:, 1], meas[:, 0]) - np.arctan2(truth[:, 1], truth[:, 0]))
    return (e + 180) % 360 - 180


def rollup(cfg):
    items = [i for i in bom() if not i.get("optional")]
    rows = []
    for i in items:
        c = i["compact"]
        if c == "keep":
            r = (i["cost_low"], i["cost_high"], i["power_w"], i["volume_cm3"])
        elif c == "replace":
            r = (i["replace_cost"], i["replace_cost"], i["replace_power"], i["replace_volume"])
        else:
            r = (0, 0, 0, 0)
        rows.append((i, c, r))
    return rows


def main():
    cfg = load_cfg()
    m = metrics()
    t, z = cfg["targets"], cfg["m6"]
    rng = np.random.default_rng(cfg["seed"])
    res, _ = sv.evaluate(cfg, m)
    white = {k: r["eta_white"] for k, r in res.items()}
    eta_line = min(white[k] for k in ("sq", "sq_alt", "dq"))
    try:
        bj = json.load(open(out("bias.json")))
        bmt, bdir = bj["field_mt"], np.array(bj["direction"])
    except FileNotFoundError:
        bmt, bdir = cfg["bias"]["field_mt"], np.array(cfg["bias"]["direction"], float)
    bdir = bdir / np.linalg.norm(bdir)
    b0 = bmt * 1e-3 * bdir
    axes, d = spin.axes(cfg), cfg["spin"]["d_mhz"]
    noise_b, cond = vector.noise_asd(b0, d, axes, eta_line)

    th, ph = np.meshgrid(np.linspace(0.05, np.pi / 2, 24), np.linspace(0, np.pi / 2, 24), indexing="ij")
    nz = np.zeros(th.shape)
    cn = np.zeros(th.shape)
    for a in range(th.shape[0]):
        for b in range(th.shape[1]):
            v = bmt * 1e-3 * np.array([np.sin(th[a, b]) * np.cos(ph[a, b]), np.sin(th[a, b]) * np.sin(ph[a, b]), np.cos(th[a, b])])
            s, c = vector.noise_asd(v, d, axes, eta_line)
            nz[a, b], cn[a, b] = s.max() * NT, c
    sigma_mhz = GAMMA * eta_line * np.sqrt(8) / np.sqrt(2 * z["t_int_s"])

    true_axes = perturb_axes(axes, z["misalign_deg"], rng)
    gscale = 1 + z["scale_err"] * rng.standard_normal(4)
    cal_scans = [scan(cfg, true_axes, axes, gscale, b0, sigma_mhz, rng, 0.0, 48, tl) for tl in z["cal_tilts_deg"]]
    bt1, bm1 = np.vstack([c[1] for c in cal_scans]), np.vstack([c[2] for c in cal_scans])
    psi, bt2, bm2 = scan(cfg, true_axes, axes, gscale, b0, sigma_mhz, rng, 1.0)
    psi0, bt0, bm0 = cal_scans[0]
    fl = cfg["m0"]["fluxgate_pt_rthz"] * 1e-12 * np.sqrt(10)
    a_mat, off = vector.fit_matrix(bm1, bt1 + fl * rng.standard_normal(bt1.shape))
    scale_e, orth_e = vector.scale_orthogonality(a_mat)
    cal = lambda b: (b - off) @ np.linalg.inv(a_mat).T
    e_raw, e_cal = heading_err(bt2, bm2), heading_err(bt2, cal(bm2))
    e_nf = heading_err(bt0, bm0)
    h_noise = np.degrees(noise_b.max() / np.sqrt(2 * z["t_int_s"]) / (z["field_ut"] * 1e-6 * np.cos(np.radians(z["dip_deg"]))))

    rows = rollup(cfg)
    subs = sorted({r[0]["subsystem"] for r in rows})
    agg = lambda sel, k: [sum(r[2][k] for r in rows if r[0]["subsystem"] == s and sel(r)) for s in subs]
    full = {k: [sum(r[0][f] for r in rows if r[0]["subsystem"] == s) for s in subs] for k, f in (("power", "power_w"), ("volume", "volume_cm3"))}
    full["cost"] = [sum(0.5 * (r[0]["cost_low"] + r[0]["cost_high"]) for r in rows if r[0]["subsystem"] == s) for s in subs]
    red = {"cost": [sum(0.5 * (r[2][0] + r[2][1]) for r in rows if r[0]["subsystem"] == s) for s in subs], "power": agg(lambda r: True, 2), "volume": agg(lambda r: True, 3)}
    with open(out("reduced_bom.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subsystem", "item", "action", "cost_low", "cost_high", "power_w", "volume_cm3"])
        for i, c, r in rows:
            if c != "drop":
                w.writerow([i["subsystem"], i["item"], c, *r])
    tot_full, tot_red = sum(full["cost"]), sum(red["cost"])
    base_b = band_all(cfg, white)
    c2 = override(cfg, z["compact_overrides"])
    m2 = run_subsystems(c2)
    r2, _ = sv.evaluate(c2, m2, n_spins=800)
    b2 = band_all(c2, {k: v["eta_white"] for k, v in r2.items()})

    table("M6: vector recovery and compact design", [
        ("bias field (mT), direction", f"{bmt}, {np.round(bdir, 3).tolist()}", "from out/bias.json if present", "INFO"),
        ("per-axis noise x/y/z (nT/rtHz)", " / ".join(f"{x * NT:.3g}" for x in noise_b), f"line eta {eta_line * NT:.3g} nT/rtHz, 8 lines time-shared", "INFO"),
        ("Jacobian condition number", cond, "", "INFO"),
        ("recovered axis scale errors (%)", ", ".join(f"{x * 100:.2f}" for x in scale_e), f"<= {t['scale_factor_tol'] * 100}", verdict(np.abs(scale_e).max() <= t["scale_factor_tol"])),
        ("recovered orthogonality errors (deg)", ", ".join(f"{x:.2f}" for x in orth_e), f"input misalign {z['misalign_deg']} deg", "INFO"),
        ("heading error before calibration, peak / rms (deg)", f"{np.abs(e_nf).max():.3f} / {np.sqrt(np.mean(e_nf**2)):.3f}", "", "INFO"),
        ("heading error after fitting axis matrix, peak (deg)", np.abs(e_cal).max(), f"<= {z['heading_need_deg']}", verdict(np.abs(e_cal).max() <= z["heading_need_deg"])),
        ("  uncalibrated validation scan, peak (deg)", np.abs(e_raw).max(), "", "INFO"),
        ("  heading noise per sqrt(Hz) from field noise (deg)", h_noise, "", "INFO"),
        ("reduced BOM cost (USD) vs full", f"{tot_red:.0f} vs {tot_full:.0f}", "", "INFO"),
        ("reduced power (W) / volume (cm3)", f"{sum(red['power']):.1f} / {sum(red['volume']):.0f}", f"full {sum(full['power']):.1f} W / {sum(full['volume']):.0f} cm3", "INFO"),
        ("eta Ramsey, full vs compact (nT/rtHz)", f"{min(base_b[k] for k in ('sq', 'sq_alt', 'dq')) * NT:.3g} vs {min(b2[k] for k in ('sq', 'sq_alt', 'dq')) * NT:.3g}", f"<= {t['eta_ramsey_nt']}", verdict(min(b2[k] for k in ("sq", "sq_alt", "dq")) * NT <= t["eta_ramsey_nt"])),
        ("eta CW, full vs compact (nT/rtHz)", f"{base_b['cw'] * NT:.3g} vs {b2['cw'] * NT:.3g}", f"<= {t['eta_cw_nt']}", verdict(b2["cw"] * NT <= t["eta_cw_nt"])),
    ])
    print("reduced BOM written to out/reduced_bom.csv")
    plotting.heat("m6_vector", [dict(x=np.degrees(ph[0]), y=np.degrees(th[:, 0]), z=nz, xlabel="azimuth (deg)", ylabel="polar (deg)", title="worst-axis noise (nT/rtHz)", log=True, mark=(np.degrees(np.arctan2(bdir[1], bdir[0])), np.degrees(np.arccos(bdir[2])))),
                                dict(x=np.degrees(ph[0]), y=np.degrees(th[:, 0]), z=cn, xlabel="azimuth (deg)", ylabel="polar (deg)", title="Jacobian condition number", log=True)])
    plotting.curves("m6_heading", [dict(curves=[(np.degrees(psi), e_raw, "before calibration"), (np.degrees(psi), e_cal, "after axis-matrix fit")], xlabel="rotation (deg)", ylabel="heading error (deg)", hlines=[z["heading_need_deg"], -z["heading_need_deg"]], title=f"{z['field_ut']} uT, drift {z['bias_drift_ppm']} ppm")])
    for k, u in (("power", "W"), ("volume", "cm3"), ("cost", "USD")):
        plotting.bars(f"m6_stack_{k}", ["full", "reduced"], [[full[k][i], red[k][i]] for i in range(len(subs))], u, f"{k} by subsystem", stacked=subs)
    plotting.scatter("m6_eta_cost", [(tot_full, min(base_b[k] for k in ("sq", "sq_alt", "dq")) * NT, "full"), (tot_red, min(b2[k] for k in ("sq", "sq_alt", "dq")) * NT, "reduced")], "total cost (USD)", "eta Ramsey (nT/rtHz)", "reduced design")


if __name__ == "__main__":
    main()
