import numpy as np

import plotting
from models import noise, sensitivity as sv, thermal
from models.common import bom_totals, load_cfg, metrics, out, override, table, verdict
from sims.subsystems import run_subsystems

NT = 1e9
TARGET = {"cw": "eta_cw_nt", "po": "eta_cw_nt", "sq": "eta_ramsey_nt", "sq_alt": "eta_ramsey_nt", "dq": "eta_ramsey_nt"}
RAMSEY = ("sq", "sq_alt", "dq")


def band(cfg, mode, white):
    lo, hi = cfg["sensitivity"]["band_hz"]
    f = np.geomspace(lo, hi, 50)
    dr = thermal.drift_asd(cfg, sv.DRIFT_KEY[mode], f)
    return float(np.sqrt(np.mean(white**2 + dr["diamond"] ** 2 + dr["magnet"] ** 2)))


def band_all(cfg, white):
    return {k: band(cfg, k, v) for k, v in white.items()}


def low(cfg, mode, white):
    return float(sv.total_asd(cfg, {}, {mode: {"eta_white": white}}, mode, cfg["sensitivity"]["low_hz"]))


def best_ramsey(cfg, white):
    b = band_all(cfg, {k: white[k] for k in RAMSEY})
    return min(b, key=b.get), b


def time_series(cfg, m, res, rng):
    """Field-equivalent series per mode, n_real independent realisations: (n_real, n) arrays."""
    th = cfg["thermal"]
    ser = {k: [] for k in sv.MODES}
    for i in range(th["n_real"]):
        z = thermal.series(cfg, m["heating_k_per_w"], rng)
        for k in sv.MODES:
            d = sv.DRIFT_KEY[k]
            w = noise.generate_noise(z["t"].size, 1 / th["sample_hz"], res[k]["eta_white"], 0.0, 1.0, rng)
            ser[k].append(w + z[f"{d}_diamond"] + z[f"{d}_magnet"] + z[f"{d}_heating"])
    return {k: np.array(v) for k, v in ser.items()}, 1 / th["sample_hz"]


def variants_study(cfg):
    rows = []
    for v in cfg["variants"]:
        c2 = override(cfg, v["overrides"])
        m2 = run_subsystems(c2)
        res, _ = sv.evaluate(c2, m2, n_spins=800)
        b = band_all(c2, {k: r["eta_white"] for k, r in res.items()})
        tot = bom_totals()
        rows.append({"name": v["name"], "eta_cw": b["cw"] * NT, "eta_ramsey": min(b[k] for k in RAMSEY) * NT,
                     "cost": tot["cost_mid"] + v["cost_delta_usd"], "power": tot["power_w"] + v["power_delta_w"], "dcost": v["cost_delta_usd"]})
    base = rows[0]["eta_ramsey"]
    for r in rows:
        r["gain_per_k"] = (base - r["eta_ramsey"]) / max(abs(r["dcost"]), 1) * 1e3
    return rows


def main():
    cfg = load_cfg()
    m = metrics()
    rng = np.random.default_rng(cfg["seed"])
    t = cfg["targets"]
    res, cal = sv.evaluate(cfg, m)
    white = {k: r["eta_white"] for k, r in res.items()}
    b110 = band_all(cfg, white)
    rows = []
    for k in sv.MODES:
        tg = t[TARGET[k]]
        rows.append((sv.NAMES[k], b110[k] * NT, f"<= {tg}", verdict(b110[k] * NT <= tg)))
        rows.append((f"  eta at 0.1 Hz", low(cfg, k, white[k]) * NT, "", "INFO"))
        rows.append((f"  tau_opt (us)", res[k]["tau_opt"], "", "INFO"))
    table("eta (1-10 Hz, nT/rtHz) per mode", rows)

    print("\n== noise budget at 1 Hz (variance fractions)")
    bud = {k: sv.budget(cfg, m, res, k) for k in sv.MODES}
    for k in sv.MODES:
        d = bud[k]
        print(f"{sv.NAMES[k]:<24}" + " ".join(f"{n}={v:.2f}" for n, v in d.items()) + f"  dominant: {max(d, key=d.get)}")
    names = list(bud["cw"])
    plotting.bars("s3_budget", [sv.NAMES[k] for k in sv.MODES], [[bud[k][n] for k in sv.MODES] for n in names], "variance fraction", "noise budget at 1 Hz", stacked=names)

    ser, dt = time_series(cfg, m, res, rng)
    taus = np.geomspace(1, 1800, 40)
    curves_a, curves_s, adev = [], [], {}
    for k in sv.MODES:
        f, a = noise.welch_asd(ser[k], dt, 2**16)
        s = (f >= 0.01) & (f <= 100)
        curves_s.append((f[s], a[s] * NT, sv.NAMES[k]))
        allans = [noise.allan(x, dt, taus) for x in ser[k]]
        tt, d = allans[0][0], np.sqrt(np.mean([a[1] ** 2 for a in allans], axis=0))
        adev[k] = (tt, d)
        curves_a.append((tt, d * NT, sv.NAMES[k]))
    plotting.curves("s3_asd_allan", [dict(curves=curves_s, xlabel="Hz", ylabel="nT/rtHz", logx=True, logy=True, title="simulated field ASD"),
                                     dict(curves=curves_a, xlabel="tau (s)", ylabel="nT", logx=True, logy=True, title="Allan deviation")])
    i = int(np.argmin(abs(adev["sq"][0] - 1000)))
    ratio = adev["sq"][1][i] / adev["dq"][1][i]
    table("drift", [("Allan SQ / DQ at 1000 s", ratio, f">= {t['dq_drift_ratio']}", verdict(ratio >= t["dq_drift_ratio"])),
                    ("Allan SQ / SQ-alt at 1000 s", adev["sq"][1][i] / adev["sq_alt"][1][i], "", "INFO")])

    bm, bb = best_ramsey(cfg, white)
    f0 = sv.scale_metric
    fct = cfg["sensitivity"]["tornado_factor"]
    keys = ["photon_rate", "contrast_pulsed", "t2star", "pol_time_us", "noise_scale", "rabi_max_mhz"]

    def eta_of(mm, ns=1.0):
        return band(cfg, bm, sv.fast(cfg, mm, cal, ns)[bm]) * NT

    base = eta_of(m)
    hi, lo_ = [], []
    for k in keys:
        hi.append(eta_of(f0(m, k, fct), fct if k == "noise_scale" else 1.0) - base)
        lo_.append(eta_of(f0(m, k, 1 / fct), 1 / fct if k == "noise_scale" else 1.0) - base)
    plotting.bars("s3_tornado", keys, [hi, lo_], f"change in eta ({bm}) nT/rtHz", f"tornado, x{fct} and x{1 / fct:.2g}", stacked=[f"x{fct}", f"x{1 / fct:.2g}"])
    print(f"\n== tornado ({sv.NAMES[bm]}, base {base:.3g} nT/rtHz)")
    for k, a, b_ in zip(keys, hi, lo_):
        print(f"{k:<16} x{fct}: {a:+.3g}   x{1 / fct:.2g}: {b_:+.3g}")

    draws = sv.monte_carlo(cfg, m, cal, rng)
    pr, dist = [], []
    for k in sv.MODES:
        e = np.array([band(cfg, k, v) for v in draws[k]]) * NT
        tg = t[TARGET[k]]
        pr.append((sv.NAMES[k], f"{np.median(e):.3g} [{np.percentile(e, 5):.3g}, {np.percentile(e, 95):.3g}]", f"<= {tg}", f"P(meet)={np.mean(e <= tg):.2f}"))
        dist.append((np.sort(e), np.linspace(0, 1, e.size), sv.NAMES[k]))
    table("Monte Carlo over metric uncertainty (median [5%, 95%] nT/rtHz)", pr)
    plotting.curves("s3_montecarlo", [dict(curves=dist, xlabel="eta (nT/rtHz)", ylabel="cumulative probability", logx=True, vlines=[t["eta_ramsey_nt"], t["eta_cw_nt"]], title="Monte Carlo")])

    print(f"\n== what must be true for {t['eta_ramsey_nt']} nT/rtHz ({sv.NAMES[bm]}; one parameter varied, others as modelled)")
    fs = np.geomspace(0.01, 1000, 300)
    for key, label, val in (("photon_rate", "photon rate (ph/s)", m["photon_rate_hz"]), ("contrast_pulsed", "pulsed contrast", m["contrast_pulsed"]), ("t2star", "T2* SQ (us)", m["t2star_sq_us"])):
        e = np.array([eta_of(f0(m, key, x)) for x in fs])
        ok = np.nonzero(e <= t["eta_ramsey_nt"])[0]
        print(f"{label:<22} " + (f"needs >= {val * fs[ok[0]]:.3g} (x{fs[ok[0]]:.3g} of modelled)" if ok.size else "not reachable by this parameter alone"))
    print("contrast factor -> required photon-rate factor:")
    for cf in (0.5, 1.0, 2.0, 4.0):
        e = np.array([eta_of(f0(f0(m, "contrast_pulsed", cf), "photon_rate", x)) for x in fs])
        ok = np.nonzero(e <= t["eta_ramsey_nt"])[0]
        print(f"  contrast x{cf}: " + (f"photon rate x{fs[ok[0]]:.3g}" if ok.size else "not reachable"))

    rows = variants_study(cfg)
    table("design variants (rank by Ramsey eta gain per $1000)", [(r["name"], f"CW {r['eta_cw']:.3g}, Ramsey {r['eta_ramsey']:.3g} nT", f"${r['cost']:.0f}, {r['power']:.1f} W", f"{r['gain_per_k']:+.3g}/k$") for r in sorted(rows, key=lambda r: -r["gain_per_k"])])
    plotting.scatter("s3_variants", [(r["cost"], r["eta_ramsey"], r["name"]) for r in rows], "total cost (USD)", "eta Ramsey (nT/rtHz)", "design variants")
    bt = bom_totals()
    print(f"\nBOM: ${bt['cost_low']:.0f}-{bt['cost_high']:.0f}, {bt['power_w']:.1f} W, {bt['volume_cm3']:.0f} cm3")
    plotting.curves("s3_eta_tau", [dict(curves=[(res[k]["tau"], res[k]["eta_tau"] * NT, k.upper()) for k in ("sq", "dq")], xlabel="tau (us)", ylabel="eta (nT/rtHz)", logy=True, title="eta vs free-evolution time")])


if __name__ == "__main__":
    main()
