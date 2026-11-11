import numpy as np

import plotting
from models import sensitivity as sv
from models.common import ROOT, load_cfg, metrics, table, verdict
from sims.system import best_ramsey

NT = 1e9


def rin_fn(cfg):
    p = ROOT / "rin.csv"
    if p.exists():
        a = np.atleast_2d(np.loadtxt(p, delimiter=",", comments="#"))
        print("using rin.csv")
        return lambda f: np.exp(np.interp(np.log(f), np.log(a[:, 0]), np.log(a[:, 1])))
    d = cfg["detection"]
    return lambda f: np.sqrt(d["rin_floor"] ** 2 + d["rin_at_1hz"] ** 2 * np.asarray(f, float) ** (-d["rin_slope"]))


def main():
    cfg = load_cfg()
    m = metrics()
    t, z, d = cfg["targets"], cfg["m0"], cfg["detection"]
    res, _ = sv.evaluate(cfg, m)
    bm, _ = best_ramsey(cfg, {k: res[k]["eta_white"] for k in sv.MODES if k in ("sq", "sq_alt", "dq")})
    from models import detection
    comp = detection.asd(d, m["photocurrent_a"], f=[res[bm]["f_eff_hz"]])
    kappa = res[bm]["eta_white"] / comp["total"][0]
    eta_t, frac = t["eta_ramsey_nt"] * 1e-9, z["budget_fraction"]
    i_allow = np.sqrt(frac) * eta_t / kappa
    rin = rin_fn(cfg)
    rej = {"single-ended": 1.0, "balanced": 10 ** (-d["balance_rejection_db"] / 20),
           "balanced + per-shot ref": 10 ** (-(d["balance_rejection_db"] + z["shot_ref_rejection_db"]) / 20)}
    fs = np.array([1.0, 10.0, 100.0, 1e3, 1e4])
    rows = []
    for name, r in rej.items():
        allowed = i_allow / (m["photocurrent_a"] * r)
        ok = bool(np.all(rin(fs[fs <= 1e3]) <= allowed))
        rows.append((f"RIN allowed, {name}", f"{allowed:.2e} /rtHz", f"cfg {rin(1.0):.1e} @1Hz", verdict(ok) if name != "single-ended" else "INFO"))
    allowed_amb = np.sqrt(frac) * eta_t
    amb = z["ambient_ut_rthz"] * 1e-6
    rows += [("ambient ASD 1-10 Hz allowed (nT)", allowed_amb * NT, f"cfg {amb * NT:.3g}", verdict(amb <= allowed_amb)),
             ("  reference suppression needed", max(1.0, amb / allowed_amb), "", "INFO"),
             ("fluxgate floor required (pT/rtHz)", eta_t / 3 * 1e12, f"cfg {z['fluxgate_pt_rthz']}", verdict(z["fluxgate_pt_rthz"] * 1e-12 <= eta_t / 3))]
    lo, hi = cfg["sensitivity"]["band_hz"]
    bw, fs_r = hi - lo, z["sample_rate_hz"]
    a_allow = eta_t * np.sqrt(frac * 2 * bw)
    for n in (1, 2, 3, 5, 10, 20):
        fl = n * z["line_hz"]
        fa = abs(fl - fs_r * round(fl / fs_r))
        a = z["line_ut"] * 1e-6 / n
        inband = fa <= hi
        rows.append((f"50 Hz harmonic {n}: {fl:.0f} Hz -> alias {fa:.1f} Hz", a * NT, f"allowed {a_allow * NT:.3g} nT" if inband else "out of band", verdict(a <= a_allow) if inband else "INFO"))
    grad_allow = allowed_amb / (z["vib_um_rthz"] * 1e-3)
    rows.append(("gradient allowed (uT/mm) at vib", grad_allow * 1e6, f"cfg {z['gradient_ut_per_mm']}", verdict(z["gradient_ut_per_mm"] * 1e-6 * z["vib_um_rthz"] * 1e-3 <= allowed_amb)))
    table(f"M0: allowed environment (best Ramsey mode {sv.NAMES[bm]}, budget {frac:.0%} of {t['eta_ramsey_nt']} nT target variance)", rows)
    f = np.geomspace(1, 1e4, 100)
    plotting.curves("m0_rin", [dict(curves=[(f, i_allow / (m["photocurrent_a"] * r) * np.ones_like(f), f"allowed, {n}") for n, r in rej.items()] + [(f, rin(f), "configured", "k--")],
                                    xlabel="Hz", ylabel="RIN (1/rtHz)", logx=True, logy=True, title="allowed vs configured laser RIN")])
    plotting.bars("m0_ambient", ["ambient cfg", "ambient allowed", "fluxgate req.", "50 Hz alias allowed"], [amb * NT, allowed_amb * NT, eta_t / 3 * NT, a_allow * NT], "nT", "ambient limits")


if __name__ == "__main__":
    main()
