import numpy as np

import plotting
from models import sensitivity as sv, spin
from models.common import load_cfg, metrics, override, table
from sims.system import band

NT = 1e9
KINDS = ("square", "bb1", "corpse")
PSI = np.linspace(0, 2 * np.pi, 12, endpoint=False)


def composite(kind, theta):
    if kind == "bb1":
        p = np.arccos(-theta / (4 * np.pi))
        return [(theta, 0.0), (np.pi, p), (2 * np.pi, 3 * p), (np.pi, p)]
    if kind == "corpse":
        k = np.arcsin(np.sin(theta / 2) / 2)
        return [(2 * np.pi + theta / 2 - k, 0.0), (2 * np.pi - 2 * k, np.pi), (theta / 2 - k, 0.0)]
    return [(theta, 0.0)]


def duration(mode, om, g, segs):
    return sum(float(spin.pulse_len(mode, om, g, th)) for th, _ in segs)


def contrast_fidelity(ens, mode, om, g, segs, ideal_ens, extra=0.0):
    """Fringe contrast of two identical pulses (peak-to-peak over the second-pulse phase) and mean state fidelity of one pulse."""
    u1 = spin.sequence_u(ens, mode, om, segs, g, extra=extra)
    p = np.array([(np.abs((spin.sequence_u(ens, mode, om, segs, g, extra=extra, psi=psi)[:, 1, :] * u1[:, :, 1]).sum(1)) ** 2).mean() for psi in PSI])
    ui = spin.sequence_u(ideal_ens, mode, om, segs, g)
    ideal = ui[:, :, 1]
    fid = (np.abs(np.einsum("nk,nk->n", ideal.conj(), u1[:, :, 1])) ** 2).mean()
    return float(np.ptp(p)), float(fid)


def ens_with(cfg, amp_spread, det_spread, rng, n=600):
    c2 = override(cfg, {"spin.amp_spread_frac": amp_spread, "spin.t2star_sq_us": 1e6, "spin.common_mode_fraction": 0.5})
    e = spin.sample(c2, rng, n=n)
    e["b"] = e["b"] + rng.normal(0, det_spread, n)
    return e


def main():
    cfg = load_cfg()
    m = metrics()
    z = cfg["robust"]
    om, g = sv.pulse_omega(cfg, m), m["dq_tone_gain"]
    rng = np.random.default_rng(cfg["seed"])
    sc = sv.spin_cfg(cfg, m)
    res, cal = sv.evaluate(cfg, m)
    ideal = ens_with(cfg, 0.0, 0.0, rng, 1)
    ideal["a"] = 0.0
    sig_det = np.sqrt(2) / (2 * np.pi * m["t2star_sq_us"])
    base_cfg_ens = lambda: spin.sample(sc, np.random.default_rng(cfg["seed"]), n=600, t2star_us=m["t2star_sq_us"])
    out = {}
    rows = []
    for mode, theta in (("sq", np.pi / 2), ("dq", np.pi)):
        sq_t = duration(mode, om, g, composite("square", theta))
        c_sq = None
        for kind in KINDS:
            segs = composite(kind, theta)
            c, f = contrast_fidelity(base_cfg_ens(), mode, om, g, segs, ideal)
            dur = duration(mode, om, g, segs)
            if kind == "square":
                c_sq = c
            ca = [contrast_fidelity(ens_with(cfg, a, 0.0, rng), mode, om, g, segs, ideal)[0] for a in z["amp_spread"]]
            cd = [contrast_fidelity(ens_with(cfg, 0.0, d, rng), mode, om, g, segs, ideal)[0] for d in z["det_spread_mhz"]]
            cfg2 = override(cfg, {"protocol.pi_overhead_us": cfg["protocol"]["pi_overhead_us"] + dur - sq_t})
            m2 = {**m, "contrast_pulsed": m["contrast_pulsed"] * c / c_sq if c_sq else m["contrast_pulsed"]}
            eta = band(cfg, mode, sv.fast(cfg2, m2, cal)[mode]) * NT
            out[(mode, kind)] = (ca, cd)
            rows.append((f"{mode.upper()} {kind}", f"contrast {c:.3f}, fidelity {f:.3f}", f"+{(dur - sq_t) * 1e3:.0f} ns/pulse ({(dur / sq_t - 1) * 100:.0f}%)", f"eta {eta:.3g} nT/rtHz"))
    table("robust pulses at modelled spreads (rabi spread %.1f%%, detuning sigma %.2f MHz, hyperfine included)" % (m["rabi_spread_frac"] * 100, sig_det), rows)
    for mode in ("sq", "dq"):
        print(f"\n{mode.upper()} contrast vs amplitude spread {z['amp_spread']} / detuning spread {z['det_spread_mhz']} MHz")
        for kind in KINDS:
            ca, cd = out[(mode, kind)]
            print(f"  {kind:<7} " + " ".join(f"{x:.2f}" for x in ca) + "  |  " + " ".join(f"{x:.2f}" for x in cd))
    plotting.curves("robust_contrast", [dict(curves=[(z["amp_spread"], out[(mode, k)][0], k) for k in KINDS], xlabel="amplitude spread", ylabel="fringe contrast", title=f"{mode.upper()}: amplitude spread") for mode in ("sq", "dq")] +
                    [dict(curves=[(z["det_spread_mhz"], out[(mode, k)][1], k) for k in KINDS], xlabel="detuning spread (MHz)", ylabel="fringe contrast", title=f"{mode.upper()}: detuning spread") for mode in ("sq", "dq")])
    over = [float(r[2].split("(")[1].split("%")[0]) for r in rows]
    plotting.scatter("robust_overhead", [(o, float(r[1].split()[1].strip(",")), r[0]) for o, r in zip(over, rows)], "sequence overhead (%)", "recovered fringe contrast", "contrast vs overhead")


if __name__ == "__main__":
    main()
