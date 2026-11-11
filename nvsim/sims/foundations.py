import numpy as np

from models import spin
from models.common import load_cfg, table


def run(cfg=None, plot=True):
    cfg = cfg or load_cfg()
    ens = spin.sample(cfg, np.random.default_rng(cfg["seed"]))
    om, t2, cm = cfg["protocol"]["hard_rabi_mhz"], cfg["spin"]["t2star_sq_us"], cfg["spin"]["common_mode_fraction"]
    tr = np.linspace(0, 3, 300)
    pr = spin.rabi(tr, om, ens)
    fr = spin.fit_rabi(tr, pr)
    det, res = 0.8, {}
    for mode, scale in (("sq", 1.0), ("dq", 1 / (2 * np.sqrt(1 - cm)))):
        tau = np.linspace(0, 4 * t2 * scale, 400)
        y = spin.ramsey(tau, om, ens, mode, det=det)
        res[mode] = (tau, y, spin.fit_ramsey(tau, y, mode, cfg["spin"]["hyperfine_mhz"], det * (2 if mode == "dq" else 1), t2 * scale))
    sq, dq = res["sq"][2], res["dq"][2]
    table(f"stage 1: spin model (Ramsey and Rabi at {om} MHz drive)", [
        ("rabi freq fit (MHz)", fr["f_mhz"], f"{om}", "INFO"),
        ("rabi decay (periods)", fr["periods"], ">=5", "INFO"),
        ("sq ramsey contrast", sq["contrast"], "", "INFO"),
        ("sq ramsey T2* (us)", sq["t2star_us"], f"{t2}", "INFO"),
        ("dq ramsey contrast", dq["contrast"], "", "INFO"),
        ("dq ramsey T2* (us)", dq["t2star_us"], "", "INFO"),
        ("fringe rate dq/sq", dq["f_mhz"] / sq["f_mhz"], "2", "INFO"),
        ("T2* ratio dq/sq fitted", dq["t2star_us"] / sq["t2star_us"], f"{1 / (2 * np.sqrt(1 - cm)):.3g} (model)", "INFO"),
    ])
    if plot:
        import plotting
        plotting.curves("stage1_spin", [
            dict(curves=[(tr, pr, "Rabi")], xlabel="pulse length (us)", ylabel="P(m_s=0)", title="Rabi"),
            dict(curves=[(res[m][0], res[m][1], m.upper()) for m in res], xlabel="tau (us)", ylabel="P(m_s=0)", title="Ramsey")])
    return res


if __name__ == "__main__":
    run()
