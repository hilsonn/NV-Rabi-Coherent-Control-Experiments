import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
GAMMA = 28000.0  # MHz/T
Q_E, K_B, H_P, C_L = 1.602176634e-19, 1.380649e-23, 6.62607015e-34, 2.99792458e8


def load_cfg(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text())


def override(cfg, ov):
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        d = cfg
        *ks, last = k.split(".")
        for s in ks:
            d = d[s]
        d[last] = v
    return cfg


def out(name):
    OUT.mkdir(exist_ok=True)
    return OUT / name


def _enc(o):
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (np.ndarray, list, tuple)):
        return [_enc(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_metrics(m):
    out("metrics.json").write_text(json.dumps(_enc(m), indent=1))


def load_metrics():
    p = out("metrics.json")
    if not p.exists():
        raise SystemExit("out/metrics.json missing: run python -m sims.subsystems first")
    return json.loads(p.read_text())


def apply_measured(m):
    p = ROOT / "measured.json"
    meas = json.loads(p.read_text()) if p.exists() else {}
    m = dict(m)
    used = [k for k in meas if not k.startswith("_")]
    m.update({k: meas[k] for k in used})
    if used:
        print("measured.json overrides:", ", ".join(used))
    return m, used


def metrics(cfg=None):
    m, _ = apply_measured(load_metrics())
    return m


def asd_arrays(m):
    a = m["noise_asd_a_rthz"]
    return np.asarray(a["f_hz"], float), np.asarray(a["asd"], float)


def bom():
    return json.loads((ROOT / "bom.json").read_text())


def bom_totals(items=None, optional=False):
    items = [i for i in (bom() if items is None else items) if optional or not i.get("optional")]
    lo, hi = sum(i["cost_low"] for i in items), sum(i["cost_high"] for i in items)
    return {"cost_low": lo, "cost_high": hi, "cost_mid": 0.5 * (lo + hi), "power_w": sum(i["power_w"] for i in items),
            "volume_cm3": sum(i.get("volume_cm3", 0) for i in items)}


def table(title, rows):
    """rows: (metric, value, target, verdict) with verdict PASS/FAIL/INFO."""
    print(f"\n== {title}")
    w = max(len(str(r[0])) for r in rows)
    print(f"{'metric':<{w}}  {'value':>14}  {'target':>16}  verdict")
    for name, v, t, ok in rows:
        vs = f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
        print(f"{name:<{w}}  {vs:>14}  {str(t):>16}  {ok}")


def verdict(ok):
    return "PASS" if ok else "FAIL"
