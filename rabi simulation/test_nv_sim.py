"""Minimal test set for full line coverage of noise.py, physics.py, plotting.py and main.py.

Two end-to-end runs of main() (everything off / everything on) exercise nearly every line; one unit
test covers the fit-failure branch. Statistical acceptance checks live in checks.py.
"""
import json

import numpy as np

import checks
import main
import physics


def _run(cfg, tmp_path, relative):
    """Run main() on cfg; output_dir is relative to the config file (relative=True) or absolute."""
    cfg["run"]["output_dir"] = "out" if relative else str(tmp_path / "out")
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return main.main(path), tmp_path / "out"


def test_defaults_all_sources_off(tmp_path):
    summary, out = _run(checks.base_config(small=True), tmp_path, relative=True)
    assert summary["corner_frequency_hz"] is None  # shot-noise only: never 1.5x the floor
    assert summary["fit_rabi_mhz_std"] is not None and abs(summary["dt_s"] - 0.1) < 1e-12
    for name in ("rabi.png", "chevron.png", "power_sweep.png", "sensitivity.png", "summary.json"):
        assert (out / name).exists()
    assert not (out / "jitter.png").exists() and not (out / "noise_traces.png").exists()


def test_all_sources_on(tmp_path, capsys):
    cfg = checks.base_config(small=True)
    cfg["jitter"].update(enabled=True, quantise_ns=50.0)
    for src in ("baseline", "omega", "detuning"):
        cfg["noise"][src]["enabled"] = True
    cfg["noise"]["detuning"]["asd_at_ref"] = 0.1  # strong low-frequency noise -> corner frequency exists
    cfg["thermal"].update(enabled=True, mw_gain_tempco_per_k=1e-3)
    cfg["thermal"]["mw_heating"]["enabled"] = True
    cfg["vibration"]["enabled"] = True
    cfg["vibration"]["resonances"].append({"f0_hz": 20.0, "q": 30.0, "asd_peak_m": 1e-8})
    cfg["sensitivity"].update(dt_s=1e-2, working_point={"mode": "manual", "tau_ns": 100.0, "detuning_mhz": 0.5})
    summary, out = _run(cfg, tmp_path, relative=False)
    assert summary["vibration_sigma_mhz_path_a"] > 0
    assert summary["corner_frequency_hz"] is not None
    assert "exceed the Nyquist" in capsys.readouterr().out  # 120 Hz and 400 Hz dropped at Nyquist 50 Hz
    for name in ("jitter.png", "noise_traces.png"):
        assert (out / name).exists()


def test_fit_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(physics, "curve_fit", boom)
    assert physics.fit_damped_rabi(np.linspace(0, 1, 10), np.arange(10.0), 5.0) is None
