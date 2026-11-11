"""Acceptance checks from the spec. Run: python checks.py   (exits non-zero if any check fails)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

import main
import noise
import physics

HERE = Path(__file__).resolve().parent


def base_config(small: bool = False) -> dict:
    """config.json with every new source off; small=True shrinks sizes for speed."""
    cfg = main.load_config(HERE / "config.json")
    if small:
        cfg["ensemble"]["n_spins"] = 300
        cfg["drive"]["n_points"] = 60
        cfg["sweeps"]["n_detuning"] = 9
        cfg["sensitivity"].update(n_samples=8192, n_realisations=4, welch_nperseg=1024, n_spins_slope=200)
    return cfg


def _ratio(f, asd, target, lo, hi):
    band = (f >= lo) & (f <= hi)
    return float(np.mean(asd[band] / target[band]))


def _series_asd(cfg, source, channel, n=65536, dt=1e-3, n_real=20, nperseg=8192):
    rngs = noise.spawn_rngs(cfg["run"]["seed"], n_real, stream=1)
    x = np.array([physics.realisation_noise(cfg, n, dt, {k: v[r] for k, v in rngs.items()}, "B")[source][channel]
                  for r in range(n_real)])
    f, asd = noise.welch_asd(x, dt, nperseg)
    return f[1:], asd[1:]


def check_regression():
    ref = json.load(open(HERE / "regression_reference.json"))
    cfg = base_config()
    cfg.update(copy.deepcopy(ref["config"]))
    cfg["sensitivity"]["n_realisations"] = 2
    a = main.simulate_path_a(cfg, np.random.default_rng(cfg["run"]["seed"]))
    assert np.array_equal(a["f_meas"][0], np.array(ref["f_meas"])), "rabi.png data differ"
    assert a["fits"][0]["rabi_mhz"] == ref["fit_rabi_mhz"] and a["fits"][0]["decay_us"] == ref["fit_decay_us"]


def check_generator_slope_and_level():
    for beta in (0, 1, 2):
        rng = np.random.default_rng(beta)
        x = np.array([noise.generate_noise(2**18, 1e-3, 2e-3, beta, 1.0, rng) for _ in range(20)])
        f, asd = noise.welch_asd(x, 1e-3, 2**14)
        f, asd = f[1:], asd[1:]
        mid = np.log10(f[0]) + 0.5 * (np.log10(f[-1]) - np.log10(f[0]))
        band = (np.log10(f) >= mid - 0.5) & (np.log10(f) <= mid + 0.5)
        slope = np.polyfit(np.log10(f[band]), np.log10(asd[band]), 1)[0]
        assert abs(slope + beta / 2) < 0.1, f"beta={beta}: slope {slope}"
        target = np.sqrt(noise.psd_model(f, 2e-3, beta, 1.0))
        assert abs(_ratio(f, asd, target, 0.8, 1.25) - 1) < 0.1, f"beta={beta}: level off"


def check_jitter():
    rng = np.random.default_rng(0)
    off, amp = rng.normal(0, 0.3, 50), np.ones(50)
    tau = np.linspace(0, 1.0, 100)
    p0 = physics.ensemble_population(tau, 5.0, 0.0, off, amp)
    tiny = physics.ensemble_population_jittered(tau, 5.0, 0.0, off, amp, 1e-6, 15)
    assert np.allclose(tiny, p0, atol=1e-9, rtol=0), "jitter limit"
    one, ones = np.zeros(1), np.ones(1)
    p1 = physics.ensemble_population(tau, 5.0, 0.0, one, ones)
    pj = physics.ensemble_population_jittered(tau, 5.0, 0.0, one, ones, 5.0, 15)
    assert np.all(np.abs(pj - 0.5) <= np.abs(p1 - 0.5) + 1e-9), "oscillation amplitude grew"


def check_shot_noise_floor():
    cfg = base_config(small=True)
    cfg["sensitivity"].update(n_realisations=20, n_samples=32768, welch_nperseg=2048)
    b = main.simulate_path_b(cfg)
    ratio = b["total_t"] / b["eta_shot"]
    assert abs(ratio.mean() - 1) < 0.1 and ratio.std() < 0.1, "shot-noise floor not flat / at eta_shot"


def check_working_point():
    cfg = base_config(small=True)
    tau = np.linspace(0, cfg["drive"]["tau_max_ns"] / 1000, cfg["drive"]["n_points"])
    det = np.linspace(*cfg["sweeps"]["detuning_range_mhz"], cfg["sweeps"]["n_detuning"])
    off, amp = physics.sample_ensemble(cfg["ensemble"], np.random.default_rng(0))
    wp = physics.find_working_point(tau, det, 5.0, off, amp, cfg["readout"]["contrast"])
    slopes = wp["slope_map"]
    i, j = np.argmin(abs(det - wp["detuning_mhz"])), np.argmin(abs(tau * 1000 - wp["tau_ns"]))
    assert slopes[i, j] > 0 and slopes[i, j] >= slopes.max(), "working point not the slope maximum"


def check_vibration_peak():
    cfg = base_config()
    cfg["vibration"].update(enabled=True, resonances=[{"f0_hz": 120.0, "q": 30.0, "asd_peak_m": 1e-8}])
    f, asd = _series_asd(cfg, "vibration", "detuning")
    peak = asd[(f > 119) & (f < 121)].max()
    expected = 1e-8 * physics.GAMMA_E_MHZ_PER_T * cfg["vibration"]["bias_gradient_t_per_m"]
    assert abs(peak / expected - 1) < 0.2, f"vibration peak {peak} vs {expected}"


def check_thermal():
    cfg = base_config()
    th = cfg["thermal"]
    th.update(enabled=True, magnet_tempco_per_k=0.0, mw_gain_tempco_per_k=0.0)
    f, asd = _series_asd(cfg, "thermal", "detuning")
    target = abs(th["diamond_d_mhz_per_k"]) * np.sqrt(noise.psd_model(f, th["temperature"]["asd_at_ref_k"],
                                                                        th["temperature"]["slope"], 1.0))
    assert abs(_ratio(f, asd, target, 0.8, 1.25) - 1) < 0.1, "thermal conversion"

    th.update(magnet_tempco_per_k=-1e-3, mw_gain_tempco_per_k=1e-3)
    ch = physics.realisation_noise(cfg, 4096, 1e-3, {k: v[0] for k, v in noise.spawn_rngs(1, 1, 1).items()}, "B")
    corr = np.corrcoef(ch["thermal"]["detuning"], ch["thermal"]["omega"])[0, 1]
    assert abs(abs(corr) - 1) < 1e-9, "thermal channels not fully correlated"


def check_heating():
    cfg = base_config()
    cfg["thermal"].update(enabled=True, magnet_tempco_per_k=0.0)
    cfg["thermal"]["temperature"]["asd_at_ref_k"] = 0.0
    cfg["thermal"]["mw_heating"].update(enabled=True, delta_t_final_k=0.5, time_constant_s=20.0)
    n, dt = 300, 0.1
    ch = physics.realisation_noise(cfg, n, dt, {k: v[0] for k, v in noise.spawn_rngs(1, 1, 0).items()}, "A")
    d = ch["thermal"]["detuning"]
    assert np.all(np.diff(np.abs(d)) > 0), "heating drift not monotonic"
    eff = np.sqrt(cfg["drive"]["rabi_mhz"] ** 2 + d**2)
    assert np.all(np.diff(eff) > 0), "effective Rabi frequency not monotonic"
    final = cfg["thermal"]["diamond_d_mhz_per_k"] * 0.5
    t = np.arange(n) * dt
    tau = -1.0 / np.polyfit(t, np.log(1.0 - d / final), 1)[0]
    assert abs(tau / 20.0 - 1) < 0.01, f"time constant {tau}"


def check_sub_nyquist_vibration():
    cfg = base_config()
    vib, dt, seq = cfg["vibration"], main.sweep_dt(cfg), cfg["acquisition"]["sequence_length_us"]

    def sigma(f0, a):
        vib["resonances"] = [{"f0_hz": f0, "q": 30.0, "asd_peak_m": a}]
        return physics.vibration_sigma_mhz(vib, 1.0, dt, seq)

    above, below = sigma(120.0, 1e-8), sigma(1.0, 1e-8)
    assert below < 1e-2 * above, "sub-Nyquist resonance contributes spread"
    assert abs(sigma(120.0, 2e-8) / above - 2) < 1e-6, "sigma does not scale with asd_peak_m"


CHECKS = [check_regression, check_generator_slope_and_level, check_jitter, check_shot_noise_floor,
          check_working_point, check_vibration_peak, check_thermal, check_heating, check_sub_nyquist_vibration]

if __name__ == "__main__":  # pragma: no cover
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
