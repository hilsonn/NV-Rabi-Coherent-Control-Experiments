"""Parameter studies for the NV Rabi report. Uses physics.py from the rabi simulation suite.

Each study changes one thing from the baseline config.json and writes a figure + numbers.
Usage: python report_experiments.py <config.json> <figure_dir>
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import physics

cfg0 = json.load(open(sys.argv[1]))
FIG = Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)
DPI = 150
SEED = cfg0["run"]["seed"]
N_SEEDS = 10
results: dict = {}


def tau_axis(cfg, tau_max_ns=None):
    tmax = tau_max_ns if tau_max_ns else cfg["drive"]["tau_max_ns"]
    return np.linspace(0.0, tmax / 1000.0, cfg["drive"]["n_points"] if not tau_max_ns else int(cfg["drive"]["n_points"] * tmax / cfg["drive"]["tau_max_ns"]))


def ens_curve(cfg, tau_us, omega=None, det=None, seed=SEED):
    rng = np.random.default_rng(seed)
    offs, amp = physics.sample_ensemble(cfg["ensemble"], rng)
    om = cfg["drive"]["rabi_mhz"] if omega is None else omega
    de = cfg["drive"]["mw_detuning_mhz"] if det is None else det
    p = physics.ensemble_population(tau_us, om, de, offs, amp)
    return physics.fluorescence(p, cfg["readout"]["contrast"]), rng


def noisy(cfg, f, rng):
    return physics.add_shot_noise(f, cfg["readout"], rng)


def _gauss_cos(t, c, a, f, td):
    return c + a * np.exp(-(t / td) ** 2) * np.cos(2 * np.pi * f * t)


def fit_gauss(tau_us, y, f_guess):
    p0 = [y.mean(), 0.5 * (y.max() - y.min()), f_guess, 0.5]
    try:
        popt, _ = curve_fit(_gauss_cos, tau_us, y, p0=p0, maxfev=20000)
    except RuntimeError:
        return None
    return {"rabi_mhz": abs(popt[2]), "decay_us": abs(popt[3])}


def seeded_fits(cfg, tau_us, omega=None, det=None, fitter="exp"):
    """Fit N_SEEDS noisy realisations (fresh ensemble draw + shot noise each); mean, std of f and Td."""
    om = cfg["drive"]["rabi_mhz"] if omega is None else omega
    fr, td = [], []
    f, _ = ens_curve(cfg, tau_us, omega, det)          # one fixed ensemble draw
    for s in range(N_SEEDS):
        rng = np.random.default_rng(SEED + 100 * (s + 1))  # fresh shot noise only
        y = noisy(cfg, f, rng) if cfg["readout"]["add_shot_noise"] else f
        r = physics.fit_damped_rabi(tau_us, y, om) if fitter == "exp" else fit_gauss(tau_us, y, om)
        if r is not None:
            fr.append(r["rabi_mhz"]); td.append(r["decay_us"])
    return float(np.mean(fr)), float(np.std(fr)), float(np.mean(td)), float(np.std(td))


def isolated(cfg, hyperfine=False, amp=0.0, t2=None):
    c = copy.deepcopy(cfg)
    c["ensemble"]["hyperfine"]["enabled"] = hyperfine
    c["ensemble"]["amplitude_spread_frac"] = amp
    if t2 is not None:
        c["ensemble"]["t2star_us"] = t2
    return c


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)


tau = tau_axis(cfg0)
tau_ns = tau * 1000
OM = cfg0["drive"]["rabi_mhz"]

# ---------------------------------------------------------------- dephasing (T2*)
t2_list = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
res_t2 = {"baseline": [], "isolated": []}
for i, t2 in enumerate(t2_list):  # colours C0..C5
    c_b = copy.deepcopy(cfg0); c_b["ensemble"]["t2star_us"] = t2
    c_i = isolated(cfg0, t2=t2)
    axs[0].plot(tau_ns, ens_curve(c_b, tau)[0], color=f"C{i}", label=f"$T_2^*$ = {t2:g} µs")
    axs[1].plot(tau_ns, ens_curve(c_i, tau)[0], color=f"C{i}", label=f"$T_2^*$ = {t2:g} µs")
    for key, c in (("baseline", c_b), ("isolated", c_i)):
        fm, fs, dm, ds = seeded_fits(c, tau)
        res_t2[key].append({"t2star_us": t2, "fit_rabi_mhz": fm, "fit_rabi_std": fs,
                            "fit_decay_us": dm, "fit_decay_std": ds,
                            "sigma_mhz": physics.dephasing_sigma_mhz(t2)})
axs[0].set_title("(a) baseline: hyperfine on, 10% $B_1$ spread")
axs[1].set_title("(b) $T_2^*$ only: hyperfine off, no $B_1$ spread")
for ax in axs[:2]:
    ax.set_xlabel("pulse length τ (ns)"); ax.set_ylabel("normalised fluorescence")
axs[0].legend(fontsize=8, ncol=2, loc="upper right")
axs[1].set_ylim(0.9685, 1.006)
WIN = tau[-1]
for key, mk, col in (("baseline", "o-", "C0"), ("isolated", "s-", "C3")):
    d = res_t2[key]
    ok = [r for r in d if r["fit_decay_us"] < 10 * WIN]
    lb = [r for r in d if r["fit_decay_us"] >= 10 * WIN]
    axs[2].errorbar([r["t2star_us"] for r in ok], [r["fit_decay_us"] for r in ok],
                    yerr=[r["fit_decay_std"] for r in ok], fmt=mk, color=col, capsize=3, label=key)
    if lb:
        axs[2].plot([r["t2star_us"] for r in lb], [10 * WIN] * len(lb), "^", color=col, ms=9,
                    label=f"{key}: no decay resolved (> {10*WIN:g} µs)")
axs[2].plot(t2_list, t2_list, "k:", label="$T_d = T_2^*$")
axs[2].set_xscale("log"); axs[2].set_yscale("log")
axs[2].set_xlabel("$T_2^*$ (µs)"); axs[2].set_ylabel("fitted Rabi decay $T_d$ (µs)")
axs[2].set_title("(c) fitted envelope decay vs $T_2^*$"); axs[2].legend(fontsize=8)
save(fig, "fig2_t2star.png")
results["t2star"] = res_t2

# ---------------------------------------------------------------- B1 amplitude spread
spreads = [0.0, 0.05, 0.10, 0.20, 0.40]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
res_amp = []
for i, s in enumerate(spreads):
    c = isolated(cfg0, amp=s, t2=10.0)
    axs[0].plot(tau_ns, ens_curve(c, tau)[0], color=f"C{i}", label=f"spread {100*s:g}%")
    if s > 0:
        fm, fs, dm, ds = seeded_fits(c, tau, fitter="gauss")
        pred = np.sqrt(2) / (2 * np.pi * OM * s)
        res_amp.append({"spread": s, "fit_gauss_decay_us": dm, "fit_std": ds, "predicted_us": pred})
axs[0].set_xlabel("pulse length τ (ns)"); axs[0].set_ylabel("normalised fluorescence")
axs[0].set_title("(a) Rabi curves vs $B_1$ amplitude spread ($T_2^*$ = 10 µs, no hyperfine)")
axs[0].legend(fontsize=8)
sx = np.array([r["spread"] for r in res_amp])
axs[1].errorbar(100 * sx, [r["fit_gauss_decay_us"] for r in res_amp], yerr=[r["fit_std"] for r in res_amp],
                fmt="o", capsize=3, color="C3", label="Gaussian-envelope fit")
ss = np.linspace(0.04, 0.42, 100)
axs[1].plot(100 * ss, np.sqrt(2) / (2 * np.pi * OM * ss), "k-", label=r"$\sqrt{2}/(2\pi\,\Omega\,s)$")
axs[1].set_xlabel("$B_1$ amplitude spread s (%)"); axs[1].set_ylabel("envelope 1/e time (µs)")
axs[1].set_yscale("log"); axs[1].set_title("(b) decay time vs spread"); axs[1].legend(fontsize=8)
save(fig, "fig3_amplitude_spread.png")
results["amplitude_spread"] = res_amp

# ---------------------------------------------------------------- 14N hyperfine
om_lo = 2.0
tau_long = np.linspace(0, 4.0, 1200)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
spec = {}
for hf, col, lab in ((False, "C0", "hyperfine off"), (True, "C3", "hyperfine on (2.16 MHz)")):
    c = isolated(cfg0, hyperfine=hf, t2=10.0)
    f = ens_curve(c, tau_long, omega=om_lo)[0]
    axs[0].plot(tau_long * 1000, f, color=col, label=lab)
    y = f - f.mean()
    win = np.hanning(y.size)
    F = np.abs(np.fft.rfft(y * win, n=16 * y.size))
    fr = np.fft.rfftfreq(16 * y.size, tau_long[1] - tau_long[0])
    axs[1].plot(fr, F / F.max(), color=col, label=lab)
    spec[lab] = fr, F
expected = [om_lo, np.hypot(om_lo, cfg0["ensemble"]["hyperfine"]["splitting_mhz"])]
for e in expected:
    axs[1].axvline(e, color="k", ls=":", lw=1)
axs[0].set_xlabel("pulse length τ (ns)"); axs[0].set_ylabel("normalised fluorescence")
axs[0].set_title(f"(a) Rabi, Ω/2π = {om_lo:g} MHz, $T_2^*$ = 10 µs, no $B_1$ spread")
axs[0].set_ylim(0.9685, 1.0065); axs[0].legend(fontsize=8, ncol=2, loc="upper center")
axs[1].set_xlim(0, 6); axs[1].set_xlabel("frequency (MHz)"); axs[1].set_ylabel("|FFT| (normalised)")
axs[1].set_title(f"(b) spectrum; dotted: {expected[0]:.2f} and {expected[1]:.2f} MHz"); axs[1].legend(fontsize=8)
save(fig, "fig4_hyperfine.png")
results["hyperfine"] = {"omega_mhz": om_lo, "expected_lines_mhz": expected}

# ---------------------------------------------------------------- detuning: Omega_eff and amplitude
dets = np.linspace(-10, 10, 41)
c = isolated(cfg0, t2=10.0)
c["readout"]["add_shot_noise"] = False
fit_f, amp_meas = [], []
for d in dets:
    f, _ = ens_curve(c, tau, det=d)
    r = physics.fit_damped_rabi(tau, f, np.hypot(OM, d))
    fit_f.append(np.nan if r is None else r["rabi_mhz"])
    amp_meas.append((1 - f.min()) / cfg0["readout"]["contrast"])
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
dd = np.linspace(-10, 10, 400)
axs[0].plot(dets, fit_f, "o", color="C3", ms=4, label="fitted (simulation)")
axs[0].plot(dd, np.hypot(OM, dd), "k-", label=r"$\sqrt{\Omega^2+\Delta^2}$")
axs[0].set_xlabel("detuning Δ (MHz)"); axs[0].set_ylabel("oscillation frequency (MHz)")
axs[0].set_title("(a) generalised Rabi frequency"); axs[0].legend(fontsize=8)
axs[1].plot(dets, amp_meas, "o", color="C3", ms=4, label="max transfer (simulation)")
axs[1].plot(dd, OM**2 / (OM**2 + dd**2), "k-", label=r"$\Omega^2/(\Omega^2+\Delta^2)$")
axs[1].set_xlabel("detuning Δ (MHz)"); axs[1].set_ylabel("max population in $m_s=-1$")
axs[1].set_title("(b) oscillation amplitude"); axs[1].legend(fontsize=8)
save(fig, "fig5b_detuning_cuts.png")
results["detuning"] = {"fwhm_expected_mhz": 2 * OM,
                       "max_abs_freq_error_mhz": float(np.nanmax(np.abs(np.array(fit_f) - np.hypot(OM, dets))))}

# ---------------------------------------------------------------- power: hyperfine bias in fitted Omega
scales = [0.1, 0.25, 0.5, 1, 2, 4, 8]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
res_pow = {}
for key, c, mk in (("baseline (hyperfine on)", copy.deepcopy(cfg0), "o"),
                   ("hyperfine off", isolated(cfg0, hyperfine=False, amp=cfg0["ensemble"]["amplitude_spread_frac"]), "s")):
    rows = []
    for s in scales:
        om = OM * np.sqrt(s)
        fm, fs, dm, ds = seeded_fits(c, tau, omega=om)
        rows.append({"scale": s, "input_mhz": om, "fit_mhz": fm, "fit_std": fs, "decay_us": dm})
    res_pow[key] = rows
    axs[0].errorbar([r["input_mhz"] for r in rows], [r["fit_mhz"] / r["input_mhz"] for r in rows],
                    yerr=[r["fit_std"] / r["input_mhz"] for r in rows], fmt=mk + "-", capsize=3, label=key)
    axs[1].plot([r["input_mhz"] for r in rows], [1000 / (2 * r["fit_mhz"]) for r in rows], mk + "-", label=key)
oo = np.linspace(OM * np.sqrt(0.1), OM * np.sqrt(8), 100)
hfs = cfg0["ensemble"]["hyperfine"]["splitting_mhz"]
w_side = oo**2 / (oo**2 + hfs**2)
axs[0].plot(oo, (oo + 2 * w_side * np.hypot(oo, hfs)) / (1 + 2 * w_side) / oo, "k:",
            label="amplitude-weighted mean of 3 lines' $\\Omega_{eff}$")
axs[0].axhline(1, color="0.6", lw=1)
axs[0].set_xlabel("input Ω/2π (MHz)"); axs[0].set_ylabel("fitted / input Rabi frequency")
axs[0].set_title("(a) fit bias vs drive strength"); axs[0].legend(fontsize=8)
axs[1].plot(oo, 1000 / (2 * oo), "k:", label="ideal $1/(2\\Omega)$")
axs[1].set_xscale("log"); axs[1].set_yscale("log")
axs[1].set_xlabel("input Ω/2π (MHz)"); axs[1].set_ylabel("π-pulse length from fit (ns)")
axs[1].set_title("(b) π-pulse calibration"); axs[1].legend(fontsize=8)
from matplotlib.ticker import ScalarFormatter, NullFormatter
for a_ in (axs[1].xaxis, axs[1].yaxis):
    a_.set_major_formatter(ScalarFormatter()); a_.set_minor_formatter(NullFormatter())
axs[1].set_xticks([1.5, 2, 3, 5, 7, 10, 14]); axs[1].set_yticks([40, 60, 100, 200, 300])
save(fig, "fig6b_power_bias.png")
results["power"] = res_pow

# ---------------------------------------------------------------- readout statistics
reps = [1000, 5000, 20000, 100000, 500000]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
res_ro = []
f_clean, _ = ens_curve(cfg0, tau)
for i, n in enumerate(reps):
    c = copy.deepcopy(cfg0); c["readout"]["n_repeats"] = n
    rng = np.random.default_rng(SEED + 7)
    y = noisy(c, f_clean, rng)
    if n in (1000, 20000, 500000):
        axs[0].plot(tau_ns, y + 0.02 * [1000, 20000, 500000].index(n), ".", ms=2.5, color=f"C{i}",
                    label=f"{n:,} repeats (offset +{0.02*[1000,20000,500000].index(n):.2f})")
    sig = np.std(y - f_clean)
    fm, fs, dm, ds = seeded_fits(c, tau)
    res_ro.append({"n_repeats": n, "noise_std": float(sig), "snr_meas": cfg0["readout"]["contrast"] / sig,
                   "snr_theory": float(physics.snr_estimate(c["readout"])), "fit_rabi_std_mhz": fs,
                   "sweep_time_s": n * cfg0["acquisition"]["sequence_length_us"] * 1e-6 * cfg0["drive"]["n_points"]})
axs[0].plot(tau_ns, f_clean, "k-", lw=0.8)
axs[0].set_xlabel("pulse length τ (ns)"); axs[0].set_ylabel("normalised fluorescence")
axs[0].set_title("(a) one sweep at three averaging levels"); axs[0].set_ylim(0.965, 1.055); axs[0].legend(fontsize=8, loc="upper right")
N = np.array([r["n_repeats"] for r in res_ro])
axs[1].loglog(N, [r["snr_meas"] for r in res_ro], "o", color="C3", label="measured SNR (C / σ)")
axs[1].loglog(N, [r["snr_theory"] for r in res_ro], "k-", label=r"$C\sqrt{n_{ph} N}$")
axs[1].set_xlabel("repeats per point N"); axs[1].set_ylabel("SNR per point")
ax2 = axs[1].twinx()
ax2.loglog(N, [1000 * r["fit_rabi_std_mhz"] for r in res_ro], "s--", color="C0", label="std of fitted Ω (kHz)")
ax2.set_ylabel("std of fitted Ω/2π (kHz)", color="C0")
axs[1].set_title("(b) SNR and Rabi-frequency precision")
h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axs[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")
save(fig, "fig7_readout.png")
results["readout"] = res_ro

json.dump(results, open(FIG / "report_results.json", "w"), indent=2, default=float)
print(json.dumps(results, indent=1, default=float))
