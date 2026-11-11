import numpy as np
from scipy.optimize import curve_fit

from models.common import GAMMA

TWO_PI = 2 * np.pi
SZ = np.diag([1.0, 0.0, -1.0])
SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)


def axes(cfg):
    a = np.array(cfg["spin"]["orientations"], float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def hamiltonian(d_mhz, b_t, axis, a_mhz=0.0, m_i=0):
    """Lab-frame spin-1 H (MHz) for field vector b_t (T) and NV axis; basis (+1, 0, -1)."""
    b = np.asarray(b_t, float)
    n = np.asarray(axis, float) / np.linalg.norm(axis)
    bz = b @ n
    bp = np.linalg.norm(b - bz * n)
    return d_mhz * SZ @ SZ + GAMMA * (bz * SZ + bp * SX) + a_mhz * m_i * SZ


def lines(d_mhz, b_t, axis_set):
    """(nu-, nu+) in MHz from exact eigenvalues per axis; b_t is (3,) or (n,3). Returns arrays (n, n_axes)."""
    b = np.atleast_2d(b_t)
    ax = np.asarray(axis_set, float)
    ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
    bz = b @ ax.T
    bp = np.linalg.norm(b[:, None, :] - bz[..., None] * ax[None], axis=-1)
    h = d_mhz * (SZ @ SZ) + GAMMA * (bz[..., None, None] * SZ + bp[..., None, None] * SX)
    w = np.linalg.eigvalsh(h)
    return w[..., 1] - w[..., 0], w[..., 2] - w[..., 0]


def sample(cfg, rng, n=None, amp=None, t2star_us=None, cm_frac=None):
    sp = cfg["spin"]
    n = int(n or sp["n_spins"])
    t2 = t2star_us or sp["t2star_sq_us"]
    f = sp["common_mode_fraction"] if cm_frac is None else cm_frac
    sig = np.sqrt(2) / (TWO_PI * t2)
    if amp is None:
        s = np.clip(1 + sp["amp_spread_frac"] * rng.standard_normal(n), 0, None)
    else:
        s = rng.choice(np.asarray(amp, float), n)
    return {"c": rng.normal(0, sig * np.sqrt(f), n), "b": rng.normal(0, sig * np.sqrt(1 - f), n),
            "m": rng.integers(-1, 2, n), "s": s, "a": sp["hyperfine_mhz"]}


def _h(ens, mode, omega, det, db, phase=0.0, gain2=1.0, extra=0.0):
    """Rotating-frame H (MHz) per spin. Tone 1 on the -1 line, tone 2 on +1 (DQ). det: tone detuning (MHz).
    extra: additional magnetic-like shift (MHz) of both lines, e.g. a detuning spread.
    phase: SQ, phase of the tone; DQ, phase of tone 2 relative to tone 1."""
    n = ens["c"].size
    mag = ens["b"] + db + extra
    d1 = det - ens["c"] + mag + ens["a"] * ens["m"]
    d2 = (-det if mode == "dq" else 0.0) - ens["c"] - mag - ens["a"] * ens["m"]
    h = np.zeros((n, 3, 3), complex)
    h[:, 0, 0], h[:, 2, 2] = -d2, -d1
    o = omega * ens["s"] / 2
    o1 = o * (1 if mode == "dq" else np.exp(1j * phase))
    h[:, 1, 2], h[:, 2, 1] = o1, np.conj(o1)
    if mode == "dq":
        o2 = gain2 * o * np.exp(1j * phase)
        h[:, 1, 0], h[:, 0, 1] = o2, np.conj(o2)
    return h, np.stack([-d2, np.zeros(n), -d1], 1)


def pulse_len(mode, omega, gain2=1.0, theta=None):
    """Pulse length (us) for rotation angle theta (SQ default pi/2; DQ default full transfer to the bright state)."""
    if mode == "dq":
        return (np.pi if theta is None else theta) / (TWO_PI * omega * np.sqrt(1 + gain2**2))
    return (np.pi / 2 if theta is None else theta) / (TWO_PI * omega)


def _u(h, t):
    w, v = np.linalg.eigh(h)
    return np.einsum("nik,nk,njk->nij", v, np.exp(-1j * TWO_PI * w * t), v.conj())


def rabi(tau, omega, ens, mode="sq", det=0.0, db=0.0, gain2=1.0):
    """Spin-averaged m_s=0 population versus pulse length tau (us)."""
    h, _ = _h(ens, mode, omega, det, db, gain2=gain2)
    w, v = np.linalg.eigh(h)
    wt = np.abs(v[:, 1, :]) ** 2
    amp = np.einsum("nk,nkt->nt", wt, np.exp(-1j * TWO_PI * w[:, :, None] * np.asarray(tau)[None, None, :]))
    return (np.abs(amp) ** 2).mean(0)


def odmr(freqs, omega, t_pulse, ens, mode="sq"):
    """Pulsed ODMR: m_s=0 population after one pulse of length t_pulse (us) versus tone detuning (MHz)."""
    return np.array([(np.abs(_u(_h(ens, mode, omega, f, 0.0)[0], t_pulse)[:, 1, 1]) ** 2).mean() for f in freqs])


def ramsey(tau, omega, ens, mode="sq", det=0.0, phase=0.0, db=0.0, gain2=1.0, extra=0.0):
    """Ramsey m_s=0 population versus free time tau (us). Pulses: SQ pi/2, DQ transfer to the bright state."""
    tp = pulse_len(mode, omega, gain2)
    h1, e = _h(ens, mode, omega, det, db, 0.0, gain2, extra)
    h2, _ = _h(ens, mode, omega, det, db, phase, gain2, extra)
    u1, u2 = _u(h1, tp), _u(h2, tp)
    psi = u1[:, :, 1][:, :, None] * np.exp(-1j * TWO_PI * e[:, :, None] * np.asarray(tau)[None, None, :])
    return (np.abs(np.einsum("nk,nkt->nt", u2[:, 1, :], psi)) ** 2).mean(0)


def ramsey_slope(tau, omega, ens, mode="sq", det=0.0, phase=None, gain2=1.0, eps=2e-3):
    """dP0/dB per tesla (phase on a grid when None); returns (slope array over tau, phase)."""
    def sl(ph):
        return (ramsey(tau, omega, ens, mode, det, ph, eps, gain2) - ramsey(tau, omega, ens, mode, det, ph, -eps, gain2)) / (2 * eps) * GAMMA
    if phase is None:
        grid = np.linspace(0, TWO_PI, 12, endpoint=False)
        probe = np.array([np.abs(sl(p)).max() for p in grid])
        phase = grid[np.argmax(probe)]
    return sl(phase), phase


def _damped(t, c, a, f, td, p):
    return c + a * np.exp(-t / td) * np.cos(TWO_PI * f * t + p)


def fit_rabi(t, y):
    t, y = np.asarray(t), np.asarray(y)
    f0 = np.fft.rfftfreq(t.size, t[1] - t[0])[np.argmax(np.abs(np.fft.rfft(y - y.mean()))[1:]) + 1]
    try:
        p, _ = curve_fit(_damped, t, y, p0=[y.mean(), 0.5 * np.ptp(y), f0, t[-1] / 3, 0.0], maxfev=20000)
    except RuntimeError:
        return {"f_mhz": f0, "decay_us": np.nan, "periods": np.nan, "contrast": np.nan}
    return {"f_mhz": abs(p[2]), "decay_us": abs(p[3]), "periods": abs(p[2] * p[3]), "contrast": 2 * abs(p[1])}


def _ramsey_model(hf, k):
    def f(t, c, a, fr, t2, p):
        beat = sum(np.cos(TWO_PI * (fr + k * hf * m) * t + p) for m in (-1, 0, 1)) / 3
        return c + a * np.exp(-((t / t2) ** 2)) * beat
    return f


def fit_ramsey(tau, y, mode, hf, f_guess, t_guess):
    k = 2 if mode == "dq" else 1
    tau, y = np.asarray(tau), np.asarray(y)
    model, best = _ramsey_model(hf, k), None
    for tg in (0.5 * t_guess, t_guess, 2 * t_guess):
        try:
            p, _ = curve_fit(model, tau, y, p0=[y.mean(), 0.5 * np.ptp(y), f_guess, tg, 0.0], maxfev=20000)
        except RuntimeError:
            continue
        r = np.sum((model(tau, *p) - y) ** 2)
        if best is None or r < best[0]:
            best = (r, p)
    if best is None:
        return {"contrast": np.nan, "t2star_us": np.nan, "f_mhz": np.nan}
    p = best[1]
    return {"contrast": 2 * abs(p[1]), "t2star_us": abs(p[3]), "f_mhz": abs(p[2]), "curve": model(tau, *p)}
