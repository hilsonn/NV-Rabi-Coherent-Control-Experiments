import numpy as np

from models.common import GAMMA

G_HZ = GAMMA * 1e6  # Hz per tesla


def gains(kp, ki_ratio):
    return kp, ki_ratio * kp**2


def response(kp, ki, t_u, f):
    """Closed-loop field-to-output transfer function of the discrete PI loop at frequencies f (Hz)."""
    z = np.exp(2j * np.pi * np.asarray(f) * t_u)
    g = (kp * (z - 1) + ki) / (z - 1) ** 2
    return g / (1 + g)


def bandwidth_hz(kp, ki, t_u):
    f = np.geomspace(1e-2 / t_u / 1e3, 0.5 / t_u, 4000)
    h = np.abs(response(kp, ki, t_u, f))
    k = np.nonzero(h < 10 ** (-3 / 20))[0]
    return float(f[k[0]]) if k.size else float(f[-1])


def tracking_noise_t(kp, ki, t_u, eta_t):
    """RMS field noise (T) of the locked output from white measurement noise of ASD eta_t (T/rtHz)."""
    f = np.linspace(1e-3, 0.5 / t_u, 20000)
    return float(eta_t * np.sqrt(np.trapezoid(np.abs(response(kp, ki, t_u, f)) ** 2, f)))


def simulate(tau_us, t_u, kp, ki, step_hz, sigma_delta_hz, t_end, field_fn, n, rng, fallback=False, tau_coarse_us=None, record=False):
    """Frequency-locked loop on n parallel cases. field_fn(t) -> (n,) field in tesla.

    Error signal e = sin(2 pi delta tau) + noise; delta_hat = e / (2 pi tau); PI step on the synthesiser (quantised).
    Lock is lost when |delta| exceeds the fine capture range 1/(4 tau); with fallback the loop switches to the coarse
    (short-tau) discriminator until |delta| is back inside half the fine range.
    Returns dict: lost (any time out of capture), final_lost, max_delta, history of delta when record.
    """
    tau_f = tau_us * 1e-6
    tau_c = (tau_coarse_us or tau_us / 8) * 1e-6
    cap = 1 / (4 * tau_f)
    sig_e = 2 * np.pi * tau_f * sigma_delta_hz
    steps = int(t_end / t_u)
    cmd = np.zeros(n)
    acc = np.zeros(n)
    coarse = np.zeros(n, bool)
    lost = np.zeros(n, bool)
    hist = np.zeros((steps, n)) if record else None
    est = np.zeros((steps, n)) if record else None
    dmax = np.zeros(n)
    for k in range(steps):
        fs = np.round(cmd / step_hz) * step_hz if step_hz > 0 else cmd
        delta = G_HZ * field_fn(k * t_u) - fs
        out = np.abs(delta) > cap
        lost |= out
        if fallback:
            coarse = np.where(out, True, np.where(np.abs(delta) < 0.5 * cap, False, coarse))
        tau = np.where(coarse, tau_c, tau_f)
        e = np.sin(2 * np.pi * delta * tau) + sig_e * rng.standard_normal(n)
        dh = e / (2 * np.pi * tau)
        cmd = cmd + kp * dh + ki * acc
        acc = acc + dh
        dmax = np.maximum(dmax, np.abs(delta))
        if record:
            hist[k], est[k] = delta, fs + dh
    fs = np.round(cmd / step_hz) * step_hz if step_hz > 0 else cmd
    final = np.abs(G_HZ * field_fn(steps * t_u) - fs) > cap
    return {"lost": lost, "final_lost": final, "max_delta_hz": dmax, "delta": hist, "estimate_hz": est, "capture_hz": cap}
