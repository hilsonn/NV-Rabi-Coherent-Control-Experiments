import numpy as np
from scipy.signal import welch


def psd_model(f, asd_at_ref, slope, f_ref):
    return asd_at_ref**2 * (np.asarray(f, float) / f_ref) ** (-slope)


def generate_noise_from_psd(n, dt, psd_fn, rng):
    f = np.fft.rfftfreq(n, dt)
    g = (rng.standard_normal(f.size - 1) + 1j * rng.standard_normal(f.size - 1)) / np.sqrt(2.0)
    if n % 2 == 0:
        g[-1] = rng.standard_normal()
    spec = np.zeros(f.size, complex)
    spec[1:] = np.sqrt(psd_fn(f[1:]) * n / (2.0 * dt)) * g
    return np.fft.irfft(spec, n)


def generate_noise(n, dt, asd_at_ref, slope, f_ref, rng):
    return generate_noise_from_psd(n, dt, lambda f: psd_model(f, asd_at_ref, slope, f_ref), rng)


def welch_asd(x, dt, nperseg):
    x = np.asarray(x, float)
    f, p = welch(x, fs=1.0 / dt, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return f, np.sqrt(p.mean(axis=0) if p.ndim == 2 else p)


def allan(x, dt, taus):
    """Overlapping Allan deviation of a rate series x at averaging times taus (s)."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    taus_out, dev = [], []
    for t in taus:
        m = int(round(t / dt))
        if m < 1 or 2 * m >= x.size:
            continue
        y = (cs[m:] - cs[:-m]) / m
        d = y[m:] - y[:-m]
        taus_out.append(m * dt)
        dev.append(np.sqrt(0.5 * np.mean(d**2)))
    return np.array(taus_out), np.array(dev)
