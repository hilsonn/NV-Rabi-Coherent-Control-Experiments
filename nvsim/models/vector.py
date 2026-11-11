import numpy as np
from scipy.optimize import least_squares

from models import spin
from models.common import GAMMA


def forward(b_t, d_mhz, axes, gamma_scale=None):
    """The eight line frequencies (MHz): nu- of the four orientations, then nu+. gamma_scale: per-orientation field scale."""
    b = np.asarray(b_t, float)
    if gamma_scale is None:
        lm, lp = spin.lines(d_mhz, b, axes)
        return np.concatenate([lm[0], lp[0]])
    lm = np.zeros(len(axes))
    lp = np.zeros(len(axes))
    for k in range(len(axes)):
        a, c = spin.lines(d_mhz, b * gamma_scale[k], axes[k:k + 1])
        lm[k], lp[k] = a[0, 0], c[0, 0]
    return np.concatenate([lm, lp])


def recover(lines, d_mhz, axes, x0, use=slice(None)):
    """Nonlinear least squares for the field vector (T) from measured lines with the exact forward model."""
    r = least_squares(lambda b: (forward(b * 1e-6, d_mhz, axes) - lines)[use], np.asarray(x0, float) * 1e6, x_scale=1.0, xtol=1e-14, ftol=1e-14)
    return r.x * 1e-6


def jacobian(b_t, d_mhz, axes, h=1e-8):
    """d lines / d B (MHz per tesla), shape (8, 3)."""
    b = np.asarray(b_t, float)
    cols = [(forward(b + h * e, d_mhz, axes) - forward(b - h * e, d_mhz, axes)) / (2 * h) for e in np.eye(3)]
    return np.stack(cols, 1)


def noise_asd(b_t, d_mhz, axes, eta_line_t, n_lines=8):
    """Per-axis field noise (T/rtHz) and condition number from the line Jacobian.

    eta_line_t: single-line field-equivalent noise (T/rtHz) if one line were measured full time; the n_lines lines are time-shared.
    """
    j = jacobian(b_t, d_mhz, axes)
    sig = GAMMA * eta_line_t * np.sqrt(n_lines)
    cov = np.linalg.inv(j.T @ j) * sig**2
    return np.sqrt(np.diag(cov)), float(np.linalg.cond(j))


def fit_matrix(meas, true):
    """Least-squares meas ~ A true + o; returns A (3,3), o (3,)."""
    x = np.hstack([true, np.ones((len(true), 1))])
    sol, *_ = np.linalg.lstsq(x, meas, rcond=None)
    return sol[:3].T, sol[3]


def scale_orthogonality(a):
    """Per-axis scale (column norm - 1) and pairwise angle error (deg) of the columns of A."""
    n = np.linalg.norm(a, axis=0)
    cols = a / n
    ang = [np.degrees(np.arccos(np.clip(cols[:, i] @ cols[:, j], -1, 1))) - 90 for i, j in ((0, 1), (0, 2), (1, 2))]
    return n - 1, np.array(ang)
