import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.common import out


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(out(name + ".png"), dpi=110)
    plt.close(fig)


def curves(name, panels):
    """panels: list of dict(curves=[(x, y, label[, fmt])], xlabel, ylabel, title, logx, logy, hlines, vlines)."""
    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    for ax, p in zip(axs.flat, panels):
        for c in p["curves"]:
            ax.plot(c[0], c[1], *([c[3]] if len(c) > 3 else []), label=c[2])
        for h in p.get("hlines", []):
            ax.axhline(h, color="k", ls="--", lw=0.8)
        for v in p.get("vlines", []):
            ax.axvline(v, color="k", ls=":", lw=0.8)
        ax.set(xlabel=p.get("xlabel", ""), ylabel=p.get("ylabel", ""), title=p.get("title", ""))
        if p.get("logx"):
            ax.set_xscale("log")
        if p.get("logy"):
            ax.set_yscale("log")
        if any(c[2] for c in p["curves"]):
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axs.flat[n:]:
        ax.axis("off")
    _save(fig, name)


def heat(name, panels):
    """panels: list of dict(x, y, z, xlabel, ylabel, title, clabel, log, levels, mark=(x, y), logx, logy)."""
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4), squeeze=False)
    for ax, p in zip(axs[0], panels):
        z = np.asarray(p["z"], float)
        im = ax.pcolormesh(p["x"], p["y"], z, shading="auto", norm="log" if p.get("log") else None, vmin=p.get("vmin"), vmax=p.get("vmax"))
        fig.colorbar(im, ax=ax, label=p.get("clabel", ""))
        if "levels" in p:
            ax.contour(p["x"], p["y"], z, levels=p["levels"], colors="w", linewidths=0.8)
        if "mark" in p:
            ax.plot(*p["mark"], "r*", ms=10)
        ax.set(xlabel=p.get("xlabel", ""), ylabel=p.get("ylabel", ""), title=p.get("title", ""))
        if p.get("logx"):
            ax.set_xscale("log")
        if p.get("logy"):
            ax.set_yscale("log")
    _save(fig, name)


def bars(name, labels, values, ylabel="", title="", stacked=None, hline=None):
    """values: 1-D, or with stacked=[series names] an (n_series, n_labels) array."""
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(labels) + 3), 4))
    if stacked:
        base = np.zeros(len(labels))
        for nm, v in zip(stacked, values):
            ax.bar(labels, v, bottom=base, label=nm)
            base = base + np.asarray(v, float)
        ax.legend(fontsize=7)
    else:
        ax.bar(labels, values)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--")
    ax.set(ylabel=ylabel, title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, name)


def scatter(name, pts, xlabel="", ylabel="", title="", logy=False):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for x, y, lab in pts:
        ax.plot(x, y, "o")
        ax.annotate(lab, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    _save(fig, name)
