#!/usr/bin/env python3
"""Lattice 2D with FCS: density maps without/with angular filter (log) and their difference.

Usage: lattice_fcs_filter_maps.py OUTDIR LABEL:OFF_RUN:FILTER_RUN [...]
Density rho = sqrt(4 pi) u0 (u0 in final.csv is the total density incl. the uncollided part).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, SymLogNorm  # noqa: E402

FLOOR = 1e-7  # lower end of the log scale, as in lattice_analysis.py


def load(run):
    run = Path(run)
    d = np.loadtxt(run / "final.csv", delimiter=",", skiprows=1, usecols=(1, 2, 4))
    x, y, u0 = d.T
    xs, ys = np.unique(x), np.unique(y)
    order = np.lexsort((x, y))
    Z = (math.sqrt(4 * math.pi) * u0[order]).reshape(len(ys), len(xs))
    cfg = json.load(open(run / "config.json"))
    summ = json.load(open(run / "summary.json")).get("lattice_fcs", {})
    return xs, ys, Z, cfg, summ


def describe(cfg):
    f = cfg.get("filter_mode", "off")
    s = "" if f == "off" else f", σ_f={cfg.get('filter_constant')}"
    return f"filter {f}{s}"


def panel_row(axes, label, off, fil, fig):
    xs, ys, Zo, co, so = off
    xs2, ys2, Zf, cf, sf = fil
    assert Zo.shape == Zf.shape and np.allclose(xs, xs2), "runs on different grids"
    vmax = max(Zo.max(), Zf.max())
    lnorm = LogNorm(vmin=FLOOR, vmax=vmax)
    D = Zf - Zo
    dmax = np.abs(D).max()
    rel = np.abs(D) / np.maximum(np.maximum(Zo, Zf), FLOOR)

    m0 = axes[0].pcolormesh(xs, ys, np.maximum(Zo, FLOOR), norm=lnorm, cmap="magma", shading="auto", rasterized=True)
    axes[0].set_title(f"{label}: FCS, ohne Filter\nMasse {so.get('total_mass_final', float('nan')):.4f}", fontsize=9)
    axes[1].pcolormesh(xs, ys, np.maximum(Zf, FLOOR), norm=lnorm, cmap="magma", shading="auto", rasterized=True)
    axes[1].set_title(f"{label}: FCS, {describe(cf)}\nMasse {sf.get('total_mass_final', float('nan')):.4f}", fontsize=9)
    m2 = axes[2].pcolormesh(xs, ys, D, norm=SymLogNorm(linthresh=1e-5, vmin=-dmax, vmax=dmax, base=10),
                            cmap="RdBu_r", shading="auto", rasterized=True)
    axes[2].set_title(f"ρ_Filter − ρ_ohne  (max |Δ| = {dmax:.2e})", fontsize=9)
    m3 = axes[3].pcolormesh(xs, ys, np.maximum(rel, 1e-6), norm=LogNorm(vmin=1e-4, vmax=max(rel.max(), 1e-3)),
                            cmap="viridis", shading="auto", rasterized=True)
    axes[3].set_title("|Δ| / max(ρ_ohne, ρ_Filter, 1e-7)", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([0, 3.5, 7])
        ax.set_yticks([0, 3.5, 7])
        ax.tick_params(labelsize=7)
    fig.colorbar(m0, ax=axes[:2], shrink=0.85, pad=0.02, label="ρ (log)")
    fig.colorbar(m2, ax=axes[2], shrink=0.85, pad=0.02, label="Δρ (symlog, linthresh 1e-5)")
    fig.colorbar(m3, ax=axes[3], shrink=0.85, pad=0.02, label="relativ (log)")
    return dict(label=label, max_abs_diff=float(dmax), l1_diff=float(np.abs(D).sum() * (xs[1] - xs[0]) ** 2),
                l1_rel=float(np.abs(D).sum() / Zo.sum()), mass_off=so.get("total_mass_final"),
                mass_filter=sf.get("total_mass_final"), absorbed_off=so.get("total_absorbed"),
                absorbed_filter=sf.get("total_absorbed"))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    pairs = [a.split(":") for a in sys.argv[2:]]
    stats = []
    big, bax = plt.subplots(len(pairs), 4, figsize=(19, 4.3 * len(pairs)), constrained_layout=True, squeeze=False)
    for row, (label, off_run, fil_run) in enumerate(pairs):
        off, fil = load(off_run), load(fil_run)
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
        stats.append(panel_row(axes, label, off, fil, fig))
        fig.savefig(out / f"lattice_fcs_filter_{label.replace('²', '').replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
        panel_row(bax[row], label, off, fil, big)
    big.savefig(out / "lattice_fcs_filter_overview.png", dpi=110)
    big.savefig(out / "lattice_fcs_filter_overview.pdf")
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
