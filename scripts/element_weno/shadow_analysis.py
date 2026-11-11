#!/usr/bin/env python3
"""3D shadow (Weih et al. 4.2.1 geometry, own beam profile) analysis from slices.csv (docs/ELEMENT_HETEROGENEOUS.md).

Usage: shadow_analysis.py OUTDIR RUN_DIR...
Per run: L1 deviation in the plane z = 0 for x in [0.2, 0.7] from the uncollided Beer-Lambert density, centre-line
density at x = 0.5 against exp(-2 sigma R), free-space beam decay at x = -0.25 (centre line, should be 1 for the
kinetic beam), total absorption, balance, cone margin, solver time. Figure: centre lines and transverse profiles
at x = 0.5 with the Beer-Lambert reference.
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nx_convergence as nxc  # noqa: E402  (shadow_metrics)


def plane(run):
    rows = [r for r in csv.DictReader(open(run / "slices.csv")) if r["plane"] == "2"]
    return [(float(r["x"]), float(r["y"]), float(r["rho"])) for r in rows]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    runs = [Path(a) for a in sys.argv[2:]]
    table = []
    for run in runs:
        c, s = json.load(open(run / "config.json")), json.load(open(run / "summary.json"))
        sigma = c["shadow_sigma"]
        m = nxc.shadow_metrics(run, sigma)
        pts = plane(run)
        centre = sorted((x, r) for x, y, r in pts if abs(y) < 1e-12)
        free = min(centre, key=lambda t: abs(t[0] + .25))[1]
        table.append(dict(run=run.name, P=c["pn_order"], n=c["elements_per_axis"], sigma=sigma,
                          moments=c["active_moments"], mirror="".join("xyz"[d] for d in range(3) if c["mirror_directions"][d]),
                          steps=s["steps"], solver_s=s["solver_seconds"], L1=m["shadow_L1_plane"],
                          centre=m["shadow_centre_x05"], centre_ref=m["shadow_centre_ref"], free=free,
                          absorbed=s["ledger"]["absorbed"], balance=s["ledger"]["balance_relative"],
                          margin=s["minimum_cone_margin"], rss_mb=s["peak_rss_kib_rank0"] / 1024))
    table.sort(key=lambda r: (r["sigma"], r["n"], r["P"]))
    with open(out / "shadow_table.md", "w") as f:
        f.write("| σ | n (Zellen x·y·z) | P | Momente | Spiegel | Schritte | Löser (s) | L1 Ebene z=0 | ρ Achse x=0.5 (Ref.) | "
                "ρ Achse x=−0.25 (Ref. 1) | Absorption | Bilanz | min. Marge |\n|---:|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|\n")
        for r in table:
            f.write(f"| {r['sigma']:g} | {r['n']} ({3*r['n']}·{r['n']}·{r['n']}) | P{r['P']} | {r['moments']} | {r['mirror']} | "
                    f"{r['steps']} | {r['solver_s']:.1f} | {r['L1']:.3f} | {r['centre']:.3f} ({r['centre_ref']:.3g}) | "
                    f"{r['free']:.3f} | {r['absorbed']:.4f} | {r['balance']:.1e} | {r['margin']:.1e} |\n")
    with open(out / "shadow_table.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sigmas = sorted({r["sigma"] for r in table})
    fig, axes = plt.subplots(2, len(sigmas), figsize=(4 * len(sigmas), 6.5), squeeze=False)
    for k, sigma in enumerate(sigmas):
        ax1, ax2 = axes[0][k], axes[1][k]
        for run in runs:
            c = json.load(open(run / "config.json"))
            if c["shadow_sigma"] != sigma:
                continue
            pts = plane(run)
            lab = f"P{c['pn_order']} n={c['elements_per_axis']}"
            cl = sorted((x, r) for x, y, r in pts if abs(y) < 1e-12)
            ax1.plot([t[0] for t in cl], [t[1] for t in cl], label=lab)
            xs = sorted({x for x, y, r in pts}); x5 = min(xs, key=lambda v: abs(v - .5))
            tr = sorted((y, r) for x, y, r in pts if x == x5)
            ax2.plot([t[0] for t in tr], [t[1] for t in tr], label=lab)
        X = [-.75 + 1.5 * i / 600 for i in range(601)]
        ax1.plot(X, [math.exp(-sigma * max(0., min(.1, x) - (-.1))) if x > -.1 else 1. for x in X], "k--", lw=1, label="Beer-Lambert")
        Y = [-.25 + .5 * i / 400 for i in range(401)]
        def prof(y):
            r = abs(y)
            p = 1. if r <= .15 else (0. if r >= .2 else math.cos(.5 * math.pi * (r - .15) / .05) ** 2)
            return p * math.exp(-sigma * 2 * math.sqrt(max(.01 - y * y, 0.)))
        ax2.plot(Y, [prof(y) for y in Y], "k--", lw=1, label="Beer-Lambert")
        ax1.set_title(f"σ = {sigma:g}: Achse y = z = 0", fontsize=9); ax1.set_xlabel("x"); ax1.set_ylabel("ρ")
        ax2.set_title(f"σ = {sigma:g}: quer, x = 0.5, z = 0", fontsize=9); ax2.set_xlabel("y"); ax2.set_ylabel("ρ")
        ax1.legend(fontsize=6); ax2.legend(fontsize=6); ax1.grid(alpha=.3); ax2.grid(alpha=.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"shadow_profiles.{ext}", dpi=150)
    print(open(out / "shadow_table.md").read())


if __name__ == "__main__":
    main()
