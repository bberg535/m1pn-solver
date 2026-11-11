#!/usr/bin/env python3
"""Figures and tables of a heterogeneous/3D NX campaign (profile hetero-paper; docs/ELEMENT_HETEROGENEOUS.md).

Usage: nx_hetero_figures.py CAMPAIGN [CAMPAIGN ...] --out DIR [--b-lineout CSV] [--b-runs CSV]

Reads runs/<label>/ and status/<label>.json of every campaign (several campaigns, e.g. a recovery campaign, are
merged by label; only runs with status 'completed' are used). Produces (PNG and PDF):

  lattice_density_maps     log density, element WENO Q1/224 (or Q1/112) for P3/P7/P11 and path B (new binary)
  lattice_convergence      deviation from Schotthoefer/Hauck (Table 1 / 2 pi) vs control points per axis, per P, Q
  lattice_error_vs_time    |deviation| vs solver time (efficiency), element WENO and path B
  lattice_lineout          density at y = 3.5: element WENO, path B new, historical path B (older binary)
  shadow_density_maps      density in the plane z = 0 for every sigma at the finest n (P3/P7/P11)
  shadow_convergence       centre-line density at x = 0.5 and free-beam density at x = -0.25 vs n (Beer-Lambert)
  shadow_profiles          transverse profiles at x = 0.5 vs Beer-Lambert, finest n
  lattice3d_slices         log density in the three mid-planes, finest run per P
  lattice3d_convergence    mass and total absorption vs resolution (self convergence)
and tables lattice.md, shadow.md, lattice3d.md. Missing families are skipped with a note, never invented.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

KIN = {"mass": 13.07 / (2 * math.pi), "absorbed": 7.034 / (2 * math.pi), "rate": 4.597 / (2 * math.pi)}


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=160)
    plt.close(fig)


def runs(campaigns):
    found = {}
    for c in campaigns:
        tasks = {t["label"]: t for t in json.load(open(c / "tasks.json"))}
        for label, t in tasks.items():
            st = c / "status" / (label + ".json")
            if st.is_file() and json.load(open(st)).get("status") == "completed":
                found[label] = (t, c / "runs" / label)
    return found


def field2d(run):
    """(x, y, rho) of final.csv (element WENO samples or path-B nodes)."""
    rows = list(csv.reader(open(run / "final.csv")))
    h = rows[0]
    ix, iy, i0 = h.index("x"), h.index("y"), h.index("u0")
    return np.array([[float(r[ix]), float(r[iy]), math.sqrt(4 * math.pi) * float(r[i0])] for r in rows[1:]])


def lattice_quantities(t, run):
    s = json.load(open(run / "summary.json"))
    if "ledger" in s:  # element WENO
        return dict(mass=s["final_density_mass"], absorbed=s["ledger"]["absorbed"], rate=s["absorption_rate_final"],
                    seconds=s["solver_seconds"])
    # path B: mass from the nodal field and lumped masses; total absorption from its reaction ledger if present
    f = field2d(run)
    m = np.array([float(r["mass"]) for r in csv.DictReader(open(run / "nodal_mass.csv"))])
    mat = {int(r["node"]): float(r["sigma_a"]) for r in csv.DictReader(open(run / "material.csv"))} \
        if (run / "material.csv").is_file() else None
    out = dict(mass=float(np.sum(m * f[:, 2])), seconds=s.get("solver_seconds", float("nan")))
    if mat:
        sa = np.array([mat[i] for i in range(len(m))])
        out["rate"] = float(np.sum(m * sa * f[:, 2]))
    return out


def plot_map(ax, pts, title, vmin=1e-7):
    xs, ys = np.unique(pts[:, 0]), np.unique(pts[:, 1])
    Z = np.full((len(ys), len(xs)), np.nan)
    xi, yi = np.searchsorted(xs, pts[:, 0]), np.searchsorted(ys, pts[:, 1])
    Z[yi, xi] = np.maximum(pts[:, 2], vmin * 1e-3)
    im = ax.pcolormesh(xs, ys, Z, norm=LogNorm(vmin=vmin, vmax=np.nanmax(Z)), shading="auto", cmap="magma")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    return im


def lattice(found, out, notes, b_lineout):
    ew = {l: v for l, v in found.items() if v[0]["family"] == "lattice"}
    bn = {l: v for l, v in found.items() if v[0]["family"] == "lattice_B"}
    if not ew:
        notes.append("lattice: no completed runs")
        return
    rows = []
    for label, (t, run) in sorted(ew.items()):
        q = lattice_quantities(t, run)
        c = json.load(open(run / "config.json"))
        rows.append(dict(label=label, P=t["pn_order"], Q=t["degree"], E=t["elements"], tag=label.split("_")[-1],
                         points=t["elements"] * t["degree"] + 1, dt=c["dt_selected"], **q,
                         dev_mass=q["mass"] / KIN["mass"] - 1, dev_absorbed=q["absorbed"] / KIN["absorbed"] - 1,
                         dev_rate=q["rate"] / KIN["rate"] - 1, method="EW"))
    for label, (t, run) in sorted(bn.items()):
        q = lattice_quantities(t, run)
        r = dict(label=label, P=t["pn_order"], Q=1, E=t["elements"], tag="B_new", points=t["elements"] + 1,
                 dt=t.get("fixed_dt"), method="B (neu)", **q)
        r["dev_mass"] = q["mass"] / KIN["mass"] - 1
        r["dev_rate"] = q["rate"] / KIN["rate"] - 1 if "rate" in q else float("nan")
        r["dev_absorbed"] = float("nan")
        rows.append(r)
    with open(out / "lattice.md", "w") as f:
        f.write("| Verfahren | P | Q/E | Punkte | dt | Löser (s) | Masse (Abw.) | Absorption gesamt (Abw.) | Rate T (Abw.) |\n"
                "|---|---|---|---:|---:|---:|---|---|---|\n")
        for r in sorted(rows, key=lambda r: (r["method"], r["P"], r["Q"], r["points"], str(r["dt"]))):
            f.write(f"| {r['method']} | P{r['P']} | Q{r['Q']}/{r['E']} {r['tag'] if r['tag'] in ('fine', 'dtB') else ''} | "
                    f"{r['points']} | {r['dt']:.3g} | {r['seconds']:.1f} | {r['mass']:.4f} ({100 * r['dev_mass']:+.2f} %) | "
                    + (f"{r['absorbed']:.4f} ({100 * r['dev_absorbed']:+.2f} %)" if 'absorbed' in r else "–") + " | "
                    + (f"{r['rate']:.4f} ({100 * r['dev_rate']:+.2f} %)" if 'rate' in r else "–") + " |\n")
        f.write(f"\nReferenz (Schotthöfer/Hauck Tab. 1 / 2π): Masse {KIN['mass']:.4f}, Absorption {KIN['absorbed']:.4f}, "
                f"Rate {KIN['rate']:.4f}. Pfad B (neu) mit aktuellem Binary und historischem dt.\n")
    # convergence: |deviation| vs control points per axis
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, name in zip(axes, ("dev_mass", "dev_absorbed", "dev_rate"),
                             ("Masse bei T", "Absorption gesamt", "Absorptionsrate bei T")):
        for P, marker in ((3, "o"), (7, "s"), (11, "^")):
            for Q, color in ((1, "C0"), (2, "C1"), (3, "C2"), (4, "C3")):
                sel = sorted((r["points"], abs(r[key])) for r in rows
                             if r["method"] == "EW" and r["P"] == P and r["Q"] == Q and r["tag"] != "dtB")
                if sel:
                    ax.loglog(*zip(*sel), marker=marker, color=color, lw=1, ms=4, label=f"P{P} Q{Q}")
            b = [(r["points"], abs(r[key])) for r in rows if r["method"] != "EW" and r["P"] == P and math.isfinite(r[key])]
            if b:
                ax.loglog(*zip(*b), marker=marker, color="k", ls="none", ms=6, mfc="none", label=f"P{P} B (neu)")
        ticks = sorted({r["points"] for r in rows})
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(ticks); ax.set_xticklabels([str(v) for v in ticks])
        ax.set_xlabel("Kontrollpunkte pro Achse"); ax.set_ylabel("|rel. Abweichung| zur kinetischen Ref."); ax.set_title(name, fontsize=9)
        ax.grid(alpha=.3, which="both")
    axes[0].legend(fontsize=6, ncol=2)
    save(fig, out, "lattice_convergence")
    # error vs time
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for r in rows:
        if math.isfinite(r["dev_mass"]):
            if r["method"] == "EW":
                ax.loglog(r["seconds"], abs(r["dev_mass"]), "o", ms=4, color={3: "C0", 7: "C1", 11: "C2"}[r["P"]])
            else:
                ax.loglog(r["seconds"], abs(r["dev_mass"]), "kx", ms=6)
    for P, col in ((3, "C0"), (7, "C1"), (11, "C2")):
        ax.plot([], [], "o", color=col, label=f"EW P{P}")
    ax.plot([], [], "kx", label="Pfad B (neu)")
    ax.set_xlabel("Lösezeit (s), 16 Kerne nx"); ax.set_ylabel("|rel. Abweichung Masse|"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    save(fig, out, "lattice_error_vs_time")
    # density maps: finest EW Q1 per P plus path B new
    panels = []
    for P in (3, 7, 11):
        cand = [(t["elements"], l) for l, (t, run) in ew.items() if t["pn_order"] == P and t["degree"] == 1 and "dtB" not in l]
        if cand:
            l = max(cand)[1]
            panels.append((f"EW Q1/{ew[l][0]['elements']} P{P}", field2d(ew[l][1])))
    for l, (t, run) in sorted(bn.items(), key=lambda kv: kv[1][0]["pn_order"]):
        panels.append((f"Pfad B Q1/{t['elements']} P{t['pn_order']} (neu)", field2d(run)))
    if panels:
        fig, axes = plt.subplots(1 + (len(panels) > 3), min(3, len(panels)), figsize=(4.2 * min(3, len(panels)), 3.8 * (1 + (len(panels) > 3))),
                                 squeeze=False)
        for ax, (title, pts) in zip(axes.flat, panels):
            fig.colorbar(plot_map(ax, pts, title), ax=ax, fraction=.046)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        save(fig, out, "lattice_density_maps")
    # line-out y = 3.5
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for P, col in ((3, "C0"), (7, "C1"), (11, "C2")):
        cand = [(t["elements"], l) for l, (t, run) in ew.items() if t["pn_order"] == P and t["degree"] == 1 and "dtB" not in l]
        if cand:
            pts = field2d(ew[max(cand)[1]][1]); line = pts[np.abs(pts[:, 1] - 3.5) < 1e-9]; line = line[np.argsort(line[:, 0])]
            ax.semilogy(line[:, 0], np.maximum(line[:, 2], 1e-12), color=col, label=f"EW Q1/{max(cand)[0]} P{P}")
        for l, (t, run) in bn.items():
            if t["pn_order"] == P:
                pts = field2d(run); line = pts[np.abs(pts[:, 1] - 3.5) < 1e-9]; line = line[np.argsort(line[:, 0])]
                ax.semilogy(line[:, 0], np.maximum(line[:, 2], 1e-12), color=col, ls=":", label=f"B neu P{P}")
        if b_lineout and b_lineout.is_file():
            b = [r for r in csv.DictReader(open(b_lineout)) if int(r["order"]) == P]
            ax.semilogy([float(r["x"]) for r in b], [float(r["density"]) for r in b], color=col, ls="--", lw=.8,
                        label=f"B historisch P{P}")
    ax.set_xlabel("x"); ax.set_ylabel("ρ (y = 3.5, t = 3.2)"); ax.legend(fontsize=6, ncol=3); ax.grid(alpha=.3)
    save(fig, out, "lattice_lineout")


def slices(run, plane):
    rows = [r for r in csv.DictReader(open(run / "slices.csv")) if r["plane"] == str(plane)]
    u, v = (1, 2) if plane == 0 else ((0, 2) if plane == 1 else (0, 1))
    keys = ("x", "y", "z")
    return np.array([[float(r[keys[u]]), float(r[keys[v]]), float(r["rho"])] for r in rows])


def beer_lambert(x, y, sigma):
    r = np.abs(y)
    prof = np.where(r <= .15, 1., np.where(r >= .2, 0., np.cos(.5 * np.pi * (r - .15) / .05) ** 2))
    half = np.sqrt(np.maximum(.01 - y * y, 0))
    chord = np.where(y * y < .01, np.maximum(0, np.minimum(half, x) - np.maximum(-half, -.75)), 0.)
    return prof * np.exp(-sigma * chord)


def shadow(found, out, notes):
    sh = {l: v for l, v in found.items() if v[0]["kind"] == "shadow"}
    if not sh:
        notes.append("shadow: no completed runs")
        return
    rows = []
    for label, (t, run) in sorted(sh.items()):
        s = json.load(open(run / "summary.json"))
        pts = slices(run, 2)
        cl = pts[np.abs(pts[:, 1]) < 1e-12]; cl = cl[np.argsort(cl[:, 0])]
        mask = (pts[:, 0] >= .2) & (pts[:, 0] <= .7)
        ref = beer_lambert(pts[:, 0], pts[:, 1], t["shadow_sigma"])
        rows.append(dict(label=label, sigma=t["shadow_sigma"], P=t["pn_order"], n=t["elements"],
                         centre=float(np.interp(.5, cl[:, 0], cl[:, 2])), free=float(np.interp(-.25, cl[:, 0], cl[:, 2])),
                         L1=float(np.sum(np.abs(pts[mask, 2] - ref[mask])) / np.sum(ref[mask])),
                         absorbed=s["ledger"]["absorbed"], balance=s["ledger"]["balance_relative"],
                         seconds=s["solver_seconds"], rss=s["peak_rss_kib_rank0"] / 2**20, run=run))
    with open(out / "shadow.md", "w") as f:
        f.write("| σ | P | n | ρ Achse x=0.5 (Ref.) | ρ Achse x=−0.25 (Ref. 1) | L1 Ebene | Absorption | Bilanz | Löser (s) | RSS (GB) |\n"
                "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda r: (r["sigma"], r["P"], r["n"])):
            f.write(f"| {r['sigma']:g} | P{r['P']} | {r['n']} | {r['centre']:.3g} ({math.exp(-.2 * r['sigma']):.3g}) | "
                    f"{r['free']:.3f} | {r['L1']:.3f} | {r['absorbed']:.4g} | {r['balance']:.1e} | {r['seconds']:.0f} | {r['rss']:.1f} |\n")
    sigmas = sorted({r["sigma"] for r in rows})
    fig, axes = plt.subplots(2, len(sigmas), figsize=(4 * len(sigmas), 6.5), squeeze=False)
    for k, sigma in enumerate(sigmas):
        for P, mk in ((3, "o"), (7, "s"), (11, "^")):
            sel = sorted((r["n"], r) for r in rows if r["sigma"] == sigma and r["P"] == P)
            if not sel:
                continue
            ns = [a for a, _ in sel]
            axes[0][k].loglog(ns, [max(abs(r["centre"] - math.exp(-.2 * sigma)), 1e-12) for _, r in sel], marker=mk, label=f"P{P}")
            axes[1][k].semilogx(ns, [r["free"] for _, r in sel], marker=mk, label=f"P{P}")
        axes[0][k].set_title(f"σ = {sigma:g}: |ρ(0.5,0,0) − Beer-Lambert|", fontsize=8)
        axes[1][k].set_title(f"σ = {sigma:g}: freier Strahl ρ(−0.25,0,0) (Ref. 1)", fontsize=8)
        axes[1][k].axhline(1, color="k", ls="--", lw=.8)
        for a in (axes[0][k], axes[1][k]):
            a.set_xlabel("n (Zellen pro 0.5)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=6)
    save(fig, out, "shadow_convergence")
    finest = {}
    for r in rows:
        key = (r["sigma"], r["P"])
        if key not in finest or r["n"] > finest[key]["n"]:
            finest[key] = r
    Ps = sorted({k[1] for k in finest})
    fig, axes = plt.subplots(len(sigmas), len(Ps), figsize=(4.6 * len(Ps), 2.2 * len(sigmas)), squeeze=False)
    for i, sigma in enumerate(sigmas):
        for j, P in enumerate(Ps):
            r = finest.get((sigma, P))
            if r is None:
                axes[i][j].axis("off"); continue
            fig.colorbar(plot_map(axes[i][j], slices(r["run"], 2), f"σ={sigma:g} P{P} n={r['n']}, z=0", vmin=1e-4),
                         ax=axes[i][j], fraction=.02)
    save(fig, out, "shadow_density_maps")
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 3.4), squeeze=False)
    yy = np.linspace(-.25, .25, 401)
    for k, sigma in enumerate(sigmas):
        ax = axes[0][k]
        for P in Ps:
            r = finest.get((sigma, P))
            if r is None:
                continue
            pts = slices(r["run"], 2)
            xs = np.unique(pts[:, 0]); x5 = xs[np.argmin(np.abs(xs - .5))]
            line = pts[pts[:, 0] == x5]; line = line[np.argsort(line[:, 1])]
            ax.plot(line[:, 1], line[:, 2], label=f"P{P} n={r['n']}")
        ax.plot(yy, beer_lambert(.5, yy, sigma), "k--", lw=1, label="Beer-Lambert")
        ax.set_title(f"σ = {sigma:g}, x = 0.5, z = 0", fontsize=8); ax.set_xlabel("y"); ax.set_ylabel("ρ")
        ax.grid(alpha=.3); ax.legend(fontsize=6)
    save(fig, out, "shadow_profiles")


def lattice3d(found, out, notes):
    l3 = {l: v for l, v in found.items() if v[0]["kind"] == "lattice3d"}
    if not l3:
        notes.append("lattice3d: no completed runs")
        return
    rows = []
    for label, (t, run) in sorted(l3.items()):
        s = json.load(open(run / "summary.json"))
        rows.append(dict(label=label, P=t["pn_order"], Q=t["degree"], E=t["elements"], points=t["elements"] * t["degree"] + 1,
                         mass=s["final_density_mass"], absorbed=s["ledger"]["absorbed"], outflow=s["ledger"]["boundary_outflow"],
                         balance=s["ledger"]["balance_relative"], margin=s["minimum_cone_margin"], seconds=s["solver_seconds"], run=run))
    with open(out / "lattice3d.md", "w") as f:
        f.write("| P | Q/E | Punkte | Masse | Absorption gesamt | Ausfluss | Bilanz | min. Marge | Löser (s) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda r: (r["P"], r["points"], r["Q"])):
            f.write(f"| P{r['P']} | Q{r['Q']}/{r['E']} | {r['points']} | {r['mass']:.5f} | {r['absorbed']:.5f} | {r['outflow']:.3e} | "
                    f"{r['balance']:.1e} | {r['margin']:.1e} | {r['seconds']:.0f} |\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for P, mk in ((3, "o"), (7, "s")):
        for Q, col in ((1, "C0"), (2, "C1")):
            sel = sorted((r["points"], r["mass"], r["absorbed"]) for r in rows if r["P"] == P and r["Q"] == Q)
            if sel:
                axes[0].plot([a for a, _, _ in sel], [b for _, b, _ in sel], marker=mk, color=col, label=f"P{P} Q{Q}")
                axes[1].plot([a for a, _, _ in sel], [c for _, _, c in sel], marker=mk, color=col, label=f"P{P} Q{Q}")
    axes[0].set_title("Masse bei T (Selbstkonvergenz)", fontsize=9); axes[1].set_title("Absorption gesamt", fontsize=9)
    for a in axes:
        a.set_xlabel("Kontrollpunkte pro Achse"); a.grid(alpha=.3); a.legend(fontsize=7)
    save(fig, out, "lattice3d_convergence")
    finest = {}
    for r in rows:
        if r["P"] not in finest or r["points"] > finest[r["P"]]["points"]:
            finest[r["P"]] = r
    fig, axes = plt.subplots(len(finest), 3, figsize=(12, 3.8 * len(finest)), squeeze=False)
    for i, (P, r) in enumerate(sorted(finest.items())):
        for plane, name in ((0, "x = 3.5"), (1, "y = 3.5"), (2, "z = 3.5")):
            fig.colorbar(plot_map(axes[i][plane], slices(r["run"], plane), f"P{P} Q{r['Q']}/{r['E']}, {name}", vmin=1e-8),
                         ax=axes[i][plane], fraction=.046)
    save(fig, out, "lattice3d_slices")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaigns", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--b-lineout", type=Path)
    ns = ap.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    found, notes = runs(ns.campaigns), []
    lattice(found, ns.out, notes, ns.b_lineout)
    shadow(found, ns.out, notes)
    lattice3d(found, ns.out, notes)
    (ns.out / "NOTES.md").write_text("\n".join(["# Hinweise", ""] + [f"- {n}" for n in notes]
                                               + [f"- {len(found)} abgeschlossene Läufe ausgewertet"]) + "\n")
    print("\n".join(notes), f"\n{len(found)} runs; figures in {ns.out}")


if __name__ == "__main__":
    main()
