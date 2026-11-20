#!/usr/bin/env python3
"""Summarize completed runs of tardis_paper_recompute.sbatch into a Markdown report."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

MC_2D = dict(mass=2.080890, absorbed=1.119098, rate=0.731555)
MC_3D = dict(mass=0.702146, absorbed=2.497854, rate=0.993430)
MC_FIELD_2D = Path("/home/user/bberg/phd/03_projects/M1PN_Solver/tmp/lattice_mc_field/lattice-paul_field224.csv")


def load_mc_field():
    if not MC_FIELD_2D.exists():
        return None
    d = np.loadtxt(MC_FIELD_2D, delimiter=",", skiprows=1)
    n = int(d[:, 0].max()) + 1
    ref = np.zeros((n, n))
    ref[d[:, 1].astype(int), d[:, 0].astype(int)] = d[:, 4]
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = (g + 1) / 2, w / 2
    h = 7.0 / n
    c = (np.arange(n)[:, None] + g[None, :]).ravel() * h
    Y, X = np.meshgrid(c, c, indexing="ij")
    pts = np.stack([Y.ravel(), X.ravel()], axis=1)
    return n, ref, ref > 1e-4, pts, w


def field_err_2d(run: Path, mc_ctx):
    csv = run / "final.csv"
    if mc_ctx is None or not csv.exists():
        return None, None
    n, ref, mask, pts, w = mc_ctx
    fd = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=(1, 2, 4))
    xs, ys = np.unique(fd[:, 0]), np.unique(fd[:, 1])
    o = np.lexsort((fd[:, 0], fd[:, 1]))
    Z = (math.sqrt(4 * math.pi) * fd[o, 2]).reshape(len(ys), len(xs))
    f = RegularGridInterpolator((ys, xs), Z, method="linear")
    A = np.einsum("iajb,a,b->ij", f(pts).reshape(n, 4, n, 4), w, w)
    rel = np.abs(A - ref)[mask] / ref[mask]
    return float(np.abs(A - ref).sum() / ref.sum()), float(np.median(rel))


def lattice_devs(s: dict, ref: dict):
    fcs = s.get("lattice_fcs")
    if fcs:
        m = fcs["total_mass_final"]
        a = fcs["total_absorbed"]
        r = fcs["total_absorption_rate_final"]
    else:
        m = s.get("final_density_mass", 0.0)
        a = s.get("ledger", {}).get("absorbed", 0.0)
        r = s.get("absorption_rate_final", 0.0)
    return (100 * (m / ref["mass"] - 1), 100 * (a / ref["absorbed"] - 1), 100 * (r / ref["rate"] - 1))


def lim_pct(s: dict) -> float:
    denom = s.get("cells", 0) * s.get("rate_calls", 0)
    return 100.0 * s.get("limited_element_stages", 0) / denom if denom else 0.0


def load_run(runs: Path, name: str):
    p = runs / name / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    campaign = Path(sys.argv[1]).resolve()
    runs = campaign / "runs"
    mc_ctx = load_mc_field()

    print(f"# Tardis Paper Recompute Report ({campaign.name})\n")

    # Block A: Dynamic adaptive PN
    print("## Block A: Dynamic tile-based adaptive PN (Table 9)\n")
    print("| Run | Total [s] | Solve [s] | Sens [s] | Active | Mean N | dM [%] | dAbs [%] | dR [%] | L1 | Median |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in [
        "dyn_gbq_p21_uni", "dyn_gbq_p21_t16", "dyn_gbq_p21_t8", "dyn_gbq_p21_t8_f01",
        "dyn_gbq_p35_uni", "dyn_gbq_p35_t16", "dyn_gbq_p35_t8", "dyn_gbq_p35_1e4_t8", "dyn_gbq_p35_t8_f01",
        "dyn_gbh_p21_uni", "dyn_gbh_p21_t16", "dyn_gbh_p21_t8", "dyn_gbh_p21_t8_f01",
        "dyn_gbh_p35_uni", "dyn_gbh_p35_t16", "dyn_gbh_p35_t8", "dyn_gbh_p35_1e4_t8", "dyn_gbh_p35_1e4_t16", "dyn_gbh_p35_t8_f01",
        "dyn_gb_p21_uni", "dyn_gb_p21_t8", "dyn_gb_p21_t8_f01",
        "dyn_gb_p35_uni", "dyn_gb_p35_1e4_t16", "dyn_gb_p35_1e4_t8", "dyn_gb_p35_t8_f01",
    ]:
        s = load_run(runs, name)
        if not s:
            continue
        ap = s.get("adaptive_pn", {})
        dm, da, dr = lattice_devs(s, MC_2D)
        l1, med = field_err_2d(runs / name, mc_ctx)
        l1s = f"{l1:.2e}" if l1 is not None else "--"
        meds = f"{100*med:.2f}%" if med is not None else "--"
        print(f"| `{name}` | {s['total_seconds']:.1f} | {s['solver_seconds']:.2f} | {ap.get('sensor_seconds', 0):.2f} | "
              f"{ap.get('mean_coeff_fraction', 1.0):.3f} | {ap.get('mean_order', 0):.1f} | "
              f"{dm:+.3f} | {da:+.3f} | {dr:+.3f} | {l1s} | {meds} |")

    # Block B: Table 8
    print("\n## Block B: Graded mesh GB & broken lines (Table 8)\n")
    print("| Run | Steps | Total [s] | dM [%] | dAbs [%] | dR [%] | L1 | Median |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in [
        "t8_u896_p21", "t8_gb_p21", "t8_gb_broken_p21", "dyn_gb_p21_uni",
        "t8_u896_p35", "t8_gb_p35", "t8_gb_broken_p35", "dyn_gb_p35_uni",
    ]:
        s = load_run(runs, name)
        if not s:
            continue
        dm, da, dr = lattice_devs(s, MC_2D)
        l1, med = field_err_2d(runs / name, mc_ctx)
        l1s = f"{l1:.2e}" if l1 is not None else "--"
        meds = f"{100*med:.2f}%" if med is not None else "--"
        print(f"| `{name}` | {s['steps']} | {s['total_seconds']:.1f} | {dm:+.3f} | {da:+.3f} | {dr:+.3f} | {l1s} | {meds} |")

    # Block D: 3D Lattice
    print("\n## Block D: 3D Lattice A100 (Table 10)\n")
    print("| Run | Steps | Total [s] | Active | dM [%] | dAbs [%] | dR [%] | Limited [%] |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    l3d = sorted(p.name for p in runs.iterdir() if p.name.startswith("l3d_") and (p / "summary.json").exists())
    for name in l3d:
        s = load_run(runs, name)
        if not s:
            continue
        dm, da, dr = lattice_devs(s, MC_3D)
        act = s.get("adaptive_pn", {}).get("mean_coeff_fraction", 1.0) if s.get("adaptive_pn", {}).get("enabled") else 1.0
        print(f"| `{name}` | {s['steps']} | {s['total_seconds']:.1f} | {act:.3f} | {dm:+.2f} | {da:+.2f} | {dr:+.2f} | "
              f"{lim_pct(s):.1f} |")

    # Block F: 2D Uniform Lattice
    print("\n## Block F: 2D Uniform Lattice (Tables 4 & 5)\n")
    print("| Run | Steps | Total [s] | Table [s] | dM [%] | dAbs [%] | dR [%] | Limited [%] |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in [
        "lat_224_p7_nofcs", "lat_448_p7_nofcs", "lat_896_p7_nofcs",
        "lat_448_p15_nofcs", "lat_896_p15_nofcs", "lat_448_p21_nofcs", "lat_896_p21_nofcs",
        "lat_896_p7_fcs", "lat_896_p7_fcs_filt",
        "lat_896_p11_fcs", "lat_896_p11_fcs_filt",
        "lat_896_p21_fcs", "lat_896_p21_fcs_filt",
        "lat_896_p35_fcs", "lat_896_p35_fcs_filt",
        "lat_896_p45_fcs_filt",
        "lat_1344_p21_fcs", "lat_1344_p21_fcs_filt",
    ]:
        s = load_run(runs, name)
        if not s:
            continue
        dm, da, dr = lattice_devs(s, MC_2D)
        ts = s.get("lattice_fcs", {}).get("table_seconds", 0.0)
        print(f"| `{name}` | {s['steps']} | {s['total_seconds']:.1f} | {ts:.1f} | {dm:+.3f} | {da:+.3f} | {dr:+.3f} | {lim_pct(s):.2f} |")

    # Block E & C summary
    print("\n## Blocks C & E: Crossing, Ablation, Line Source (completion & timings)\n")
    print("| Run | Status | Steps | Total [s] | Solve [s] | Limited [%] |")
    print("|---|---|---:|---:|---:|---:|")
    for p in sorted(runs.iterdir()):
        if not (p / "summary.json").exists():
            continue
        if p.name.startswith(("ls_", "cross_", "abl_", "gen_")):
            s = json.load(open(p / "summary.json"))
            print(f"| `{p.name}` | {s.get('status', '?')} | {s.get('steps', 0)} | "
                  f"{s.get('total_seconds', 0):.1f} | {s.get('solver_seconds', 0):.2f} | {lim_pct(s):.2f} |")


if __name__ == "__main__":
    main()
