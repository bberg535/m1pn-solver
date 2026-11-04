#!/usr/bin/env python3
"""Rerunnable post-processing for PN mass-correction campaigns.

Incomplete runs are retained in campaign_status.csv and are excluded from
final-state comparisons. Analytic kinetic-density errors and method-A-relative
differences are reported as separate quantities.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, eigvalsh


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        if not fields:
            f.write("status\n")
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def number(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_field(folder: Path, filename: str = "final.csv") -> tuple[np.ndarray, np.ndarray] | None:
    try:
        arr = np.loadtxt(folder / filename, delimiter=",", skiprows=1, ndmin=2)
        if arr.shape[1] < 5:
            return None
        return arr[:, 1:4], arr[:, 4:]
    except (OSError, ValueError):
        return None


def campaign_runs(campaign_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    statuses: list[dict[str, Any]] = []
    runs: dict[str, dict[str, Any]] = {}
    for base in campaign_dirs:
        if not base.exists():
            statuses.append({"campaign": str(base), "run": "", "status": "directory_missing"})
            continue
        children = [p for p in base.iterdir() if p.is_dir() and (p / "config.json").exists()]
        for folder in sorted(children):
            cfg = read_json(folder / "config.json") or {}
            proc = read_json(folder / "process.json") or {}
            summary = read_json(folder / "summary.json") or {}
            progress = read_json(folder / "progress.json") or {}
            planned = number(cfg.get("final_time"))
            reached = number(summary.get("time"), number(progress.get("time"), 0.))
            status = str(summary.get("status", proc.get("status", progress.get("status", "unknown"))))
            complete = status == "completed" and (folder / "final.csv").exists()
            if not complete and status == "unknown":
                status = "incomplete" if progress else "missing_status"
            record = {
                "campaign": base.name,
                "run": folder.name,
                "variant": proc.get("variant", folder.name.rsplit("_", 1)[-1]),
                "status": "completed" if complete else status,
                "complete": complete,
                "time_reached": reached,
                "planned_time": planned,
                "steps": summary.get("steps", progress.get("steps")),
                "folder": str(folder),
                "rejected_attempts": summary.get("rejected_attempts"),
                "solver_seconds": summary.get("solver_seconds"),
            }
            statuses.append(record)
            key = folder.name
            runs[key] = {"folder": folder, "config": cfg, "process": proc, "summary": summary,
                         "progress": progress, "status": record}
    return statuses, runs


def crossing_analysis(runs: dict[str, dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for variant in ["A", "Aex", "B", "C"]:
        name = f"crossing64_P7_{variant}"
        run = runs.get(name)
        if run and run["status"]["complete"]:
            selected[variant] = run
    if not selected:
        return []

    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for variant, run in selected.items():
        data = load_field(run["folder"])
        try:
            mass = np.loadtxt(run["folder"] / "nodal_mass.csv", delimiter=",", skiprows=1, usecols=1)
        except (OSError, ValueError):
            continue
        if data is None or len(mass) != data[0].shape[0]:
            continue
        x, u = data
        cfg = run["config"]
        t = number(run["summary"].get("time"), number(cfg.get("final_time"), .32))
        geometry = cfg.get("beam_geometry", {})
        bg = number(geometry.get("background"), .01)
        beams = geometry.get("beams", [{"center": [.3 + t, .5, 0.]}, {"center": [.5, .3 + t, 0.]}])
        width_exponent = number(geometry.get("width_exponent"), -50.)
        exact = np.full(len(x), bg)
        for beam in beams:
            center = np.asarray(beam.get("center", [0., 0., 0.]), dtype=float)
            direction = np.asarray(beam.get("direction", [0., 0., 0.]), dtype=float)
            # The crossing-transport manufactured kinetic density translates
            # each beam along its configured direction at unit speed.
            center = center + t * direction
            exact += np.exp(width_exponent * np.sum((x - center[None, :]) ** 2, axis=1))
        rho = u[:, 0] * math.sqrt(4 * math.pi)
        arrays[variant] = (x, rho, exact, mass)
        analytic_error = rho - exact
        denom = float(mass @ np.abs(exact))
        region_masks = {
            "all": np.ones(len(x), dtype=bool),
            "boundary": np.any((x[:, :2] < 1e-12) | (x[:, :2] > 1 - 1e-12), axis=1),
            "boundary_zone": np.any((x[:, :2] < .1) | (x[:, :2] > .9), axis=1),
            "crossing_zone": np.linalg.norm(x[:, :2] - .5, axis=1) < .2,
        }
        row: dict[str, Any] = {
            "case": "crossing64_P7", "variant": variant, "time": t,
            "analytic_reference": "analytic kinetic density (includes PN model error)",
            "analytic_L1_numerator": float(mass @ np.abs(analytic_error)),
            "analytic_L1_denominator": denom,
            "analytic_relative_L1": float(mass @ np.abs(analytic_error)) / denom if denom else math.nan,
            "analytic_L2": float(np.sqrt(mass @ (analytic_error ** 2))),
            "analytic_Linf": float(np.max(np.abs(analytic_error))),
        }
        for label, mask in region_masks.items():
            row[f"analytic_Linf_{label}"] = float(np.max(np.abs(analytic_error[mask]))) if mask.any() else math.nan
        rows.append(row)

    if "A" in arrays:
        xa, rho_a, _, wa = arrays["A"]
        for row in rows:
            variant = row["variant"]
            if variant == "A" or variant not in arrays:
                continue
            x, rho, _, mass = arrays[variant]
            if x.shape == xa.shape and np.allclose(x, xa, rtol=0, atol=1e-14):
                delta = rho - rho_a
                den = float(mass @ np.abs(rho_a))
                row["method_A_relative_L1_numerator"] = float(mass @ np.abs(delta))
                row["method_A_relative_L1_denominator"] = den
                row["method_A_relative_L1"] = row["method_A_relative_L1_numerator"] / den if den else math.nan
                row["method_A_relative_L2"] = float(np.sqrt(mass @ (delta ** 2)))
                row["method_A_relative_Linf"] = float(np.max(np.abs(delta)))
                for region, mask in {
                    "boundary": np.any((x[:, :2] < 1e-12) | (x[:, :2] > 1 - 1e-12), axis=1),
                    "boundary_zone": np.any((x[:, :2] < .1) | (x[:, :2] > .9), axis=1),
                    "crossing_zone": np.linalg.norm(x[:, :2] - .5, axis=1) < .2,
                }.items():
                    row[f"method_A_relative_Linf_{region}"] = float(np.max(np.abs(delta[mask]))) if mask.any() else math.nan

    # Concentrated exportable maps: method density, analytic error, and
    # method-minus-A, with a horizontal y=0.5 lineout table.
    map_dir = out / "crossing_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    xref, _, exact, _ = next(iter(arrays.values()))
    for variant, (x, rho, ex, _) in arrays.items():
        fields = [(rho, f"{variant} density"), (ex, "analytic kinetic rho"),
                  (rho - ex, f"{variant} minus analytic kinetic rho")]
        if "A" in arrays:
            fields.append((rho - arrays["A"][1], f"{variant} minus A"))
        fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 3.6), constrained_layout=True, squeeze=False)
        for ax, (value, title) in zip(axes[0], fields):
            im = ax.tricontourf(x[:, 0], x[:, 1], value, levels=30, cmap="coolwarm" if "minus" in title else "viridis")
            ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=.8)
        fig.savefig(map_dir / f"crossing_{variant}.png", dpi=160); fig.savefig(map_dir / f"crossing_{variant}.pdf"); plt.close(fig)
    if "A" in arrays:
        line_rows = []
        x0 = arrays["A"][0]
        yband = np.abs(x0[:, 1] - .5)
        ysel = yband <= max(1e-12, np.min(yband) + 1e-12)
        ix = np.flatnonzero(ysel)[np.argsort(x0[ysel, 0])]
        for i in ix:
            row = {"x": float(x0[i, 0]), "y": float(x0[i, 1]), "analytic_rho": float(arrays["A"][2][i])}
            for variant, data in arrays.items():
                row[f"rho_{variant}"] = float(data[1][i])
                row[f"error_{variant}"] = float(data[1][i] - data[2][i])
                if "A" in arrays:
                    row[f"difference_{variant}_A"] = float(data[1][i] - arrays["A"][1][i])
            line_rows.append(row)
        write_csv(map_dir / "lineout_y_mid.csv", line_rows)
    return rows


def study_csvs(study_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    convergence: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    # Later final data supersedes standard data, but the latter remains useful
    # while final runs are still being collected.
    files: dict[tuple[str, str], Path] = {}
    for base in study_dirs:
        if not base.exists():
            continue
        for suite in ["convergence", "patches", "spectrum"]:
            for path in base.glob(f"{suite}_*.csv"):
                variant = path.stem[len(suite) + 1:]
                key = (suite, variant)
                rank = 2 if "final" in base.name else 1
                previous = files.get(key)
                old_rank = 2 if previous and "final" in previous.parent.name else 1
                if previous is None or rank >= old_rank:
                    files[key] = path
    for (suite, variant), path in files.items():
        rows = read_csv(path)
        dest = {"convergence": convergence, "patches": patches, "spectrum": spectra}[suite]
        for r in rows:
            item: dict[str, Any] = {"source": str(path), **r}
            for k, v in list(item.items()):
                if k in {"source"}:
                    continue
                try:
                    item[k] = float(v) if any(c in v for c in ".eE") else int(v)
                except (ValueError, TypeError):
                    pass
            dest.append(item)
    # Add observed spatial convergence rates independently within each mode.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in convergence:
        key = (str(r.get("variant")), str(r.get("boundary")), str(r.get("suite")))
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        group.sort(key=lambda r: number(r.get("cells")))
        for j, row in enumerate(group):
            for col in ["pn_l1", "pn_l2", "pn_linf", "density_l1", "density_l2", "density_linf"]:
                rate = math.nan
                if j and number(group[j-1].get(col)) > 0 and number(row.get(col)) > 0:
                    rate = math.log(number(group[j-1][col]) / number(row[col])) / math.log(number(row["cells"]) / number(group[j-1]["cells"]))
                row[f"rate_{col}"] = rate
    return convergence, patches, spectra


def spatial_mass(nodes: int, cells: int, periodic: bool) -> np.ndarray:
    m = np.zeros((nodes, nodes))
    h = 1. / cells
    for e in range(cells):
        i, j = e, (e + 1) % nodes
        m[i, i] += h / 3; m[j, j] += h / 3
        m[i, j] += h / 6; m[j, i] += h / 6
    return m


def spectrum_summary(spectra: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for r in spectra:
        key = (str(r.get("variant")), str(r.get("boundary")), int(r.get("order", 1)), int(r.get("cells", 4)))
        groups.setdefault(key, []).append(r)
    result = []
    eigen_rows: list[dict[str, Any]] = []
    for (variant, boundary, order, cells), rows in groups.items():
        n = max(max(int(r["row"]), int(r["column"])) for r in rows) + 1
        G = np.zeros((n, n))
        for r in rows:
            G[int(r["row"]), int(r["column"])] = float(r["value"])
        periodic = boundary == "periodic"
        nodes = n // ((order + 1) ** 2)
        if nodes * ((order + 1) ** 2) != n:
            continue
        Mc = spatial_mass(nodes, cells, periodic)
        M = Mc.copy()
        # Current study variants A and PN use the consistent spatial mass.
        # B/C are lumped-mass methods; the same block lumped metric is used
        # for their energy and symmetric-part diagnostics.
        mass_mode = "lumped" if variant in {"B", "C"} else "consistent"
        if mass_mode == "lumped":
            M = np.diag(M.sum(axis=1))
        W = np.kron(M, np.eye((order + 1) ** 2))
        lam = eigvals(G)
        H = .5 * (W @ G + G.T @ W)
        sym = eigvalsh(H, W)
        pn_rows=groups.get(("PN",boundary,order,cells),[])
        stabilization={}
        if pn_rows:
            Gpn=np.zeros((n,n))
            for rr in pn_rows:Gpn[int(rr["row"]),int(rr["column"])]=float(rr["value"])
            q=(order+1)**2
            Z=W@G-np.kron(Mc,np.eye(q))@Gpn
            zs=eigvalsh(.5*(Z+Z.T))
            conservation=np.kron(np.ones((1,nodes)),np.eye(q))@Z
            stabilization=dict(stabilization_symmetric_max=float(zs[-1]),stabilization_symmetric_min=float(zs[0]),stabilization_positive_directions=int(np.sum(zs>1e-10)),stabilization_global_conservation_inf=float(np.max(abs(conservation))))
            for index,value in enumerate(zs):
                eigen_rows.append(dict(variant=variant,boundary=boundary,order=order,cells=cells,mass_metric="Euclidean coefficient work a^T Z",spectrum="stabilization_symmetric_part",index=index,real=float(value),imag=0.,magnitude=float(abs(value))))
        for index, value in enumerate(lam):
            eigen_rows.append({"variant": variant, "boundary": boundary, "order": order,
                               "cells": cells, "mass_metric": mass_mode, "spectrum": "generator",
                               "index": index, "real": float(value.real), "imag": float(value.imag),
                               "magnitude": float(abs(value))})
        for index, value in enumerate(sym):
            eigen_rows.append({"variant": variant, "boundary": boundary, "order": order,
                               "cells": cells, "mass_metric": mass_mode, "spectrum": "weighted_symmetric_part",
                               "index": index, "real": float(value), "imag": 0., "magnitude": float(abs(value))})
        result.append({
            "variant": variant, "boundary": boundary, "order": order, "cells": cells,
            "mass_metric": mass_mode, "generator_dimension": n, **stabilization,
            "max_real_eigenvalue": float(np.max(lam.real)),
            "spectral_radius": float(np.max(np.abs(lam))),
            "max_weighted_symmetric_rate": float(sym[-1]),
            "min_weighted_symmetric_rate": float(sym[0]),
            "weighted_symmetric_positive_directions": int(np.sum(sym > 1e-10)),
            "max_abs_eigenvalue_imag": float(np.max(np.abs(lam.imag))),
        })
    return result, eigen_rows


def plots(convergence: list[dict[str, Any]], patches: list[dict[str, Any]], out: Path) -> None:
    if convergence:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for variant in sorted({str(r.get("variant")) for r in convergence}):
            for boundary, ax in [("periodic", axes[0]), ("open", axes[1])]:
                rr = sorted([r for r in convergence if str(r.get("variant")) == variant and str(r.get("boundary")) == boundary], key=lambda r: number(r.get("cells")))
                if rr:
                    ax.loglog([r["cells"] for r in rr], [max(number(r.get("pn_l2")), 1e-18) for r in rr], "o-", label=variant)
                    ax.loglog([r["cells"] for r in rr], [max(number(r.get("density_l2")), 1e-18) for r in rr], "--", label=f"{variant} density")
        for ax, boundary in zip(axes, ["periodic", "open"]):
            ax.set_title(boundary); ax.set_xlabel("cells"); ax.set_ylabel("PN L2 error"); ax.grid(True, which="both"); ax.legend()
        fig.savefig(out / "convergence.png", dpi=160); fig.savefig(out / "convergence.pdf"); plt.close(fig)
    if patches:
        selected = [r for r in patches if r.get("patch") in {"affine", "smooth"}]
        if selected:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for ax, kind in zip(axes, ["affine", "smooth"]):
                for variant in sorted({str(r.get("variant")) for r in selected}):
                    for codim in sorted({r.get("codimension") for r in selected if r.get("patch") == kind}):
                        rr = sorted([r for r in selected if r.get("patch") == kind and str(r.get("variant")) == variant and r.get("codimension") == codim], key=lambda r: number(r.get("cells")))
                        if rr:
                            ax.loglog([r["cells"] for r in rr], [max(number(r.get("max_rate_error")), 1e-18) for r in rr], "o-", label=f"{variant}, codim {codim}")
                ax.set(title=f"{kind} derivative error by stratum", xlabel="cells", ylabel="max rate error"); ax.grid(True, which="both"); ax.legend(fontsize=7)
            fig.savefig(out / "boundary_error_strata.png", dpi=160); fig.savefig(out / "boundary_error_strata.pdf"); plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=Path, default=root)
    ap.add_argument("--campaign-dir", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path, default=root / "results" / "comparison")
    a = ap.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    study_dirs = [a.study_dir / "tmp" / "study_final", a.study_dir / "tmp" / "study_standard", a.study_dir / "results" / "study_final", a.study_dir / "results" / "study_standard"]
    campaign_dirs = a.campaign_dir or [a.study_dir / "tmp" / x for x in ["main_biharmonic", "main_B", "main_C", "performance", "boundary", "cfl", "regression_v1"]]
    statuses, runs = campaign_runs(campaign_dirs)
    conv, patch_rows, spectra = study_csvs(study_dirs)
    cross = crossing_analysis(runs, out)
    spec, eigen_rows = spectrum_summary(spectra)
    campaign_summary: list[dict[str, Any]] = []
    for name, run in runs.items():
        config, process, summary = run["config"], run["process"], run["summary"]
        row: dict[str, Any] = {"run": name, "campaign": run["folder"].parent.name,
                               "status": run["status"]["status"], "complete": run["status"]["complete"],
                               "variant": process.get("variant", name.rsplit("_", 1)[-1]),
                               "case": config.get("case"), "dimension": config.get("dimension"),
                               "cells": config.get("elements_per_axis"), "pn_order": config.get("pn_order"),
                               "final_time": summary.get("time", config.get("final_time")),
                               "steps": summary.get("steps"), "solver_seconds": summary.get("solver_seconds"),
                               "total_seconds": summary.get("total_seconds"),
                               "mass_seconds": summary.get("mass_seconds"),
                               "pn_predictor_seconds": summary.get("pn_predictor_seconds"),
                               "pn_mass_correction_seconds": summary.get("pn_mass_correction_seconds"),
                               "biharmonic_seconds": summary.get("biharmonic_seconds"),
                               "boundary_seconds": summary.get("boundary_seconds"),
                               "mcl_seconds": summary.get("mcl_seconds"),
                               "seconds_per_SSP_stage": process.get("seconds_per_SSP_stage"),
                               "gamma": config.get("gamma"), "stabilization": config.get("stabilization"),
                               "mass_mode": config.get("mass_mode"), "pn_boundary_quadrature": config.get("pn_boundary_quadrature"),
                               "pn_norm_max_dofs": config.get("pn_norm_max_dofs"),
                               "rejected_attempts": summary.get("rejected_attempts"),
                               "minimum_cone_margin": summary.get("minimum_cone_margin"),
                               "maximum_step_balance": summary.get("maximum_step_balance"),
                               "maximum_mass_residual": summary.get("maximum_mass_residual"),
                               "active_pair_fraction": summary.get("active_pair_fraction"),
                               "active_pair_evaluations": summary.get("active_pair_evaluations"),
                               "active_face_evaluations": summary.get("active_face_evaluations"),
                               "peak_rss_kib_rank0": summary.get("peak_rss_kib_rank0"),
                               "folder": str(run["folder"])}
        for key in ["rho_L1", "rho_L2", "rho_Linf", "relative_L1", "relative_L1_denominator",
                    "boundary_Linf", "boundary_zone_Linf", "crossing_zone_Linf", "min_density",
                    "negative_nodes", "cone_margin_min", "high_moment_L2", "high_moment_Linf"]:
            if key in process:
                row[key] = process[key]
        campaign_summary.append(row)
    write_csv(out / "campaign_status.csv", statuses)
    write_csv(out / "campaign_summary.csv", campaign_summary)
    write_csv(out / "crossing_errors.csv", cross)
    write_csv(out / "study_convergence.csv", conv)
    write_csv(out / "study_patches.csv", patch_rows)
    write_csv(out / "study_spectrum_summary.csv", spec)
    write_csv(out / "study_spectrum_eigenvalues.csv", eigen_rows)
    # Retain legacy regression equivalence results verbatim when available.
    reg = next((d / "regression.json" for d in campaign_dirs if (d / "regression.json").exists()), None)
    if reg:
        (out / "regression.json").write_text(reg.read_text())
    plots(conv, patch_rows, out)
    metadata = {
        "study_sources": [str(p) for p in study_dirs if p.exists()],
        "campaign_sources": [str(p) for p in campaign_dirs],
        "campaign_runs_seen": len(statuses),
        "campaign_runs_complete": sum(bool(r.get("complete")) for r in statuses),
        "campaign_runs_incomplete_or_failed": sum(not bool(r.get("complete")) for r in statuses),
        "crossing_final_variants": sorted({r["variant"] for r in cross}),
        "analytic_density_reference_note": "Analytic kinetic density includes finite-PN model error; method-A-relative differences are separately normalized.",
        "spectrum_note": "Generalized symmetric-part eigenvalues use consistent FE block mass for A/PN and lumped row-sum block mass for B/C.",
    }
    (out / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
