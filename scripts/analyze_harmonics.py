#!/usr/bin/env python3
"""Degree-resolved PN diagnostics against projected analytic kinetic data.

The reference is the analytic translating kinetic density projected into the
stored PN basis. It therefore includes finite-PN model error and is not an
exact solution of the discrete PN equations. No method-to-method difference is
used in these diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_NOTE = "Projected analytic kinetic reference; includes PN model error and is not the exact PN PDE solution."


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if raw.shape[1] < 6:
        raise ValueError(f"expected coordinates plus PN coefficients in {path}")
    return raw[:, 1:4], raw[:, 4:]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def beam_amplitudes(x: np.ndarray, beams: list[dict[str, Any]], exponent: float,
                    time: float = 0.) -> np.ndarray:
    columns = []
    for beam in beams:
        center = np.asarray(beam["center"], dtype=float)
        if time:
            center = center + time * np.asarray(beam.get("direction", [0., 0., 0.]), dtype=float)
        columns.append(np.exp(exponent * np.sum((x - center[None, :]) ** 2, axis=1)))
    return np.column_stack(columns)


def weighted_norm(values: np.ndarray, mass: np.ndarray) -> float:
    return float(np.sqrt(np.sum(mass[:, None] * values * values)))


def analyze_one(folder: Path, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = read_json(folder / "config.json")
    summary = read_json(folder / "summary.json")
    if summary.get("status") != "completed" or not (folder / "final.csv").exists():
        raise RuntimeError("run is incomplete")
    x0, u0 = load_state(folder / "initial.csv")
    xf, actual = load_state(folder / "final.csv")
    if x0.shape != xf.shape or not np.allclose(x0, xf, rtol=0., atol=1e-14):
        raise ValueError("initial/final coordinate arrays differ")
    mass = np.loadtxt(folder / "nodal_mass.csv", delimiter=",", skiprows=1, usecols=1, ndmin=1)
    if len(mass) != len(x0) or not np.all(mass > 0):
        raise ValueError("invalid nodal mass data")
    order = int(cfg["pn_order"])
    q = (order + 1) ** 2
    if u0.shape[1] != q or actual.shape[1] != q:
        raise ValueError(f"PN coefficient count mismatch: expected {q}")

    geometry = cfg["beam_geometry"]
    bg = float(geometry["background"])
    beams = geometry["beams"]
    exponent = float(geometry["width_exponent"])
    e0 = np.zeros(q)
    e0[0] = 1. / math.sqrt(4 * math.pi)
    D0 = beam_amplitudes(x0, beams, exponent)
    perturbation = u0 - bg * e0[None, :]
    Y, _, rank, singular_values = np.linalg.lstsq(D0, perturbation, rcond=None)
    reconstruction = bg * e0[None, :] + D0 @ Y
    residual = u0 - reconstruction
    residual_max = float(np.max(np.abs(residual)))
    residual_l2 = weighted_norm(residual, mass)
    if residual_max >= 1e-12:
        raise AssertionError(f"initial harmonic-vector inference residual {residual_max:.4e} >= 1e-12")

    final_time = float(summary["time"])
    Dfinal = beam_amplitudes(xf, beams, exponent, final_time)
    reference = bg * e0[None, :] + Dfinal @ Y
    difference = actual - reference

    rows: list[dict[str, Any]] = []
    for ell in range(order + 1):
        lo, hi = ell * ell, (ell + 1) * (ell + 1)
        a = actual[:, lo:hi]
        r = reference[:, lo:hi]
        d = difference[:, lo:hi]
        actual_l2 = weighted_norm(a, mass)
        ref_l2 = weighted_norm(r, mass)
        error_l2 = weighted_norm(d, mass)
        rows.append({
            "variant": variant, "degree": ell, "mode_count": hi - lo,
            "actual_modal_weighted_L2": actual_l2,
            "reference_modal_weighted_L2": ref_l2,
            "absolute_modal_error_weighted_L2": error_l2,
            "relative_error_denominator": ref_l2,
            "relative_error_denominator_nonzero": ref_l2 > 0,
            "relative_modal_error": error_l2 / ref_l2 if ref_l2 > 0 else math.nan,
            "modal_error_Linf_node_Euclidean": float(np.max(np.linalg.norm(d, axis=1))),
            "modal_coefficient_error_Linf": float(np.max(np.abs(d))),
            "reference_type": REFERENCE_NOTE,
        })

    actual_l2 = weighted_norm(actual, mass)
    ref_l2 = weighted_norm(reference, mass)
    error_l2 = weighted_norm(difference, mass)
    rows.append({
        "variant": variant, "degree": "all", "mode_count": q,
        "actual_modal_weighted_L2": actual_l2,
        "reference_modal_weighted_L2": ref_l2,
        "absolute_modal_error_weighted_L2": error_l2,
        "relative_error_denominator": ref_l2,
        "relative_error_denominator_nonzero": ref_l2 > 0,
        "relative_modal_error": error_l2 / ref_l2 if ref_l2 > 0 else math.nan,
        "modal_error_Linf_node_Euclidean": float(np.max(np.linalg.norm(difference, axis=1))),
        "modal_coefficient_error_Linf": float(np.max(np.abs(difference))),
        "reference_type": REFERENCE_NOTE,
    })

    provenance = {
        "variant": variant,
        "run_folder": str(folder.resolve()),
        "config_sha256": digest(folder / "config.json"),
        "initial_sha256": digest(folder / "initial.csv"),
        "final_sha256": digest(folder / "final.csv"),
        "nodal_mass_sha256": digest(folder / "nodal_mass.csv"),
        "pn_order": order,
        "pn_coefficient_count": q,
        "nodes": int(len(x0)),
        "final_time": final_time,
        "beam_background": bg,
        "beam_width_exponent": exponent,
        "beam_centers_at_t0": [b["center"] for b in beams],
        "beam_directions": [b.get("direction", [0., 0., 0.]) for b in beams],
        "least_squares_rank": int(rank),
        "least_squares_singular_values": [float(v) for v in singular_values],
        "inferred_harmonic_vectors": Y.tolist(),
        "initial_reconstruction_residual_max_abs": residual_max,
        "initial_reconstruction_residual_weighted_L2": residual_l2,
        "reference_type": REFERENCE_NOTE,
        "method_relative_comparisons_computed": False,
    }
    return rows, provenance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["variant", "degree", "mode_count", "actual_modal_weighted_L2",
              "reference_modal_weighted_L2", "absolute_modal_error_weighted_L2",
              "relative_error_denominator", "relative_error_denominator_nonzero",
              "relative_modal_error", "modal_error_Linf_node_Euclidean",
              "modal_coefficient_error_Linf", "reference_type"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def plot_spectra(rows: list[dict[str, Any]], out: Path) -> None:
    modal = [r for r in rows if r["degree"] != "all"]
    if not modal:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for variant in sorted({r["variant"] for r in modal}):
        rr = sorted([r for r in modal if r["variant"] == variant], key=lambda r: r["degree"])
        degree = [r["degree"] for r in rr]
        for ax, key, label in [
            (axes[0], "actual_modal_weighted_L2", "actual"),
            (axes[0], "reference_modal_weighted_L2", "reference"),
            (axes[1], "absolute_modal_error_weighted_L2", "absolute error"),
            (axes[2], "relative_modal_error", "relative error"),
        ]:
            ax.semilogy(degree, [max(float(r[key]), 1e-30) for r in rr], "o-", label=f"{variant} {label}")
        ell7 = next((r for r in rr if r["degree"] == 7), None)
        if ell7 is not None:
            axes[1].annotate(f"{variant} ell=7", (7, max(ell7["absolute_modal_error_weighted_L2"], 1e-30)),
                             xytext=(-42, 9), textcoords="offset points", fontsize=7)
    axes[0].set_title("Modal amplitude")
    axes[1].set_title("Absolute modal error")
    axes[2].set_title("Relative modal error")
    for ax in axes:
        ax.set_xlabel("harmonic degree ell"); ax.set_ylabel("weighted L2 coefficient norm")
        ax.grid(True, which="both"); ax.legend(fontsize=7)
    fig.savefig(out / "degree_spectrum.png", dpi=180)
    fig.savefig(out / "degree_spectrum.pdf")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=Path, default=root)
    ap.add_argument("--output", type=Path, default=root / "results" / "harmonics")
    a = ap.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    root_dir = a.study_dir.resolve()
    candidates = {
        "A": root_dir / "tmp" / "main_biharmonic" / "crossing64_P7_A",
        "Aex": root_dir / "tmp" / "main_biharmonic" / "crossing64_P7_Aex",
        "B": root_dir / "tmp" / "main_B" / "crossing64_P7_B",
        "C": root_dir / "tmp" / "main_C" / "crossing64_P7_C",
    }
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"reference_type": REFERENCE_NOTE, "runs": {}, "skipped": {}}
    for variant, folder in candidates.items():
        summary = None
        try:
            summary = read_json(folder / "summary.json")
        except (OSError, json.JSONDecodeError):
            pass
        if not summary or summary.get("status") != "completed" or not (folder / "final.csv").exists():
            provenance["skipped"][variant] = {
                "run_folder": str(folder),
                "status": summary.get("status") if summary else "not_completed_or_missing",
            }
            continue
        result, run_info = analyze_one(folder, variant)
        rows.extend(result)
        provenance["runs"][variant] = run_info
    write_csv(out / "degree_errors.csv", rows)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_spectra(rows, out)
    print(json.dumps({"output": str(out), "completed_variants": sorted(provenance["runs"]),
                      "skipped_variants": provenance["skipped"],
                      "ell7_rows": [r for r in rows if r["degree"] == 7]}, indent=2))


if __name__ == "__main__":
    main()
