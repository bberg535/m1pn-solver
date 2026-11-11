#!/usr/bin/env python3
"""Before/after comparison of the boundary fixes. 'legacy' = all fix flags off (bit-identical to commit 8efe517,
checked against the pre-fix results in audit/results), 'new' = production defaults. Writes results/fix/comparison.json."""
import json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
R = here / "results"; F = R / "fix"
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
out = {}
# 1. legacy flags reproduce 8efe517 exactly
for name, old, leg in [("small", R / "small.jsonl", F / "small_legacy.jsonl"), ("crossing64", R / "crossing64_p1_P7.jsonl", F / "crossing64_legacy.jsonl")]:
    a, b = load(old), load(leg)
    diff = sum(1 for x, y in zip(a, b) for k, v in x.items() if k != "wall_s" and isinstance(v, float) and v != y.get(k))
    out[f"legacy_bitwise_identical_to_8efe517_{name}"] = {"records": len(a), "differing_float_fields": diff}
# 2. crossing
keys = ["steps", "min_cone_margin_abs_all_stages", "min_cone_margin_rel_all_stages", "cone_violations_euler_partial", "cone_violations_ssp_stage",
        "silent_nudges_ulp_repairs", "max_ulps_per_nudge", "max_raw_zero_sum_rel", "max_limited_zero_sum_rel", "min_element_factor", "min_boundary_theta",
        "limited_frac_interior_elements", "limited_frac_boundary_elements", "limited_frac_boundary_nodes_theta", "energy_rate_max_rel_stabilization",
        "energy_rate_max_rel_boundary_term", "energy_rate_max_rel_sync_full", "energy_integral_volume", "energy_integral_stabilization", "energy_integral_boundary",
        "energy_integral_sync", "energy_integral_total", "energy_budget_closure_err_rel", "sync_defect_rel_norm_max", "mass_balance_resid_rel_max",
        "global_mass_rate_accepted_max", "global_mass_rate_pn_target_max", "pn_target_vs_accepted_total_mass_rate_abs_dev_max", "pn_target_vs_accepted_boundary_L1_dev_max",
        "accepted_boundary_rho_rate_min", "accepted_boundary_rho_rate_max", "rho_min_final", "rho_max_final", "mass_final", "rel_L2_density_diff_to_full", "wall_s"]
for tag in ("legacy", "new", "sym"):
    p = F / f"crossing64_{tag}.jsonl"
    if p.exists():
        out[f"crossing64_{tag}"] = {r["label"]: {k: r.get(k) for k in keys} for r in load(p)}
# 3. small tests T1/T2/T3/T8
def pick(rows, test, **kw):
    return [r for r in rows if r["test"] == test and all(r.get(k) == v for k, v in kw.items())]
for tag in ("legacy", "new"):
    rows = load(F / f"small_{tag}.jsonl")
    out[f"T1_{tag}"] = [{k: r[k] for k in ("audit_low_order", "max_abs_rate_all_coeffs")} for r in rows if r["test"] == "T1_isotropic_stationary"]
    out[f"T2_{tag}"] = [{k: r.get(k) for k in ("label", "steps_completed", "global_mass_rate_accepted_max", "global_mass_rate_pn_target_max", "accepted_boundary_rho_rate_max",
                                                "cone_violations_euler_partial", "silent_nudges_ulp_repairs", "max_ulps_per_nudge", "min_cone_margin_rel_all_stages",
                                                "energy_rate_max_rel_sync_full", "energy_integral_sync", "energy_integral_total", "mass_balance_resid_rel_max")}
                        for r in rows if r["test"] == "T2_vacuum_single_node"]
    out[f"T3_{tag}"] = [r for r in rows if r["test"].startswith("T3_polynomial_beam_stationary")]
    out[f"T8_{tag}"] = [{k: r.get(k) for k in ("scale", "half_vacuum", "rel_dev_from_scale1_after_3_steps", "gamma_min", "frac_limited_el", "cone_viol", "nudges")} for r in rows if r["test"] == "T8_scale"]
import math
def dens(name):
    q = F / name
    return [float(x) for x in open(q)] if q.exists() else None
for variant in ("full", "no_weno", "low_order_control"):
    ref = dens(f"dens_legacy_{variant}.txt")
    for tag in ("new", "sym"):
        v = dens(f"dens_{tag}_{variant}.txt")
        if ref and v:
            num = math.sqrt(sum((a - b) ** 2 for a, b in zip(ref, v))); den = math.sqrt(sum(a * a for a in ref))
            out.setdefault("crossing64_density_rel_L2_difference_to_legacy", {})[f"{tag}/{variant}"] = num / den
if (F / "c6.jsonl").exists():
    out["C6"] = load(F / "c6.jsonl")
(F / "comparison.json").write_text(json.dumps(out, indent=1))
print("wrote", F / "comparison.json")
