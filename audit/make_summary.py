#!/usr/bin/env python3
"""Collect machine-readable provenance and the headline numbers of the audit into results/summary.json.
Every number is read from the JSONL files written by the audit binaries; nothing is typed in by hand."""
import json, subprocess, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
res = here / "results"
def rows(name):
    return [json.loads(l) for l in open(res / name) if l.strip()]
def pick(rs, **kw):
    out = [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
    return out
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=here).stdout.strip()
ops = rows("ops.jsonl")
small = rows("small.jsonl")
cross = rows("crossing64_p1_P7.jsonl")
t6 = rows("T6_pn3.jsonl")
t7 = rows("T7.jsonl")
er = rows("energy_random.jsonl")
probe = rows("probe.jsonl")
S = {}
S["provenance"] = {
    "code_commit": git("rev-parse", "HEAD"), "branch_of_audit_worktree": git("rev-parse", "--abbrev-ref", "HEAD"),
    "code_branch": "experiment/element-weno", "production_files_modified": False,
    "compiler": "g++ (GCC) 13.2.0 via OpenMPI 5.0.10 mpic++, -O3 -DNDEBUG -std=gnu++17 -march=x86-64-v3 -ffp-contract=fast -fopenmp, OMP_NUM_THREADS=1",
    "instrumentation": "audit/make_audit_source.py patches a COPY of src/element_weno.cpp (diagnostic hooks only); src/ untouched",
}
A = [r for r in ops if r["test"] == "A_boundary_operator"]
S["A_boundary_operator"] = {
    "mode_counts_pos_neg_zero": {f"P{r['pn_order']}": [r["n_pos"], r["n_neg"], r["n_zero"]] for r in A if r["normal"] == "ex"},
    "max_rel_Hp_minus_Aplus": {f"P{n}": max(r["rel_Hp_minus_Aplus"] for r in A if r["pn_order"] == n) for n in (3, 5, 7)},
    "kinetic_outflow_energy_of_null_modes_ZtHpZ_range": {f"P{n}": [min(r["Zt_Hp_Z_min"] for r in A if r["pn_order"] == n), max(r["Zt_Hp_Z_max"] for r in A if r["pn_order"] == n)] for n in (3, 5, 7)},
    "lambda_max_Q_implemented_LLF_exterior_max": max(r["lambda_max_Q_implemented"] for r in A),
    "lambda_max_Q_characteristic_C1_flux_Boundary_flux": {f"P{n}": max(r["lambda_max_Q_characteristic"] for r in A if r["pn_order"] == n) for n in (3, 5, 7)},
    "rel_diff_implemented_flux_vs_characteristic_flux": {f"P{n}": max(r["Feff_minus_char_flux_rel"] for r in A if r["pn_order"] == n) for n in (3, 5, 7)},
    "null_mode_boundary_energy_abs_max": max(r["null_energy_ZtQZ_abs"] for r in A),
    "discrete_face_energy_random_max_rate_over_trace_L2": max(r["max_rate_over_trace_L2"] for r in er),
}
B = [r for r in ops if r["test"] == "B_stabilization_matrix"]
S["B_stabilization"] = {
    "max_abs_row_sum_S_gamma_rel": max(max(r["S1_row_sum_max"], r["Sgamma_random_row_sum_max"]) for r in B),
    "S_gamma1_sym_defect_rel_max": max(r["S1_sym_defect_rel"] for r in B),
    "S_gamma_random_sym_defect_rel_range": [min(r["Srandom_sym_defect_rel"] for r in B), max(r["Srandom_sym_defect_rel"] for r in B)],
    "binary_pattern_sym_defect_rel_max": max(r["binary_pattern_worst_sym_defect_rel"] for r in B),
    "min_lambda_sym_part_over_all_gamma_vertices_rel": min(r["binary_pattern_worst_lambda_min_rel"] for r in B),
    "poly_reproduction_err_max": max(max(r["poly_reproduction_err_gamma1"], r["poly_reproduction_err_weno"]) for r in B),
    "effective_min_rayleigh_over_hillclimb": min(r["min_Rayleigh_uSu_over_uS0u"] for r in ops if r["test"] == "B_stabilization_effective_energy"),
    "weno_sensor_scale_dependence_gamma_min": {str(r["scale"]): r["gamma_min"] for r in ops if r["test"] == "T8_weno_sensor_scale"},
}
cf = {r["label"]: r for r in cross}
S["crossing_64x64_P7_p1_T0.32_cfl0.5"] = {k: {kk: v.get(kk) for kk in [
    "steps", "dt", "cfl_fraction_of_step_bound", "min_cone_margin_abs_all_stages", "min_cone_margin_rel_all_stages",
    "cone_violations_euler_partial", "cone_violations_ssp_stage", "silent_nudges_ulp_repairs", "max_raw_zero_sum_rel", "max_limited_zero_sum_rel",
    "min_element_factor", "min_boundary_theta", "limited_frac_boundary_elements", "limited_frac_interior_elements", "limited_frac_boundary_nodes_theta",
    "gamma_min_boundary_elems", "gamma_min_interior_elems", "frac_gamma_lt_0p99_boundary", "frac_gamma_lt_0p99_interior",
    "energy_rate_max_rel_volume", "energy_rate_max_rel_stabilization", "energy_rate_max_rel_boundary_term", "energy_rate_max_rel_sync_full",
    "energy_rate_max_rel_sync_element_limiter_only", "energy_rate_max_rel_sync_boundary_limiter_only", "energy_rate_max_rel_total",
    "energy_integral_volume", "energy_integral_stabilization", "energy_integral_boundary", "energy_integral_sync", "energy_integral_total",
    "energy_budget_closure_err_rel", "sync_defect_rel_norm_max", "mass_balance_resid_rel_max", "global_mass_rate_accepted_max",
    "global_mass_rate_pn_target_max", "pn_target_vs_accepted_total_mass_rate_abs_dev_max", "pn_target_vs_accepted_boundary_L1_dev_max",
    "accepted_boundary_rho_rate_min", "accepted_boundary_rho_rate_max", "rel_L2_density_diff_to_full", "rho_min_final", "rho_max_final", "mass_final", "wall_s"]}
    for k, v in cf.items()}
S["T1_isotropic"] = [r for r in small if r["test"] == "T1_isotropic_stationary"]
S["T2_vacuum"] = [{k: r.get(k) for k in ["label", "steps_completed", "min_cone_margin_rel_all_stages", "cone_violations_euler_partial", "silent_nudges_ulp_repairs",
                                          "global_mass_rate_accepted_max", "global_mass_rate_pn_target_max", "accepted_boundary_rho_rate_min", "accepted_boundary_rho_rate_max",
                                          "energy_rate_max_rel_sync_full", "energy_integral_sync", "energy_integral_total", "mass_initial", "mass_final"]}
                  for r in small if r["test"] == "T2_vacuum_single_node"]
S["T3"] = [r for r in small if r["test"].startswith("T3_polynomial_beam_stationary")]
S["T6_convergence_pn3_p2"] = [{k: r.get(k) for k in ["variant", "elements", "dt", "rms_err_density_boundary_layer", "rms_err_density_interior", "order_boundary", "order_interior"]} for r in t6]
S["T7"] = [{k: r.get(k) for k in ["label", "rel_L2_error_vs_E48", "rel_L2_error_vs_E48_x_gt_0p75", "min_element_factor", "min_boundary_theta",
                                   "gamma_min_boundary_elems", "gamma_min_interior_elems", "limited_frac_boundary_elements", "limited_frac_interior_elements"]} for r in t7]
S["T8"] = [r for r in small if r["test"] == "T8_scale"]
S["boundary_compat_probe_failures"] = [{k: r[k] for k in ["pn_order", "normal", "psi", "ext_minus_u_norm", "u_norm"]} for r in probe if r["ext_minus_u_norm"] > 1e-9 * max(1, r["u_norm"])]
(res / "summary.json").write_text(json.dumps(S, indent=1))
print("wrote", res / "summary.json")
