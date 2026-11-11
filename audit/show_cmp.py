import json, sys
c = json.load(open('audit/results/fix/comparison.json'))
print(c['legacy_bitwise_identical_to_8efe517_small'], c['legacy_bitwise_identical_to_8efe517_crossing64'])
show = ('min_cone_margin_rel_all_stages', 'cone_violations_euler_partial', 'cone_violations_ssp_stage', 'silent_nudges_ulp_repairs', 'min_boundary_theta',
        'limited_frac_boundary_elements', 'limited_frac_interior_elements', 'energy_rate_max_rel_sync_full', 'energy_integral_total', 'sync_defect_rel_norm_max',
        'mass_balance_resid_rel_max', 'global_mass_rate_accepted_max', 'accepted_boundary_rho_rate_max', 'rho_min_final', 'mass_final', 'rel_L2_density_diff_to_full', 'wall_s')
for tag in sys.argv[1:]:
    print('==', tag)
    for lab, v in c[f'crossing64_{tag}'].items():
        print(lab[:34], {k: (float('%.4g' % x) if isinstance(x, float) else x) for k, x in v.items() if k in show})
