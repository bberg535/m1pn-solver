#!/usr/bin/env python3
"""Create an isolated native study driver; never edits the parent solver."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def main():
    text = (PARENT / 'nx/src/study_main.cpp').read_text()
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError(f'Parent driver anchor changed: {old[:100]}')
        text = text.replace(old, new)
    replace('#include "pn_uncoupled.hpp"', '#include "pn_uncoupled.hpp"\n#include "filter.hpp"')
    replace('    MeshSpec mesh;', '''    std::string filter_mode = "off", filter_model;
    double filter_strength = 0;
    MeshSpec mesh;''')
    replace('                   " --cuda on|off --mcl-mode eager|lazy\\n";', '''                   " --cuda on|off --mcl-mode eager|lazy\\n"
                   " --filter off|constant|nn --filter-strength S --filter-model FILE\\n";''')
    replace('        if (key == "--pn-order")', '''        if (key == "--filter") o.filter_mode = v;
        else if (key == "--filter-strength") o.filter_strength = real();
        else if (key == "--filter-model") o.filter_model = v;
        else if (key == "--pn-order")''')
    replace('    require(o.final_time >= 0', '''    require(o.filter_mode == "off" || o.filter_mode == "constant" || o.filter_mode == "nn",
            "Invalid --filter mode");
    require(o.filter_strength >= 0, "Negative filter strength");
    if (o.filter_mode != "off") {
        require(o.mesh.dimension == 2 && o.num.r2 && o.num.stabilization == "m1pn_mass_correction",
                "This filter experiment requires 2D R2 standard B mass correction");
        require(!o.cuda_on, "Filter experiment requires CPU evolution");
    }
    if (o.filter_model.empty())
        o.filter_model = std::string(NN_FILTER_ROOT) + "/models/model_N" + std::to_string(o.order) + ".txt";
    require(o.final_time >= 0''')
    # Check model before output directory creation so bad input leaves no run.
    replace('        Options o = parse(argc, argv);', '''        Options o = parse(argc, argv);
        nnfilter::Filter filter(o.order, o.filter_mode, o.filter_strength, o.filter_model);''')
    replace('        Field boundary = Field::Zero(fe.nodes, pn.q), reaction_ledger = boundary;',
            '        Field boundary = Field::Zero(fe.nodes, pn.q), reaction_ledger = boundary, filter_ledger = boundary;')
    replace('        while (t < o.final_time) {', '''        Field filtered, trial_filter;
        while (t < o.final_time) {''')
    replace('                    if (splitting) {\n                        trial_reaction', '''                    const bool filtering = filter.active();
                    if (filtering) {
                        nnfilter::reset_ledger(trial_filter, fe.nodes, pn.q);
                        filter.apply_into(filtered,pn,fe,u,absorption,scattering,src,dt/2,trial_filter);
                        if (!pn_only) nnfilter::check_state_parallel(filtered);
                    }
                    const Field &before_reaction = filtering ? filtered : u;
                    if (splitting) {
                        trial_reaction''')
    replace('candidate = reaction(u, src, absorption, scattering, o.anisotropy, dt / 2);',
            'candidate = reaction(before_reaction, src, absorption, scattering, o.anisotropy, dt / 2);')
    replace('trial_reaction += fe.mass.asDiagonal() * (candidate - u);',
            'trial_reaction += fe.mass.asDiagonal() * (candidate - before_reaction);')
    replace('const Field &stage_input = splitting ? candidate : u;',
            'const Field &stage_input = splitting ? candidate : before_reaction;')
    replace('                    if (pn_only) pn_only_require_finite(candidate, fe, pn);', '''                    if (filtering)
                        filter.apply_into(candidate,pn,fe,candidate,absorption,scattering,src,dt/2,trial_filter);
                    if (pn_only) pn_only_require_finite(candidate, fe, pn);''')
    replace('                    else if (splitting) tr.check_state(candidate);',
            '                    else if (filtering) nnfilter::check_state_parallel(candidate);\n                    else if (splitting) tr.check_state(candidate);')
    replace('            if (splitting) reaction_ledger += trial_reaction;',
            '            if (splitting) reaction_ledger += trial_reaction;\n            if (filter.active()) nnfilter::add_ledger(filter_ledger, trial_filter);')
    replace('fe.mass.asDiagonal() * (u - u0) + boundary - reaction_ledger)',
            'fe.mass.asDiagonal() * (u - u0) + boundary - reaction_ledger - filter_ledger)')
    replace('            write_field(o.output + "/reaction_integral.csv", fe, reaction_ledger);', '''            write_field(o.output + "/reaction_integral.csv", fe, reaction_ledger);
            write_field(o.output + "/filter_integral.csv", fe, filter_ledger);
            const Vec final_strength = filter.strength(pn,fe,u,absorption,scattering,src);
            std::ofstream sf(o.output + "/filter_strength.csv");
            sf << std::setprecision(17) << "node,sigma_f\\n";
            for (int i=0; i<fe.nodes; ++i) sf << i << "," << final_strength[i] << "\\n";''')
    replace('                   << ",\\"pn_order\\":" << pn.order', '''                   << ",\\"filter_mode\\":" << std::quoted(o.filter_mode)
                   << ",\\"filter_constant\\":" << o.filter_strength
                   << ",\\"filter_model\\":" << std::quoted(o.filter_model)
                   << ",\\"filter_order\\":4,\\"filter_density_coefficient\\":0"
                   << ",\\"filter_integration\\":\\"symmetric_exponential_midpoint_splitting\\""
                   << ",\\"filter_features\\":\\"upstream_2d_spatial_sample_normalized_native_fe_derivatives\\""
                   << ",\\"pn_order\\":" << pn.order''')
    replace('                   << ",\\"solver_seconds\\":" << solver_seconds', '''                   << ",\\"filter_seconds_including_retries\\":" << filter.elapsed
                   << ",\\"filter_evaluations_including_retries\\":" << filter.evaluations
                   << ",\\"filter_density_ledger_max\\":" << maxabs(filter_ledger.col(0))
                   << ",\\"final_filter_strength_min\\":" << final_strength.minCoeff()
                   << ",\\"final_filter_strength_max\\":" << final_strength.maxCoeff()
                   << ",\\"solver_seconds\\":" << solver_seconds''')
    (ROOT / 'src/main.cpp').write_text('// Generated isolated driver: scripts/prepare_driver.py\n' + text)
    paths = list((PARENT / 'src').glob('*')) + [PARENT / 'CMakeLists.txt', PARENT / 'nx/src/study_main.cpp', PARENT / 'nx/src/pn_uncoupled.hpp']
    manifest = {'parent_commit': subprocess.check_output(['git','-C',str(PARENT),'rev-parse','HEAD'],text=True).strip(),
                'parent_worktree_has_existing_edits': True,
                'sha256': {str(p.relative_to(PARENT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}}
    (ROOT / 'parent_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
