#!/usr/bin/env python3
"""Create build/element_weno_audit.cpp from the UNMODIFIED src/element_weno.cpp.
Only diagnostic hooks are inserted (all default to production behaviour):
  audit_low_order            : drop every antidiffusive element/boundary correction (low-order control)
  audit_skip_element_limit   : element contributions applied raw (boundary limiter unchanged)
  audit_skip_boundary_limit  : boundary corrections applied raw (element limiter unchanged)
  audit_factor/theta/scale   : last element factors, boundary thetas, per-element contribution scale
The patch is applied to a copy at build time; src/ is never touched."""
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
def sub(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sub("namespace m1pn {\nnamespace {",
    "namespace m1pn {\nbool audit_low_order=false, audit_skip_element_limit=false, audit_skip_boundary_limit=false;\n"
    "std::vector<double> audit_factor, audit_theta, audit_scale;\nnamespace {")
sub("    Field element_limited(cells, width), element_raw;",
    "    audit_scale.assign(cells,0.); audit_theta.assign(nodes,1.);\n    Field element_limited(cells, width), element_raw;")
sub("        if (options.limit) {\n            // Step 1",
    "        if (options.limit && !audit_skip_element_limit) {\n            // Step 1")
sub("        Four after = Four::Zero();",
    "        { double sc=0; for (const auto &f : contribution) sc=std::max(sc, f.cwiseAbs().maxCoeff()); audit_scale[e]=sc; }\n"
    "        if (audit_low_order) { for (auto &f : limited) f.setZero(); element_factor[e]=0; }\n"
    "        Four after = Four::Zero();")
sub("            if (options.limit) {\n                const Four g = correction / boundary_area[i];",
    "            if (options.limit && !audit_skip_boundary_limit) {\n                const Four g = correction / boundary_area[i];")
sub("            const Four base = boundary_area[i] * (bar - ui);",
    "            if (audit_low_order) theta = 0;\n            audit_theta[i] = theta;\n            const Four base = boundary_area[i] * (bar - ui);")
sub("    if (stats) {\n        stats->min_gamma",
    "    audit_factor = element_factor;\n    if (stats) {\n        stats->min_gamma")
pathlib.Path(sys.argv[2]).write_text(src)
