#!/usr/bin/env python3
"""Task list of the line-source runs (same format as the crossing campaign).

GH: Garrett-Hauck benchmark, [-1.5,1.5]^2 (centred at 0.5 via --domain-length 3),
    sigma_s = 1, T = 1, Gaussian width 0.03, floor 1e-4. Variant B (needs
    --domain-length); element-weno runs it as GHS below. Reference: Ganapol.
VOID: same problem without material, scaled to the unit square
    (x' = (x+1.5)/3, t' = t/3, width 0.01, floor 9e-4 = 9 * 1e-4).
    B and element-weno. Reference: analytic uncollided line source.
GHS: the GH benchmark for element-weno, which lives on [0,1]^2: exact rescaling
    x' = (x+1.5)/3, t' = t/3, sigma_s' = 3, width .01, floor 9e-4 and filter
    strength x3 (sigma_f is a rate). phi' = 9 phi. Appended last so that the
    indices of the earlier tasks stay valid.
Filter strengths as in the crossing campaign (paper Table 1, P11: 6).
"""
import sys
out = sys.argv[1]
SIGMA = {3: 31.75, 7: 16.52, 11: 6.0}
GH = "--case line-source --dimension 2 --domain-length 3 --pulse-width .03 --background 1e-4 --sigma-s 1 --time 1 --memory-mib 49152"
VOID = "--case line-source --dimension 2 --pulse-width .01 --background 9e-4 --time 0.3333333333333333 --memory-mib 49152"
lines = []
for order in (3, 7, 11):
    modes = ["off", "constant"] + (["nn"] if order in (3, 7) else [])
    for grid in (256, 512):
        for mode in modes:
            extra = f" --filter {mode}" + (f" --filter-strength {SIGMA[order]}" if mode == "constant" else "")
            lines.append((f"GH_B_{mode}_g{grid}_P{order}", f"{GH} --pn-order {order} --elements {grid}{extra}"))
    grid = 256
    for mode in modes:
        extra = f" --filter {mode}" + (f" --filter-strength {SIGMA[order]}" if mode == "constant" else "")
        lines.append((f"VOID_B_{mode}_g{grid}_P{order}", f"{VOID} --pn-order {order} --elements {grid}{extra}"))
    for degree in (1, 2, 3, 4):
        for mode in ("off", "constant"):
            extra = f" --filter constant --filter-strength {SIGMA[order]}" if mode == "constant" else ""
            lines.append((f"VOID_EW_p{degree}_{mode}_g{grid}_P{order}",
                          f"{VOID} --pn-order {order} --stabilization element-weno --degree {degree} "
                          f"--elements {round(grid / degree)}{extra}"))
GHS = "--case line-source --dimension 2 --pulse-width .01 --background 9e-4 --sigma-s 3 --time 0.3333333333333333 --memory-mib 49152"
for grid in (256, 512):
    for order in (3, 7, 11):
        for degree in (1, 2, 3, 4):
            for mode in ("off", "constant"):
                extra = f" --filter constant --filter-strength {3 * SIGMA[order]:g}" if mode == "constant" else ""
                lines.append((f"GHS_EW_p{degree}_{mode}_g{grid}_P{order}",
                              f"{GHS} --pn-order {order} --stabilization element-weno --degree {degree} "
                              f"--elements {round(grid / degree)}{extra}"))
with open(out, "w") as f:
    for name, args in lines:
        f.write(f"{name}\t{args}\n")
print(len(lines))
