#!/usr/bin/env python3
"""Write the task list of the 256/512 crossing campaign (one line per run)."""
import sys
out = sys.argv[1]
SIGMA = {3: 31.75, 7: 16.52, 11: 6.0}  # paper table 1 (2D); P11: value of nx511 runs
common = "--case crossing-study --inflow crossing-study-transport --dimension 2 --time .32 --memory-mib 49152"
lines = []
for grid in (256, 512):
    for order in (3, 7, 11):
        base = f"{common} --pn-order {order}"
        modes = ["off", "constant"] + (["nn"] if order in (3, 7) else [])
        for mode in modes:
            extra = f" --filter {mode}" + (f" --filter-strength {SIGMA[order]}" if mode == "constant" else "")
            lines.append((f"B_{mode}_g{grid}_P{order}", f"{base} --elements {grid}{extra}"))
        for degree in (1, 2, 3, 4):
            n = round(grid / degree)
            for mode in ("off", "constant"):
                extra = f" --filter constant --filter-strength {SIGMA[order]}" if mode == "constant" else ""
                lines.append((f"EW_p{degree}_{mode}_g{grid}_P{order}",
                              f"{base} --stabilization element-weno --degree {degree} --elements {n}{extra}"))
with open(out, "w") as f:
    for name, args in lines:
        f.write(f"{name}\t{args}\n")
print(len(lines))
