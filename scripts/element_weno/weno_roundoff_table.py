#!/usr/bin/env python3
"""Tables of the WENO rounding-limit study (m1pn_weno_roundoff_study JSON lines, docs/ELEMENT_WENO.md).

    python3 -I scripts/element_weno/weno_roundoff_table.py DIR/base.jsonl DIR/q3.jsonl ... > table.md

One block per variant: L2 error and EOC per degree over the grids, and the minimum gamma of the run.
"""
import json
import math
import sys
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if d.get("trace") or "run" not in d:
            continue
        rows[(d["p"], d["n"])] = d
    return rows


def main():
    out = ["| Variante | p | " + " | ".join(f"n={n}" for n in (32, 64, 128, 256, 512)) + " | EOC letzte | min γ (n=512) |",
           "|---|---:|" + "---:|" * 5 + "---:|---:|"]
    for path in sys.argv[1:]:
        name = Path(path).stem
        rows = load(path)
        for p in sorted({k[0] for k in rows}):
            ns = sorted(n for (q, n) in rows if q == p)
            errs = [rows[(p, n)]["run"]["l2_error"] for n in ns]
            cells = [f"{e:.2e}" for e in errs]
            eoc = (math.log(errs[-2] / errs[-1]) / math.log(ns[-1] / ns[-2])) if len(errs) > 1 else float("nan")
            g = rows[(p, ns[-1])]["run"]["min_gamma_stages"]
            out.append(f"| {name} | {p} | " + " | ".join(cells) + f" | {eoc:.2f} | {g:.6f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
