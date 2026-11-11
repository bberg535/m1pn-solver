#!/usr/bin/env python3
"""C1: re-evaluate the failing stable_base_bar input in 100-digit arithmetic (mpmath) from the exact binary64 inputs.
Decides whether the exact bar lies in the M1 cone (mathematical truth) and whether the binary64 rounding of it fails.
Run:  python3 -I audit/c1_highprecision.py audit/results/c1_probe.jsonl"""
import json, sys, struct
from mpmath import mp, mpf, sqrt
mp.dps = 100
def H(x):  # hexfloat string -> exact mpf
    return mpf(float.fromhex(x))
def split(u, t, qt, sign):
    rho = u[0]
    if rho == 0:
        return [mpf(0)] * 4
    J = u[1:]
    w = [j / rho for j in J]
    q = (rho ** 2 - sum(j ** 2 for j in J)) / rho ** 2          # normalized gap
    mu = sum(wi * ti for wi, ti in zip(w, t))
    s = sqrt(1 + 3 * q)
    c = q / (1 + s)
    k = 3 / (2 + s)
    tau = (q + qt + sum((wi + sign * ti) ** 2 for wi, ti in zip(w, t))) / 2
    out = [rho * tau]
    for wi, ti in zip(w, t):
        out.append(rho * (tau * wi + sign * c * (ti - k * mu * wi)))
    return out
def ulp_up(x):
    return struct.unpack('<d', struct.pack('<q', struct.unpack('<q', struct.pack('<d', x))[0] + 1))[0]
def incone(v):  # exact cone test of a quadruple of exact numbers
    return v[0] >= 0 and v[0] ** 2 - sum(x ** 2 for x in v[1:]) >= 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["test"] != "C1_probe":
        continue
    ui = [H(x) for x in r["ui"]]; uj = [H(x) for x in r["uj"]]; edge = [H(x) for x in r["edge"]]; d = H(r["d"])
    t = [e / d for e in edge]
    qt = (d ** 2 - sum(e ** 2 for e in edge)) / d ** 2
    L = split(ui, t, qt, +1); R = split(uj, t, qt, -1)
    bar = [(a + b) / 2 for a, b in zip(L, R)]
    gap = bar[0] ** 2 - sum(x ** 2 for x in bar[1:])
    # nearest binary64 rounding of each component
    rd = [float(x) for x in bar]
    rd_gap = mpf(rd[0]) ** 2 - sum(mpf(x) ** 2 for x in rd[1:])
    # minimal number of binary64 ulps that rho must be raised (components rounded to nearest) to land in the stored cone
    n = 0
    cand = list(rd)
    while not incone([mpf(x) for x in cand]) and n < 64:
        cand[0] = ulp_up(cand[0]); n += 1
    print(json.dumps({
        "where": r["where"], "pair": [r["element"], r["i"], r["j"]],
        "input_gap_rel": [float((u[0] ** 2 - sum(x ** 2 for x in u[1:])) / u[0] ** 2) for u in (ui, uj)],
        "split_gaps_rel_exact": [float((s[0] ** 2 - sum(x ** 2 for x in s[1:])) / max(s[0] ** 2, mpf(10) ** -300)) for s in (L, R)],
        "exact_bar_rho": mp.nstr(bar[0], 25), "exact_bar_abs_J": mp.nstr(sqrt(sum(x ** 2 for x in bar[1:])), 25),
        "exact_bar_gap_rho2_minus_J2": mp.nstr(gap, 10), "exact_bar_in_cone": bool(gap >= 0),
        "rounded_to_nearest_double_gap": mp.nstr(rd_gap, 10), "rounded_in_cone": bool(rd_gap >= 0),
        "ulps_rho_must_be_raised": n, "relative_size_of_raise": float(mpf(cand[0]) / mpf(rd[0]) - 1),
        "classification": "rounding_only" if gap >= 0 else "MATHEMATICAL_VIOLATION"}, indent=None))
