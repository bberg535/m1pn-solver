#!/usr/bin/env python3
"""Static PN order map from a Monte Carlo pilot (m1pn_lattice_mc --moment-output, docs/STATIC_HP_BLOCKS.md).

Per pixel the degree energies E_l = sum_m u_lm^2 of the tallied (collided) moments, corrected for the statistical
bias E[hat u^2] = u^2 + Var: E_l <- max(0, E_l - sum_m stderr_lm^2). The tail T_n = sqrt(sum_{l>n} E_l) and the order
n(x) = min{n >= 1 : T_n <= tol * max rho_c} as for the deterministic pilot (order_map_from_spectrum.py); a block gets the
largest order of its pixels. Orders are capped at the tallied order L (the map reports how many blocks hit L).

Usage: order_map_from_mc.py MOMENTS.csv TOL NB OUT.txt [--no-bias-correction] [--length 7]
"""
import argparse
import math

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("moments")
ap.add_argument("tol", type=float)
ap.add_argument("nb", type=int)
ap.add_argument("out")
ap.add_argument("--no-bias-correction", action="store_true")
ap.add_argument("--length", type=float, default=7.)
a = ap.parse_args()
d = np.loadtxt(a.moments, delimiter=",", skiprows=1)
ix, iy, ls, mean, err = d[:, 0].astype(int), d[:, 1].astype(int), d[:, 5].astype(int), d[:, 6], d[:, 7]
P, L = ix.max() + 1, ls.max()
E = np.zeros((P, P, L + 1))
V = np.zeros((P, P, L + 1))
np.add.at(E, (iy, ix, ls), mean ** 2)
np.add.at(V, (iy, ix, ls), err ** 2)
if not a.no_bias_correction:
    E = np.maximum(0., E - V)
rho = np.zeros((P, P))
m0 = ls == 0
rho[iy[m0], ix[m0]] = math.sqrt(4 * math.pi) * mean[m0]
tail = np.sqrt(np.cumsum(E[:, :, ::-1], axis=2)[:, :, ::-1])  # tail[..., l] = sqrt(sum_{k >= l} E_k)
bound = a.tol * rho.max()
ok = np.concatenate([tail[:, :, 1:] <= bound, np.ones((P, P, 1), bool)], axis=2)
need = np.maximum(np.argmax(ok, axis=2), 1)
noise = np.sqrt(V[:, :, 2:].sum(axis=2))  # noise level of the tail beyond l = 1
g = np.ones((a.nb, a.nb), int)
c = (np.arange(P) + .5) * a.length / P
bi = np.minimum((c / a.length * a.nb).astype(int), a.nb - 1)
for j in range(P):
    for i in range(P):
        g[bi[j], bi[i]] = max(g[bi[j], bi[i]], need[j, i])
np.savetxt(a.out, g, fmt="%d")
frac = ((g + 1) ** 2).sum() / (a.nb * a.nb * (L + 1) ** 2)
print(f"{a.out}: L = {L}, pixels {P}^2, state fraction (vs P{L}) {frac:.2f}, blocks at L {np.mean(g == L):.0%}, "
      f"max tail noise / bound {noise.max() / bound:.2f}")
