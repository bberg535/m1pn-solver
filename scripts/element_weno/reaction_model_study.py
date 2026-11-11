#!/usr/bin/env python3
"""Nodal-average reaction vs consistent FE reaction on 1D Bernstein Q_p (docs/ELEMENT_HETEROGENEOUS.md, Sec. 3).

Independent of the C++ solver (NumPy only). Periodic mesh of E elements on [0,1], Bernstein degree p.

  nodal:       d a_i/dt = -sigma_i a_i,           sigma_i = int sigma phi_i / int phi_i  (positive averages)
  consistent:  M d a/dt = -M_sigma a,             (M_sigma)_ij = int sigma phi_i phi_j

Reported: (1) reaction-rate consistency ||r_h - P(-sigma u)||_L2, (2) error of the reaction solution at t = 1
against the L2 projection of e^{-sigma t} u, both for a smooth sigma; (3) L1 error for a sigma jump that is not
aligned with element boundaries; (4) min Bernstein coefficient (positivity) of the reacted state for a nonnegative
initial state with a jump. Usage: reaction_model_study.py [--csv FILE]
"""
import argparse
import math
import numpy as np
from numpy.polynomial.legendre import leggauss


def bernstein(p, x):
    """Values B_{i,p}(x), i = 0..p, for an array x (shape (p+1, len(x)))."""
    return np.array([math.comb(p, i) * x**i * (1 - x)**(p - i) for i in range(p + 1)])


class Mesh:
    def __init__(self, E, p, qpts):
        self.E, self.p, self.n = E, p, E * p  # periodic: E*p nodes
        g, w = leggauss(qpts)
        self.xq, self.wq = (g + 1) / 2, w / 2
        self.B = bernstein(p, self.xq)  # (p+1, q)
        self.h = 1 / E

    def nodes(self, e):
        return [(e * self.p + i) % self.n for i in range(self.p + 1)]

    def points(self, e, xq=None):
        return (e + (self.xq if xq is None else xq)) * self.h

    def assemble(self, f=None, sub=1):
        """M (f=None) or M_f = int f phi_i phi_j, with `sub` sub-cells per element for discontinuous f."""
        A = np.zeros((self.n, self.n))
        for e in range(self.E):
            idx = self.nodes(e)
            for c in range(sub):
                xs = (c + self.xq) / sub
                B = bernstein(self.p, xs)
                w = self.wq / sub * self.h * (f(self.points(e, xs)) if f else 1)
                A[np.ix_(idx, idx)] += (B * w) @ B.T
        return A

    def load(self, f, sub=1):
        b = np.zeros(self.n)
        for e in range(self.E):
            idx = self.nodes(e)
            for c in range(sub):
                xs = (c + self.xq) / sub
                b[idx] += bernstein(self.p, xs) @ (self.wq / sub * self.h * f(self.points(e, xs)))
        return b

    def l2_project(self, f, sub=1):
        return np.linalg.solve(self.assemble(), self.load(f, sub))

    def error(self, a, f, norm=2, sub=8):
        tot = 0.
        for e in range(self.E):
            for c in range(sub):
                xs = (c + self.xq) / sub
                vals = a[self.nodes(e)] @ bernstein(self.p, xs)
                diff = np.abs(vals - f(self.points(e, xs)))
                tot += np.sum(self.wq / sub * self.h * diff**norm)
        return tot ** (1 / norm)


def study(csv=None):
    sigma = lambda x: 1 + .5 * np.sin(2 * np.pi * x)
    u0 = lambda x: 1 + .3 * np.cos(2 * np.pi * x) + .1 * np.sin(6 * np.pi * x)
    jump = lambda x: np.where(x > .3 + 1 / 17, 10., 1.)  # never on an element boundary for E = 2^k
    rows = []
    print(f"{'p':>2} {'E':>4} | {'rate nodal':>10} {'EOC':>5} {'rate cons':>10} {'EOC':>5} | {'t=1 nodal':>10} {'EOC':>5}"
          f" {'t=1 cons':>10} {'EOC':>5} | {'jump nodal':>10} {'jump cons':>10} | {'min coef nodal':>14} {'min coef cons':>14}")
    for p in range(1, 5):
        prev = None
        for E in (8, 16, 32, 64, 128):
            m = Mesh(E, p, p + 6)
            M = m.assemble()
            a = m.l2_project(u0)
            lumped = M.sum(axis=1)
            sig_n = m.load(sigma, sub=1) / lumped
            # (1) rate consistency
            r_nodal = -sig_n * a
            r_cons = -np.linalg.solve(M, m.assemble(sigma) @ a)
            exact_rate = lambda x: -sigma(x) * u0(x)
            e1n, e1c = m.error(r_nodal, exact_rate), m.error(r_cons, exact_rate)
            # (2) exact reaction solution at t = 1
            exact_t = lambda x: np.exp(-sigma(x)) * u0(x)
            a_nodal = np.exp(-sig_n) * a
            w, V = np.linalg.eig(np.linalg.solve(M, m.assemble(sigma)))
            a_cons = np.real(V @ np.diag(np.exp(-w)) @ np.linalg.solve(V, a))
            e2n, e2c = m.error(a_nodal, exact_t), m.error(a_cons, exact_t)
            # (3)/(4) unaligned jump, reaction time 1, L1 error; positivity of the coefficients
            sig_j = m.load(jump, sub=64) / lumped
            exact_j = lambda x: np.exp(-jump(x)) * u0(x)
            aj_n = np.exp(-sig_j) * a
            wj, Vj = np.linalg.eig(np.linalg.solve(M, m.assemble(jump, sub=64)))
            aj_c = np.real(Vj @ np.diag(np.exp(-wj)) @ np.linalg.solve(Vj, a))
            e3n, e3c = m.error(aj_n, exact_j, norm=1), m.error(aj_c, exact_j, norm=1)
            # positivity: start from a nonnegative bump with zero coefficients elsewhere
            bump = np.where(np.abs(np.arange(m.n) / m.n - .4) < .1, 1., 0.)
            min_n = np.min(np.exp(-sig_j) * bump)
            min_c = np.min(np.real(Vj @ np.diag(np.exp(-wj * 5)) @ np.linalg.solve(Vj, bump)))
            eoc = lambda new, old: math.log2(old / new) if old else float('nan')
            row = dict(p=p, E=E, rate_nodal=e1n, rate_cons=e1c, sol_nodal=e2n, sol_cons=e2c,
                       jump_nodal_L1=e3n, jump_cons_L1=e3c, min_coef_nodal=min_n, min_coef_cons=min_c)
            if prev:
                row.update(eoc_rate_nodal=eoc(e1n, prev['rate_nodal']), eoc_rate_cons=eoc(e1c, prev['rate_cons']),
                           eoc_sol_nodal=eoc(e2n, prev['sol_nodal']), eoc_sol_cons=eoc(e2c, prev['sol_cons']),
                           eoc_jump_nodal=eoc(e3n, prev['jump_nodal_L1']), eoc_jump_cons=eoc(e3c, prev['jump_cons_L1']))
            rows.append(row)
            f = lambda k: f"{row.get(k, float('nan')):5.2f}"
            print(f"{p:2d} {E:4d} | {e1n:10.3e} {f('eoc_rate_nodal')} {e1c:10.3e} {f('eoc_rate_cons')} | {e2n:10.3e} "
                  f"{f('eoc_sol_nodal')} {e2c:10.3e} {f('eoc_sol_cons')} | {e3n:10.3e} {e3c:10.3e} | {min_n:14.3e} {min_c:14.3e}")
            prev = row
    if csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('p', 'E'), k))
        with open(csv, 'w') as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    study(ap.parse_args().csv)
