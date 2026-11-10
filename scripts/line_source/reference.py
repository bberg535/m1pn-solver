#!/usr/bin/env python3
"""Exact scalar flux of the line-source benchmark with a Gaussian initial condition.

Ganapol et al. (2001), AZURV1; cylindrical form and Gaussian smoothing as in
Garrett & Hauck, TTSP 42 (2013) and Bennett & McClarren, arXiv:2205.15783,
Eqs. (38)-(41) with xi from Eq. (8).  Infinite homogeneous medium, unit speed,
isotropic scattering with sigma_t = sigma_s = 1 (--medium scattering) or a
void without any interaction (--medium void, uncollided part only, no e^-t).

  phi(x, 0) = floor + exp(-r^2 / (2 s^2)) / (2 pi s^2)

The floor stays constant in an infinite medium in both cases.
NumPy only (Gauss-Legendre quadrature).

Usage: reference.py --time 1 --sigma .03 --medium scattering --floor 1e-4 --out ref.csv [--check]
Writes columns r, phi, phi_uncollided, phi_collided.
"""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--time", type=float, default=1.)
p.add_argument("--sigma", type=float, default=.03)
p.add_argument("--medium", choices=["scattering", "void"], default="scattering")
p.add_argument("--floor", type=float, default=1e-4)
p.add_argument("--rmax", type=float, default=1.5)
p.add_argument("--points", type=int, default=1501)
p.add_argument("--out", required=True)
p.add_argument("--check", action="store_true")
a = p.parse_args()
t, s = a.time, a.sigma
c = 1. if a.medium == "scattering" else 0.
att = np.exp(-t) if c == 1 else 1.  # uncollided attenuation


def gauss(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * (x + 1) / 2, w * (hi - lo) / 2


def phi_pt_collided(r):
    """Collided point-source flux, Eq. (41), for 0 < r < t (vectorised over r)."""
    eta = r / t
    q = (1 + eta) / (1 - eta)
    pre = np.exp(-t) / (4 * np.pi * r * t * t)
    first = pre * c * t * np.log(q)
    u, w = gauss(400, 0, np.pi)
    tn = np.tan(u / 2)[None, :]
    e = eta[:, None]
    xi = (np.log(q)[:, None] + 1j * u[None, :]) / (e + 1j * tn)
    val = (1 / np.cos(u / 2)[None, :]) ** 2 * ((e + 1j * tn) * xi ** 3 *
                                               np.exp(c * t / 2 * (1 - e * e) * xi)).real
    second = pre / (2 * np.pi) * (c * t / 2) ** 2 * (1 - eta * eta) * (val @ w)
    return first + second


def phi_line_collided(rho):
    """Collided line-source flux, Eq. (40); w = wmax sin(b) removes the endpoint singularity."""
    eta = rho / t
    wmax = np.sqrt(1 - eta * eta)
    b, wb = gauss(200, 0, np.pi / 2)
    rr = t * np.sqrt(eta * eta + (wmax * np.sin(b)) ** 2)
    return 2 * t * np.sum(phi_pt_collided(rr) * wmax * np.cos(b) * wb)


def i0e(z):
    """exp(-z) I0(z) for z >= 0."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    small = z < 500
    out[small] = np.i0(z[small]) * np.exp(-z[small])
    zl = z[~small]
    out[~small] = (1 + 1 / (8 * zl) + 9 / (128 * zl * zl) + 225 / (3072 * zl ** 3)) / np.sqrt(2 * np.pi * zl)
    return out


def ring(r, rho):
    """Mean of the normalised Gaussian over a circle of radius rho at distance r from its centre."""
    return np.exp(-(r - rho) ** 2 / (2 * s * s)) * i0e(r * rho / (s * s)) / (2 * np.pi * s * s)


def trap(y, x):
    return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2


r = np.linspace(0, a.rmax, a.points)
# Uncollided: radial distance t sin(al) with weight sin(al) d al (free streaming, Eq. (38)).
al, wal = gauss(2000, 0, np.pi / 2)
unc = att * np.array([np.sum(np.sin(al) * ring(x, t * np.sin(al)) * wal) for x in r])
coll = np.zeros_like(r)
if c == 1:
    rho = np.linspace(0, t, 2401)[1:-1]
    lc = np.array([phi_line_collided(x) for x in rho])
    rho = np.concatenate([[0], rho, [t]])
    lc = np.concatenate([[lc[0]], lc, [lc[-1]]])   # bounded at the ends; endpoints carry no weight
    coll = np.array([trap(2 * np.pi * rho * lc * ring(x, rho), rho) for x in r])
floor = a.floor
phi = floor + unc + coll
np.savetxt(a.out, np.c_[r, phi, unc, coll], delimiter=",", header="r,phi,phi_uncollided,phi_collided",
           comments="", fmt="%.12e")
if a.check:
    if c == 1:
        print(f"collided Dirac mass {trap(2 * np.pi * rho * lc, rho):.6f}, expected 1-e^-t = {1 - np.exp(-t):.6f}")
        print(f"smoothed collided mass {trap(2 * np.pi * r * coll, r):.6f}")
    print(f"smoothed uncollided mass {trap(2 * np.pi * r * unc, r):.6f}, expected {att:.6f}")
    for x in (0, .2, .4, .6, .8, .9, .95, 1., 1.05):
        i = np.argmin(abs(r - x)); print(f"r={r[i]:.3f} phi={phi[i]:.6f} unc={unc[i]:.6f} coll={coll[i]:.6f}")
