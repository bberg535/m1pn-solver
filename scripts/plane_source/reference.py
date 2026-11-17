#!/usr/bin/env python3
"""Exact scalar flux of the plane-source benchmark with a Gaussian initial condition.

Infinite homogeneous medium, unit speed, isotropic scattering with sigma_t = sigma_s = 1 (Ganapol, AZURV1).
The plane kernel is the point kernel integrated over the plane (Ganapol 2001; Bennett & McClarren,
arXiv:2205.15783):
  phi_c,plane(x) = 2 pi int_{|x|}^{t} r phi_c,point(r) dr          (collided, |x| < t)
  phi_u,plane(x) = e^{-t} / (2 t) on |x| < t                         (uncollided)
The initial condition is rho_0(x) = exp(-x^2 / (2 s^2)) (peak one, mass sqrt(2 pi) s), as m1pn_fast
--case plane-source --plane-width; the uncollided part is convolved analytically (erf), the collided part
numerically. The collided point kernel is the one of scripts/line_source/reference.py (Eq. (41)).

Usage: reference.py --time 1 --sigma .03 --out ref.csv [--check]
Writes columns x, phi, phi_uncollided, phi_collided on x in [0, xmax] (the solution is even).
"""
import argparse
import math
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--time", type=float, default=1.)
p.add_argument("--sigma", type=float, default=.03)
p.add_argument("--xmax", type=float, default=1.5)
p.add_argument("--points", type=int, default=3001)
p.add_argument("--out", required=True)
p.add_argument("--check", action="store_true")
a = p.parse_args()
t, s = a.time, a.sigma
c = 1.


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


def phi_plane_collided(x):
    """2 pi int_{|x|}^t r phi_pt(r) dr; r phi_pt is bounded (log singularity at r -> t is integrable)."""
    lo = abs(x)
    if lo >= t:
        return 0.
    # r = t - (t - lo) (1 - v)^2 clusters points at the front, where log((1+eta)/(1-eta)) is singular.
    v, w = gauss(300, 0, 1)
    r = t - (t - lo) * (1 - v) ** 2
    jac = 2 * (t - lo) * (1 - v)
    return 2 * np.pi * np.sum(r * phi_pt_collided(r) * jac * w)


def trap(y, x):
    return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2


# Dirac collided kernel on [0, t), even; tabulated on a front-clustered grid.
xi_ = t * (1 - (1 - np.linspace(0, 1, 1601)) ** 2)[:-1]
kc = np.array([phi_plane_collided(x) for x in xi_])
y = np.concatenate([-xi_[::-1], xi_[1:]])
ky = np.concatenate([kc[::-1], kc[1:]])
y = np.concatenate([[-t], y, [t]])
ky = np.concatenate([[ky[0]], ky, [ky[-1]]])          # bounded at the front; the endpoints carry no weight
mass0 = math.sqrt(2 * math.pi) * s
x = np.linspace(0, a.xmax, a.points)
# Convolution with rho_0: fine uniform grid in y (the kernel is smooth on the scale of s except at +-t).
yf = np.linspace(-t, t, 40001)
kf = np.interp(yf, y, ky)
coll = np.array([trap(kf * np.exp(-(xx - yf) ** 2 / (2 * s * s)), yf) for xx in x])
unc = np.exp(-t) / (2 * t) * mass0 / 2 * np.array(
    [math.erf((xx + t) / (math.sqrt(2) * s)) - math.erf((xx - t) / (math.sqrt(2) * s)) for xx in x])
phi = unc + coll
np.savetxt(a.out, np.c_[x, phi, unc, coll], delimiter=",", header="x,phi,phi_uncollided,phi_collided",
           comments="", fmt="%.12e")
if a.check:
    print(f"collided Dirac mass {trap(ky, y):.6f}, expected 1-e^-t = {1 - np.exp(-t):.6f}")
    print(f"smoothed collided mass / m0 {2 * trap(coll, x) / mass0:.6f}, uncollided {2 * trap(unc, x) / mass0:.6f}"
          f" (expected e^-t = {np.exp(-t):.6f})")
    for xx in (0, .2, .4, .6, .8, .9, .95, 1.):
        i = np.argmin(abs(x - xx))
        print(f"x={x[i]:.3f} Dirac collided {np.interp(x[i], y, ky):.6f} phi/m0={phi[i] / mass0:.6f}")
