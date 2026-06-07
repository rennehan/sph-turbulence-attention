"""Faithful a-priori test of the Rennehan et al. (2019) dynamic model, using
THEIR smoothing operator (XSPH, eq 20, eps=0.8, harmonic-mean filter-scale
density) rather than a Shepard convolution.

Reports, for the smoothed (resolved) field at grid scale h_bar and test scale
h_hat = 2 h_bar:
  corr(L, tau_true)  - is the SGS stress predictable by similarity?
  corr(M, tau_true)  - does the strain (Smagorinsky) basis carry signal?
  corr(L, M)         - are they orthogonal (=> dynamic C ~ 0)?
  dynamic C          - your eq 14 coefficient (local, entire-term averaged)

Compared against the Shepard-filter numbers (0.926 / -0.067 / 0.002).
"""
import sys
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from load_snapshots import load
from sph_ops import xsph_smooth, sph_gradient


def dev(t):
    t = t.copy(); tr = t[:, 0, 0] + t[:, 1, 1]
    t[:, 0, 0] -= 0.5 * tr; t[:, 1, 1] -= 0.5 * tr
    return t

def corr(a, b):
    return np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]

def strain(sn, v, Hg, m, L):
    gx = sph_gradient(sn, v[:, 0], Hg, m, L, renorm=True)
    gy = sph_gradient(sn, v[:, 1], Hg, m, L, renorm=True)
    G = np.stack([gx, gy], 1)
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    Sd = dev(S)
    return Sd, np.sqrt(2 * np.einsum("nij,nij->n", Sd, Sd))

b = load(sys.argv[1] if len(sys.argv) > 1 else "data/res_N16384.sph")
meta = b.meta; h = meta["h"]; m = meta["m"]; L = meta["L"]
Hbar = 2 * h          # grid filter (smoothing length)
Hhat = 2 * Hbar       # test filter h_hat = 2 h_bar
Hgrad = 2 * h
s = b.snapshots[len(b.snapshots) // 2]
pts = np.column_stack([s.x, s.y]).astype(np.float64)
tree = cKDTree(pts % L, boxsize=L)
pairs = tree.query_ball_point(pts % L, 2 * Hhat)     # support of the widest (test) filter
deg = np.array([len(p) - 1 for p in pairs])
off = np.zeros(len(pts) + 1, np.int32); off[1:] = np.cumsum(deg)
idx = np.empty(off[-1], np.int32); w = 0
for i, p in enumerate(pairs):
    for j in p:
        if j != i:
            idx[w] = j; w += 1
sn = replace(s, nbr_offsets=off, nbr_indices=idx)
u = np.stack([s.vx, s.vy], 1).astype(np.float64)

# --- grid-smoothed (resolved) field and TRUE SGS stress, all with XSPH ---
ub = xsph_smooth(sn, u, Hbar, m, L)
tau = dev(xsph_smooth(sn, np.einsum("ni,nj->nij", u, u), Hbar, m, L)
          - np.einsum("ni,nj->nij", ub, ub))

# --- single-scale Smagorinsky tensor (the model basis) ---
Sd, Smag = strain(sn, ub, Hgrad, m, L)
M = dev(-2 * Hbar ** 2 * Smag[:, None, None] * Sd)

# --- test-smoothed field, Leonard tensor, Germano alpha/beta ---
ubh = xsph_smooth(sn, ub, Hhat, m, L)
Ld = dev(xsph_smooth(sn, np.einsum("ni,nj->nij", ub, ub), Hhat, m, L)
         - np.einsum("ni,nj->nij", ubh, ubh))
Sdh, Smagh = strain(sn, ubh, Hgrad, m, L)
alpha = Hhat ** 2 * Smagh[:, None, None] * Sdh
beta_hat = xsph_smooth(sn, Smag[:, None, None] * Sd, Hhat, m, L)   # ^(|S| S^d), entire term
Mger = dev(-2 * (alpha - Hbar ** 2 * beta_hat))

# --- dynamic C, eq 14: local, test-filter the ENTIRE contracted scalars ---
num = xsph_smooth(sn, np.einsum("nij,nij->n", Ld, Mger), Hhat, m, L)
den = xsph_smooth(sn, np.einsum("nij,nij->n", Mger, Mger), Hhat, m, L)
C = num / np.where(np.abs(den) > 1e-30, den, 1.0)

print(f"{sys.argv[1] if len(sys.argv)>1 else 'data/res_N16384.sph'}  t={s.t:.2f}  XSPH smoothing (eps=0.8, harmonic rho)")
print(f"  corr(L, tau_true) = {corr(Ld, tau):+.3f}   (Shepard: 0.926)")
print(f"  corr(M, tau_true) = {corr(M, tau):+.3f}   (Shepard: -0.067)")
print(f"  corr(L, M)        = {corr(Ld, M):+.3f}   (Shepard: 0.002)")
print(f"  dynamic C: mean={C.mean():+.4f}  median={np.median(C):+.4f}  backscatter={100*np.mean(C<0):.0f}%")
