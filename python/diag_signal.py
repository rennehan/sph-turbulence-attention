"""Signal-existence test + positive control.

Compares two a-priori models against the true deviatoric SGS stress tau^d:
  Smagorinsky  M = -2 Delta^2 |S| S^d        (strain-based eddy viscosity)
  scale-sim    L = <ubar_i ubar_j>^ - <ubar_i>^<ubar_j>^   (Bardina, test filter ^)

corr(tau^d, M) ~ 0 we already saw. If corr(tau^d, L) is HIGH (~0.7-0.9), the
SGS stress IS predictable -> the pipeline is sound, Smagorinsky is just the wrong
model, and a learned closure has a real ceiling to reach. If BOTH are ~0, the
stress is genuinely unpredictable at this resolution.
"""
import sys
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from load_snapshots import load
from sph_ops import sph_filter, sph_gradient

path = sys.argv[1]
b = load(path); meta = b.meta
h = meta["h"]; m = meta["m"]; L = meta["L"]

def dev(t):
    t = t.copy(); tr = t[:, 0, 0] + t[:, 1, 1]
    t[:, 0, 0] -= 0.5 * tr; t[:, 1, 1] -= 0.5 * tr
    return t

def corr(a, b_):
    return np.corrcoef(a.reshape(-1), b_.reshape(-1))[0, 1]

print(f"{path}  N={meta['N']}")
print(f"{'H/h':>4} {'corr(tau,Smag)':>15} {'corr(tau,similarity)':>21}")

for snap in [b.snapshots[len(b.snapshots) // 2]]:
    pts = (np.column_stack([snap.x, snap.y]).astype(np.float64)) % L
    tree = cKDTree(pts, boxsize=L)
    u = np.stack([snap.vx, snap.vy], 1).astype(np.float64)
    for Hh in [2, 4]:
        H = Hh * h
        rad = 4 * H                              # supports both H and 2H filters
        pairs = tree.query_ball_point(pts, rad)
        deg = np.array([len(p) - 1 for p in pairs])
        off = np.zeros(len(pts) + 1, np.int32); off[1:] = np.cumsum(deg)
        idx = np.empty(off[-1], np.int32); w = 0
        for i, p in enumerate(pairs):
            for j in p:
                if j != i:
                    idx[w] = j; w += 1
        sn = replace(snap, nbr_offsets=off, nbr_indices=idx)

        ubar = sph_filter(sn, u, H, m, L)
        tau = sph_filter(sn, np.einsum("ni,nj->nij", u, u), H, m, L) \
            - np.einsum("ni,nj->nij", ubar, ubar)
        taud = dev(tau)

        # Smagorinsky
        gx = sph_gradient(sn, ubar[:, 0], H, m, L, renorm=True)
        gy = sph_gradient(sn, ubar[:, 1], H, m, L, renorm=True)
        G = np.stack([gx, gy], 1)
        S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
        Sd = dev(S)
        Smag = np.sqrt(2 * np.einsum("nij,nij->n", Sd, Sd))
        M = dev(-2 * H * H * Smag[:, None, None] * Sd)

        # scale similarity (Bardina): test-filter ubar at width 2H
        Ht = 2 * H
        ut = sph_filter(sn, ubar, Ht, m, L)
        Lstress = sph_filter(sn, np.einsum("ni,nj->nij", ubar, ubar), Ht, m, L) \
            - np.einsum("ni,nj->nij", ut, ut)
        Ld = dev(Lstress)

        print(f"{Hh:>4} {corr(taud, M):>15.3f} {corr(taud, Ld):>21.3f}")
