"""Decisive test: sweep the FILTER WIDTH at fixed resolution.

Resolution alone (40^2 -> 128^2 at a 2h filter) didn't recover a positive
Smagorinsky correlation. But the 2h filter shrinks with resolution, so it may
still sit too close to the resolution scale. Here we hold the snapshot fixed
and widen the filter H = {2,3,4,5,6} h, rebuilding neighbours at radius 2H with
a periodic KD-tree (we dumped all positions, so no re-simulation needed).

If corr(tau^d, Smagorinsky) climbs toward +0.3-0.5 as H grows -> the 2h filter
was just too narrow. If it stays ~0/negative at all widths -> it's 2D
inverse-cascade backscatter, intrinsic to the flow, not a filter artifact.
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
# mid-window snapshot
s = b.snapshots[len(b.snapshots) // 2]
print(f"{path}  N={meta['N']}  snapshot t={s.t:.2f}")
print(f"{'H/h':>4} {'Delta':>7} {'nbrs':>6} {'corr':>7} {'C_d':>9} {'k_sgs/k_res':>11}")

pts = np.column_stack([s.x, s.y]).astype(np.float64) % L
tree = cKDTree(pts, boxsize=L)

def neighbors_csr(radius):
    pairs = tree.query_ball_point(pts, radius)  # includes self
    deg = np.array([len(p) - 1 for p in pairs])  # drop self
    offsets = np.zeros(len(pts) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(deg)
    idx = np.empty(offsets[-1], dtype=np.int32)
    w = 0
    for i, p in enumerate(pairs):
        for j in p:
            if j != i:
                idx[w] = j; w += 1
    return offsets, idx

u = np.stack([s.vx, s.vy], 1).astype(np.float64)
for Hh in [2, 3, 4, 5, 6]:
    H = Hh * h
    Delta = H
    off, idx = neighbors_csr(2 * H)            # filter support = 2H
    sn = replace(s, nbr_offsets=off, nbr_indices=idx)
    ubar = sph_filter(sn, u, H, m, L)
    gux = sph_gradient(sn, ubar[:, 0], H, m, L, renorm=True)
    guy = sph_gradient(sn, ubar[:, 1], H, m, L, renorm=True)
    G = np.stack([gux, guy], axis=1)            # G[:,a,b] = d ubar_a / dx_b
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    tr = S[:, 0, 0] + S[:, 1, 1]
    Sdev = S.copy(); Sdev[:, 0, 0] -= 0.5 * tr; Sdev[:, 1, 1] -= 0.5 * tr
    Smag = np.sqrt(2 * np.einsum("nij,nij->n", Sdev, Sdev))
    prod = sph_filter(sn, np.einsum("ni,nj->nij", u, u), H, m, L)
    tau = prod - np.einsum("ni,nj->nij", ubar, ubar)
    ksgs = 0.5 * np.mean(tau[:, 0, 0] + tau[:, 1, 1])
    kres = 0.5 * np.mean(ubar[:, 0] ** 2 + ubar[:, 1] ** 2)
    taud = tau.copy(); t2 = tau[:, 0, 0] + tau[:, 1, 1]
    taud[:, 0, 0] -= 0.5 * t2; taud[:, 1, 1] -= 0.5 * t2
    M = -2 * Delta * Delta * Smag[:, None, None] * Sdev
    corr = np.corrcoef(taud.reshape(-1), M.reshape(-1))[0, 1]
    Cd = np.sum(taud * M) / np.sum(M * M)
    print(f"{Hh:>4} {Delta:7.4f} {int(np.mean(np.diff(off))):>6} {corr:7.3f} {Cd:9.4f} {ksgs/kres:11.3f}")
