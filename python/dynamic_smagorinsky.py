"""Phase 2 baseline: local dynamic Smagorinsky (Germano-Lilly), done properly.

Grid filter at Delta = 2h (the resolved field <u>); test filter ^ at
Delta_hat = alpha*Delta = 4h (alpha=2). Local coefficient with the test filter
applied to the ENTIRE contracted scalar (not to L and M separately):

    C(x) = ^( L^d_ij M_ij ) / ^( M_ij M_ij )
    M_ij = -2[ Delta_hat^2 |S^| S^^d_ij - Delta^2 ^( |S| S^d_ij ) ]
    L^d  = ( ^(ub_i ub_j) - ^ub_i ^ub_j )^d

where ub = <u> (grid filtered), S = strain(ub), S^ = strain(^ub). C_s^2 = C.

Neighbours for all filters are built once at the test-filter support (2*4h=8h)
with a periodic KD-tree on the dumped positions (no re-simulation).
"""
import sys
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from load_snapshots import load
from sph_ops import sph_filter, sph_gradient


def _dev(t):
    t = t.copy(); tr = t[:, 0, 0] + t[:, 1, 1]
    t[:, 0, 0] -= 0.5 * tr; t[:, 1, 1] -= 0.5 * tr
    return t


def _grad_tensor(sn, v, Hg, m, L):
    gx = sph_gradient(sn, v[:, 0], Hg, m, L, renorm=True)
    gy = sph_gradient(sn, v[:, 1], Hg, m, L, renorm=True)
    return np.stack([gx, gy], axis=1)            # G[:,a,b] = dv_a/dx_b


def _strain(G):
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    Sd = _dev(S)
    Smag = np.sqrt(2.0 * np.einsum("nij,nij->n", Sd, Sd))
    return Sd, Smag


def _neighbors(pts, L, radius):
    tree = cKDTree(pts % L, boxsize=L)
    pairs = tree.query_ball_point(pts % L, radius)
    deg = np.array([len(p) - 1 for p in pairs])
    off = np.zeros(len(pts) + 1, np.int32); off[1:] = np.cumsum(deg)
    idx = np.empty(off[-1], np.int32); w = 0
    for i, p in enumerate(pairs):
        for j in p:
            if j != i:
                idx[w] = j; w += 1
    return off, idx


def dynamic_coefficient(snap, meta, alpha=2.0):
    """Return dict with C(x), the dynamic model stress, and the true tau^d."""
    h = meta["h"]; m = meta["m"]; L = meta["L"]
    Hg = 2 * h                      # grid filter width Delta
    Ht = alpha * Hg                 # test filter width Delta_hat
    Hgrad = 2 * h                   # gradient kernel (fixed differentiation scale)
    pts = np.column_stack([snap.x, snap.y]).astype(np.float64)
    off, idx = _neighbors(pts, L, 2 * Ht)        # support of widest filter
    sn = replace(snap, nbr_offsets=off, nbr_indices=idx)
    u = np.stack([snap.vx, snap.vy], 1).astype(np.float64)

    # resolved field and TRUE grid-scale SGS stress (for comparison only)
    ub = sph_filter(sn, u, Hg, m, L)
    tau_true = _dev(sph_filter(sn, np.einsum("ni,nj->nij", u, u), Hg, m, L)
                    - np.einsum("ni,nj->nij", ub, ub))

    # grid strain and single-scale Smagorinsky tensor (the model that gets applied)
    Sd, Smag = _strain(_grad_tensor(sn, ub, Hgrad, m, L))
    Msmag = -2 * Hg ** 2 * Smag[:, None, None] * Sd          # tau ~ C * Msmag

    # test-filtered resolved field and its strain
    ubh = sph_filter(sn, ub, Ht, m, L)
    Sdh, Smagh = _strain(_grad_tensor(sn, ubh, Hgrad, m, L))

    # Germano M_ij = -2[ Dh^2 |S^| S^^d - D^2 ^(|S| S^d) ]   (entire-term test filter)
    gridterm = sph_filter(sn, Smag[:, None, None] * Sd, Ht, m, L)  # ^(|S| S^d)
    M = -2 * (Ht ** 2 * Smagh[:, None, None] * Sdh - Hg ** 2 * gridterm)
    M = _dev(M)

    # Leonard stress L^d = (^(ub ub) - ^ub ^ub)^d
    Ld = _dev(sph_filter(sn, np.einsum("ni,nj->nij", ub, ub), Ht, m, L)
              - np.einsum("ni,nj->nij", ubh, ubh))

    # local coefficient: test-filter the ENTIRE contracted scalars
    num = sph_filter(sn, np.einsum("nij,nij->n", Ld, M), Ht, m, L)
    den = sph_filter(sn, np.einsum("nij,nij->n", M, M), Ht, m, L)
    C = num / np.where(np.abs(den) > 1e-30, den, 1.0)         # C_s^2

    tau_dyn = _dev(C[:, None, None] * Msmag)                  # dynamic model stress
    return dict(C=C, Sd=Sd, Smag=Smag, Msmag=Msmag, tau_true=tau_true,
                tau_dyn=tau_dyn, Hg=Hg)


if __name__ == "__main__":
    b = load(sys.argv[1] if len(sys.argv) > 1 else "data/res_N16384.sph")
    meta = b.meta
    Cs2_all, eps_t, eps_d, corr_d = [], [], [], []
    for s in b.snapshots:
        r = dynamic_coefficient(s, meta, alpha=2.0)
        C, Sd, tt, td = r["C"], r["Sd"], r["tau_true"], r["tau_dyn"]
        Cs2_all.append(C)
        eps_t.append(-np.mean(np.einsum("nij,nij->n", tt, Sd)))   # true SGS dissipation
        eps_d.append(-np.mean(np.einsum("nij,nij->n", td, Sd)))   # dynamic model dissipation
        corr_d.append(np.corrcoef(tt.reshape(-1), td.reshape(-1))[0, 1])
    C = np.concatenate(Cs2_all)
    print(f"{sys.argv[1] if len(sys.argv)>1 else 'data/res_N16384.sph'}  N={meta['N']}  alpha=2  local test-filter avg")
    print(f"dynamic C (=C_s^2):  mean={C.mean():+.4f}  median={np.median(C):+.4f}  "
          f"backscatter(C<0)={100*np.mean(C<0):.0f}%  ->  C_s~{np.sqrt(max(C.mean(),0)):.3f}")
    eps_t = np.array(eps_t); eps_d = np.array(eps_d)
    print(f"mean SGS dissipation:  true={eps_t.mean():+.3e}  dynamic={eps_d.mean():+.3e}  "
          f"ratio={eps_d.mean()/eps_t.mean():+.2f}")
    print(f"pointwise corr(tau_dynamic, tau_true) = {np.mean(corr_d):+.3f}  "
          f"(vs scale-similarity 0.93)")
