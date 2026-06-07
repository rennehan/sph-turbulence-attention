"""Diagnose the SGS signal vs time on a sweep bundle: for each snapshot report
the developed-turbulence indicators and whether Smagorinsky has any skill.

  vrms/U          turbulence intensity (is the flow alive?)
  k_sgs/k_res     subgrid energy fraction at the 2h filter
  corr            a-priori correlation( tau^d , Smagorinsky model M )
  C_d (Lilly)     -<tau^d:M>/<M:M> for this snapshot
"""
import sys
import numpy as np
from load_snapshots import load
from phase1 import resolved_velocity, velocity_gradient, _strain, sgs_stress_deviatoric
from sph_ops import sph_filter

path = sys.argv[1]
b = load(path); meta = b.meta
H = 2 * meta["h"]; m = meta["m"]; L = meta["L"]; U = meta["U"]; Delta = 2 * meta["h"]
print(f"{path}   alpha={meta['av_alpha']}  N={meta['N']}")
print(f"{'t':>6} {'vrms/U':>7} {'k_sgs/k_res':>11} {'corr':>7} {'C_d':>9}")
for s in b.snapshots:
    ub = resolved_velocity(s, meta)
    G = velocity_gradient(s, meta, ub); Sdev, Smag = _strain(G)
    tau = sgs_stress_deviatoric(s, meta, ub)
    M = -2 * Delta * Delta * Smag[:, None, None] * Sdev
    # k_sgs from the full (pre-deviatoric) trace
    u = np.stack([s.vx, s.vy], 1).astype(float)
    full = sph_filter(s, np.einsum("ni,nj->nij", u, u), H, m, L) - np.einsum("ni,nj->nij", ub, ub)
    ksgs = 0.5 * np.mean(full[:, 0, 0] + full[:, 1, 1])
    kres = 0.5 * np.mean(ub[:, 0] ** 2 + ub[:, 1] ** 2)
    corr = np.corrcoef(tau.reshape(-1), M.reshape(-1))[0, 1]
    Cd = np.sum(tau * M) / np.sum(M * M)
    vrms = np.sqrt(np.mean(s.vx ** 2 + s.vy ** 2))
    print(f"{s.t:6.2f} {vrms/U:7.3f} {ksgs/kres:11.3f} {corr:7.3f} {Cd:9.4f}")
