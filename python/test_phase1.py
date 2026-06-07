"""Sanity checks for the Phase 1 target + features on a real bundle.

  - <u> is smoother than u (smaller gradients)
  - tau^d is traceless
  - C_d: distribution, backscatter fraction, weight non-negative
  - features finite, sensible ranges

Run: python python/test_phase1.py [bundle.sph]
"""

import sys

import numpy as np

from load_snapshots import load
from phase1 import (resolved_velocity, sgs_stress_deviatoric,
                    coefficient_target, invariant_features)
from sph_ops import sph_filter

path = sys.argv[1] if len(sys.argv) > 1 else "data/test_bundle.sph"
b = load(path)
meta = b.meta
s = b.snapshots[0]

fail = 0
def ok(cond, msg):
    global fail
    print(f"{'  ok  ' if cond else ' FAIL '} {msg}")
    if not cond:
        fail += 1

# resolved field smoother than raw
ubar = resolved_velocity(s, meta)
raw_spread = np.std(np.stack([s.vx, s.vy], 1))
fil_spread = np.std(ubar)
ok(fil_spread <= raw_spread, f"<u> not amplified vs u (std {fil_spread:.4g} <= {raw_spread:.4g})")

# tau^d traceless
tau_d = sgs_stress_deviatoric(s, meta)
trace = tau_d[:, 0, 0] + tau_d[:, 1, 1]
scale = np.sqrt(np.mean(tau_d ** 2))
ok(np.max(np.abs(trace)) < 1e-6 * (scale + 1e-30),
   f"tau^d traceless (max|tr|/rms = {np.max(np.abs(trace))/(scale+1e-30):.2e})")
ok(np.allclose(tau_d[:, 0, 1], tau_d[:, 1, 0]), "tau^d symmetric")

# full SGS stress trace = 2 * k_sgs must be >= 0 everywhere (it's a variance)
ub0 = resolved_velocity(s, meta)
u0 = np.stack([s.vx, s.vy], 1).astype(float)
H = 2 * meta["h"]
full = sph_filter(s, np.einsum("ni,nj->nij", u0, u0), H, meta["m"], meta["L"]) \
    - np.einsum("ni,nj->nij", ub0, ub0)
ftr = full[:, 0, 0] + full[:, 1, 1]
ok(ftr.min() >= -1e-12 * abs(ftr.mean()), f"SGS trace (2 k_sgs) >= 0 everywhere (min {ftr.min():.2e})")

# C_d distribution across ALL snapshots (weighted by the Lilly weight)
Cd_all, w_all = [], []
for sn in b.snapshots:
    c, w = coefficient_target(sn, meta)
    Cd_all.append(c); w_all.append(w)
Cd = np.concatenate(Cd_all); w = np.concatenate(w_all)
ok(np.all(np.isfinite(Cd)), "C_d all finite")
ok(np.all(w >= 0), "weight non-negative")
wmean = np.average(Cd, weights=w + 1e-300)          # weighted mean coefficient
frac_back = np.average((Cd < 0), weights=w + 1e-300)  # weighted backscatter fraction
print(f"\nC_d  weighted-mean = {wmean:.4f}  (Smagorinsky C_s^2 ~ 0.026)"
      f"  -> C_s ~ {np.sqrt(abs(wmean)):.3f}")
print(f"C_d  unweighted median = {np.median(Cd):.4f}   "
      f"p5/p95 = {np.percentile(Cd,5):.3f}/{np.percentile(Cd,95):.3f}")
print(f"backscatter (C_d<0): {100*np.mean(Cd<0):.1f}% unweighted, "
      f"{100*frac_back:.1f}% weighted")
# NOTE: the SIGN/magnitude of <C_d> is a property of the DATA, not the code.
# A negative or ~0 weighted mean flags a non-physical SGS signal (decayed window,
# marginal resolution, or 2D backscatter) — a data-quality readout, not a code bug.
phys = 0.0 < wmean < 0.2
print(f"[{'physical' if phys else 'CHECK DATA'}] weighted-mean C_d = {wmean:.4f} "
      f"({'forward cascade' if phys else 'net backscatter / no clean signal'})")

# features
feats = invariant_features(s, meta)
for k, v in feats.items():
    ok(np.all(np.isfinite(v)), f"feature '{k}' finite  [{v.min():.3g}, {v.max():.3g}]")

print("\n" + ("ALL PASS" if fail == 0 else f"{fail} FAILURES"))
sys.exit(0 if fail == 0 else 1)
