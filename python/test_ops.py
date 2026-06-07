"""Correctness checks for the generic SPH operators, on a real bundle.

  filter(const)  == const            (Shepard normalization)
  gradient(x)    == (1, 0)           (renorm reproduces linear fields)
  gradient(2y)   == (0, 2)

Also reports rho'/rho0, the number you use to justify a plain (non-Favre)
filter at this Mach number.

Run: python python/test_ops.py [bundle.sph]   (defaults to data/test_bundle.sph)
"""

import sys
from pathlib import Path

import numpy as np

from load_snapshots import load
from sph_ops import sph_filter, sph_gradient

path = sys.argv[1] if len(sys.argv) > 1 else "data/test_bundle.sph"
b = load(path)
m, L = b.meta["m"], b.meta["L"]
h = b.meta["h"]
H = 2.0 * h  # 2h test filter
s = b.snapshots[0]

fail = 0
def ok(cond, msg):
    global fail
    print(f"{'  ok  ' if cond else ' FAIL '} {msg}")
    if not cond:
        fail += 1

# filter reproduces a constant exactly
c = np.full(s.x.size, 3.7)
fc = sph_filter(s, c, H, m, L)
ok(np.allclose(fc, 3.7, atol=1e-9), f"filter(const) == const (max dev {np.abs(fc-3.7).max():.2e})")

# renormalized gradient reproduces linear fields. A globally-linear field (x) is
# discontinuous across the periodic seam, so only test particles whose entire
# neighbourhood (support radius 2H) is away from the wrap. Real fields are
# periodic, so the operator itself has no seam issue.
margin = 2.0 * H
interior = (s.x > margin) & (s.x < L - margin) & (s.y > margin) & (s.y < L - margin)
gx = sph_gradient(s, s.x.astype(np.float64), H, m, L, renorm=True)
err_x = np.abs(gx[interior] - np.array([1.0, 0.0]))
ok(err_x.max() < 1e-3, f"grad(x) == (1,0) interior (max err {err_x.max():.2e})")

gy = sph_gradient(s, 2.0 * s.y.astype(np.float64), H, m, L, renorm=True)
err_y = np.abs(gy[interior] - np.array([0.0, 2.0]))
ok(err_y.max() < 1e-3, f"grad(2y) == (0,2) interior (max err {err_y.max():.2e})")

# density fluctuation: justifies plain vs Favre filter
rms = np.concatenate([sn.rho for sn in b.snapshots])
rho0 = b.meta["rho0"]
frac = np.std(rms) / rho0
print(f"\nrho'/rho0 (rms over all snapshots) = {frac*100:.2f}%  "
      f"[M={b.meta['mach']:.2g}]  -> Favre corrections ~ O((rho'/rho0)^2) = {frac**2*100:.3f}%")

print("\n" + ("ALL PASS" if fail == 0 else f"{fail} FAILURES"))
sys.exit(0 if fail == 0 else 1)
