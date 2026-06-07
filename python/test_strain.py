"""Analytic validation of the gradient->strain path (the one path the
scale-similarity positive control does NOT exercise).

Assign synthetic linear velocity fields to the real particle positions and
check the recovered strain/vorticity against exact answers. Linear fields are
reproduced exactly by the renormalized gradient, so these should match to ~1e-9.

  pure shear   u = (gamma*y, 0):  S_xy = gamma/2, |S| = gamma, vorticity = -gamma
  solid rotation u = (-Om*y, Om*x): S = 0 (no strain!), vorticity = +2*Om
  pure strain  u = (gamma*x, -gamma*y): S = diag(gamma,-gamma), |S| = 2*gamma, vort = 0
"""
import sys
import numpy as np
from load_snapshots import load
from phase1 import velocity_gradient, _strain

b = load(sys.argv[1] if len(sys.argv) > 1 else "data/res_N16384.sph")
meta = b.meta; L = meta["L"]; H = 2 * meta["h"]
s = b.snapshots[0]
# interior particles only (linear fields are discontinuous across the periodic seam)
interior = (s.x > 2 * H) & (s.x < L - 2 * H) & (s.y > 2 * H) & (s.y < L - 2 * H)
x, y = s.x.astype(float), s.y.astype(float)

fail = 0
def chk(name, got, exp, tol=1e-6):
    global fail
    d = np.max(np.abs(got[interior] - exp))
    print(f"{'  ok  ' if d < tol else ' FAIL '} {name}: max dev {d:.2e}")
    if d >= tol:
        fail += 1

def strain_of(u):
    G = velocity_gradient(s, meta, ubar=u)
    Sdev, Smag = _strain(G)
    vort = G[:, 1, 0] - G[:, 0, 1]
    Sxy = 0.5 * (G[:, 0, 1] + G[:, 1, 0])
    return G, Sdev, Smag, vort, Sxy

g = 0.7
G, Sdev, Smag, vort, Sxy = strain_of(np.stack([g * y, 0 * x], 1))
print("pure shear u=(0.7 y, 0):")
chk("  |S| == gamma", Smag, g)
chk("  S_xy == gamma/2", Sxy, g / 2)
chk("  vorticity == -gamma", vort, -g)

Om = 0.5
G, Sdev, Smag, vort, Sxy = strain_of(np.stack([-Om * y, Om * x], 1))
print("solid-body rotation u=(-0.5 y, 0.5 x):")
chk("  |S| == 0 (no strain)", Smag, 0.0)
chk("  vorticity == +2*Om", vort, 2 * Om)

G, Sdev, Smag, vort, Sxy = strain_of(np.stack([g * x, -g * y], 1))
print("pure strain u=(0.7 x, -0.7 y):")
chk("  |S| == 2*gamma", Smag, 2 * g)
chk("  vorticity == 0", vort, 0.0)

print("\n" + ("ALL PASS" if fail == 0 else f"{fail} FAILURES"))
sys.exit(0 if fail == 0 else 1)
