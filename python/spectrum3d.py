"""3D kinetic-energy spectrum of an SPH bundle, to check for a real turbulent
cascade (inertial range ~ k^-5/3) vs an over-dissipated / decayed field.

Volume-weighted CIC deposit of velocity onto an Ng^3 grid, 3D FFT, shell-average.
"""
import sys, json
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def cic(x, y, z, vx, vy, vz, vol, Ng, L):
    momx = np.zeros((Ng, Ng, Ng)); momy = np.zeros((Ng, Ng, Ng)); momz = np.zeros((Ng, Ng, Ng))
    wt = np.zeros((Ng, Ng, Ng))
    sc = Ng / L
    for i in range(x.size):   # not prange: scatter race; keep serial (fast enough)
        fx = x[i] * sc; fy = y[i] * sc; fz = z[i] * sc
        ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
        dx = fx - ix; dy = fy - iy; dz = fz - iz
        w = vol[i]
        for ox in range(2):
            wx = (1 - dx) if ox == 0 else dx
            cx = (ix + ox) % Ng
            for oy in range(2):
                wy = (1 - dy) if oy == 0 else dy
                cy = (iy + oy) % Ng
                for oz in range(2):
                    wz = (1 - dz) if oz == 0 else dz
                    cz = (iz + oz) % Ng
                    ww = w * wx * wy * wz
                    wt[cx, cy, cz] += ww
                    momx[cx, cy, cz] += ww * vx[i]
                    momy[cx, cy, cz] += ww * vy[i]
                    momz[cx, cy, cz] += ww * vz[i]
    return momx, momy, momz, wt


def spectrum(s, meta, Ng=64):
    L, m = meta["L"], meta["m"]
    vol = (m / s["rho"]).astype(np.float64)
    momx, momy, momz, wt = cic(s["x"].astype(np.float64), s["y"].astype(np.float64),
                               s["z"].astype(np.float64), s["vx"].astype(np.float64),
                               s["vy"].astype(np.float64), s["vz"].astype(np.float64),
                               vol, Ng, L)
    wt[wt < 1e-12] = 1e-12
    vgx = momx / wt; vgy = momy / wt; vgz = momz / wt
    # remove mean
    for v in (vgx, vgy, vgz):
        v -= v.mean()
    fx = np.fft.fftn(vgx); fy = np.fft.fftn(vgy); fz = np.fft.fftn(vgz)
    E3 = 0.5 * (np.abs(fx) ** 2 + np.abs(fy) ** 2 + np.abs(fz) ** 2) / Ng ** 6
    k = np.fft.fftfreq(Ng, d=1.0 / Ng)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    kr = np.round(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)).astype(int)
    nb = Ng // 2
    E = np.zeros(nb + 1)
    for kk in range(1, nb + 1):
        E[kk] = E3[kr == kk].sum()
    return np.arange(nb + 1), E


if __name__ == "__main__":
    z = np.load(sys.argv[1], allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = meta["n_snapshots"]
    # use a mid/developed snapshot
    i = n // 2
    s = {k: z[f"snap{i}_{k}"] for k in ("x", "y", "z", "vx", "vy", "vz", "rho", "P")}
    kk, E = spectrum(s, meta, Ng=64)
    t = float(z[f"snap{i}_t"])
    print(f"{sys.argv[1]}  t={t:.2f}  {round(meta['N']**(1/3))}^3")
    print(f"k_h (filter ~2h) ~ {round(meta['L']/(2*meta['h']))}")
    print(f"{'k':>4} {'E(k)':>12} {'E*k^(5/3)':>12}")
    for j in range(1, len(E)):
        if E[j] > 0:
            print(f"{kk[j]:4d} {E[j]:12.3e} {E[j]*kk[j]**(5/3):12.3e}")
