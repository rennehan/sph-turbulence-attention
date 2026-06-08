"""Independent grid/FFT a-priori SGS check (uses NO SPH operators).

Interpolate the SPH velocity onto a regular grid (CIC), then do the whole
a-priori calculation spectrally:
  - Gaussian filter at width Delta via FFT
  - resolved strain via spectral derivatives (i*k)
  - tau_ij = G*(u_i u_j) - ub_i ub_j
  - eps = -<tau^d : S^d>,  corr(M, tau)

If this agrees with the SPH pipeline (corr~0, eps<0), the SPH gradient/filter
are correct and the result is physics, not a method bug.
"""
import sys, json
import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    snaps = [{k: z[f"snap{i}_{k}"] for k in ("x", "y", "vx", "vy", "rho", "P")} | {"t": float(z[f"snap{i}_t"])}
             for i in range(meta["n_snapshots"])]
    return meta, snaps


def cic(x, y, f, Ng, L):
    sc = Ng / L
    fx = (x * sc) % Ng; fy = (y * sc) % Ng
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int)
    dx = fx - ix; dy = fy - iy
    g = np.zeros(Ng * Ng)
    for ox in (0, 1):
        wx = (1 - dx) if ox == 0 else dx
        cx = (ix + ox) % Ng
        for oy in (0, 1):
            wy = (1 - dy) if oy == 0 else dy
            cy = (iy + oy) % Ng
            np.add.at(g, cx * Ng + cy, f * wx * wy)
    return g.reshape(Ng, Ng)


def deviatoric(T):  # T dict of 'xx','xy','yy' arrays
    tr = 0.5 * (T["xx"] + T["yy"])
    return {"xx": T["xx"] - tr, "xy": T["xy"], "yy": T["yy"] - tr}


def ddot(A, B):  # symmetric 2x2 double contraction
    return A["xx"] * B["xx"] + 2 * A["xy"] * B["xy"] + A["yy"] * B["yy"]


def analyze_grid(s, meta, Ng=256, Delta_over_h=2.0):
    L, m, h = meta["L"], meta["m"], meta["h"]
    Delta = Delta_over_h * h
    x = s["x"].astype(np.float64); y = s["y"].astype(np.float64)
    vol = (m / s["rho"]).astype(np.float64)
    wt = cic(x, y, vol, Ng, L); wt[wt < 1e-12] = 1e-12
    ux = cic(x, y, vol * s["vx"], Ng, L) / wt
    uy = cic(x, y, vol * s["vy"], Ng, L) / wt

    k1 = 2 * np.pi * np.fft.fftfreq(Ng, d=L / Ng)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    Gh = np.exp(-k2 * Delta ** 2 / 24.0)            # Gaussian filter transfer

    def filt(field):
        return np.real(np.fft.ifft2(np.fft.fft2(field) * Gh))
    def ddx(field):
        return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(field)))
    def ddy(field):
        return np.real(np.fft.ifft2(1j * ky * np.fft.fft2(field)))

    ubx, uby = filt(ux), filt(uy)
    # true SGS stress (Gaussian filter of products minus product of filtered)
    tau = {"xx": filt(ux * ux) - ubx * ubx,
           "xy": filt(ux * uy) - ubx * uby,
           "yy": filt(uy * uy) - uby * uby}
    taud = deviatoric(tau)
    # resolved strain (spectral derivatives of filtered velocity)
    dxx = ddx(ubx); dxy = ddy(ubx); dyx = ddx(uby); dyy = ddy(uby)
    S = {"xx": dxx, "xy": 0.5 * (dxy + dyx), "yy": dyy}
    Sd = deviatoric(S)
    Smag = np.sqrt(2 * ddot(Sd, Sd))
    M = {k: -2 * Delta ** 2 * Smag * Sd[k] for k in Sd}
    # flatten symmetric tensors (weight xy by sqrt2 so dot products match) for corr
    def flat(T):
        return np.concatenate([T["xx"].ravel(), np.sqrt(2) * T["xy"].ravel(), T["yy"].ravel()])
    corr = np.corrcoef(flat(M), flat(taud))[0, 1]
    eps = -np.mean(ddot(taud, Sd))
    ksgs = 0.5 * np.mean(tau["xx"] + tau["yy"])
    kres = 0.5 * np.mean(ubx ** 2 + uby ** 2)
    return corr, eps, ksgs / kres


if __name__ == "__main__":
    meta, snaps = load(sys.argv[1])
    print(f"{sys.argv[1]}  grid/FFT a-priori (independent of SPH ops)")
    print(f"{'t':>5} {'Delta/h':>8} {'corr(M,tau)':>12} {'eps':>11} {'ksgs/kres':>10}")
    for s in snaps:
        for dh in (2.0, 4.0):
            c, e, kr = analyze_grid(s, meta, Ng=256, Delta_over_h=dh)
            print(f"{s['t']:5.2f} {dh:8.1f} {c:12.3f} {e:+11.2e} {kr:10.3f}")
