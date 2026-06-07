"""Numba-accelerated 3D weakly-compressible SPH (headless data generation).

Same scheme as sim3d.js — periodic box, cubic-spline kernel, isothermal EOS
(P = cs^2 rho), Monaghan AV (standard mode), KDK leapfrog, 3D shear layer with
spanwise perturbations -> forward energy cascade. Hot loops are @njit(parallel).

Dumps raw primitives (x,y,z,vx,vy,vz,rho,P) to a .npz bundle; neighbours are
built downstream in Python (KD-tree) on a particle subset.

Usage:
    python sph3d_numba.py N t_warm dt_snap n_snap [seed] [alpha] [out.npz]
"""
import sys
import json
import time

import numpy as np
from numba import njit, prange


@njit(inline="always")
def _mimg(d, L):
    if d > 0.5 * L:
        return d - L
    if d < -0.5 * L:
        return d + L
    return d


@njit
def build_grid(x, y, z, L, cell, ncx, head, nxt):
    for c in range(head.size):
        head[c] = -1
    for i in range(x.size):
        cx = int(x[i] // cell) % ncx
        cy = int(y[i] // cell) % ncx
        cz = int(z[i] // cell) % ncx
        c = (cz * ncx + cy) * ncx + cx
        nxt[i] = head[c]
        head[c] = i


@njit(parallel=True)
def density(x, y, z, rho, P, m, h, supp, sigma, cs02, L, cell, ncx, head, nxt):
    supp2 = supp * supp
    W0 = sigma
    for i in prange(x.size):
        xi, yi, zi = x[i], y[i], z[i]
        cx = int(xi // cell) % ncx
        cy = int(yi // cell) % ncx
        cz = int(zi // cell) % ncx
        acc = m * W0
        for dk in range(-1, 2):
            ccz = (cz + dk) % ncx
            for dj in range(-1, 2):
                ccy = (cy + dj) % ncx
                for di in range(-1, 2):
                    ccx = (cx + di) % ncx
                    j = head[(ccz * ncx + ccy) * ncx + ccx]
                    while j != -1:
                        if j != i:
                            dxx = _mimg(xi - x[j], L)
                            dyy = _mimg(yi - y[j], L)
                            dzz = _mimg(zi - z[j], L)
                            r2 = dxx * dxx + dyy * dyy + dzz * dzz
                            if 1e-14 < r2 < supp2:
                                q = np.sqrt(r2) / h
                                if q < 1.0:
                                    acc += m * sigma * (1 - 1.5 * q * q + 0.75 * q * q * q)
                                else:
                                    a = 2 - q
                                    acc += m * sigma * 0.25 * a * a * a
                        j = nxt[j]
        rho[i] = acc
        P[i] = cs02 * acc


@njit(parallel=True)
def forces(x, y, z, vx, vy, vz, ax, ay, az, rho, P, m, h, supp, sigma, cs0,
           alpha, eta2, L, cell, ncx, head, nxt):
    supp2 = supp * supp
    beta = 2 * alpha
    for i in prange(x.size):
        xi, yi, zi = x[i], y[i], z[i]
        vxi, vyi, vzi = vx[i], vy[i], vz[i]
        Pir2 = P[i] / (rho[i] * rho[i])
        cx = int(xi // cell) % ncx
        cy = int(yi // cell) % ncx
        cz = int(zi // cell) % ncx
        fx = 0.0; fy = 0.0; fz = 0.0
        for dk in range(-1, 2):
            ccz = (cz + dk) % ncx
            for dj in range(-1, 2):
                ccy = (cy + dj) % ncx
                for di in range(-1, 2):
                    ccx = (cx + di) % ncx
                    j = head[(ccz * ncx + ccy) * ncx + ccx]
                    while j != -1:
                        if j != i:
                            dxx = _mimg(xi - x[j], L)
                            dyy = _mimg(yi - y[j], L)
                            dzz = _mimg(zi - z[j], L)
                            r2 = dxx * dxx + dyy * dyy + dzz * dzz
                            if 1e-14 < r2 < supp2:
                                r = np.sqrt(r2); q = r / h
                                if q < 1.0:
                                    dw = sigma * (-3 * q + 2.25 * q * q) / h
                                else:
                                    a = 2 - q; dw = sigma * (-0.75 * a * a) / h
                                gwx = dw * dxx / r; gwy = dw * dyy / r; gwz = dw * dzz / r
                                Pjr2 = P[j] / (rho[j] * rho[j])
                                dvx = vxi - vx[j]; dvy = vyi - vy[j]; dvz = vzi - vz[j]
                                coef = -(Pir2 + Pjr2)
                                vr = dvx * dxx + dvy * dyy + dvz * dzz
                                if vr < 0:
                                    muav = h * vr / (r2 + eta2)
                                    rhobar = 0.5 * (rho[i] + rho[j])
                                    coef -= (-alpha * cs0 * muav + beta * muav * muav) / rhobar
                                fx += m * coef * gwx
                                fy += m * coef * gwy
                                fz += m * coef * gwz
                        j = nxt[j]
        ax[i] = fx; ay[i] = fy; az[i] = fz


def make_ic(N, U, csfac, seed):
    L, rho0, delta = 1.0, 1.0, 0.04
    nx = round(N ** (1 / 3))
    Nact = nx ** 3
    dx = L / nx
    m = rho0 * L ** 3 / Nact
    h = 1.3 * dx
    rng = np.random.default_rng(seed)
    lin = (np.arange(nx) + 0.5) * dx
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    x = (gx.ravel() + (rng.random(Nact) - 0.5) * 0.02 * dx) % L
    y = (gy.ravel() + (rng.random(Nact) - 0.5) * 0.02 * dx) % L
    z = (gz.ravel() + (rng.random(Nact) - 0.5) * 0.02 * dx) % L
    tp = 2 * np.pi
    sh = np.tanh((y - 0.25) / delta) - np.tanh((y - 0.75) / delta) - 1
    vx = U * sh
    env = np.exp(-((y - 0.25) ** 2) / (2 * 0.05 ** 2)) + np.exp(-((y - 0.75) ** 2) / (2 * 0.05 ** 2))
    vy = 0.06 * U * np.sin(tp * 2 * x) * np.cos(tp * 1 * z) * env
    vz = 0.06 * U * np.sin(tp * 1 * x) * np.sin(tp * 2 * z) * env
    cs0 = csfac * U
    dt = 0.10 * h / (cs0 + 2 * U)
    supp = 2 * h
    ncx = max(3, int(L / supp))
    cell = L / ncx
    sigma = 1.0 / (np.pi * h ** 3)
    meta = dict(dim=3, L=L, N=Nact, m=m, h=h, supp=supp, dt=dt, U=U, csfac=csfac,
                mach=1 / csfac, rho0=rho0, cs0=cs0, eos="isothermal", kernel="cubic-spline",
                h_constant=True, sigma=sigma, ncx=ncx, cell=cell)
    state = dict(x=x.astype(np.float64), y=y.astype(np.float64), z=z.astype(np.float64),
                 vx=vx.astype(np.float64), vy=vy.astype(np.float64), vz=vz.astype(np.float64),
                 ax=np.zeros(Nact), ay=np.zeros(Nact), az=np.zeros(Nact),
                 rho=np.zeros(Nact), P=np.zeros(Nact),
                 head=np.empty(ncx ** 3, np.int64), nxt=np.empty(Nact, np.int64))
    return state, meta


def _compute(s, meta, alpha):
    build_grid(s["x"], s["y"], s["z"], meta["L"], meta["cell"], meta["ncx"], s["head"], s["nxt"])
    density(s["x"], s["y"], s["z"], s["rho"], s["P"], meta["m"], meta["h"], meta["supp"],
            meta["sigma"], meta["cs0"] ** 2, meta["L"], meta["cell"], meta["ncx"], s["head"], s["nxt"])
    forces(s["x"], s["y"], s["z"], s["vx"], s["vy"], s["vz"], s["ax"], s["ay"], s["az"],
           s["rho"], s["P"], meta["m"], meta["h"], meta["supp"], meta["sigma"], meta["cs0"],
           alpha, 0.01 * meta["h"] ** 2, meta["L"], meta["cell"], meta["ncx"], s["head"], s["nxt"])


def step(s, meta, alpha):
    dt, L = meta["dt"], meta["L"]
    _compute(s, meta, alpha)
    s["vx"] += 0.5 * dt * s["ax"]; s["vy"] += 0.5 * dt * s["ay"]; s["vz"] += 0.5 * dt * s["az"]
    s["x"] = (s["x"] + dt * s["vx"]) % L
    s["y"] = (s["y"] + dt * s["vy"]) % L
    s["z"] = (s["z"] + dt * s["vz"]) % L
    _compute(s, meta, alpha)
    s["vx"] += 0.5 * dt * s["ax"]; s["vy"] += 0.5 * dt * s["ay"]; s["vz"] += 0.5 * dt * s["az"]


def ke(s, meta):
    return 0.5 * meta["m"] * np.sum(s["vx"] ** 2 + s["vy"] ** 2 + s["vz"] ** 2)


def run(N, U, csfac, seed, alpha, warmup_steps, snap_steps, n_snap, onprog=None):
    s, meta = make_ic(N, U, csfac, seed)
    snaps = []
    nstep = 0
    for k in range(n_snap):
        target = warmup_steps if k == 0 else snap_steps
        for _ in range(target):
            step(s, meta, alpha); nstep += 1
            if onprog and nstep % 50 == 0:
                onprog(nstep, meta["dt"])
        snaps.append({n: s[n].astype(np.float32).copy() for n in
                      ("x", "y", "z", "vx", "vy", "vz", "rho", "P")})
        snaps[-1]["t"] = nstep * meta["dt"]
    return snaps, meta


if __name__ == "__main__":
    N = int(sys.argv[1]); t_warm = float(sys.argv[2]); dt_snap = float(sys.argv[3])
    n_snap = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    alpha = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
    out = sys.argv[7] if len(sys.argv) > 7 else f"data/sph3d_N{N}_s{seed}.npz"
    _, m0 = make_ic(min(N, 4096), U=0.5, csfac=5, seed=seed)  # warm up JIT on small N
    print(f"N={round(N**(1/3))}^3  dt depends on N; compiling JIT...", flush=True)
    dt = (0.10 * 1.3 * (1 / round(N ** (1 / 3))) / (5 * 0.5 + 2 * 0.5))
    ws = round(t_warm / dt); ss = round(dt_snap / dt)
    t0 = time.time()
    def prog(nstep, dtv):
        el = time.time() - t0
        print(f"  step {nstep}  t={nstep*dtv:.2f}  {el:.0f}s  ({nstep/el:.1f} steps/s)", flush=True)
    snaps, meta = run(N, 0.5, 5, seed, alpha, ws, ss, n_snap, onprog=prog)
    arrs = {}
    for i, sn in enumerate(snaps):
        for k in ("x", "y", "z", "vx", "vy", "vz", "rho", "P"):
            arrs[f"snap{i}_{k}"] = sn[k]
        arrs[f"snap{i}_t"] = np.float64(sn["t"])
    meta_out = {**meta, "av_alpha": alpha, "seed": seed, "n_snapshots": len(snaps)}
    np.savez(out, meta=json.dumps(meta_out), **arrs)
    print(f"wrote {out}  ({len(snaps)} snaps, {round(N**(1/3))}^3, {time.time()-t0:.0f}s)", flush=True)
