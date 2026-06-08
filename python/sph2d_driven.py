"""Numba 2D weakly-compressible SPH with solenoidal OU forcing -> statistically
steady driven turbulence (a proper cascade, unlike the decaying shear layer).

Isothermal EOS, Balsara-GATED Monaghan AV (fires on compression, not shear, so
it doesn't kill the turbulent cascade), large-scale linear drag (Ekman friction
to balance the 2D inverse cascade), KDK leapfrog. Dumps primitives to .npz.

Usage: python sph2d_driven.py N t_warm dt_snap n_snap [seed] [out.npz]
"""
import sys, json, time
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
def build_grid(x, y, L, radius):
    ncx = max(3, int(L / radius)); cell = L / ncx
    head = np.full(ncx * ncx, -1, np.int64); nxt = np.empty(x.size, np.int64)
    for i in range(x.size):
        c = (int(y[i] // cell) % ncx) * ncx + (int(x[i] // cell) % ncx)
        nxt[i] = head[c]; head[c] = i
    return head, nxt, ncx, cell


@njit(parallel=True)
def density_grad(x, y, vx, vy, rho, P, divv, curl, m, h, sigma, cs02, L, head, nxt, ncx, cell):
    supp2 = (2 * h) ** 2
    for i in prange(x.size):
        xi, yi = x[i], y[i]; vxi, vyi = vx[i], vy[i]
        cx = int(xi // cell) % ncx; cy = int(yi // cell) % ncx
        acc = m * sigma
        # first pass for density needs rho_j; do density with self, then grads use rho after.
        for dj in range(-1, 2):
            ccy = (cy + dj) % ncx
            for di in range(-1, 2):
                j = head[ccy * ncx + ((cx + di) % ncx)]
                while j != -1:
                    if j != i:
                        dxx = _mimg(xi - x[j], L); dyy = _mimg(yi - y[j], L)
                        r2 = dxx * dxx + dyy * dyy
                        if 1e-14 < r2 < supp2:
                            q = np.sqrt(r2) / h
                            if q < 1:
                                acc += m * sigma * (1 - 1.5 * q * q + 0.75 * q ** 3)
                            else:
                                a = 2 - q; acc += m * sigma * 0.25 * a ** 3
                    j = nxt[j]
        rho[i] = acc; P[i] = cs02 * acc


@njit(parallel=True)
def grads(x, y, vx, vy, rho, divv, curl, m, h, sigma, L, head, nxt, ncx, cell):
    supp2 = (2 * h) ** 2
    for i in prange(x.size):
        xi, yi = x[i], y[i]; vxi, vyi = vx[i], vy[i]
        cx = int(xi // cell) % ncx; cy = int(yi // cell) % ncx
        dv = 0.0; cu = 0.0
        for dj in range(-1, 2):
            ccy = (cy + dj) % ncx
            for di in range(-1, 2):
                j = head[ccy * ncx + ((cx + di) % ncx)]
                while j != -1:
                    if j != i:
                        dxx = _mimg(xi - x[j], L); dyy = _mimg(yi - y[j], L)
                        r2 = dxx * dxx + dyy * dyy
                        if 1e-14 < r2 < supp2:
                            r = np.sqrt(r2); q = r / h
                            if q < 1:
                                dw = sigma * (-3 * q + 2.25 * q * q) / h
                            else:
                                a = 2 - q; dw = sigma * (-0.75 * a * a) / h
                            gwx = dw * dxx / r; gwy = dw * dyy / r
                            vol = m / rho[j]
                            dvx = vx[j] - vxi; dvy = vy[j] - vyi
                            dv += vol * (dvx * gwx + dvy * gwy)
                            cu += vol * (dvy * gwx - dvx * gwy)
                    j = nxt[j]
        divv[i] = dv; curl[i] = cu


@njit(parallel=True)
def forces(x, y, vx, vy, ax, ay, rho, P, divv, curl, m, h, sigma, cs0, alpha, eta2,
           L, head, nxt, ncx, cell):
    supp2 = (2 * h) ** 2; beta = 2 * alpha
    for i in prange(x.size):
        xi, yi = x[i], y[i]; vxi, vyi = vx[i], vy[i]
        Pir2 = P[i] / (rho[i] * rho[i])
        fbi = abs(divv[i]) / (abs(divv[i]) + abs(curl[i]) + 1e-4 * cs0 / h)
        cx = int(xi // cell) % ncx; cy = int(yi // cell) % ncx
        fx = 0.0; fy = 0.0
        for dj in range(-1, 2):
            ccy = (cy + dj) % ncx
            for di in range(-1, 2):
                j = head[ccy * ncx + ((cx + di) % ncx)]
                while j != -1:
                    if j != i:
                        dxx = _mimg(xi - x[j], L); dyy = _mimg(yi - y[j], L)
                        r2 = dxx * dxx + dyy * dyy
                        if 1e-14 < r2 < supp2:
                            r = np.sqrt(r2); q = r / h
                            if q < 1:
                                dw = sigma * (-3 * q + 2.25 * q * q) / h
                            else:
                                a = 2 - q; dw = sigma * (-0.75 * a * a) / h
                            gwx = dw * dxx / r; gwy = dw * dyy / r
                            coef = -(Pir2 + P[j] / (rho[j] * rho[j]))
                            dvx = vxi - vx[j]; dvy = vyi - vy[j]
                            vr = dvx * dxx + dvy * dyy
                            if vr < 0:
                                muav = h * vr / (r2 + eta2)
                                rhobar = 0.5 * (rho[i] + rho[j])
                                fbj = abs(divv[j]) / (abs(divv[j]) + abs(curl[j]) + 1e-4 * cs0 / h)
                                PiG = (-alpha * cs0 * muav + beta * muav * muav) / rhobar
                                coef -= PiG * 0.5 * (fbi + fbj)        # Balsara-gated
                            fx += m * coef * gwx; fy += m * coef * gwy
                    j = nxt[j]
        ax[i] = fx; ay[i] = fy


class Driven2D:
    def __init__(self, N, U=0.5, csfac=5, seed=1, alpha=1.0,
                 f_amp=0.4, f_T=0.5, drag=0.3, kmax2=4):
        self.L = 1.0; rho0 = 1.0
        nx = round(N ** 0.5); self.Nact = nx * nx
        dx = self.L / nx
        self.m = rho0 * self.L ** 2 / self.Nact
        self.h = 1.3 * dx; self.supp = 2 * self.h
        self.sigma = 10.0 / (7 * np.pi * self.h ** 2)
        self.cs0 = csfac * U; self.cs02 = self.cs0 ** 2
        self.dt = 0.10 * self.h / (self.cs0 + 2 * U)
        self.alpha = alpha; self.eta2 = 0.01 * self.h ** 2
        self.mach = U / self.cs0
        rng = np.random.default_rng(seed)
        lin = (np.arange(nx) + 0.5) * dx
        gx, gy = np.meshgrid(lin, lin, indexing="ij")
        self.x = ((gx.ravel() + (rng.random(self.Nact) - 0.5) * 0.02 * dx) % self.L)
        self.y = ((gy.ravel() + (rng.random(self.Nact) - 0.5) * 0.02 * dx) % self.L)
        self.vx = np.zeros(self.Nact); self.vy = np.zeros(self.Nact)
        self.ax = np.zeros(self.Nact); self.ay = np.zeros(self.Nact)
        self.rho = np.zeros(self.Nact); self.P = np.zeros(self.Nact)
        self.divv = np.zeros(self.Nact); self.curl = np.zeros(self.Nact)
        # forcing modes (low-k, solenoidal), OU-evolved
        modes = []
        K = int(np.sqrt(kmax2))
        for kx in range(-K, K + 1):
            for ky in range(-K, K + 1):
                k2 = kx * kx + ky * ky
                if 1 <= k2 <= kmax2 and (kx > 0 or (kx == 0 and ky > 0)):  # half-space
                    modes.append((kx, ky))
        self.modes = np.array(modes, dtype=np.float64)
        nm = len(modes)
        self.fa = np.zeros((nm, 2)); self.fb = np.zeros((nm, 2))  # OU amplitudes (cos, sin)
        self.f_amp = f_amp; self.f_T = f_T; self.drag = drag
        self.rng = rng; self.t = 0.0

    def _project(self, vecs):
        # make each amplitude perpendicular to its k (solenoidal)
        k = self.modes
        kn = k / np.linalg.norm(k, axis=1, keepdims=True)
        dot = np.sum(vecs * kn, axis=1, keepdims=True)
        return vecs - dot * kn

    def _ou_step(self):
        c = self.dt / self.f_T
        s = self.f_amp * np.sqrt(2 * self.dt / self.f_T)
        self.fa = self._project(self.fa * (1 - c) + s * self.rng.standard_normal(self.fa.shape))
        self.fb = self._project(self.fb * (1 - c) + s * self.rng.standard_normal(self.fb.shape))

    def _force_accel(self):
        ax = np.zeros(self.Nact); ay = np.zeros(self.Nact)
        for mi in range(len(self.modes)):
            kx, ky = self.modes[mi]
            th = 2 * np.pi * (kx * self.x + ky * self.y)
            ct = np.cos(th); st = np.sin(th)
            ax += self.fa[mi, 0] * ct + self.fb[mi, 0] * st
            ay += self.fa[mi, 1] * ct + self.fb[mi, 1] * st
        return ax, ay

    def _compute(self):
        g = build_grid(self.x, self.y, self.L, self.supp)
        density_grad(self.x, self.y, self.vx, self.vy, self.rho, self.P, self.divv, self.curl,
                     self.m, self.h, self.sigma, self.cs02, self.L, *g)
        grads(self.x, self.y, self.vx, self.vy, self.rho, self.divv, self.curl,
              self.m, self.h, self.sigma, self.L, *g)
        forces(self.x, self.y, self.vx, self.vy, self.ax, self.ay, self.rho, self.P,
               self.divv, self.curl, self.m, self.h, self.sigma, self.cs0, self.alpha,
               self.eta2, self.L, *g)

    def step(self):
        dt = self.dt
        self._compute()
        fax, fay = self._force_accel()
        axt = self.ax + fax - self.drag * self.vx
        ayt = self.ay + fay - self.drag * self.vy
        self.vx += 0.5 * dt * axt; self.vy += 0.5 * dt * ayt
        self.x = (self.x + dt * self.vx) % self.L
        self.y = (self.y + dt * self.vy) % self.L
        self._ou_step()
        self._compute()
        fax, fay = self._force_accel()
        axt = self.ax + fax - self.drag * self.vx
        ayt = self.ay + fay - self.drag * self.vy
        self.vx += 0.5 * dt * axt; self.vy += 0.5 * dt * ayt
        self.t += dt

    def vrms(self):
        return np.sqrt(np.mean(self.vx ** 2 + self.vy ** 2))


if __name__ == "__main__":
    N = int(sys.argv[1]); t_warm = float(sys.argv[2]); dt_snap = float(sys.argv[3]); n_snap = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    out = sys.argv[6] if len(sys.argv) > 6 else f"data/driven2d_N{N}_s{seed}.npz"
    sim = Driven2D(N, seed=seed)
    ws = round(t_warm / sim.dt); ss = round(dt_snap / sim.dt)
    t0 = time.time()
    sim.step()  # JIT
    snaps = {}; meta = dict(dim=2, L=sim.L, N=sim.Nact, m=sim.m, h=sim.h, supp=sim.supp,
                            dt=sim.dt, cs0=sim.cs0, mach=sim.mach, rho0=1.0, eos="isothermal",
                            kernel="cubic-spline", forcing="OU-solenoidal", drag=sim.drag,
                            av_alpha=sim.alpha, seed=seed)
    print(f"N={round(N**0.5)}^2  dt={sim.dt:.2e}  warmup {ws} steps", flush=True)
    for st in range(ws):
        sim.step()
        if st % 200 == 0:
            print(f"  warmup {st}/{ws}  t={sim.t:.2f}  vrms={sim.vrms():.3f} (M={sim.vrms()/sim.cs0:.2f})  {time.time()-t0:.0f}s", flush=True)
    for k in range(n_snap):
        for kk in ("x", "y", "vx", "vy", "rho", "P"):
            snaps[f"snap{k}_{kk}"] = getattr(sim, kk).astype(np.float32).copy()
        snaps[f"snap{k}_t"] = np.float64(sim.t)
        print(f"  snap {k} t={sim.t:.2f} vrms={sim.vrms():.3f}", flush=True)
        if k < n_snap - 1:
            for _ in range(ss):
                sim.step()
    meta["n_snapshots"] = n_snap
    np.savez(out, meta=json.dumps(meta), **snaps)
    print(f"wrote {out}  ({n_snap} snaps, {time.time()-t0:.0f}s)", flush=True)
