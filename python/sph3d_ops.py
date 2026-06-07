"""Numba 3D SPH operators for a-priori analysis: XSPH smoothing (Rennehan+2019
eq 20) and the renormalized gradient. Cell-list neighbour search, so they run
over all ~262k particles in seconds. Used to build the resolved field, strain,
Leonard/Germano tensors, and the dynamic coefficient in 3D.
"""
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
def build_grid(x, y, z, L, radius):
    ncx = max(3, int(L / radius))
    cell = L / ncx
    head = np.full(ncx ** 3, -1, np.int64)
    nxt = np.empty(x.size, np.int64)
    for i in range(x.size):
        cx = int(x[i] // cell) % ncx
        cy = int(y[i] // cell) % ncx
        cz = int(z[i] // cell) % ncx
        c = (cz * ncx + cy) * ncx + cx
        nxt[i] = head[c]
        head[c] = i
    return head, nxt, ncx, cell


@njit(parallel=True)
def density_at(x, y, z, m, H, L, head, nxt, ncx, cell):
    """SPH density at smoothing length H (cubic spline), including self."""
    sigma = 1.0 / (np.pi * H ** 3)
    supp2 = (2 * H) ** 2
    rho = np.empty(x.size)
    for i in prange(x.size):
        xi, yi, zi = x[i], y[i], z[i]
        cx = int(xi // cell) % ncx; cy = int(yi // cell) % ncx; cz = int(zi // cell) % ncx
        acc = m * sigma
        for dk in range(-1, 2):
            ccz = (cz + dk) % ncx
            for dj in range(-1, 2):
                ccy = (cy + dj) % ncx
                for di in range(-1, 2):
                    j = head[(ccz * ncx + ((cy + dj) % ncx)) * ncx + ((cx + di) % ncx)]
                    while j != -1:
                        if j != i:
                            dxx = _mimg(xi - x[j], L); dyy = _mimg(yi - y[j], L); dzz = _mimg(zi - z[j], L)
                            r2 = dxx * dxx + dyy * dyy + dzz * dzz
                            if 1e-14 < r2 < supp2:
                                q = np.sqrt(r2) / H
                                if q < 1:
                                    acc += m * sigma * (1 - 1.5 * q * q + 0.75 * q ** 3)
                                else:
                                    a = 2 - q; acc += m * sigma * 0.25 * a ** 3
                        j = nxt[j]
        rho[i] = acc
    return rho


@njit(parallel=True)
def xsph_filter(x, y, z, field, H, m, L, eps, rhoH, head, nxt, ncx, cell):
    """XSPH smoothing (eq 20): f_bar = f + eps*sum_b (m/<rho_ab>)(f_b-f_a)W(r,H),
    <rho_ab> harmonic mean of filter-scale densities rhoH. field is (N, K)."""
    sigma = 1.0 / (np.pi * H ** 3)
    supp2 = (2 * H) ** 2
    N, K = field.shape
    out = field.copy()
    for i in prange(N):
        xi, yi, zi = x[i], y[i], z[i]
        cx = int(xi // cell) % ncx; cy = int(yi // cell) % ncx; cz = int(zi // cell) % ncx
        acc = np.zeros(K)
        for dk in range(-1, 2):
            ccz = (cz + dk) % ncx
            for dj in range(-1, 2):
                ccy = (cy + dj) % ncx
                for di in range(-1, 2):
                    j = head[(ccz * ncx + ccy) * ncx + ((cx + di) % ncx)]
                    while j != -1:
                        if j != i:
                            dxx = _mimg(xi - x[j], L); dyy = _mimg(yi - y[j], L); dzz = _mimg(zi - z[j], L)
                            r2 = dxx * dxx + dyy * dyy + dzz * dzz
                            if 1e-14 < r2 < supp2:
                                q = np.sqrt(r2) / H
                                if q < 1:
                                    w = sigma * (1 - 1.5 * q * q + 0.75 * q ** 3)
                                else:
                                    a = 2 - q; w = sigma * 0.25 * a ** 3
                                vol = m / (2.0 * rhoH[i] * rhoH[j] / (rhoH[i] + rhoH[j]))
                                wv = vol * w
                                for k in range(K):
                                    acc[k] += wv * (field[j, k] - field[i, k])
                        j = nxt[j]
        for k in range(K):
            out[i, k] = field[i, k] + eps * acc[k]
    return out


@njit(parallel=True)
def grad_renorm(x, y, z, field, Hg, m, L, rho, head, nxt, ncx, cell):
    """Renormalized (Bonet-Lok) gradient of field (N,K) -> (N,K,3).
    grad[a,k,:] = M^{-1} sum_b vol_b (f_b-f_a) gradW,  M = sum_b vol_b gradW (r_b-r_a)."""
    sigma = 1.0 / (np.pi * Hg ** 3)
    supp2 = (2 * Hg) ** 2
    N, K = field.shape
    out = np.zeros((N, K, 3))
    for i in prange(N):
        xi, yi, zi = x[i], y[i], z[i]
        cx = int(xi // cell) % ncx; cy = int(yi // cell) % ncx; cz = int(zi // cell) % ncx
        raw = np.zeros((K, 3))
        Mm = np.zeros((3, 3))
        for dk in range(-1, 2):
            ccz = (cz + dk) % ncx
            for dj in range(-1, 2):
                ccy = (cy + dj) % ncx
                for di in range(-1, 2):
                    j = head[(ccz * ncx + ccy) * ncx + ((cx + di) % ncx)]
                    while j != -1:
                        if j != i:
                            dx = _mimg(xi - x[j], L); dy = _mimg(yi - y[j], L); dz = _mimg(zi - z[j], L)
                            r2 = dx * dx + dy * dy + dz * dz
                            if 1e-14 < r2 < supp2:
                                r = np.sqrt(r2); q = r / Hg
                                if q < 1:
                                    dw = sigma * (-3 * q + 2.25 * q * q) / Hg
                                else:
                                    a = 2 - q; dw = sigma * (-0.75 * a * a) / Hg
                                gx = dw * dx / r; gy = dw * dy / r; gz = dw * dz / r
                                vol = m / rho[j]
                                g0, g1, g2 = gx, gy, gz
                                # r_b - r_a = -(dx,dy,dz)
                                Mm[0, 0] += vol * g0 * (-dx); Mm[0, 1] += vol * g0 * (-dy); Mm[0, 2] += vol * g0 * (-dz)
                                Mm[1, 0] += vol * g1 * (-dx); Mm[1, 1] += vol * g1 * (-dy); Mm[1, 2] += vol * g1 * (-dz)
                                Mm[2, 0] += vol * g2 * (-dx); Mm[2, 1] += vol * g2 * (-dy); Mm[2, 2] += vol * g2 * (-dz)
                                for k in range(K):
                                    df = field[j, k] - field[i, k]
                                    raw[k, 0] += vol * df * g0
                                    raw[k, 1] += vol * df * g1
                                    raw[k, 2] += vol * df * g2
                        j = nxt[j]
        # invert 3x3 Mm (closed form); fall back to identity if singular
        a, b, c = Mm[0, 0], Mm[0, 1], Mm[0, 2]
        d, e, f = Mm[1, 0], Mm[1, 1], Mm[1, 2]
        g, hh, ii = Mm[2, 0], Mm[2, 1], Mm[2, 2]
        det = a * (e * ii - f * hh) - b * (d * ii - f * g) + c * (d * hh - e * g)
        if abs(det) > 1e-20:
            inv = np.empty((3, 3))
            inv[0, 0] = (e * ii - f * hh) / det; inv[0, 1] = (c * hh - b * ii) / det; inv[0, 2] = (b * f - c * e) / det
            inv[1, 0] = (f * g - d * ii) / det; inv[1, 1] = (a * ii - c * g) / det; inv[1, 2] = (c * d - a * f) / det
            inv[2, 0] = (d * hh - e * g) / det; inv[2, 1] = (b * g - a * hh) / det; inv[2, 2] = (a * e - b * d) / det
            for k in range(K):
                for r_ in range(3):
                    out[i, k, r_] = inv[r_, 0] * raw[k, 0] + inv[r_, 1] * raw[k, 1] + inv[r_, 2] * raw[k, 2]
        else:
            for k in range(K):
                for r_ in range(3):
                    out[i, k, r_] = raw[k, r_]
    return out
