"""Generic SPH numerical operators (plumbing) for Phase 1.

These are domain-agnostic: a Shepard-normalized kernel filter and an SPH
gradient, plus the cubic-spline kernel and periodic minimum-image helpers.
They take raw snapshot arrays + the constant mass/box and return fields.

They deliberately do NOT know about SGS stress, strain invariants, or the
target coefficient — that physics (the filter->stress->coefficient chain and
the invariant feature set) is built on top of these, by hand, in phase1.py.
"""

from __future__ import annotations

import numpy as np


# --- cubic spline kernel (2D), smoothing length H, support 2H ----------------
def kernel_w(r: np.ndarray, H: float) -> np.ndarray:
    q = np.asarray(r) / H
    sig = 10.0 / (7.0 * np.pi * H * H)
    w = np.zeros_like(q, dtype=np.float64)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    w[m1] = sig * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
    a = 2.0 - q[m2]
    w[m2] = sig * 0.25 * a ** 3
    return w


def kernel_dwdr(r: np.ndarray, H: float) -> np.ndarray:
    """Scalar dW/dr; gradW = dwdr * (r_vec / |r|)."""
    q = np.asarray(r) / H
    sig = 10.0 / (7.0 * np.pi * H * H)
    d = np.zeros_like(q, dtype=np.float64)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    d[m1] = sig * (-3.0 * q[m1] + 2.25 * q[m1] ** 2) / H
    a = 2.0 - q[m2]
    d[m2] = sig * (-0.75 * a ** 2) / H
    return d


def _pairs(snap):
    """Flatten the CSR neighbour list into (src, dst) index arrays."""
    off = snap.nbr_offsets
    deg = np.diff(off)
    src = np.repeat(np.arange(snap.x.size), deg)
    dst = snap.nbr_indices
    return src, dst


def _disp(snap, src, dst, L):
    """Periodic minimum-image displacement r_src - r_dst, and its length."""
    dx = snap.x[src] - snap.x[dst]
    dy = snap.y[src] - snap.y[dst]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    r = np.sqrt(dx * dx + dy * dy)
    return dx, dy, r


def sph_filter(snap, field: np.ndarray, H: float, m: float, L: float) -> np.ndarray:
    """Shepard-normalized SPH filter <field> at width H (use H = 2h for a 2h test
    filter). `field` is (N,) or (N, k). Self term (W(0)) is included explicitly.
    Returns the filtered field, same shape as `field`.
    """
    field = np.asarray(field, dtype=np.float64)
    N = snap.x.size
    vec = field.reshape(N, -1)
    src, dst = _pairs(snap)
    _, _, r = _disp(snap, src, dst, L)
    w = kernel_w(r, H)
    vol = m / snap.rho[dst].astype(np.float64)
    wv = vol * w  # per-pair weight

    num = np.zeros_like(vec)
    den = np.zeros(N)
    np.add.at(num, src, wv[:, None] * vec[dst])
    np.add.at(den, src, wv)

    # self contribution
    w0 = float(kernel_w(np.array([0.0]), H)[0])
    vol_i = m / snap.rho.astype(np.float64)
    num += (vol_i * w0)[:, None] * vec
    den += vol_i * w0

    out = num / den[:, None]
    return out.reshape(field.shape)


def xsph_smooth(snap, field: np.ndarray, H: float, m: float, L: float,
                eps: float = 0.8) -> np.ndarray:
    """XSPH smoothing of Rennehan et al. (2019), eq 20:

        f_bar_a = f_a + eps * sum_b (m_b / <rho_ab>) (f_b - f_a) W(r_ab, H)

    with <rho_ab> the HARMONIC mean of the filter-scale densities rho(H), and
    eps=0.8 (high-k modes damped to (1-eps), not removed). Constants and (to
    first order) linear fields are preserved. `field` is (N,) or (N, ...).
    Apply twice (H, then 2H) for the doubly-filtered/test-filtered quantities.
    """
    field = np.asarray(field, dtype=np.float64)
    N = snap.x.size
    vec = field.reshape(N, -1)
    src, dst = _pairs(snap)
    _, _, r = _disp(snap, src, dst, L)
    w = kernel_w(r, H)
    # density at the filter scale H (include self term)
    w0 = float(kernel_w(np.array([0.0]), H)[0])
    rhoH = np.full(N, m * w0)
    np.add.at(rhoH, src, m * w)
    rho_ab = 2.0 * rhoH[src] * rhoH[dst] / (rhoH[src] + rhoH[dst])   # harmonic mean
    wv = (m / rho_ab) * w
    delta = np.zeros_like(vec)
    np.add.at(delta, src, wv[:, None] * (vec[dst] - vec[src]))
    out = vec + eps * delta
    return out.reshape(field.shape)


def sph_gradient(snap, field: np.ndarray, H: float, m: float, L: float,
                 renorm: bool = True) -> np.ndarray:
    """SPH gradient of a scalar `field` (N,). Returns (N, 2) = [dA/dx, dA/dy].

    Uses the difference form (A_j - A_i) grad W_ij. With renorm=True applies the
    Bonet-Lok first-order correction matrix B_i so that linear fields are
    reproduced exactly on disordered particles (recommended). `field` must be
    a scalar field; call once per velocity component to assemble the Jacobian.
    """
    field = np.asarray(field, dtype=np.float64)
    N = snap.x.size
    src, dst = _pairs(snap)
    dx, dy, r = _disp(snap, src, dst, L)
    safe = r > 1e-12
    dwdr = kernel_dwdr(r, H)
    gx = np.zeros_like(r); gy = np.zeros_like(r)
    gx[safe] = dwdr[safe] * dx[safe] / r[safe]      # grad W components (r_i - r_j)
    gy[safe] = dwdr[safe] * dy[safe] / r[safe]
    vol = m / snap.rho[dst].astype(np.float64)

    dA = field[dst] - field[src]
    grad = np.zeros((N, 2))
    np.add.at(grad[:, 0], src, vol * dA * gx)
    np.add.at(grad[:, 1], src, vol * dA * gy)

    if not renorm:
        return grad

    # raw_grad = M @ true_grad for linear fields, with
    #   M_ab = sum_j vol_j (gradW_ij)_a (r_j - r_i)_b ,  r_j - r_i = -(dx,dy)
    # so the correction is B = M^{-1}.
    B = np.zeros((N, 2, 2))
    np.add.at(B[:, 0, 0], src, vol * gx * (-dx))
    np.add.at(B[:, 0, 1], src, vol * gx * (-dy))
    np.add.at(B[:, 1, 0], src, vol * gy * (-dx))
    np.add.at(B[:, 1, 1], src, vol * gy * (-dy))
    # invert per-particle 2x2; fall back to identity where singular
    det = B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] * B[:, 1, 0]
    good = np.abs(det) > 1e-10
    Binv = np.zeros_like(B)
    Binv[good, 0, 0] = B[good, 1, 1] / det[good]
    Binv[good, 1, 1] = B[good, 0, 0] / det[good]
    Binv[good, 0, 1] = -B[good, 0, 1] / det[good]
    Binv[good, 1, 0] = -B[good, 1, 0] / det[good]
    Binv[~good] = np.eye(2)
    return np.einsum("nij,nj->ni", Binv, grad)
