"""Phase 1 — SGS target + invariant features.

Built to the author's spec (author is the domain expert on the SPH/SGS physics;
the ML in Phases 3-4 is the author-owned part). Decisions baked in here:

  - one explicit filter at width H = 2h (support 4h = dumped neighbour radius)
  - plain (non-Favre) filter, justified by measured rho'/rho0 ~ 1.4% (test_ops.py)
  - resolved velocity gradient taken at the filter scale H = 2h
  - filter width Delta = 2h  (only rescales C_d; must match the Phase 2 baselines)
  - C_d conditioning via the Lilly weight M:M (returned alongside C_d)
  - features {|S|, |omega|, div<u>, rho/rho0, |omega|/|S|}; tr(S^2) and h dropped

All quantities are per particle. Build on the generic operators in sph_ops.
"""

from __future__ import annotations

import numpy as np

from sph_ops import sph_filter, sph_gradient

_EPS = 1e-30  # guards strain-norm divisions; with M:M weighting these points vanish anyway


def _HmL(meta):
    h = meta["h"]
    return 2.0 * h, meta["m"], meta["L"]


def resolved_velocity(snap, meta) -> np.ndarray:
    """Resolved (2h-filtered) velocity <u>, shape (N, 2). Plain filter."""
    H, m, L = _HmL(meta)
    u = np.stack([snap.vx, snap.vy], axis=1).astype(np.float64)
    return sph_filter(snap, u, H, m, L)


def velocity_gradient(snap, meta, ubar: np.ndarray | None = None) -> np.ndarray:
    """Gradient tensor of the resolved field, G[n, a, b] = d<u>_a / dx_b, (N, 2, 2).
    Gradient taken at the filter scale H = 2h with the renormalized operator."""
    H, m, L = _HmL(meta)
    if ubar is None:
        ubar = resolved_velocity(snap, meta)
    gux = sph_gradient(snap, ubar[:, 0], H, m, L, renorm=True)  # [d ux/dx, d ux/dy]
    guy = sph_gradient(snap, ubar[:, 1], H, m, L, renorm=True)  # [d uy/dx, d uy/dy]
    N = snap.x.size
    G = np.empty((N, 2, 2))
    G[:, 0, 0] = gux[:, 0]; G[:, 0, 1] = gux[:, 1]
    G[:, 1, 0] = guy[:, 0]; G[:, 1, 1] = guy[:, 1]
    return G


def _strain(G: np.ndarray):
    """Deviatoric resolved strain S^d (N,2,2) and its norm |S| = sqrt(2 S^d:S^d) (N,)."""
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    tr = S[:, 0, 0] + S[:, 1, 1]
    Sdev = S.copy()
    Sdev[:, 0, 0] -= 0.5 * tr
    Sdev[:, 1, 1] -= 0.5 * tr
    Smag = np.sqrt(2.0 * np.einsum("nij,nij->n", Sdev, Sdev))
    return Sdev, Smag


def sgs_stress_deviatoric(snap, meta, ubar: np.ndarray | None = None) -> np.ndarray:
    """True deviatoric SGS stress tau^d (N, 2, 2).
    tau_ij = <u_i u_j> - <u_i><u_j>,  tau^d_ij = tau_ij - 1/2 tau_kk delta_ij (d=2)."""
    H, m, L = _HmL(meta)
    if ubar is None:
        ubar = resolved_velocity(snap, meta)
    u = np.stack([snap.vx, snap.vy], axis=1).astype(np.float64)
    prod = np.einsum("ni,nj->nij", u, u)        # raw u_i u_j
    prod_filt = sph_filter(snap, prod, H, m, L)  # <u_i u_j>
    tau = prod_filt - np.einsum("ni,nj->nij", ubar, ubar)
    tr = tau[:, 0, 0] + tau[:, 1, 1]
    tau[:, 0, 0] -= 0.5 * tr
    tau[:, 1, 1] -= 0.5 * tr
    return tau


def coefficient_target(snap, meta):
    """Dimensionless Smagorinsky coefficient target.

    Returns (C_d, weight), each (N,):
      C_d    = - (tau^d : S^d) / (Delta^2 |S|^3)        (C_d = C_s^2; <0 = backscatter)
      weight = M:M = 2 Delta^4 |S|^4   with M = -2 Delta^2 |S| S^d

    `weight` is the Lilly least-squares weight: minimizing sum_i weight_i
    (C_pred - C_d)^2 is equivalent to least squares on the stress itself, and
    automatically suppresses the low-|S| points where C_d is ill-conditioned.
    (Note weight ~ |S|^4, the exact M:M form, not the looser |S|^3 shorthand.)
    """
    H, m, L = _HmL(meta)
    Delta = 2.0 * meta["h"]
    ubar = resolved_velocity(snap, meta)
    G = velocity_gradient(snap, meta, ubar)
    Sdev, Smag = _strain(G)
    tau_d = sgs_stress_deviatoric(snap, meta, ubar)

    num = np.einsum("nij,nij->n", tau_d, Sdev)          # tau^d : S^d
    denom = Delta * Delta * Smag ** 3
    C_d = np.where(denom > _EPS, -num / np.where(denom > _EPS, denom, 1.0), 0.0)
    weight = 2.0 * Delta ** 4 * Smag ** 4               # M:M
    return C_d, weight


def invariant_features(snap, meta) -> dict[str, np.ndarray]:
    """Galilean- & rotation-invariant features from the resolved field, each (N,).

    Keys: Smag |S|, vortmag |omega|, divu div<u>, rho_norm rho/rho0,
          vort_strain_ratio |omega|/|S|.
    (tr(S^2)=1/2|S|^2 and constant h are intentionally excluded.)
    """
    ubar = resolved_velocity(snap, meta)
    G = velocity_gradient(snap, meta, ubar)
    Sdev, Smag = _strain(G)
    vortmag = np.abs(G[:, 1, 0] - G[:, 0, 1])           # |d uy/dx - d ux/dy|
    divu = G[:, 0, 0] + G[:, 1, 1]
    rho_norm = snap.rho.astype(np.float64) / meta["rho0"]
    ratio = vortmag / np.sqrt(Smag ** 2 + _EPS)
    return {
        "Smag": Smag,
        "vortmag": vortmag,
        "divu": divu,
        "rho_norm": rho_norm,
        "vort_strain_ratio": ratio,
    }
