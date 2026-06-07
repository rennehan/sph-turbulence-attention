"""Phase 1 — SGS target + invariant features.  *** AUTHOR-OWNED ***

This is the physics-ML boundary. The generic SPH operators in sph_ops.py
(sph_filter, sph_gradient) are plumbing you build ON TOP OF — but the target
definition and the invariant feature construction below are yours to implement
and defend. The bodies are intentionally left unimplemented.

Reference formulas (from the Phase 1 walkthrough) are in each docstring. The
decisions flagged are the ones to pin down and justify in the writeup.

Convention used throughout: one explicit filter at width H = 2h (support 4h),
matching the dumped neighbour radius. Same H in the target and in every
baseline (Phase 2) — its absolute value only rescales C_d.
"""

from __future__ import annotations

import numpy as np

from sph_ops import sph_filter, sph_gradient  # your building blocks


def resolved_velocity(snap, meta) -> np.ndarray:
    """The resolved (2h-filtered) velocity field <u>, shape (N, 2).

    This is the field an LES at scale 2h would carry; features AND the target
    are defined consistently from it.  Plain filter is justified here by the
    measured rho'/rho0 ~ 0.8% (see test_ops.py) -> Favre ≈ plain. If you switch
    to Favre, density-weight: <rho u>/<rho>.

        <u>_i = sph_filter(snap, u, H=2h, ...)
    """
    raise NotImplementedError("author-owned (Phase 1)")


def sgs_stress_deviatoric(snap, meta) -> np.ndarray:
    """True deviatoric SGS stress tau^d per particle, shape (N, 2, 2).

    Plain filter:   tau_ij = <u_i u_j> - <u_i><u_j>
    Deviatoric (2D, trace factor 1/d = 1/2):
                    tau^d_ij = tau_ij - 1/2 tau_kk delta_ij

    DECISION: Favre vs plain (plain, justified by rho'/rho0). The trace tau_kk
    is ~2x the subgrid KE and is folded into a modified pressure — only tau^d
    is modelled by the eddy-viscosity form.
    """
    raise NotImplementedError("author-owned (Phase 1)")


def coefficient_target(snap, meta) -> np.ndarray:
    """Dimensionless target coefficient C_d per particle, shape (N,).

    Resolved strain  S̄_ij = 1/2 (d<u>_i/dx_j + d<u>_j/dx_i), deviatoric S̄^d,
    norm |S̄| = sqrt(2 S̄^d:S̄^d). Per-particle Lilly projection of the true
    deviatoric stress onto the Smagorinsky direction gives the closed form:

        C_d = - ( tau^d : S̄^d ) / ( Delta^2 |S̄|^3 )           (C_d = C_s^2)

    C_d < 0 marks backscatter (energy subgrid->resolved) — the learned model
    can represent it; analytical Smagorinsky clips it to >= 0.

    DECISION (conditioning): the denominator -> 0 in low-strain regions, so C_d
    is noisy there. Prefer a strain-weighted loss (weight ~ |S̄|^3 or M:M, the
    Lilly-averaged estimator) over masking. Decide and justify.

    HONEST LIMITATION to state: targeting the scalar C_d concedes the tensor
    direction to S̄^d; the model learns amplitude+sign, not Smagorinsky's known
    structural/eigenvector misalignment. Fixing that needs a tensor-valued
    equivariant model = out of scope / future work.
    """
    raise NotImplementedError("author-owned (Phase 1)")


def invariant_features(snap, meta) -> dict[str, np.ndarray]:
    """Galilean- & rotation-invariant input features, each shape (N,).

    From the RESOLVED field <u> (scale consistency with the target):
        |S̄|        sqrt(2 S̄^d:S̄^d)        strain magnitude
        |omegā|    |d<u>_y/dx - d<u>_x/dy| filtered vorticity magnitude
        div<u>      d<u>_x/dx + d<u>_y/dy   dilatation (weakly compressible)
        rho/rho0                            dimensionless density

    PRUNED from the original list (justify in the writeup):
        tr(S^2) = 1/2 |S̄|^2  -> redundant with |S̄| in 2D. DROP.
        h        -> constant in this solver (no adaptive smoothing). DROP.

    DECISION: final feature set + normalization. Standardize (mean 0, var 1)
    using TRAIN-SPLIT statistics only (avoid leakage). |omegā|/|S̄| is a
    physically meaningful dimensionless alternative for one feature.
    """
    raise NotImplementedError("author-owned (Phase 1)")


if __name__ == "__main__":
    raise SystemExit(
        "phase1.py is the author-owned target/feature module — implement the "
        "functions above using sph_ops as building blocks."
    )
