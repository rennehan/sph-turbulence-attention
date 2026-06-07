"""A-priori SGS analysis of 3D bundles, faithful to Rennehan+2019 (XSPH smoothing
eps=0.8, grid h_bar=2h, test h_hat=2 h_bar, local entire-term dynamic coeff).

Per snapshot reports:
  vrms/U, k_sgs/k_res            - flow state / subgrid fraction
  corr(M, tau)                   - does the strain (Smagorinsky) basis carry signal?
  corr(L, tau)                   - scale-similarity skill (positive control)
  corr(L, M)                     - orthogonality
  dynamic C                      - the 2019 coefficient (mean, %backscatter)
  eps = -<tau^d:S^d>             - mean SGS dissipation (>0 forward, <0 backscatter)

The 2D result: corr(M,tau)~0, corr(L,M)~0, backscatter. Expectation in 3D:
corr(M,tau)~0.3, C_s~0.1, net forward dissipation.
"""
import sys, json
import numpy as np
from sph3d_ops import build_grid, density_at, xsph_filter, grad_renorm


def dev(T):
    tr = (T[:, 0, 0] + T[:, 1, 1] + T[:, 2, 2]) / 3.0
    Td = T.copy()
    for d in range(3):
        Td[:, d, d] -= tr
    return Td

def corr(a, b):
    return np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]

def C(a):
    return np.ascontiguousarray(a, dtype=np.float64)

def load_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = meta["n_snapshots"]
    snaps = []
    for i in range(n):
        snaps.append({k: z[f"snap{i}_{k}"] for k in ("x", "y", "z", "vx", "vy", "vz", "rho", "P")}
                     | {"t": float(z[f"snap{i}_t"])})
    return meta, snaps


def analyze(s, meta):
    L, m, h = meta["L"], meta["m"], meta["h"]
    Hb, Hh, Hg = 2 * h, 4 * h, 2 * h          # grid filter, test filter, gradient scale
    eps = 0.8
    x, y, zc = C(s["x"]), C(s["y"]), C(s["z"])
    rho = C(s["rho"])
    u = np.ascontiguousarray(np.stack([s["vx"], s["vy"], s["vz"]], 1), dtype=np.float64)

    g4 = build_grid(x, y, zc, L, 4 * h)       # supports grid filter (Hb) + gradient (Hg)
    g8 = build_grid(x, y, zc, L, 8 * h)       # supports test filter (Hh)
    rhoHb = density_at(x, y, zc, m, Hb, L, *g4)
    rhoHh = density_at(x, y, zc, m, Hh, L, *g8)

    # resolved field and TRUE SGS stress
    ub = xsph_filter(x, y, zc, u, Hb, m, L, eps, rhoHb, *g4)
    uu = np.ascontiguousarray(np.einsum("ni,nj->nij", u, u).reshape(-1, 9))
    tau = (xsph_filter(x, y, zc, uu, Hb, m, L, eps, rhoHb, *g4).reshape(-1, 3, 3)
           - np.einsum("ni,nj->nij", ub, ub))
    taud = dev(tau)

    # grid strain + single-scale Smagorinsky tensor
    G = grad_renorm(x, y, zc, ub, Hg, m, L, rho, *g4)          # (N,3,3): d ub_k/dx_r
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    Sd = dev(S); Smag = np.sqrt(2 * np.einsum("nij,nij->n", Sd, Sd))
    M = dev(-2 * Hb ** 2 * Smag[:, None, None] * Sd)

    # test-scale: Leonard tensor, Germano alpha/beta, dynamic coefficient
    ubh = xsph_filter(x, y, zc, ub, Hh, m, L, eps, rhoHh, *g8)
    uubar = np.ascontiguousarray(np.einsum("ni,nj->nij", ub, ub).reshape(-1, 9))
    Ld = dev(xsph_filter(x, y, zc, uubar, Hh, m, L, eps, rhoHh, *g8).reshape(-1, 3, 3)
             - np.einsum("ni,nj->nij", ubh, ubh))
    Gh = grad_renorm(x, y, zc, ubh, Hg, m, L, rho, *g4)
    Sh = 0.5 * (Gh + np.transpose(Gh, (0, 2, 1)))
    Sdh = dev(Sh); Smagh = np.sqrt(2 * np.einsum("nij,nij->n", Sdh, Sdh))
    alpha = Hh ** 2 * Smagh[:, None, None] * Sdh
    beta = np.ascontiguousarray((Smag[:, None, None] * Sd).reshape(-1, 9))
    beta_h = xsph_filter(x, y, zc, beta, Hh, m, L, eps, rhoHh, *g8).reshape(-1, 3, 3)
    Mger = dev(-2 * (alpha - Hb ** 2 * beta_h))
    LM = C(np.einsum("nij,nij->n", Ld, Mger)[:, None])
    MM = C(np.einsum("nij,nij->n", Mger, Mger)[:, None])
    num = xsph_filter(x, y, zc, LM, Hh, m, L, eps, rhoHh, *g8)[:, 0]
    den = xsph_filter(x, y, zc, MM, Hh, m, L, eps, rhoHh, *g8)[:, 0]
    Cd = num / np.where(np.abs(den) > 1e-30, den, 1.0)

    ksgs = 0.5 * np.mean(tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2])
    kres = 0.5 * np.mean(np.sum(ub ** 2, 1))
    vrms = np.sqrt(np.mean(s["vx"] ** 2 + s["vy"] ** 2 + s["vz"] ** 2))
    return dict(t=s["t"], vrms_U=vrms / meta["U"], ksgs_kres=ksgs / kres,
                cMtau=corr(M, taud), cLtau=corr(Ld, taud), cLM=corr(Ld, M),
                Cmean=Cd.mean(), Cback=100 * np.mean(Cd < 0),
                eps=-np.mean(np.einsum("nij,nij->n", taud, Sd)))


if __name__ == "__main__":
    meta, snaps = load_npz(sys.argv[1])
    print(f"{sys.argv[1]}  {round(meta['N']**(1/3)):.0f}^3  M={meta['mach']:.2g}  alpha={meta['av_alpha']}")
    print(f"{'t':>5} {'vrms/U':>7} {'ksgs/kr':>8} {'cor(M,t)':>9} {'cor(L,t)':>9} {'cor(L,M)':>9} {'Cmean':>8} {'back%':>6} {'eps':>10}")
    for s in snaps:
        r = analyze(s, meta)
        print(f"{r['t']:5.2f} {r['vrms_U']:7.3f} {r['ksgs_kres']:8.3f} {r['cMtau']:9.3f} "
              f"{r['cLtau']:9.3f} {r['cLM']:9.3f} {r['Cmean']:8.4f} {r['Cback']:6.0f} {r['eps']:+10.2e}")
