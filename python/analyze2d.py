"""A-priori SGS analysis of 2D driven-turbulence bundles (.npz).

Reports the energy spectrum (is there a cascade?) and, per snapshot, the SGS
correlations using BOTH gradient estimators:
  M_dW  : strain from the kernel-derivative (Bonet-Lok) gradient
  M_mls : strain from the GIZMO moving-least-squares gradient (eq 12 of the paper)

  corr(M,tau), corr(L,tau), corr(L,M), eps = -<tau^d:S^d>, dynamic C
"""
import sys, json
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from sph_ops import xsph_smooth, sph_gradient, mls_gradient


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    snaps = [{k: z[f"snap{i}_{k}"] for k in ("x", "y", "vx", "vy", "rho", "P")} | {"t": float(z[f"snap{i}_t"])}
             for i in range(meta["n_snapshots"])]
    return meta, snaps


def make_snap(s, L, radius):
    x = s["x"].astype(np.float64); y = s["y"].astype(np.float64)
    tree = cKDTree(np.column_stack([x % L, y % L]), boxsize=L)
    pairs = tree.query_ball_point(np.column_stack([x % L, y % L]), radius)
    deg = np.array([len(p) - 1 for p in pairs])
    off = np.zeros(len(x) + 1, np.int32); off[1:] = np.cumsum(deg)
    idx = np.empty(off[-1], np.int32); w = 0
    for i, p in enumerate(pairs):
        for j in p:
            if j != i:
                idx[w] = j; w += 1
    return SimpleNamespace(x=x, y=y, rho=s["rho"].astype(np.float64),
                           nbr_offsets=off, nbr_indices=idx)


def dev2(T):
    tr = 0.5 * (T[:, 0, 0] + T[:, 1, 1])
    Td = T.copy(); Td[:, 0, 0] -= tr; Td[:, 1, 1] -= tr
    return Td

def corr(a, b):
    return np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]

def strain_from_G(gx, gy):
    G = np.stack([gx, gy], 1)                       # (N,2,2): d v_k/dx_r
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    Sd = dev2(S); Sm = np.sqrt(2 * np.einsum("nij,nij->n", Sd, Sd))
    return Sd, Sm


def spectrum(s, meta, Ng=128):
    L, m = meta["L"], meta["m"]
    x = s["x"].astype(np.float64); y = s["y"].astype(np.float64)
    vol = (m / s["rho"]).astype(np.float64)
    sc = Ng / L
    fx = x * sc; fy = y * sc
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int)
    dx = fx - ix; dy = fy - iy
    momx = np.zeros(Ng * Ng); momy = np.zeros(Ng * Ng); wt = np.zeros(Ng * Ng)
    for ox in (0, 1):
        wx = (1 - dx) if ox == 0 else dx
        cx = (ix + ox) % Ng
        for oy in (0, 1):
            wy = (1 - dy) if oy == 0 else dy
            cy = (iy + oy) % Ng
            cidx = cx * Ng + cy; ww = vol * wx * wy
            np.add.at(wt, cidx, ww)
            np.add.at(momx, cidx, ww * s["vx"]); np.add.at(momy, cidx, ww * s["vy"])
    wt[wt < 1e-12] = 1e-12
    vgx = (momx / wt).reshape(Ng, Ng); vgy = (momy / wt).reshape(Ng, Ng)
    vgx -= vgx.mean(); vgy -= vgy.mean()
    Fx = np.fft.fft2(vgx); Fy = np.fft.fft2(vgy)
    E2 = 0.5 * (np.abs(Fx) ** 2 + np.abs(Fy) ** 2) / Ng ** 4
    k = np.fft.fftfreq(Ng, d=1.0 / Ng)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kr = np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
    E = np.array([E2[kr == kk].sum() for kk in range(Ng // 2 + 1)])
    return E


def analyze(s, meta):
    L, m, h = meta["L"], meta["m"], meta["h"]
    Hb, Hh = 2 * h, 4 * h
    sn = make_snap(s, L, 8 * h)
    u = np.stack([s["vx"], s["vy"]], 1).astype(np.float64)
    rhoHb_dummy = None
    ub = xsph_smooth(sn, u, Hb, m, L)
    tau = dev2(xsph_smooth(sn, np.einsum("ni,nj->nij", u, u).reshape(-1, 4), Hb, m, L).reshape(-1, 2, 2)
               - np.einsum("ni,nj->nij", ub, ub))
    Delta = Hb
    out = {"t": s["t"]}
    # both gradient estimators
    for tag, gradfn in [("dW", lambda f: sph_gradient(sn, f, Hb, m, L, renorm=True)),
                        ("mls", lambda f: mls_gradient(sn, f, Hb, L))]:
        Sd, Sm = strain_from_G(gradfn(ub[:, 0]), gradfn(ub[:, 1]))
        M = -2 * Delta ** 2 * Sm[:, None, None] * Sd
        out[f"cMtau_{tag}"] = corr(M, tau)
        out[f"eps_{tag}"] = -np.mean(np.einsum("nij,nij->n", tau, Sd))
    # scale similarity (Leonard)
    ubh = xsph_smooth(sn, ub, Hh, m, L)
    Ld = dev2(xsph_smooth(sn, np.einsum("ni,nj->nij", ub, ub).reshape(-1, 4), Hh, m, L).reshape(-1, 2, 2)
              - np.einsum("ni,nj->nij", ubh, ubh))
    Sd, Sm = strain_from_G(mls_gradient(sn, ub[:, 0], Hb, L), mls_gradient(sn, ub[:, 1], Hb, L))
    M = -2 * Delta ** 2 * Sm[:, None, None] * Sd
    out["cLtau"] = corr(Ld, tau); out["cLM"] = corr(Ld, M)
    ksgs = 0.5 * (tau[:, 0, 0] + tau[:, 1, 1]).mean(); kres = 0.5 * np.mean(np.sum(ub ** 2, 1))
    out["ksgs_kres"] = ksgs / kres
    out["vrms_M"] = np.sqrt(np.mean(s["vx"] ** 2 + s["vy"] ** 2)) / meta["cs0"]
    return out


if __name__ == "__main__":
    meta, snaps = load(sys.argv[1])
    print(f"{sys.argv[1]}  {round(meta['N']**0.5)}^2  forcing={meta.get('forcing')}")
    E = spectrum(snaps[-1], meta)
    kD = round(meta["L"] / (2 * meta["h"]))
    print(f"spectrum (last snap)  k_filter(2h)~{kD}:")
    print("   " + "  ".join(f"k{k}={E[k]:.1e}" for k in (1, 2, 4, 8, 16, 32) if k < len(E)))
    sl = np.polyfit(np.log([4, 8, 16]), np.log([E[4], E[8], E[16]]), 1)[0]
    print(f"   spectral slope (k=4..16) ~ {sl:.2f}   (2D enstrophy cascade ~ -3, energy ~ -5/3)")
    print(f"{'t':>5} {'M':>5} {'ksgs/kr':>8} {'cMt_dW':>7} {'cMt_mls':>8} {'cLtau':>7} {'cLM':>7} "
          f"{'eps_dW':>10} {'eps_mls':>10}")
    for s in snaps:
        r = analyze(s, meta)
        print(f"{r['t']:5.2f} {r['vrms_M']:5.2f} {r['ksgs_kres']:8.3f} {r['cMtau_dW']:7.3f} "
              f"{r['cMtau_mls']:8.3f} {r['cLtau']:7.3f} {r['cLM']:7.3f} {r['eps_dW']:+10.2e} {r['eps_mls']:+10.2e}")
