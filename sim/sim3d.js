// ===========================================================================
// 3D weakly-compressible SPH — headless data generation (no React, no dashboard).
//
// Periodic box, cubic-spline kernel, summation density, ISOTHERMAL EOS
// (P = cs^2 rho, matching the 3D ideal-isothermal-gas tests of Rennehan et al.
// 2019), symmetric pressure force, blunt Monaghan AV ("standard" mode only),
// KDK leapfrog. 3D shear layer with spanwise perturbations -> 3D turbulence
// with a forward energy cascade (unlike the 2D inverse-cascade case).
//
// Dumps RAW PRIMITIVES ONLY (x,y,z,vx,vy,vz,rho,P). In 3D the neighbour counts
// at the test-filter scale are huge, so neighbours are built downstream in
// Python (KD-tree) on a subset of target particles. Reuses serializeBundle.
// ===========================================================================

import { mulberry32, serializeBundle } from "./src/sim.js";

export function createSim3D(opts) {
  const L = 1.0;
  const N = opts.N;
  const rho0 = 1.0;
  const U = opts.U;
  const delta = 0.04;                       // shear-layer thickness
  const nx = Math.round(Math.cbrt(N));
  const Nact = nx * nx * nx;
  const dx = L / nx;
  const m = (rho0 * L * L * L) / Nact;
  const h = 1.3 * dx;
  const supp = 2 * h;
  const csfac = opts.csfac || 5;
  const cs0 = csfac * U;                     // isothermal sound speed; Mach = 1/csfac
  const dt = 0.10 * h / (cs0 + 2 * U);
  const eta2 = 0.01 * h * h;
  const sigma = 1.0 / (Math.PI * h * h * h); // 3D cubic-spline normalization

  const x = new Float32Array(Nact), y = new Float32Array(Nact), z = new Float32Array(Nact);
  const vx = new Float32Array(Nact), vy = new Float32Array(Nact), vz = new Float32Array(Nact);
  const ax = new Float32Array(Nact), ay = new Float32Array(Nact), az = new Float32Array(Nact);
  const rho = new Float32Array(Nact), P = new Float32Array(Nact);

  // initialise: lattice + double-tanh shear (in y) + 3D seed perturbations
  const rng = mulberry32(opts.seed | 0);
  let p = 0;
  for (let k = 0; k < nx; k++) {
    for (let j = 0; j < nx; j++) {
      for (let i = 0; i < nx; i++) {
        const px = (i + 0.5) * dx + (rng() - 0.5) * 0.02 * dx;
        const py = (j + 0.5) * dx + (rng() - 0.5) * 0.02 * dx;
        const pz = (k + 0.5) * dx + (rng() - 0.5) * 0.02 * dx;
        x[p] = (px % L + L) % L; y[p] = (py % L + L) % L; z[p] = (pz % L + L) % L;
        const sh = Math.tanh((y[p] - 0.25) / delta) - Math.tanh((y[p] - 0.75) / delta) - 1;
        vx[p] = U * sh;
        const e1 = Math.exp(-((y[p] - 0.25) ** 2) / (2 * 0.05 * 0.05));
        const e2 = Math.exp(-((y[p] - 0.75) ** 2) / (2 * 0.05 * 0.05));
        const env = e1 + e2;
        // 3D perturbation: x- and z-dependent modes seed spanwise (3D) breakdown
        vy[p] = 0.06 * U * Math.sin(TWO_PI * 2 * x[p]) * Math.cos(TWO_PI * 1 * z[p]) * env;
        vz[p] = 0.06 * U * Math.sin(TWO_PI * 1 * x[p]) * Math.sin(TWO_PI * 2 * z[p]) * env;
        p++;
      }
    }
  }

  const ncx = Math.max(3, Math.floor(L / supp));
  const cell = L / ncx;
  const head = new Int32Array(ncx * ncx * ncx);
  const next = new Int32Array(Nact);
  const cidx = (cx, cy, cz) => ((cz % ncx + ncx) % ncx) * ncx * ncx
    + ((cy % ncx + ncx) % ncx) * ncx + ((cx % ncx + ncx) % ncx);

  function buildGrid() {
    head.fill(-1);
    for (let i = 0; i < Nact; i++) {
      const c = cidx(Math.floor(x[i] / cell), Math.floor(y[i] / cell), Math.floor(z[i] / cell));
      next[i] = head[c]; head[c] = i;
    }
  }
  const mimg = (d) => (d > 0.5 * L ? d - L : d < -0.5 * L ? d + L : d);

  function kernelW(r) {
    const q = r / h;
    if (q >= 2) return 0;
    if (q < 1) return sigma * (1 - 1.5 * q * q + 0.75 * q * q * q);
    const a = 2 - q; return sigma * 0.25 * a * a * a;
  }
  function kernelDW(r) {
    const q = r / h;
    if (q >= 2) return 0;
    let f;
    if (q < 1) f = -3 * q + 2.25 * q * q;
    else { const a = 2 - q; f = -0.75 * a * a; }
    return sigma * f / h;
  }

  function forEachNeighbor(i, cb) {
    const cx = Math.floor(x[i] / cell), cy = Math.floor(y[i] / cell), cz = Math.floor(z[i] / cell);
    for (let dk = -1; dk <= 1; dk++)
      for (let dj = -1; dj <= 1; dj++)
        for (let di = -1; di <= 1; di++) {
          let jp = head[cidx(cx + di, cy + dj, cz + dk)];
          while (jp !== -1) { cb(jp); jp = next[jp]; }
        }
  }

  function density() {
    for (let i = 0; i < Nact; i++) {
      let r = m * kernelW(0);
      forEachNeighbor(i, (j) => {
        if (j === i) return;
        const dxx = mimg(x[i] - x[j]), dyy = mimg(y[i] - y[j]), dzz = mimg(z[i] - z[j]);
        const r2 = dxx * dxx + dyy * dyy + dzz * dzz;
        if (r2 < supp * supp && r2 > 1e-14) r += m * kernelW(Math.sqrt(r2));
      });
      rho[i] = r;
      P[i] = cs0 * cs0 * r;            // isothermal EOS
    }
  }

  function forces(ctrl) {
    const alpha = ctrl.alpha, beta = 2 * alpha;
    for (let i = 0; i < Nact; i++) {
      let fxi = 0, fyi = 0, fzi = 0;
      const Pi_rho2 = P[i] / (rho[i] * rho[i]);
      forEachNeighbor(i, (j) => {
        if (j === i) return;
        const dxx = mimg(x[i] - x[j]), dyy = mimg(y[i] - y[j]), dzz = mimg(z[i] - z[j]);
        const r2 = dxx * dxx + dyy * dyy + dzz * dzz;
        if (r2 >= supp * supp || r2 < 1e-14) return;
        const r = Math.sqrt(r2);
        const dw = kernelDW(r);
        const gwx = dw * dxx / r, gwy = dw * dyy / r, gwz = dw * dzz / r;
        const Pj_rho2 = P[j] / (rho[j] * rho[j]);
        const dvx = vx[i] - vx[j], dvy = vy[i] - vy[j], dvz = vz[i] - vz[j];
        let coef = -(Pi_rho2 + Pj_rho2);
        const vr = dvx * dxx + dvy * dyy + dvz * dzz;     // approaching if < 0
        if (vr < 0) {
          const muav = h * vr / (r2 + eta2);
          const rhobar = 0.5 * (rho[i] + rho[j]);
          const PiG = (-alpha * cs0 * muav + beta * muav * muav) / rhobar;
          coef -= PiG;
        }
        fxi += m * coef * gwx; fyi += m * coef * gwy; fzi += m * coef * gwz;
      });
      ax[i] = fxi; ay[i] = fyi; az[i] = fzi;
    }
  }

  function compute(ctrl) { buildGrid(); density(); forces(ctrl); }
  function halfKick() {
    for (let i = 0; i < Nact; i++) {
      vx[i] += 0.5 * dt * ax[i]; vy[i] += 0.5 * dt * ay[i]; vz[i] += 0.5 * dt * az[i];
    }
  }
  function step(ctrl) {
    compute(ctrl); halfKick();
    for (let i = 0; i < Nact; i++) {
      x[i] = ((x[i] + dt * vx[i]) % L + L) % L;
      y[i] = ((y[i] + dt * vy[i]) % L + L) % L;
      z[i] = ((z[i] + dt * vz[i]) % L + L) % L;
    }
    compute(ctrl); halfKick();
    sim.t += dt;
  }
  function prime() { buildGrid(); density(); }
  function diag() {
    let ke = 0; for (let i = 0; i < Nact; i++) ke += 0.5 * m * (vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2);
    return { ke };
  }

  const sim = {
    L, Nact, h, supp, dt, m, U, csfac, mach: 1 / csfac, rho0, cs0, t: 0,
    x, y, z, vx, vy, vz, rho, P, step, prime, diag,
  };
  return sim;
}

const TWO_PI = Math.PI * 2;

export function captureSnapshot3D(sim, index, seed) {
  return {
    seed, index, t: sim.t,
    arrays: {
      x: sim.x.slice(), y: sim.y.slice(), z: sim.z.slice(),
      vx: sim.vx.slice(), vy: sim.vy.slice(), vz: sim.vz.slice(),
      rho: sim.rho.slice(), P: sim.P.slice(),
    },
  };
}

export function generateBundle3D(config, onProgress) {
  const snapshots = [];
  let last = null;
  for (const seed of config.seeds) {
    const sim = createSim3D({ N: config.build.N, U: config.build.U, csfac: config.build.csfac, seed });
    last = sim;
    sim.prime();
    for (let s = 0; s < config.warmupSteps; s++) {
      sim.step(config.ctrl);
      if (onProgress && s % 25 === 0) onProgress({ seed, phase: "warmup", s, total: config.warmupSteps });
    }
    for (let k = 0; k < config.nSnapshots; k++) {
      snapshots.push(captureSnapshot3D(sim, k, seed));
      if (onProgress) onProgress({ seed, phase: "capture", s: k + 1, total: config.nSnapshots });
      if (k < config.nSnapshots - 1) for (let s = 0; s < config.snapSteps; s++) sim.step(config.ctrl);
    }
  }
  const meta = {
    dim: 3, L: last.L, N: last.Nact, m: last.m, h: last.h, supp: last.supp, dt: last.dt,
    U: last.U, csfac: last.csfac, mach: last.mach, rho0: last.rho0, cs0: last.cs0,
    eos: "isothermal", mode: config.ctrl.mode, av_alpha: config.ctrl.alpha, kernel: "cubic-spline",
    h_constant: true, seeds: config.seeds.slice(),
    warmup_steps: config.warmupSteps, snap_steps: config.snapSteps,
    t_warm: config.warmupSteps * last.dt, dt_snap: config.snapSteps * last.dt,
    n_snapshots_per_seed: config.nSnapshots, n_snapshots_total: snapshots.length,
  };
  return { meta, snapshots };
}

export { serializeBundle };
