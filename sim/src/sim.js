// ===========================================================================
// 2D weakly-compressible SPH — pure physics core (no React, no DOM).
//
// Shared by the dashboard (visual control surface) and by headless runners
// (browser Web Worker for dumping, or Node for batch/scale). Keeping this
// React-free is what lets the dashboard pick parameters and the dump path
// re-run the *same* physics reproducibly.
//
// Periodic box, cubic-spline kernel, summation density, symmetric pressure
// force, KDK leapfrog. Dissipation modes:
//   "standard"   : blunt Monaghan AV (over-dissipative; the clean "truth" run)
//   "twobracket" : Balsara-gated AV + transported k_sgs + EOS-curvature P_sgs
//   "momonly"    : Balsara-gated AV + algebraic Smagorinsky (momentum-only)
//
// For a-priori SGS data, dump from "standard" with AV as low as stable: we
// want a resolved field that has NOT had an SGS model applied to it.
// ===========================================================================

export const TWO_PI = Math.PI * 2;

// deterministic PRNG so a given seed reproduces identical initial conditions
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// in-place iterative radix-2 FFT (forward, unnormalized)
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const vr = re[b] * cwr - im[b] * cwi, vi = re[b] * cwi + im[b] * cwr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] = re[a] + vr; im[a] = im[a] + vi;
        const ncwr = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = ncwr;
      }
    }
  }
}
export function fft2d(re, im, Ng) {
  const tr = new Float64Array(Ng), ti = new Float64Array(Ng);
  for (let r = 0; r < Ng; r++) {
    for (let c = 0; c < Ng; c++) { tr[c] = re[r * Ng + c]; ti[c] = im[r * Ng + c]; }
    fft(tr, ti);
    for (let c = 0; c < Ng; c++) { re[r * Ng + c] = tr[c]; im[r * Ng + c] = ti[c]; }
  }
  for (let c = 0; c < Ng; c++) {
    for (let r = 0; r < Ng; r++) { tr[r] = re[r * Ng + c]; ti[r] = im[r * Ng + c]; }
    fft(tr, ti);
    for (let r = 0; r < Ng; r++) { re[r * Ng + c] = tr[r]; im[r * Ng + c] = ti[r]; }
  }
}

export function createSim(opts) {
  const L = 1.0;
  const N = opts.N;
  const rho0 = 1.0;
  const U = opts.U;                       // shear half-amplitude
  const delta = 0.04;                     // shear-layer thickness
  const gamma = 5 / 3;                     // ideal-gas adiabatic index
  const m = (rho0 * L * L) / N;           // particle mass
  const nx = Math.round(Math.sqrt(N));
  const ny = Math.round(N / nx);
  const Nact = nx * ny;
  const dx = L / nx;
  const h = 1.3 * dx;                      // smoothing length (constant: no adaptive smoothing)
  const supp = 2 * h;                      // kernel support
  const csfac = opts.csfac || 5;
  const cs0 = csfac * U;                   // initial sound speed; cs/U = csfac sets the Mach number
  const u0 = cs0 * cs0 / (gamma * (gamma - 1)); // uniform initial specific internal energy
  const dt = 0.10 * h / (cs0 + 2 * U);     // CFL (tighter, paired with leapfrog)
  const eta2 = 0.01 * h * h;
  const sigma = 10 / (7 * Math.PI * h * h);

  // state arrays
  const x = new Float32Array(Nact), y = new Float32Array(Nact);
  const vx = new Float32Array(Nact), vy = new Float32Array(Nact);
  const ax = new Float32Array(Nact), ay = new Float32Array(Nact);
  const adx = new Float32Array(Nact), ady = new Float32Array(Nact); // dissipative part only
  const rho = new Float32Array(Nact), P = new Float32Array(Nact), cs = new Float32Array(Nact);
  const u = new Float32Array(Nact), dudt = new Float32Array(Nact);   // internal energy + its rate
  const ksg = new Float32Array(Nact), dkdt = new Float32Array(Nact); // subgrid kinetic energy + rate
  const sigrho2 = new Float32Array(Nact), psgs = new Float32Array(Nact); // subgrid density variance + pressure
  const kfloor = 1e-7;                                                // tiny seed so production can bootstrap
  const Lxx = new Float32Array(Nact), Lxy = new Float32Array(Nact);
  const Lyx = new Float32Array(Nact), Lyy = new Float32Array(Nact);
  const divv = new Float32Array(Nact), vort = new Float32Array(Nact), Smag = new Float32Array(Nact);

  // initialise: lattice + double-tanh shear + single-mode seed perturbation
  const rng = mulberry32(opts.seed | 0);
  let p = 0;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const px = (i + 0.5) * dx + (rng() - 0.5) * 0.02 * dx;
      const py = (j + 0.5) * (L / ny) + (rng() - 0.5) * 0.02 * dx;
      x[p] = (px % L + L) % L;
      y[p] = (py % L + L) % L;
      // periodic double shear layer (interfaces at y=0.25 and y=0.75, no seam jump)
      const sh = Math.tanh((y[p] - 0.25) / delta) - Math.tanh((y[p] - 0.75) / delta) - 1;
      vx[p] = U * sh;
      // seed: vertical velocity bumps localised on both interfaces, 2 wavelengths
      const e1 = Math.exp(-((y[p] - 0.25) * (y[p] - 0.25)) / (2 * 0.05 * 0.05));
      const e2 = Math.exp(-((y[p] - 0.75) * (y[p] - 0.75)) / (2 * 0.05 * 0.05));
      vy[p] = 0.06 * U * Math.sin(TWO_PI * 2 * x[p]) * (e1 + e2);
      u[p] = u0;
      p++;
    }
  }

  // linked-cell grid (periodic)
  const ncx = Math.max(3, Math.floor(L / supp));
  const cell = L / ncx;
  const head = new Int32Array(ncx * ncx);
  const next = new Int32Array(Nact);

  function buildGrid() {
    head.fill(-1);
    for (let i = 0; i < Nact; i++) {
      let cx = Math.floor(x[i] / cell); cx = ((cx % ncx) + ncx) % ncx;
      let cy = Math.floor(y[i] / cell); cy = ((cy % ncx) + ncx) % ncx;
      const c = cy * ncx + cx;
      next[i] = head[c];
      head[c] = i;
    }
  }

  function mimg(d) { if (d > 0.5 * L) return d - L; if (d < -0.5 * L) return d + L; return d; }

  function kernelDW(r) {
    // returns dW/dr (scalar). gradW = dWdr * (dr/|r|)
    const q = r / h;
    if (q >= 2) return 0;
    let f;
    if (q < 1) f = -3 * q + 2.25 * q * q;
    else { const a = 2 - q; f = -0.75 * a * a; }
    return sigma * f / h;
  }
  function kernelW(r) {
    const q = r / h;
    if (q >= 2) return 0;
    if (q < 1) return sigma * (1 - 1.5 * q * q + 0.75 * q * q * q);
    const a = 2 - q; return sigma * 0.25 * a * a * a;
  }

  function forEachNeighbor(i, cb) {
    let cx = Math.floor(x[i] / cell); cx = ((cx % ncx) + ncx) % ncx;
    let cy = Math.floor(y[i] / cell); cy = ((cy % ncx) + ncx) % ncx;
    for (let dj = -1; dj <= 1; dj++) {
      const ccy = ((cy + dj) % ncx + ncx) % ncx;
      for (let di = -1; di <= 1; di++) {
        const ccx = ((cx + di) % ncx + ncx) % ncx;
        let jp = head[ccy * ncx + ccx];
        while (jp !== -1) { cb(jp); jp = next[jp]; }
      }
    }
  }

  function density() {
    for (let i = 0; i < Nact; i++) {
      let r = m * kernelW(0); // self
      forEachNeighbor(i, (j) => {
        if (j === i) return;
        const dxx = mimg(x[i] - x[j]), dyy = mimg(y[i] - y[j]);
        const r2 = dxx * dxx + dyy * dyy;
        if (r2 < supp * supp && r2 > 1e-14) r += m * kernelW(Math.sqrt(r2));
      });
      rho[i] = r;
      const ui = u[i] > 1e-8 ? u[i] : 1e-8;
      P[i] = (gamma - 1) * r * ui;                       // ideal gas
      cs[i] = Math.sqrt(gamma * (gamma - 1) * ui);        // adiabatic sound speed
    }
  }

  function gradients() {
    for (let i = 0; i < Nact; i++) {
      let lxx = 0, lxy = 0, lyx = 0, lyy = 0;
      // Shepard-weighted kernel moments of rho for the subgrid density variance
      const W0 = kernelW(0);
      let A = m * W0, B = m * rho[i] * W0, C = (m / rho[i]) * W0; // self terms
      forEachNeighbor(i, (j) => {
        if (j === i) return;
        const dxx = mimg(x[i] - x[j]), dyy = mimg(y[i] - y[j]);
        const r2 = dxx * dxx + dyy * dyy;
        if (r2 >= supp * supp || r2 < 1e-14) return;
        const r = Math.sqrt(r2);
        const w = kernelW(r);
        A += m * w; B += m * rho[j] * w; C += (m / rho[j]) * w;
        const dw = kernelDW(r);
        const gwx = dw * dxx / r, gwy = dw * dyy / r;
        const vol = m / rho[j];
        const dvx = vx[j] - vx[i], dvy = vy[j] - vy[i];
        lxx += vol * dvx * gwx; lxy += vol * dvx * gwy;
        lyx += vol * dvy * gwx; lyy += vol * dvy * gwy;
      });
      Lxx[i] = lxx; Lxy[i] = lxy; Lyx[i] = lyx; Lyy[i] = lyy;
      divv[i] = lxx + lyy;
      vort[i] = lyx - lxy;                       // dvy/dx - dvx/dy
      const sxy = 0.5 * (lxy + lyx);
      Smag[i] = Math.sqrt(2 * (lxx * lxx + lyy * lyy + 2 * sxy * sxy));
      const rAvg = A / C, r2Avg = B / C;         // kernel-weighted <rho>, <rho^2>
      const v = r2Avg - rAvg * rAvg;             // sub-kernel density variance
      sigrho2[i] = v > 0 ? v : 0;
    }
  }

  function forces(ctrl) {
    const alpha = ctrl.alpha, beta = 2 * alpha;
    const mode = ctrl.mode;
    const sgs = mode !== "standard";          // Balsara gating active in both two-bracket modes
    const useK = mode === "twobracket";       // transported k + eddy viscosity from k (conservative)
    const useSmag = mode === "momonly";       // algebraic Smagorinsky, momentum-only, NOT energy-conserving
    const usePsgs = useK;                      // subgrid pressure only in the conservative mode
    const Cnu = ctrl.Cnu, Ceps = ctrl.Ceps;
    const Cs2h2 = (ctrl.Cs * h) * (ctrl.Cs * h);
    // subgrid pressure from EOS curvature: P_sgs = 1/2 P_rho_rho sigma_rho^2,
    // with P_rho_rho|_s = gamma(gamma-1) P / rho^2  (ideal gas adiabat)
    const cPsgs = 0.5 * gamma * (gamma - 1);
    for (let i = 0; i < Nact; i++) {
      psgs[i] = usePsgs ? cPsgs * P[i] * sigrho2[i] / (rho[i] * rho[i]) : 0;
    }
    function nut(i) {
      if (useK) return Cnu * h * Math.sqrt((ksg[i] > 0 ? ksg[i] : 0) + kfloor);
      if (useSmag) return Cs2h2 * Smag[i];
      return 0;
    }
    for (let i = 0; i < Nact; i++) {
      let fxi = 0, fyi = 0;   // pressure (reversible)
      let dxi = 0, dyi = 0;   // dissipative (AV + SGS eddy viscosity)
      let dui = 0;            // dInternalEnergy/dt
      let dki = 0;            // dSubgridKE/dt
      const Pi_rho2 = (P[i] + psgs[i]) / (rho[i] * rho[i]);   // effective pressure incl. subgrid term
      const fbi = sgs ? Math.abs(divv[i]) / (Math.abs(divv[i]) + Math.abs(vort[i]) + 1e-4 * cs[i] / h) : 1;
      const mui = rho[i] * nut(i);
      forEachNeighbor(i, (j) => {
        if (j === i) return;
        const dxx = mimg(x[i] - x[j]), dyy = mimg(y[i] - y[j]);
        const r2 = dxx * dxx + dyy * dyy;
        if (r2 >= supp * supp || r2 < 1e-14) return;
        const r = Math.sqrt(r2);
        const dw = kernelDW(r);
        const gwx = dw * dxx / r, gwy = dw * dyy / r;
        const Pj_rho2 = (P[j] + psgs[j]) / (rho[j] * rho[j]); // effective pressure incl. subgrid term
        const dvx = vx[i] - vx[j], dvy = vy[i] - vy[j];
        const vrelGW = dvx * gwx + dvy * gwy;           // (v_a - v_b) . grad W

        // pressure force (reversible / Hamiltonian)
        const coefP = -(Pi_rho2 + Pj_rho2);
        fxi += m * coefP * gwx;
        fyi += m * coefP * gwy;

        // artificial viscosity (Monaghan), Balsara-gated in two-bracket modes
        const vr = dvx * dxx + dvy * dyy;
        let PiG = 0;
        if (vr < 0) {
          const muav = h * vr / (r2 + eta2);
          const rhobar = 0.5 * (rho[i] + rho[j]);
          const cbar = 0.5 * (cs[i] + cs[j]);
          PiG = (-alpha * cbar * muav + beta * muav * muav) / rhobar;
          if (sgs) {
            const fbj = Math.abs(divv[j]) / (Math.abs(divv[j]) + Math.abs(vort[j]) + 1e-4 * cs[j] / h);
            PiG *= 0.5 * (fbi + fbj);
          }
          dxi += m * (-PiG) * gwx;
          dyi += m * (-PiG) * gwy;
        }

        // pdV work + AV heating -> internal energy
        dui += 0.5 * m * (Pi_rho2 + Pj_rho2 + PiG) * vrelGW;

        // SGS eddy viscosity (Morris) on momentum
        if (useK || useSmag) {
          const muj = rho[j] * nut(j);
          const g = (dxx * gwx + dyy * gwy) / (r2 + eta2);   // r.gradW / (r^2+eta^2) < 0
          const fac = m * (mui + muj) / (rho[i] * rho[j]) * g;
          dxi += fac * dvx;
          dyi += fac * dvy;
          if (useK) {
            dki += -0.5 * fac * (dvx * dvx + dvy * dvy);
            dki += m * (nut(i) + nut(j)) / (rho[i] * rho[j]) * (ksg[i] - ksg[j]) * g;
          }
        }
      });
      if (useK) {
        const ki = ksg[i] > 0 ? ksg[i] : 0;
        const eps = Ceps * ki * Math.sqrt(ki) / h;
        dki -= eps;
        dui += eps;          // subgrid cascade terminates as internal energy
      }
      ax[i] = fxi + dxi; ay[i] = fyi + dyi;
      adx[i] = dxi; ady[i] = dyi;
      dudt[i] = dui;
      dkdt[i] = dki;
    }
  }

  function compute(ctrl) { buildGrid(); density(); gradients(); forces(ctrl); }
  function halfKick() {
    for (let i = 0; i < Nact; i++) {
      vx[i] += 0.5 * dt * ax[i];
      vy[i] += 0.5 * dt * ay[i];
      u[i] += 0.5 * dt * dudt[i];
      ksg[i] += 0.5 * dt * dkdt[i];
      if (u[i] < 1e-8) u[i] = 1e-8;
      if (ksg[i] < 0) ksg[i] = 0;
    }
  }
  function step(ctrl) {
    // kick–drift–kick leapfrog
    compute(ctrl);
    halfKick();
    for (let i = 0; i < Nact; i++) {
      x[i] += dt * vx[i];
      y[i] += dt * vy[i];
      x[i] = (x[i] % L + L) % L;
      y[i] = (y[i] % L + L) % L;
    }
    compute(ctrl);
    halfKick();
    sim.t += dt;
  }

  // velocity energy spectrum E(k) via SPH interpolation onto an Ng x Ng grid + 2D FFT
  function spectrum(Ng) {
    const NN = Ng * Ng;
    const rex = new Float64Array(NN), imx = new Float64Array(NN);
    const rey = new Float64Array(NN), imy = new Float64Array(NN);
    const wsum = new Float64Array(NN);
    const dxg = L / Ng;
    const reach = Math.ceil(supp / dxg);
    for (let q = 0; q < Nact; q++) {
      const vol = m / rho[q];
      const ci = Math.floor(x[q] / dxg), cj = Math.floor(y[q] / dxg);
      for (let dj = -reach; dj <= reach; dj++) {
        const gj = ((cj + dj) % Ng + Ng) % Ng;
        for (let di = -reach; di <= reach; di++) {
          const gi = ((ci + di) % Ng + Ng) % Ng;
          const ddx = mimg(gi * dxg - x[q]), ddy = mimg(gj * dxg - y[q]);
          const r2 = ddx * ddx + ddy * ddy;
          if (r2 >= supp * supp) continue;
          const w = vol * kernelW(Math.sqrt(r2));
          const idx = gj * Ng + gi;
          rex[idx] += w * vx[q]; rey[idx] += w * vy[q]; wsum[idx] += w;
        }
      }
    }
    for (let i = 0; i < NN; i++) { if (wsum[i] > 1e-9) { rex[i] /= wsum[i]; rey[i] /= wsum[i]; } }
    fft2d(rex, imx, Ng); fft2d(rey, imy, Ng);
    const nb = Ng / 2;
    const E = new Float64Array(nb + 1);
    for (let r = 0; r < Ng; r++) for (let c = 0; c < Ng; c++) {
      const kx = c <= Ng / 2 ? c : c - Ng, ky = r <= Ng / 2 ? r : r - Ng;
      const kr = Math.round(Math.sqrt(kx * kx + ky * ky));
      if (kr >= 1 && kr <= nb) {
        const idx = r * Ng + c;
        E[kr] += 0.5 * ((rex[idx] * rex[idx] + imx[idx] * imx[idx]) + (rey[idx] * rey[idx] + imy[idx] * imy[idx])) / (NN * NN);
      }
    }
    return E;
  }

  function prime() { buildGrid(); density(); gradients(); }

  function diag() {
    let ke = 0, ens = 0, uint = 0, ksum = 0, psum = 0, Psum = 0;
    for (let i = 0; i < Nact; i++) {
      ke += 0.5 * m * (vx[i] * vx[i] + vy[i] * vy[i]);
      ens += 0.5 * m * vort[i] * vort[i];
      uint += m * u[i];
      ksum += m * ksg[i];
      psum += psgs[i]; Psum += P[i];
    }
    return { ke, ens, uint, ksum, psgsFrac: Psum > 0 ? psum / Psum : 0 };
  }

  const sim = {
    L, Nact, h, supp, dt, m, U, csfac, mach: 1 / csfac, rho0, u0, gamma, cs0,
    t: 0, uint0: 0, ke0: 1,
    kh: L / (2 * h), nyq: 32,
    x, y, vx, vy, u, ksg, rho, vort, Smag, divv, cs, sigrho2, psgs, P,
    step, diag, prime, spectrum,
  };
  return sim;
}

// ---------------------------------------------------------------------------
// Snapshot dumping (plumbing). Raw primitives only — NO derived/filtered
// quantities. Phase 1 (the SGS target + invariant features) is built in
// Python from exactly these arrays.
// ---------------------------------------------------------------------------

// Neighbour search to an arbitrary radius (default 4h, so a 2h test filter,
// whose kernel support reaches 4h, has every neighbour it needs). Returns CSR:
//   offsets : Int32Array(N+1)   row i is indices[offsets[i] .. offsets[i+1])
//   indices : Int32Array(total) neighbour particle ids (self EXCLUDED)
// Minimum-image is applied here only to decide membership; relative vectors
// are NOT dumped (consumer recomputes them from positions + L if needed).
export function extractNeighbors(sim, radius) {
  const { x, y, L, Nact } = sim;
  const ncx = Math.max(3, Math.floor(L / radius));
  const cell = L / ncx;                 // cell >= radius, so a 3x3 stencil suffices
  const head = new Int32Array(ncx * ncx).fill(-1);
  const next = new Int32Array(Nact);
  for (let i = 0; i < Nact; i++) {
    let cx = Math.floor(x[i] / cell); cx = ((cx % ncx) + ncx) % ncx;
    let cy = Math.floor(y[i] / cell); cy = ((cy % ncx) + ncx) % ncx;
    const c = cy * ncx + cx;
    next[i] = head[c]; head[c] = i;
  }
  const mimg = (d) => (d > 0.5 * L ? d - L : d < -0.5 * L ? d + L : d);
  const r2max = radius * radius;
  const offsets = new Int32Array(Nact + 1);
  const lists = new Array(Nact);
  for (let i = 0; i < Nact; i++) {
    const out = [];
    let cx = Math.floor(x[i] / cell); cx = ((cx % ncx) + ncx) % ncx;
    let cy = Math.floor(y[i] / cell); cy = ((cy % ncx) + ncx) % ncx;
    for (let dj = -1; dj <= 1; dj++) {
      const ccy = ((cy + dj) % ncx + ncx) % ncx;
      for (let di = -1; di <= 1; di++) {
        const ccx = ((cx + di) % ncx + ncx) % ncx;
        let jp = head[ccy * ncx + ccx];
        while (jp !== -1) {
          if (jp !== i) {
            const dxx = mimg(x[i] - x[jp]), dyy = mimg(y[i] - y[jp]);
            if (dxx * dxx + dyy * dyy < r2max) out.push(jp);
          }
          jp = next[jp];
        }
      }
    }
    lists[i] = out;
    offsets[i + 1] = offsets[i] + out.length;
  }
  const total = offsets[Nact];
  const indices = new Int32Array(total);
  let w = 0;
  for (let i = 0; i < Nact; i++) for (const j of lists[i]) indices[w++] = j;
  return { offsets, indices };
}

// Copy current sim state + neighbour CSR into a plain snapshot object.
// Arrays are sliced (copied) because the sim mutates its buffers in place.
export function captureSnapshot(sim, radius, index, seed) {
  const nbr = extractNeighbors(sim, radius);
  return {
    seed, index, t: sim.t,
    arrays: {
      x: sim.x.slice(), y: sim.y.slice(),
      vx: sim.vx.slice(), vy: sim.vy.slice(),
      rho: sim.rho.slice(), P: sim.P.slice(),
      nbr_offsets: nbr.offsets, nbr_indices: nbr.indices,
    },
  };
}

// Run one seed: build, warm up, then capture nSnapshots spaced by snapSteps.
export function simulateSeed(config, seed, onProgress) {
  const { build, ctrl, warmupSteps, snapSteps, nSnapshots, nbrRadius } = config;
  const sim = createSim({ ...build, seed });
  sim.prime();
  for (let s = 0; s < warmupSteps; s++) {
    sim.step(ctrl);
    if (onProgress && (s % 50 === 0)) onProgress({ seed, phase: "warmup", s, total: warmupSteps });
  }
  const snaps = [];
  for (let k = 0; k < nSnapshots; k++) {
    snaps.push(captureSnapshot(sim, nbrRadius, k, seed));
    if (onProgress) onProgress({ seed, phase: "capture", s: k + 1, total: nSnapshots });
    if (k < nSnapshots - 1) for (let s = 0; s < snapSteps; s++) sim.step(ctrl);
  }
  return { sim, snaps };
}

// Build a full multi-seed bundle. Returns { meta, snapshots } ready to serialize.
export function generateBundle(config, onProgress) {
  const snapshots = [];
  let lastSim = null;
  for (const seed of config.seeds) {
    const { sim, snaps } = simulateSeed(config, seed, onProgress);
    lastSim = sim;
    for (const s of snaps) snapshots.push(s);
  }
  const meta = {
    L: lastSim.L, N: lastSim.Nact, m: lastSim.m, h: lastSim.h, supp: lastSim.supp,
    dt: lastSim.dt, U: lastSim.U, csfac: lastSim.csfac, mach: lastSim.mach,
    gamma: lastSim.gamma, rho0: lastSim.rho0, cs0: lastSim.cs0,
    mode: config.ctrl.mode, av_alpha: config.ctrl.alpha,
    kernel: "cubic-spline", nbr_radius_h: config.nbrRadius / lastSim.h,
    nbr_radius: config.nbrRadius, h_constant: true,
    seeds: config.seeds.slice(),
    warmup_steps: config.warmupSteps, snap_steps: config.snapSteps,
    t_warm: config.warmupSteps * lastSim.dt, dt_snap: config.snapSteps * lastSim.dt,
    n_snapshots_per_seed: config.nSnapshots, n_snapshots_total: snapshots.length,
  };
  return { meta, snapshots };
}

// ---------------------------------------------------------------------------
// Binary container — single self-describing file:
//   "SPHD" | u32 version | u32 manifestLen | manifest(JSON utf8) | pad4 | payload
// Manifest holds meta + per-snapshot array {dtype,offset,length}; offsets are
// relative to the (4-byte-aligned) payload start. f4 = float32, i4 = int32, LE.
// ---------------------------------------------------------------------------
const MAGIC = [0x53, 0x50, 0x48, 0x44]; // "SPHD"

export function serializeBundle(bundle) {
  let off = 0;                       // running offset within payload
  const flat = [];                  // {offset, arr} in write order
  const snapMeta = bundle.snapshots.map((s) => {
    const a = {};
    for (const key of Object.keys(s.arrays)) {
      const arr = s.arrays[key];
      const o = off;
      a[key] = { dtype: arr instanceof Float32Array ? "f4" : "i4", offset: o, length: arr.length };
      flat.push({ offset: o, arr });
      off += arr.byteLength;
    }
    return { seed: s.seed, index: s.index, t: s.t, arrays: a };
  });
  const payloadBytes = off;
  const manifest = { format: "sph-dump-v1", meta: bundle.meta, snapshots: snapMeta };
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  const headerLen = 12 + manifestBytes.length;
  const payloadStart = Math.ceil(headerLen / 4) * 4;
  const total = payloadStart + payloadBytes;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) dv.setUint8(i, MAGIC[i]);
  dv.setUint32(4, 1, true);
  dv.setUint32(8, manifestBytes.length, true);
  new Uint8Array(buf, 12, manifestBytes.length).set(manifestBytes);
  const u8 = new Uint8Array(buf);
  for (const { offset, arr } of flat) {
    const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    u8.set(bytes, payloadStart + offset);
  }
  return new Uint8Array(buf);
}
