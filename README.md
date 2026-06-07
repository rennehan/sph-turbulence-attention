# sph-turbulence-attention

*A priori* study of machine-learned subgrid-scale (SGS) closures for 2D SPH
turbulence: can a learned closure (MLP, then attention-over-neighbors) predict
the SGS stress better than analytical Smagorinsky / dynamic Smagorinsky?

> **Scientific framing & findings — TODO (Phase 5, author's own words).**
> This README currently documents only the data-generation plumbing.

## Layout

```
sim/                  2D weakly-compressible SPH solver + dashboard + dumper (JS)
  src/sim.js          pure physics core (no React) — also runs headless in Node
  src/SPHSolver.jsx   dashboard: explore params, then dump a-priori snapshots
  src/dumpWorker.js   runs snapshot generation off the UI thread
  test_sim.mjs        headless physics sanity + dump-format round-trip
python/
  load_snapshots.py   reads .sph bundles -> numpy (raw primitives + neighbour CSR)
  sph_ops.py          generic SPH operators: Shepard filter, renormalized gradient
  test_ops.py         operator checks (filter(const), grad(linear)) + rho'/rho0
  phase1.py           SGS target + invariant features — AUTHOR-OWNED stubs
data/                 dumped .sph bundles land here (gitignored)
```

## Run the dashboard

```
cd sim && npm install && npm run dev
```

Explore the scheme/sliders to find a developed-turbulence window (watch the
spectrum and vorticity field). Then, in **Standard AV** mode, set the dump
controls and hit **dump** — it re-runs each seed from t=0 (warmup → capture)
and downloads a `.sph` bundle. Reproducible from (seed, params, warmup).

Headless sanity check + format round-trip: `cd sim && npm test`.

## Snapshot bundle (`.sph`)

Self-describing binary: `"SPHD" | u32 ver | u32 manifestLen | manifest(JSON) | payload`.
Each snapshot carries **raw primitives only** — `x, y, vx, vy, rho, P` — plus a
neighbour list in CSR form (`nbr_offsets` length N+1, `nbr_indices` flattened,
self excluded) out to **4h** (so a 2h test filter, support 4h, has its
neighbours). Load with `python/load_snapshots.py`.

The SGS filter, the SGS-stress target, and the invariant input features are
**not** in the dump — they are constructed downstream (Phase 1), which keeps the
physics-ML boundary clean.

## Notes / limitations (data side)

- Free shear layer (Kelvin–Helmholtz), **decaying** — harvest snapshots in the
  developed-mixing window, not a forced steady state.
- Constant smoothing length `h` (no adaptive smoothing) — single resolution.
- Default Mach ≈ 0.2 (`cs/U = 5`); deviatoric-only SGS target is justified here.
- Dump from **Standard AV** so the resolved field has no SGS model baked in
  (it is still the SPH solution, not a true DNS — state this in limitations).
