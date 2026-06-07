// Headless time-sweep: dump a fine early-time series so we can locate the
// developed-turbulence window and measure whether there's a learnable SGS
// signal (vs the dead t>=4 window we dumped from the dashboard).
import { writeFileSync, mkdirSync } from "node:fs";
import { createSim, generateBundle, serializeBundle } from "./src/sim.js";

const N = 1600, U = 0.5, csfac = 5;
const ref = createSim({ N, U, csfac, seed: 1 });
const dt = ref.dt;
const t0 = 0.2, dtSnap = 0.15, nSnap = 40;     // covers t ~ 0.2 .. 6.05

mkdirSync(new URL("../data/", import.meta.url), { recursive: true });
for (const alpha of [1.0, 0.25]) {
  const cfg = {
    build: { N, U, csfac },
    ctrl: { mode: "standard", alpha, Cnu: 0.07, Ceps: 1.0, Cs: 0.15 },
    warmupSteps: Math.round(t0 / dt),
    snapSteps: Math.round(dtSnap / dt),
    nSnapshots: nSnap,
    nbrRadius: 4 * ref.h,
    seeds: [1],
  };
  const bundle = generateBundle(cfg, () => {});
  const bytes = serializeBundle(bundle);
  const out = new URL(`../data/sweep_a${alpha}.sph`, import.meta.url);
  writeFileSync(out, bytes);
  console.log(`wrote sweep_a${alpha}.sph  (${bundle.snapshots.length} snaps, ${(bytes.byteLength/1e6).toFixed(1)} MB)`);
}
