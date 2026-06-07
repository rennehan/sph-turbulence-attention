// Headless resolution sweep: run the solver at a higher particle count
// (default 128x128 = 16384) across the developed-turbulence window and dump
// snapshots, to check whether a real SGS signal emerges with proper scale
// separation. Usage: node res_sweep.mjs [N] [t0] [dtSnap] [nSnap]
import { writeFileSync, mkdirSync } from "node:fs";
import { createSim, generateBundle, serializeBundle } from "./src/sim.js";

const N = parseInt(process.argv[2] || "16384", 10);
const t0 = parseFloat(process.argv[3] || "0.4");
const dtSnap = parseFloat(process.argv[4] || "0.2");
const nSnap = parseInt(process.argv[5] || "14", 10);
const U = 0.5, csfac = 5, alpha = 0.25;   // low AV to preserve small-scale content

const ref = createSim({ N, U, csfac, seed: 1 });
const dt = ref.dt;
const warmupSteps = Math.round(t0 / dt);
const snapSteps = Math.round(dtSnap / dt);
const totalSteps = warmupSteps + snapSteps * nSnap;
console.log(`N=${ref.Nact} (nx=${Math.round(Math.sqrt(N))}) h=${ref.h.toExponential(2)} dt=${dt.toExponential(2)}`);
console.log(`window t=${t0}..${(t0 + dtSnap * nSnap).toFixed(2)}  ~${totalSteps} steps total\n`);

const t_start = Date.now();
let lastPct = -1;
const cfg = {
  build: { N, U, csfac },
  ctrl: { mode: "standard", alpha, Cnu: 0.07, Ceps: 1.0, Cs: 0.15 },
  warmupSteps, snapSteps, nSnapshots: nSnap, nbrRadius: 4 * ref.h, seeds: [1],
};
const bundle = generateBundle(cfg, (p) => {
  const done = p.phase === "warmup" ? p.s : warmupSteps + p.s * snapSteps;
  const pct = Math.floor((done / totalSteps) * 100 / 10) * 10;
  if (pct !== lastPct) {
    lastPct = pct;
    console.log(`  ${pct}%  (${p.phase})  ${((Date.now() - t_start) / 1000).toFixed(0)}s`);
  }
});
const bytes = serializeBundle(bundle);
mkdirSync(new URL("../data/", import.meta.url), { recursive: true });
const out = new URL(`../data/res_N${N}.sph`, import.meta.url);
writeFileSync(out, bytes);
console.log(`\nwrote res_N${N}.sph  (${bundle.snapshots.length} snaps, ${(bytes.byteLength / 1e6).toFixed(1)} MB, ${((Date.now() - t_start) / 1000).toFixed(0)}s)`);
