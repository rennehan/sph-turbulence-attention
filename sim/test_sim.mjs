// Headless sanity check for the physics core + dump format round-trip.
// Run: npm test   (from sim/).  Also proves the Node "scale" path works.
import { writeFileSync, mkdirSync } from "node:fs";
import { createSim, extractNeighbors, generateBundle, serializeBundle } from "./src/sim.js";

let fail = 0;
const ok = (cond, msg) => { console.log(`${cond ? "  ok  " : " FAIL "} ${msg}`); if (!cond) fail++; };

// --- 1. build + step: density ~ rho0, energies finite ---
const sim = createSim({ N: 256, U: 0.5, seed: 7, csfac: 5 });
sim.prime();
const ctrl = { mode: "standard", alpha: 1.0, Cnu: 0.07, Ceps: 1.0, Cs: 0.15 };
const d0 = sim.diag();
ok(Number.isFinite(d0.ke) && d0.ke > 0, `initial KE finite & positive (${d0.ke.toExponential(2)})`);
let rhoMean = 0; for (let i = 0; i < sim.Nact; i++) rhoMean += sim.rho[i]; rhoMean /= sim.Nact;
ok(Math.abs(rhoMean - sim.rho0) < 0.05 * sim.rho0, `<rho> ~ rho0 (${rhoMean.toFixed(4)} vs ${sim.rho0})`);

for (let s = 0; s < 200; s++) sim.step(ctrl);
const d1 = sim.diag();
let allFinite = true;
for (let i = 0; i < sim.Nact; i++) if (!Number.isFinite(sim.vx[i]) || !Number.isFinite(sim.x[i])) allFinite = false;
ok(allFinite, "state finite after 200 steps");
ok(d1.ke < d0.ke, `KE decayed (dissipative): ${d0.ke.toExponential(2)} -> ${d1.ke.toExponential(2)}`);

// --- 2. neighbour extraction: count ~ pi r^2 n, self excluded, symmetric ---
const radius = 4 * sim.h;
const { offsets, indices } = extractNeighbors(sim, radius);
ok(offsets.length === sim.Nact + 1, `CSR offsets length N+1 (${offsets.length})`);
ok(offsets[sim.Nact] === indices.length, "CSR offsets[N] == indices.length");
let deg = 0, selfFound = false;
for (let i = 0; i < sim.Nact; i++) {
  for (let p = offsets[i]; p < offsets[i + 1]; p++) if (indices[p] === i) selfFound = true;
  deg += offsets[i + 1] - offsets[i];
}
const meanDeg = deg / sim.Nact;
const expDeg = Math.PI * radius * radius * (sim.Nact / (sim.L * sim.L));
ok(!selfFound, "self excluded from neighbour lists");
ok(Math.abs(meanDeg - expDeg) < 0.3 * expDeg, `mean degree ~ pi r^2 n (${meanDeg.toFixed(1)} vs ${expDeg.toFixed(1)})`);
// symmetry: i in nbr(j) iff j in nbr(i)
const set = (i) => new Set(indices.slice(offsets[i], offsets[i + 1]));
let symOk = true;
for (let i = 0; i < Math.min(sim.Nact, 50); i++) {
  for (const j of set(i)) if (!set(j).has(i)) { symOk = false; break; }
}
ok(symOk, "neighbour relation symmetric (periodic min-image)");

// --- 3. dump + serialize, then round-trip parse the manifest ---
const bundle = generateBundle(
  { build: { N: 256, U: 0.5, csfac: 5 }, ctrl, warmupSteps: 100, snapSteps: 40, nSnapshots: 2, nbrRadius: radius, seeds: [11, 22] },
  () => {}
);
ok(bundle.snapshots.length === 4, `bundle has 2 seeds x 2 snaps = 4 (${bundle.snapshots.length})`);
ok(bundle.snapshots[0].seed === 11 && bundle.snapshots[2].seed === 22, "snapshots tagged by seed");

const bytes = serializeBundle(bundle);
ok(bytes[0] === 0x53 && bytes[1] === 0x50 && bytes[2] === 0x48 && bytes[3] === 0x44, "magic 'SPHD'");
const dv = new DataView(bytes.buffer);
const mlen = dv.getUint32(8, true);
const manifest = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + mlen)));
ok(manifest.meta.N === 256 && manifest.meta.mode === "standard", "manifest meta carries run params");
ok(manifest.snapshots.length === 4 && "nbr_offsets" in manifest.snapshots[0].arrays, "manifest indexes arrays");

mkdirSync(new URL("../data/", import.meta.url), { recursive: true });
const out = new URL("../data/test_bundle.sph", import.meta.url);
writeFileSync(out, bytes);
console.log(`\nwrote ${out.pathname} (${(bytes.byteLength / 1e3).toFixed(1)} kB) for the Python round-trip\n`);

console.log(fail === 0 ? "ALL PASS" : `${fail} FAILURES`);
process.exit(fail === 0 ? 0 : 1);
