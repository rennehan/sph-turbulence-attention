import React, { useRef, useEffect, useState, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, ReferenceLine } from "recharts";
import { Play, Pause, RotateCcw, Activity, Download } from "lucide-react";
import { createSim, TWO_PI } from "./sim.js";

// ---- colour maps (precomputed palettes of "rgb()" strings) ----
function lerp(a, b, t) { return a + (b - a) * t; }
function buildPalette(stops) {
  const pal = new Array(256);
  for (let k = 0; k < 256; k++) {
    const t = k / 255;
    let s = 0;
    while (s < stops.length - 2 && t > stops[s + 1][0]) s++;
    const [t0, c0] = stops[s], [t1, c1] = stops[s + 1];
    const u = (t - t0) / (t1 - t0 || 1);
    const r = Math.round(lerp(c0[0], c1[0], u));
    const g = Math.round(lerp(c0[1], c1[1], u));
    const b = Math.round(lerp(c0[2], c1[2], u));
    pal[k] = `rgb(${r},${g},${b})`;
  }
  return pal;
}
const PAL_VORT = buildPalette([[0.0, [38, 132, 235]], [0.5, [16, 18, 26]], [1.0, [240, 120, 40]]]);
const PAL_SPEED = buildPalette([[0.0, [18, 20, 30]], [0.5, [30, 158, 150]], [1.0, [240, 224, 120]]]);
const PAL_DENS = buildPalette([[0.0, [40, 110, 220]], [0.5, [225, 225, 225]], [1.0, [200, 50, 50]]]);
const PAL_HEAT = buildPalette([[0.0, [14, 16, 24]], [0.45, [120, 40, 30]], [0.75, [225, 110, 30]], [1.0, [250, 235, 150]]]);

export default function SPHSolver() {
  const canvasRef = useRef(null);
  const simRef = useRef(null);
  const controlsRef = useRef(null);
  const frameRef = useRef(0);
  const prevRef = useRef([]);
  const specEmaRef = useRef(null);
  const prevSpecRef = useRef(null);
  const workerRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState("standard");
  const [alpha, setAlpha] = useState(1.0);
  const [Cnu, setCnu] = useState(0.07);
  const [Ceps, setCeps] = useState(1.0);
  const [Cs, setCs] = useState(0.15);
  const [U, setU] = useState(0.5);
  const [csfac, setCsfac] = useState(5);
  const [Np, setNp] = useState(1600);
  const [seed, setSeed] = useState(12345);
  const [substeps, setSubsteps] = useState(3);
  const [colorBy, setColorBy] = useState("vorticity");
  const [diag, setDiag] = useState({ t: 0, ke: 1, tot: 1, mech: 1, ksgs: 0, therm: 0, ens: 0, fps: 0 });
  const [series, setSeries] = useState([]);
  const [specData, setSpecData] = useState([]);

  // dump controls
  const [tWarm, setTWarm] = useState(4.0);
  const [nSnaps, setNSnaps] = useState(30);
  const [dtSnap, setDtSnap] = useState(0.5);
  const [seedsStr, setSeedsStr] = useState("1,2,3");
  const [dumpStatus, setDumpStatus] = useState(null); // null | {busy, msg}

  const buildSpecData = (ema, prev) => {
    const len = Math.max(ema ? ema.length : 0, prev ? prev.length : 0);
    const arr = [];
    for (let k = 1; k < len; k++) {
      const E = ema && ema[k] > 0 ? ema[k] : null;
      const Ep = prev && prev[k] > 0 ? prev[k] : null;
      arr.push({ k, E, Eprev: Ep });
    }
    return arr;
  };

  const build = useCallback(() => {
    const sim = createSim({ N: Np, U, seed, csfac });
    simRef.current = sim;
    sim.prime();
    const d0 = sim.diag();
    sim.ke0 = d0.ke || 1;
    sim.uint0 = d0.uint;
    sim.ksum0 = d0.ksum;
    frameRef.current = 0;
    specEmaRef.current = null;
    setSeries([]);
    setSpecData(buildSpecData(null, prevSpecRef.current));
    setDiag({ t: 0, ke: 1, tot: 1, mech: 1, ksgs: 0, therm: 0, ens: d0.ens, fps: 0 });
    renderFrame();
  }, [Np, U, seed, csfac]);

  useEffect(() => {
    controlsRef.current = { mode, alpha, Cnu, Ceps, Cs, substeps, colorBy };
  }, [mode, alpha, Cnu, Ceps, Cs, substeps, colorBy]);

  useEffect(() => { build(); /* eslint-disable-next-line */ }, [Np, U, seed, csfac]);

  const renderFrame = useCallback(() => {
    const sim = simRef.current; const canvas = canvasRef.current;
    if (!sim || !canvas) return;
    const ctrl = controlsRef.current || { colorBy };
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssSize = canvas.clientWidth || 460;
    if (canvas.width !== Math.floor(cssSize * dpr)) {
      canvas.width = Math.floor(cssSize * dpr);
      canvas.height = Math.floor(cssSize * dpr);
    }
    const ctx = canvas.getContext("2d");
    const S = canvas.width;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0a0c12";
    ctx.fillRect(0, 0, S, S);
    const sc = S / sim.L;
    const pr = Math.max(1.4 * dpr, (S / Math.sqrt(sim.Nact)) * 0.42);
    const cb = ctrl.colorBy || colorBy;
    let pal, scale;
    if (cb === "vorticity") { pal = PAL_VORT; scale = (sim.U / 0.04) * 1.2; }
    else if (cb === "speed") { pal = PAL_SPEED; scale = sim.U * 1.6; }
    else if (cb === "internal e") { pal = PAL_HEAT; scale = 0.025 * sim.u0; }
    else if (cb === "subgrid k") { pal = PAL_SPEED; scale = 0.03 * sim.U * sim.U; }
    else if (cb === "ρ-var") { pal = PAL_HEAT; scale = 0.06 * sim.rho0; }
    else { pal = PAL_DENS; }
    for (let i = 0; i < sim.Nact; i++) {
      let idx;
      if (cb === "vorticity") idx = Math.max(0, Math.min(255, Math.round(((sim.vort[i] / scale) * 0.5 + 0.5) * 255)));
      else if (cb === "speed") { const sp = Math.sqrt(sim.vx[i] * sim.vx[i] + sim.vy[i] * sim.vy[i]); idx = Math.max(0, Math.min(255, Math.round((sp / scale) * 255))); }
      else if (cb === "internal e") idx = Math.max(0, Math.min(255, Math.round(((sim.u[i] - sim.u0) / scale) * 255)));
      else if (cb === "subgrid k") idx = Math.max(0, Math.min(255, Math.round((sim.ksg[i] / scale) * 255)));
      else if (cb === "ρ-var") idx = Math.max(0, Math.min(255, Math.round((Math.sqrt(sim.sigrho2[i]) / scale) * 255)));
      else { idx = Math.max(0, Math.min(255, Math.round(((sim.rho[i] - sim.rho0) / (0.03 * sim.rho0) * 0.5 + 0.5) * 255))); }
      ctx.fillStyle = pal[idx];
      const px = sim.x[i] * sc, py = (sim.L - sim.y[i]) * sc;
      ctx.beginPath();
      ctx.arc(px, py, pr, 0, TWO_PI);
      ctx.fill();
    }
  }, [colorBy]);

  useEffect(() => {
    if (!running) return;
    let raf, last = performance.now(), fps = 0;
    const loop = () => {
      const sim = simRef.current; const ctrl = controlsRef.current;
      if (sim && ctrl) {
        for (let s = 0; s < ctrl.substeps; s++) sim.step(ctrl);
        renderFrame();
        frameRef.current++;
        const now = performance.now();
        fps = 0.9 * fps + 0.1 * (1000 / Math.max(1, now - last));
        last = now;
        if (frameRef.current % 6 === 0) {
          const d = sim.diag();
          const dU = d.uint - sim.uint0;
          const ksgs = d.ksum;
          const tot = d.ke + ksgs + dU;
          const k0 = sim.ke0;
          setDiag({ t: sim.t, ke: d.ke / k0, mech: d.ke / k0, ksgs: ksgs / k0, therm: dU / k0, tot: tot / k0, ens: d.ens, psgs: d.psgsFrac, fps });
          setSeries((prev) => {
            const nxt = prev.length > 320 ? prev.slice(-320) : prev.slice();
            nxt.push({
              t: +sim.t.toFixed(3),
              ke: +(d.ke / k0).toFixed(4),
              tot: +(tot / k0).toFixed(4),
              mech: +(d.ke / k0).toFixed(4),
              ksgs: +(ksgs / k0).toFixed(4),
              therm: +(dU / k0).toFixed(4),
            });
            return nxt;
          });
        }
        if (frameRef.current % 15 === 0) {
          const E = sim.spectrum(64);
          let ema = specEmaRef.current;
          if (!ema) ema = E.slice();
          else for (let k = 0; k < E.length; k++) ema[k] = 0.85 * ema[k] + 0.15 * E[k];
          specEmaRef.current = ema;
          setSpecData(buildSpecData(ema, prevSpecRef.current));
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, renderFrame]);

  const reset = () => {
    setRunning(false);
    if (series.length > 4) prevRef.current = series.map((d) => ({ t: d.t, kePrev: d.ke }));
    if (specEmaRef.current) prevSpecRef.current = specEmaRef.current.slice();
    build();
  };

  // ---- snapshot dump ----
  const startDump = () => {
    const sim = simRef.current;
    if (!sim || (dumpStatus && dumpStatus.busy)) return;
    const seeds = seedsStr.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n));
    if (!seeds.length) { setDumpStatus({ busy: false, msg: "no valid seeds" }); return; }
    const dt = sim.dt;
    const config = {
      build: { N: Np, U, csfac },
      ctrl: { mode, alpha, Cnu, Ceps, Cs },
      warmupSteps: Math.max(0, Math.round(tWarm / dt)),
      snapSteps: Math.max(1, Math.round(dtSnap / dt)),
      nSnapshots: nSnaps,
      nbrRadius: 4 * sim.h,
      seeds,
    };
    setRunning(false);
    setDumpStatus({ busy: true, msg: "starting…" });
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("./dumpWorker.js", import.meta.url), { type: "module" });
    }
    const worker = workerRef.current;
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "progress") {
        setDumpStatus({ busy: true, msg: `seed ${d.seed} · ${d.phase} ${d.s}/${d.total}` });
      } else if (d.type === "done") {
        const blob = new Blob([d.buffer], { type: "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const mach = (1 / csfac).toFixed(2);
        a.href = url;
        a.download = `sph_N${config.build.N}_U${U}_M${mach}_${seeds.length}seeds_${nSnaps}x.sph`;
        a.click();
        URL.revokeObjectURL(url);
        const mb = (d.buffer.byteLength / 1e6).toFixed(1);
        setDumpStatus({ busy: false, msg: `done · ${d.meta.n_snapshots_total} snaps · ${mb} MB` });
      } else if (d.type === "error") {
        setDumpStatus({ busy: false, msg: `error: ${d.message}` });
      }
    };
    worker.postMessage(config);
  };

  useEffect(() => () => { if (workerRef.current) workerRef.current.terminate(); }, []);

  // ---- chart data ----
  const keData = (() => {
    const out = series.map((d) => ({ t: d.t, ke: d.ke }));
    const prev = prevRef.current;
    for (let i = 0; i < out.length && i < prev.length; i++) out[i].kePrev = prev[i].kePrev;
    if (prev.length > out.length) for (let i = out.length; i < prev.length; i++) out.push({ t: prev[i].t, kePrev: prev[i].kePrev });
    return out;
  })();
  const energyData = series;
  const specChart = (() => {
    const anchor = specData.find((d) => d.k === 3 && d.E) || specData.find((d) => d.E);
    const C = anchor && anchor.E ? anchor.E * Math.pow(anchor.k, 5 / 3) : null;
    return specData.map((d) => ({ ...d, ref: C ? C * Math.pow(d.k, -5 / 3) : null }));
  })();
  const khMode = simRef.current ? Math.round(simRef.current.kh) : 17;

  const mono = '"SF Mono","JetBrains Mono",ui-monospace,Menlo,monospace';
  const sans = 'ui-sans-serif,system-ui,-apple-system,sans-serif';
  const panel = { background: "#10131b", border: "1px solid #222838", borderRadius: 10, padding: 14 };
  const label = { fontFamily: mono, fontSize: 11, letterSpacing: "0.04em", color: "#7c879e", textTransform: "uppercase" };
  const accent = "#36c9bd";

  const Slider = ({ name, val, set, min, max, stepv, fmt, live }) => (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={label}>{name}{!live && <span style={{ color: "#54607a" }}> ·reset</span>}</span>
        <span style={{ fontFamily: mono, fontSize: 12, color: accent }}>{fmt ? fmt(val) : val}</span>
      </div>
      <input type="range" min={min} max={max} step={stepv} value={val}
        onChange={(e) => set(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: accent }} />
    </div>
  );

  const NumField = ({ name, val, set, step, hint }) => (
    <div style={{ marginBottom: 9 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
        <span style={label}>{name}</span>
        {hint && <span style={{ fontFamily: mono, fontSize: 10, color: "#54607a" }}>{hint}</span>}
      </div>
      <input type="number" value={val} step={step}
        onChange={(e) => set(e.target.value === "" ? "" : parseFloat(e.target.value))}
        style={{ width: "100%", boxSizing: "border-box", background: "#0d111a", color: "#cdd4e3", border: "1px solid #232a3b", borderRadius: 6, padding: "6px 8px", fontFamily: mono, fontSize: 12 }} />
    </div>
  );

  return (
    <div style={{ fontFamily: sans, background: "#070910", color: "#cdd4e3", padding: 18, borderRadius: 14, border: "1px solid #1a2030" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
        <div>
          <div style={{ fontFamily: mono, fontSize: 17, color: "#eef2fb", letterSpacing: "-0.01em" }}>
            2D SPH · free shear layer
          </div>
          <div style={{ fontFamily: mono, fontSize: 11, color: "#5a647d", marginTop: 2 }}>
            ideal gas γ=5/3 · periodic · {simRef.current ? simRef.current.Nact : Np} particles
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setRunning((r) => !r)}
            style={{ display: "flex", alignItems: "center", gap: 6, background: running ? "#2a2030" : accent, color: running ? "#e6b0b0" : "#04110f", border: "none", borderRadius: 8, padding: "9px 16px", fontFamily: mono, fontSize: 13, cursor: "pointer", fontWeight: 600 }}>
            {running ? <Pause size={15} /> : <Play size={15} />}{running ? "pause" : "run"}
          </button>
          <button onClick={reset}
            style={{ display: "flex", alignItems: "center", gap: 6, background: "#161b28", color: "#aeb7cc", border: "1px solid #2a3245", borderRadius: 8, padding: "9px 14px", fontFamily: mono, fontSize: 13, cursor: "pointer" }}>
            <RotateCcw size={14} />reset
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 268px", gap: 16, alignItems: "start" }}>
        <div>
          <div style={{ position: "relative", borderRadius: 10, overflow: "hidden", border: "1px solid #1c2333", background: "#0a0c12" }}>
            <canvas ref={canvasRef} style={{ width: "100%", aspectRatio: "1 / 1", display: "block" }} />
            <div style={{ position: "absolute", top: 8, left: 10, fontFamily: mono, fontSize: 11, color: "#46506a" }}>
              {colorBy} field
            </div>
          </div>

          <div style={{ ...panel, marginTop: 12, padding: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <Activity size={13} color={accent} />
              <span style={label}>kinetic energy / KE₀ &nbsp;(faded = previous run)</span>
            </div>
            <div style={{ height: 148 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={keData} margin={{ top: 6, right: 8, bottom: 2, left: -18 }}>
                  <XAxis dataKey="t" tick={{ fill: "#56607a", fontSize: 10, fontFamily: mono }} stroke="#222838" tickFormatter={(v) => v.toFixed(1)} />
                  <YAxis domain={[0, 1.05]} tick={{ fill: "#56607a", fontSize: 10, fontFamily: mono }} stroke="#222838" />
                  <Tooltip contentStyle={{ background: "#10131b", border: "1px solid #2a3245", borderRadius: 8, fontFamily: mono, fontSize: 11 }} labelStyle={{ color: "#7c879e" }} />
                  <Legend wrapperStyle={{ fontFamily: mono, fontSize: 11 }} />
                  <Line type="monotone" dataKey="kePrev" name="previous run" stroke="#46506e" dot={false} strokeWidth={1.3} strokeDasharray="4 3" isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="ke" name="current run" stroke={accent} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ ...panel, marginTop: 12, padding: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <Activity size={13} color={accent} />
              <span style={label}>energy / KE₀ &nbsp;·&nbsp; total = kinetic + subgrid k + Δ internal</span>
            </div>
            <div style={{ height: 158 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={energyData} margin={{ top: 6, right: 8, bottom: 2, left: -18 }}>
                  <XAxis dataKey="t" tick={{ fill: "#56607a", fontSize: 10, fontFamily: mono }} stroke="#222838" tickFormatter={(v) => v.toFixed(1)} />
                  <YAxis domain={[0, 1.15]} tick={{ fill: "#56607a", fontSize: 10, fontFamily: mono }} stroke="#222838" />
                  <Tooltip contentStyle={{ background: "#10131b", border: "1px solid #2a3245", borderRadius: 8, fontFamily: mono, fontSize: 11 }} labelStyle={{ color: "#7c879e" }} />
                  <Legend wrapperStyle={{ fontFamily: mono, fontSize: 11 }} />
                  <Line type="monotone" dataKey="mech" name="kinetic" stroke="#3a8ce0" dot={false} strokeWidth={1.6} isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="ksgs" name="subgrid k" stroke="#36c9bd" dot={false} strokeWidth={1.4} isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="therm" name="Δ internal (heat)" stroke="#e0902a" dot={false} strokeWidth={1.6} isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="tot" name="total" stroke="#cdd4e3" dot={false} strokeWidth={2.2} isAnimationActive={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ ...panel, marginTop: 12, padding: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <Activity size={13} color={accent} />
              <span style={label}>energy spectrum E(k) &nbsp;·&nbsp; log–log, time-averaged &nbsp;(faded = previous run)</span>
            </div>
            <div style={{ height: 168 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={specChart} margin={{ top: 6, right: 10, bottom: 2, left: -6 }}>
                  <XAxis dataKey="k" type="number" scale="log" domain={[1, 32]} ticks={[1, 2, 4, 8, 16, 32]} tick={{ fill: "#56607a", fontSize: 10, fontFamily: mono }} stroke="#222838" allowDataOverflow />
                  <YAxis scale="log" domain={["auto", "auto"]} tick={{ fill: "#56607a", fontSize: 10, fontFamily: mono }} stroke="#222838" tickFormatter={(v) => v.toExponential(0)} width={48} />
                  <Tooltip contentStyle={{ background: "#10131b", border: "1px solid #2a3245", borderRadius: 8, fontFamily: mono, fontSize: 11 }} labelStyle={{ color: "#7c879e" }} formatter={(v) => (v == null ? "" : Number(v).toExponential(2))} />
                  <Legend wrapperStyle={{ fontFamily: mono, fontSize: 11 }} />
                  <ReferenceLine x={khMode} stroke="#54607a" strokeDasharray="3 3" label={{ value: "k_Δ", fill: "#7c879e", fontSize: 10, fontFamily: mono, position: "top" }} />
                  <Line type="monotone" dataKey="ref" name="−5/3" stroke="#54607a" dot={false} strokeWidth={1} strokeDasharray="5 4" isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="Eprev" name="previous run" stroke="#46506e" dot={false} strokeWidth={1.3} isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="E" name="current run" stroke={accent} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={panel}>
            <div style={{ ...label, marginBottom: 9 }}>dissipation scheme</div>
            {[["standard", "Standard AV", "blunt Monaghan viscosity · no SGS"],
              ["twobracket", "Two-bracket (conservative)", "gated AV + transported k + P_sgs · energy-conserving"],
              ["momonly", "Two-bracket (lossy SGS)", "gated AV + Smagorinsky momentum-only · SGS energy not conserved"]].map(([id, t, sub]) => (
              <button key={id} onClick={() => setMode(id)}
                style={{ width: "100%", textAlign: "left", marginBottom: 7, padding: "9px 11px", borderRadius: 8, cursor: "pointer",
                  background: mode === id ? "#13251f" : "#0d111a",
                  border: mode === id ? `1px solid ${accent}` : "1px solid #232a3b" }}>
                <div style={{ fontFamily: mono, fontSize: 13, color: mode === id ? accent : "#c5cde0" }}>{t}</div>
                <div style={{ fontFamily: mono, fontSize: 10.5, color: "#5a647d", marginTop: 2 }}>{sub}</div>
              </button>
            ))}
          </div>

          <div style={panel}>
            <Slider name="AV α" val={alpha} set={setAlpha} min={0} max={2.5} stepv={0.05} fmt={(v) => v.toFixed(2)} live />
            <Slider name="SGS Cν (eddy visc)" val={Cnu} set={setCnu} min={0} max={0.2} stepv={0.005} fmt={(v) => v.toFixed(3)} live />
            <Slider name="SGS Cε (dissip)" val={Ceps} set={setCeps} min={0.3} max={2.0} stepv={0.05} fmt={(v) => v.toFixed(2)} live />
            <Slider name="Smagorinsky Cs (lossy mode)" val={Cs} set={setCs} min={0} max={0.4} stepv={0.01} fmt={(v) => v.toFixed(2)} live />
            <Slider name="shear U" val={U} set={setU} min={0.2} max={1.0} stepv={0.05} fmt={(v) => v.toFixed(2)} />
            <Slider name="sound cs/U (↓ = higher Mach)" val={csfac} set={setCsfac} min={2.5} max={12} stepv={0.5} fmt={(v) => `${v} · M${(1 / v).toFixed(2)}`} />
            <Slider name="particles" val={Np} set={setNp} min={600} max={3000} stepv={100} fmt={(v) => v} />
            <Slider name="substeps/frame" val={substeps} set={setSubsteps} min={1} max={10} stepv={1} live />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
              <span style={label}>seed <span style={{ color: accent }}>{seed}</span> <span style={{ color: "#54607a" }}>·frozen</span></span>
              <button onClick={() => setSeed(Math.floor(Math.random() * 1e6))}
                style={{ background: "#0d111a", color: "#8b95ad", border: "1px solid #232a3b", borderRadius: 7, padding: "5px 10px", fontFamily: mono, fontSize: 11, cursor: "pointer" }}>
                new seed
              </button>
            </div>
          </div>

          <div style={panel}>
            <div style={{ ...label, marginBottom: 9 }}>dump snapshots (a-priori data)</div>
            <NumField name="warmup t" val={tWarm} set={setTWarm} step={0.5} hint="sim time" />
            <NumField name="snapshots / seed" val={nSnaps} set={setNSnaps} step={1} />
            <NumField name="Δt between snaps" val={dtSnap} set={setDtSnap} step={0.1} hint="sim time" />
            <div style={{ marginBottom: 9 }}>
              <div style={{ ...label, marginBottom: 3 }}>seeds (comma-sep)</div>
              <input type="text" value={seedsStr} onChange={(e) => setSeedsStr(e.target.value)}
                style={{ width: "100%", boxSizing: "border-box", background: "#0d111a", color: "#cdd4e3", border: "1px solid #232a3b", borderRadius: 6, padding: "6px 8px", fontFamily: mono, fontSize: 12 }} />
            </div>
            <div style={{ fontFamily: mono, fontSize: 10, color: "#54607a", marginBottom: 8, lineHeight: 1.5 }}>
              neighbours dumped to 4h · uses current scheme/params · re-runs each seed from t=0
            </div>
            {mode !== "standard" && (
              <div style={{ fontFamily: mono, fontSize: 10.5, color: "#e0902a", marginBottom: 8, lineHeight: 1.5 }}>
                ⚠ not in “standard” mode — the dumped field has an SGS model applied. For a-priori truth, switch to Standard AV.
              </div>
            )}
            <button onClick={startDump} disabled={dumpStatus && dumpStatus.busy}
              style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                background: dumpStatus && dumpStatus.busy ? "#1a2230" : accent,
                color: dumpStatus && dumpStatus.busy ? "#7c879e" : "#04110f",
                border: "none", borderRadius: 8, padding: "9px 14px", fontFamily: mono, fontSize: 13,
                cursor: dumpStatus && dumpStatus.busy ? "default" : "pointer", fontWeight: 600 }}>
              <Download size={14} />{dumpStatus && dumpStatus.busy ? "dumping…" : "dump"}
            </button>
            {dumpStatus && (
              <div style={{ fontFamily: mono, fontSize: 10.5, color: dumpStatus.msg.startsWith("error") ? "#e07070" : "#7c879e", marginTop: 7 }}>
                {dumpStatus.msg}
              </div>
            )}
          </div>

          <div style={panel}>
            <div style={{ ...label, marginBottom: 8 }}>colour field</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {["vorticity", "speed", "subgrid k", "ρ-var", "internal e", "density"].map((c) => (
                <button key={c} onClick={() => { setColorBy(c); renderFrame(); }}
                  style={{ flex: "1 0 30%", padding: "7px 0", borderRadius: 7, cursor: "pointer", fontFamily: mono, fontSize: 11,
                    background: colorBy === c ? "#13251f" : "#0d111a",
                    color: colorBy === c ? accent : "#8b95ad",
                    border: colorBy === c ? `1px solid ${accent}` : "1px solid #232a3b" }}>{c}</button>
              ))}
            </div>
          </div>

          <div style={{ ...panel, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px 8px" }}>
            {[["t", diag.t.toFixed(2)], ["KE/KE₀", diag.ke.toFixed(3)], ["k_sgs/KE₀", (diag.ksgs || 0).toFixed(3)],
              ["heat/KE₀", diag.therm.toFixed(3)], ["Psgs/P %", ((diag.psgs || 0) * 100).toFixed(2)],
              ["E_tot/E₀", diag.tot.toFixed(3)], ["drift %", ((diag.tot - 1) * 100).toFixed(2)], ["fps", diag.fps.toFixed(0)]].map(([k, v]) => (
              <div key={k}>
                <div style={{ ...label, fontSize: 10 }}>{k}</div>
                <div style={{ fontFamily: mono, fontSize: 15, color: k === "drift %" ? (Math.abs(diag.tot - 1) > 0.05 ? "#e0902a" : accent) : "#e6ebf6", marginTop: 1 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ fontFamily: mono, fontSize: 11, color: "#4e5772", marginTop: 12, lineHeight: 1.6 }}>
        Explore with the schemes/sliders, find a developed-turbulence window, then dump a-priori snapshots from Standard AV.
        Each dump re-runs every seed from t=0 (warmup → capture), so the data is reproducible from (seed, params, warmup).
        Raw primitives only (x, v, ρ, P) + 4h neighbour lists — the SGS filter, target, and invariant features are built downstream.
      </div>
    </div>
  );
}
