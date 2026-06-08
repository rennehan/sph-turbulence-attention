# System Prompt — A-priori ML Subgrid-Scale (SGS) Closures for Driven Turbulence

## Context and goal

This project is an *a priori* study of machine-learned SGS stress closures for
compressible turbulence. The scientific question:

> **Can a learned closure (MLP on invariants, then attention-over-neighbours)
> predict the SGS stress better than the analytical Smagorinsky / dynamic
> Smagorinsky models — and specifically, can it recover the part those models
> get wrong?**

The data engine is moving to **Athena++** (finite-volume grid code) with driven
turbulence. The SGS math below is code-agnostic: it needs only a filter and a
gradient, both of which are cleaner on a grid (FFT/finite-difference) than in a
particle code. *A priori* means: take well-resolved snapshots as surrogate-DNS,
filter them, compute the exact SGS stress, and ask how well each model reproduces
it. No online/a-posteriori coupling.

---

## Turbulence setup (Athena++)

- **Driven, statistically stationary turbulence**, periodic box. Use a sustained
  solenoidal forcing (Athena++ turbulence driver / Ornstein–Uhlenbeck on low
  wavenumbers). A *sustained cascade* is essential — decaying turbulence relaxes
  to near-zero net SGS transfer and is not a valid testbed.
- **3D is the primary regime.** In 3D the energy cascade is forward, so the
  eddy-viscosity premise holds and Smagorinsky has a genuine coefficient
  (C_s ≈ 0.1–0.16). In 2D the energy cascade is *inverse*: the net SGS energy
  transfer is **backscatter**, and an eddy-viscosity (strictly dissipative) model
  is the wrong sign on average. 2D is fine as a fast methodological testbed if you
  remember this.
- **Subsonic, M ≈ 0.2** (so a deviatoric-only stress target is justified; the
  trace/dilatational part is small and folded into a modified pressure).
- **Scale separation is mandatory.** The filter width Δ must sit well above the
  grid/dissipation scale, in the inertial range. If Δ is too close to the
  resolution, the subgrid content k_sgs is at the noise floor and every
  correlation is a correlation-with-noise. Target **k_sgs/k_res ≳ 0.1–0.3**.
  Verify with the energy spectrum: there must be a real inertial range
  (3D ~ k^−5/3) between the forcing scale and Δ, and between Δ and the grid.
- Sample **decorrelated snapshots** (spacing ≳ one large-eddy turnover).

---

## The LES filter and test filter

Let `⟨·⟩` denote the grid filter at width Δ (box or Gaussian), and `^(·)` the test
filter at width Δ̂ = αΔ with **α = 2**. On a periodic grid both are cheap in
Fourier space:

- Gaussian: transfer function `Ĝ(k) = exp(−k²Δ²/24)`; `⟨f⟩ = IFFT(Ĝ · FFT(f))`.
- Box: `Ĝ(k) = sinc(kΔ/2)`.

For **compressible** flow use Favre (density-weighted) filtering:
`ũ_i = ⟨ρ u_i⟩ / ⟨ρ⟩`. At M ≈ 0.2 with ρ′/ρ ≲ 1%, plain ≈ Favre; either is
defensible if the ρ′/ρ value is quoted.

---

## Correct definitions (d = spatial dimension; 3 in Athena++)

**Resolved field:** `ū = ⟨u⟩` (or Favre `ũ`).

**True SGS stress (the target's ground truth):**

    τ_ij = ⟨u_i u_j⟩ − ū_i ū_j          (Favre: ⟨ρ u_i u_j⟩/⟨ρ⟩ − ũ_i ũ_j)

**Deviatoric (trace-free) part — the modelled quantity:**

    τᵈ_ij = τ_ij − (1/d) τ_kk δ_ij      (τ_kk = 2·k_sgs ≥ 0)

**Resolved strain rate and its invariants (from gradients of ū):**

    S̄_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i)
    S̄ᵈ_ij = S̄_ij − (1/d) S̄_kk δ_ij
    |S̄| = sqrt(2 S̄ᵈ:S̄ᵈ)               (A:B ≡ Σ_ij A_ij B_ij)

**Smagorinsky model tensor** (coefficient factored out; C_d = C_s²):

    ν_t = (C_s Δ)² |S̄|
    τᵈ_ij ≈ −2 ν_t S̄ᵈ_ij = C_d · M_ij ,   M_ij ≡ −2 Δ² |S̄| S̄ᵈ_ij

**SGS energy dissipation (forward > 0, backscatter < 0):**

    ε = −⟨τᵈ_ij S̄ᵈ_ij⟩

**Scale-similarity (Leonard / Bardina) tensor** — computable from ū alone:

    Lᵈ_ij = ( ^(ū_i ū_j) − ^ū_i ^ū_j )ᵈ

**Dynamic Smagorinsky coefficient (Germano–Lilly, localized).** Use the test
filter `^` and average the **entire contracted scalar** (not L and M separately):

    C(x) = ^( Lᵈ_ij 𝓜_ij ) / ^( 𝓜_ij 𝓜_ij )
    𝓜_ij = −2[ Δ̂² |^S̄| ^S̄ᵈ_ij − Δ² ^( |S̄| S̄ᵈ_ij ) ]      (entire-term test filter)
    C_s = sqrt( max(C, 0) )

Notes: the second term in `𝓜` is the test filter applied to the *product*
`|S̄|S̄ᵈ`, not `|^S̄|·^S̄ᵈ`. Numerator and denominator are filtered separately;
the denominator `^(𝓜:𝓜)` is positive-definite and does not vanish, so C is
well-defined. Clip C ∈ [0, C_s,max] only when *using* it as a model (negative C
= backscatter, which eddy viscosity cannot represent); for *a-priori analysis*
keep the unclipped C to see the backscatter.

---

## Established results (what the analysis shows — keep these in mind)

These are robust across resolution, filter width, filter type, and two
independent gradient estimators, and were confirmed by an independent grid/FFT
a-priori pipeline.

1. **Eddy viscosity gets the MAGNITUDE right, the DIRECTION wrong.** Decompose
   the model skill into two separate correlations:
   - `corr(|τᵈ|, |S̄|) ≈ 0.65` — strong and stable. The stress is large where the
     strain is large; the eddy-viscosity *scaling* `|τ| ~ |S̄|` is correct, and
     this is **not** noise-limited (it holds as k_sgs grows).
   - `corr(M_ij, τᵈ_ij) ≈ 0` — the full-tensor *alignment*. `τᵈ` is not parallel
     to `S̄ᵈ`; Smagorinsky points the stress the wrong way. Reporting only this
     pooled tensor-Pearson hides the magnitude correlation — always report both.
2. **Scale similarity captures magnitude AND direction:** `corr(Lᵈ, τᵈ) ≈ 0.7–0.9`.
   The SGS stress lives in the velocity-gradient / similarity subspace, not the
   strain-eigenframe.
3. **2D is backscatter-dominated** (`ε < 0`); 3D forced turbulence is
   forward-cascade (`ε > 0`, C_s ≈ 0.1).
4. Tensor *alignment* metrics must be evaluated where there is real signal —
   ensure k_sgs is well above the noise floor (scale separation), or the test is
   degenerate.

**The ML thesis, in one line:** keep the `|S̄|` magnitude scaling (already
correct) and **learn the tensor direction** from the neighbour / scale-similarity
structure. This is exactly what an attention-over-neighbours model can do that a
purely local strain model cannot — and it defines the central MLP-vs-attention
ablation: *does non-local information recover the direction?*

---

## ML target and input features (the physics–ML boundary — author-owned)

**Target = a single dimensionless coefficient** (avoids equivariance machinery;
the tensor structure is supplied by a known basis). Two principled choices:

- **Smagorinsky coefficient** (project τᵈ onto the strain basis):
  `C_d = −(τᵈ:S̄ᵈ)/(Δ²|S̄|³)`. Simple, but inherits the directional ceiling above.
- **Similarity / mixed coefficient** (project onto the Leonard basis):
  `C = (τᵈ:Lᵈ)/(Lᵈ:Lᵈ)`. Keeps the correct direction (≈0.9 ceiling). Preferred.

Either way, train with a loss **weighted by the Lilly weight** (the denominator,
`Δ²|S̄|³` or `𝓜:𝓜` or `Lᵈ:Lᵈ`) so ill-conditioned low-strain points are
down-weighted; equivalently, regress in stress space and minimize
`Σ_i w_i (C_pred − C_true)²`.

**Input features — all Galilean- and rotation-invariant** (computed from the
resolved field at the filter scale):
- `|S̄|` (strain magnitude), `|ω̄|` (vorticity magnitude), `∇·ū` (dilatation,
  weakly compressible), `ρ/ρ_0` (dimensionless density).
- In 2D, `tr(S̄²) = ½|S̄|²` is redundant with `|S̄|`. Drop pure constants (e.g. a
  fixed smoothing length).
- Standardize (mean 0, var 1) using **train-split statistics only**.

**No raw positions, raw velocities, or raw coordinate-frame tensor/vector
components as inputs.** Flag any feature that breaks Galilean/rotation invariance.

---

## Models, baselines, protocol

- **MLP**: invariant features → coefficient (local, de-risking baseline).
- **Attention**: single layer, single head, neighbour gather with masking;
  invariant relative-position encoding (distance + projections onto the strain or
  similarity frame). Same invariant target. The point is whether neighbour info
  recovers the direction. **Stop at single-layer/single-head** — no multi-head,
  no full GNN, no a-posteriori coupling (those are "future work").
- **Baselines**: static Smagorinsky (C_s ≈ 0.1–0.16), dynamic Smagorinsky
  (Germano–Lilly above), scale-similarity / dynamic mixed model.
- **Split by snapshot (and/or by forcing seed), never by cell/particle** — avoids
  leakage between train and test.
- **Evaluation**: report magnitude correlation AND tensor alignment AND
  eigenvector-alignment, the SGS dissipation `ε` (and its PDF / backscatter
  fraction), and reconstructed-stress error. Always alongside the spectrum and
  k_sgs/k_res so the regime is documented.

---

## Validation practices (always do these)

- **Validate every operator against an analytic field**: a filter must reproduce
  a constant exactly; a gradient must reproduce a linear field exactly; the strain
  of a solid-body rotation must be zero; the strain of a pure shear must give the
  known tensor. (On a grid, spectral derivatives are exact to machine precision.)
- **Cross-check the a-priori pipeline with an independent method** (e.g. a second
  filter type, or FFT vs finite-difference). Match grid resolution to data
  resolution when interpolating, or interpolation noise will inflate small-scale
  energy.
- **Separate magnitude from direction** in every correlation.
- **Confirm a real inertial range** (spectrum) and **substantial k_sgs** before
  trusting any SGS correlation.

---

## Scope and division of labour

- **Must-ship core**: data pipeline → invariant target+features → baselines →
  MLP → comparison. The MLP-vs-baselines comparison alone is a complete artifact.
  The attention model is the ambitious upside; never let it jeopardize shipping.
- **Author owns** (defensibility / learning goal): the neural-network definitions,
  the training loop, and the SGS *target* + *invariant feature* definitions.
- **Assistant builds**: the Athena++ data-generation + snapshot pipeline, the
  filter/gradient/SGS operators, the analytical baselines (given the formulas),
  diagnostics, plotting, and evaluation harnesses.
- Default to the simplest thing that answers the question. When a modelling choice
  has a physical justification, attach it. Flag any comparison that changes two
  variables at once (e.g. learned-vs-analytical AND local-vs-nonlocal) so the
  ablation stays clean.
