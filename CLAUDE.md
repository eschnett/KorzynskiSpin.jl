# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Julia package that computes the quasi-local spin of a (possibly
non-axisymmetric) apparent horizon using Korzyński's conformal decomposition
(CQG 24, 5935 (2007), arXiv:0707.2824). The main entry point is
`horizon_spin(...)` → `SpinResult` (`src/invariants.jl`). The optional
follow-up `horizon_multipoles(::SpinResult)` → `HorizonMultipoles`
(`src/multipoles.jl`) computes the shape and current multipoles.

## Commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                  # full test suite (test/runtests.jl)
julia -e 'using JuliaFormatter; format(".")'                  # format (settings in .JuliaFormatter.toml: blue style, margin 132)
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl                            # build Documenter site
(cd docs.tex && latexmk -pdf algorithm.tex)                   # rebuild the canonical TeX write-up
```

- All tests sit in one file, `test/runtests.jl`, as nested `@testset`s.
  There is no test filter, so to run one testset, run that block by itself
  in a REPL where the test environment is active. The test deps
  (ApparentHorizonFinder, SpacetimeMetrics) come from `test/Project.toml`.
  `test/kerr_schild.jl` builds the analytic Kerr–Schild Cauchy data.
- CI runs Julia 1.11 and 1.12 on Linux, macOS and Windows (x64, arm64, x86).
- `viz/` is a separate Makie sandbox with its own `Project.toml` and is not
  part of the package (see `viz/README.md`).

## Documentation conventions

- `docs.tex/algorithm.tex` and `docs.tex/related_work.tex` are the
  **canonical** specification. `docs/src/algorithm.md` mirrors them by hand,
  with hand-numbered `\tag{N}` equation numbers that must match the compiled
  PDF. Adding or removing a numbered display in the TeX shifts every later
  number, and untagged `\begin{equation}` blocks use up numbers too. Check
  against `\newlabel{eq:...}` in the `.aux` file.
- After editing the TeX, copy the rebuilt PDF to **both**
  `docs.tex/algorithm.pdf` and `docs/src/assets/algorithm.pdf` (both are
  committed).
- A local docs build can rewrite the tracked `docs/Project.toml` (it adds a
  `[sources]` entry) and leaves a `docs/build/` directory behind. Revert or
  delete them before committing.
- Source comments cite section and equation numbers as
  "docs/algorithm.tex §3.x" or "docs/src/algorithm.md". The file is
  actually at `docs.tex/algorithm.tex`.

## Architecture

The pipeline runs in `horizon_spin_geom` (`src/invariants.jl`). Each step
matches a section of `algorithm.tex`, and the step outputs are the fields of
`SpinResult`:

1. **Geometry** (`geometry.jl`): the embedding (a callable `(θ,ϕ)->x⃗`, a
   point matrix, or an ApparentHorizonFinder NamedTuple with
   `origin/grid/hlm`) gives the induced metric `q`, the area and the
   rotation one-form `ω_a = −K_ij e_a^i s^j`. The embedding is
   differentiated spectrally, so no derivatives of the Cauchy data are
   needed. `metric3`/`excurv3` use a **batched** interface (all points at
   once). Per-point callables are detected by the `::SVector{3}` argument
   annotation (via `applicable`), and untyped ones are wrapped with
   `pointwise`.
2. **Operators** (`calculus.jl`): `MetricOps(q)` caches `q^{ab}`, the
   difference tensor `C^a_bc` relative to the unit round coordinate sphere
   `q̂`, and `√det q`. From these it builds the Laplacian, divergence and
   scalar curvature. `operator_matrix` turns an operator into a dense
   mode-space matrix. The `Δ_q` matrix is built once and reused, scaled for
   the area-normalized metric `q̄ = (4π/𝒜) q`.
3. **Hodge gauge fix** (`hodge_fix`): `ω^inv = ω − dg` with `Δ_q g = D·ω`.
   The spin is slicing-independent only because of this step.
4. **Uniformization** (`uniformize.jl`): a Gundlach-style fast flow solves
   the Liouville equation `Δ̄u − R̄/2 + e^{2u} = 0`, which makes
   `q̊ = e^{2u} q̄` unit-round **in the original chart**. The ℓ=1 modes are
   the Möbius gauge kernel, so their model eigenvalue is replaced by the ℓ=2
   value. If the flow stalls, it returns `converged=false`, which becomes
   `SpinResult.success`.
5. **Eigenfunctions** (`eigenfunctions.jl`): the triple `χ_i` (the ℓ=1
   eigenfunctions of `Δ̊`) comes from a dense generalized eigenproblem
   `Δ̄χ = −2e^{2u}χ`. The steps after that are: take a real basis of the
   degenerate cluster, Löwdin-orthonormalize with respect to the round
   measure, and fix the handedness. This leaves an arbitrary global SO(3)
   rotation, and all outputs are covariant under it. The Möbius generators
   are `φ_i = ε̊^{ab}∂_bχ_i` (rotations) and `ξ_i = −q̊^{ab}∂_bχ_i` (proper
   conformal generators). No explicit conformally spherical coordinates are
   ever built.
6. **Invariants** (`invariants.jl`): `J⃗`, `K⃗` (integrals of `ω^inv` against
   `φ_i`/`ξ_i` with the *physical* area form), the SO(1,3) invariants `A`,
   `B`, the spin `J`, the Möbius boost to the frame where `J⃗′ ∥ K⃗′`, the
   axis, and the axial field `Σ axis_i φ̃_i`.
7. **Multipoles** (`multipoles.jl`, optional, `horizon_multipoles(res)`):
   Möbius transformations act *pointwise* on `(1, χ)` as Lorentz matrices Λ,
   with `u′ = u − log (ΛX)₀`. `balance_frame` runs a Newton iteration to the
   unique round metric with `∮ χ_i ε_q = 0` (AKKL balancing). `orient_frame`
   then points ẑ along the current dipole. `I_lm = ¼∮R Y_lm ε` and
   `L_lm = ½∮ dY_lm ∧ ω` use `sYlm` evaluated at the balanced χ. The Kerr
   references are in `test/kerr_schild.jl` (GLC arXiv:2602.05823, eq. 6.42).

Field representation (`fields.jl`): fields are AbstractSphericalHarmonics
`Tensor{D}`s on a `SphereGrid` (default `EquiangularGrid`, backend
FastSphericalHarmonics). Components are stored in the orthonormal dyad
`(e_θ, e_ϕ)` of the unit coordinate sphere, so upper and lower indices agree
with respect to `q̂`, and `grad` is `∇̂` with the derivative index last.
Values are stored as `ComplexF64` even for real fields. Use `real_part` and
`imag_norm` to check and drop imaginary parts. `SpinResult.diagnostics`
collects the residuals of internal identities (Gauss–Bonnet, round
curvature, eigenvalue offset, Takahashi `Σχ_i² = 1`, Hodge, boost
consistency). Tests assert on these, and they are the first place to look
when accuracy degrades.
