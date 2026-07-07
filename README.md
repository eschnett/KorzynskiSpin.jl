# KorzynskiSpin

[![CI](https://github.com/eschnett/KorzynskiSpin.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/KorzynskiSpin.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/eschnett/KorzynskiSpin.jl/actions/workflows/docs.yml/badge.svg)](https://eschnett.github.io/KorzynskiSpin.jl/)

Quasi-local spin of a (possibly non-axisymmetric) apparent horizon via
Korzyński's conformal decomposition:

- Mikołaj Korzyński, *Quasi-local angular momentum of non-symmetric isolated
  and dynamical horizons from the conformal decomposition of the metric*,
  Class. Quantum Grav. **24**, 5935 (2007),
  [arXiv:0707.2824 [gr-qc]](https://arxiv.org/abs/0707.2824).

The algorithm — induced metric and rotation one-form from Cauchy data, Hodge
gauge fixing, uniformization by a Gundlach-style fast flow
([arXiv:gr-qc/9707050](https://arxiv.org/abs/gr-qc/9707050)), the ℓ=1
eigenfunctions of the round Laplacian, the Möbius generators
in the original chart, and the invariants $\vec J$, $\vec K$, the spin
$J = \sqrt{(A+\sqrt{A^2+4B^2})/2}$, the spin axis, and the axial vector
field — is worked out in detail in
[docs.tex/algorithm.tex](docs.tex/algorithm.tex) (also rendered in the
package documentation).  A survey of related methods is in
[docs.tex/related_work.tex](docs.tex/related_work.tex).

## Usage

```julia
using KorzynskiSpin
using StaticArrays

# Inputs: a parametrized surface and the Cauchy data (γ_ij, K_ij) as
# functions of the Cartesian slice coordinates.  The extrinsic curvature
# convention is K_ij = −(1/2) £_n γ_ij (the ADM/NR standard).
embedding(θ, ϕ) = SVector(...)          # surface point x⃗(θ, ϕ)
metric3(x⃗)  = SMatrix{3,3}(...)         # γ_ij at x⃗
excurv3(x⃗) = SMatrix{3,3}(...)          # K_ij at x⃗

result = horizon_spin(embedding, metric3, excurv3; lmax=24)
result.J               # the spin (G = 1)
result.axis_embedding  # spin axis as a spatial direction
result.axial           # axial vector field on the surface (dyad components)
result.diagnostics     # residuals of the internal identities
```

`metric3` and `excurv3` use a *batched* interface: they may be given as
callables `Xs -> array` that receive **all** collocation points at once (a
grid-shaped `Matrix{SVector{3}}`) and return γ_ij / K_ij in an array of the
same shape, so the caller can parallelize the evaluation (threads, `pmap`, GPU,
batched autodiff).  A per-point callable is still accepted: one whose argument
is annotated `x::SVector{3}` (as above) is detected and wrapped automatically,
and a bare untyped closure can be wrapped explicitly with `pointwise`.

For a horizon given as a radial shape function $h(\theta,\phi)$ about a
centre (as produced by an apparent horizon finder), use
`shape_embedding(h; center=c)`.

All angular derivatives are taken pseudospectrally (spin-weighted spherical
harmonics via
[AbstractSphericalHarmonics.jl](https://github.com/eschnett/AbstractSphericalHarmonics.jl)
/ [FastSphericalHarmonics.jl](https://github.com/eschnett/FastSphericalHarmonics.jl));
no derivatives of the Cauchy data are required.

## Using ApparentHorizonFinder

[ApparentHorizonFinder.jl](https://github.com/eschnett/ApparentHorizonFinder)
(v2) finds the horizon shape on the same
[AbstractSphericalHarmonics](https://github.com/eschnett/AbstractSphericalHarmonics.jl)
grid objects, so its result feeds in directly — no interpolation:

```julia
using ApparentHorizonFinder, KorzynskiSpin

horizon = find_horizon(admvars, guess_origin, EquiangularGrid(15), guess_radius)
result  = horizon_spin(horizon, metric3, excurv3)
# or at a different resolution (spectral resampling):
result  = horizon_spin(horizon, metric3, excurv3; grid=EquiangularGrid(23))
```

A matrix of surface points at the collocation points (e.g. from
`horizon_points`) is accepted as well:
`horizon_spin(points, metric3, excurv3; grid=...)`.

## Validation

`Pkg.test()` checks, among others (see `test/runtests.jl`):

- round and off-center spheres and ellipsoids in flat space ($J = 0$,
  Gauss–Bonnet, uniformization and Takahashi-rigidity residuals,
  reconstruction of the conformal coordinates),
- an analytic rotation one-form on the round sphere (machine-precision
  agreement of $\vec J$ with the closed-form answer, gauge invariance under
  $\omega \to \omega + \mathrm{d}g$),
- Kerr in the Kerr–Schild slicing (analytic data from
  [SpacetimeMetrics.jl](https://github.com/eschnett/SpacetimeMetrics),
  including its `rotate`/`translate` transformers): $J = Ma$ to machine
  precision ($\sim 10^{-15}$ at `lmax=20`), $\vec K = 0$, spin axis
  recovered, also under rotations+translations of the data, with spectral
  convergence in `lmax`,
- a rotated, translated, **boosted** Kerr black hole whose shape is found
  numerically by ApparentHorizonFinder: the boost genuinely changes the
  slicing, and $J = Ma$ and the area still hold to $\sim 10^{-10}$ — the
  "tilted foliation" acid test of Korzyński's gauge fixing.

## Related work

To our knowledge this is the first implementation of Korzyński's
conformal-decomposition spin; the citation record of the paper contains
theory papers and codes that chose other spin definitions.  A detailed
survey is in [docs.tex/related_work.tex](docs.tex/related_work.tex); in
brief:

- *Closest in spirit*: Jasiulek's invariant-integral method
  ([arXiv:0906.1228](https://arxiv.org/abs/0906.1228)) computes
  quasi-local spin pseudospectrally without solving a Killing equation,
  but from moments of curvature invariants rather than conformal
  decomposition; no public code.  Jasiulek & Korzyński's spectral
  embedding flow ([arXiv:1111.6523](https://arxiv.org/abs/1111.6523))
  solves the related isometric-embedding problem, not uniformization.
- *Production codes* evaluate the same surface integral
  $J_\phi = -\frac{1}{8\pi}\oint\omega(\phi)\,\epsilon$ with a
  different choice of rotation vector $\phi$: SpEC uses
  Cook–Whiting/Owen approximate Killing vectors
  ([arXiv:0706.0199](https://arxiv.org/abs/0706.0199),
  [arXiv:1708.07325](https://arxiv.org/abs/1708.07325)); the Einstein
  Toolkit's
  [QuasiLocalMeasures](https://einsteintoolkit.org/thornguide/EinsteinAnalysis/QuasiLocalMeasures/documentation.html)
  uses Killing transport (Dreyer–Krishnan–Schnetter–Shoemaker,
  [arXiv:gr-qc/0206008](https://arxiv.org/abs/gr-qc/0206008));
  Caudill–Cook–Grigsby–Pfeiffer used flat-space conformal Killing
  vectors ([arXiv:gr-qc/0605053](https://arxiv.org/abs/gr-qc/0605053)),
  which Korzyński notes is equivalent to his definition only in special
  cases.  Rácz's axial-vector construction
  ([arXiv:2401.14251](https://arxiv.org/abs/2401.14251)) is a recent
  analytic alternative.
- *Building blocks*: surface uniformization by discrete Ricci flow is
  standard in computational conformal geometry (Gu–Luo et al.), and the
  eth/spin-weighted-harmonics infrastructure follows Gómez et al.
  ([arXiv:gr-qc/9702002](https://arxiv.org/abs/gr-qc/9702002)).

Since no reference implementation of this method exists, validation
rests on analytic limits (Kerr, gauge distortions) and cross-checks
against the other definitions on identical data.
