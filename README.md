# KorzyńskiSpin

Quasi-local spin of a (possibly non-axisymmetric) apparent horizon via
Korzyński's conformal decomposition:

- Mikołaj Korzyński, *Quasi-local angular momentum of non-symmetric isolated
  and dynamical horizons from the conformal decomposition of the metric*,
  Class. Quantum Grav. **24**, 5935 (2007),
  [arXiv:0707.2824 [gr-qc]](https://arxiv.org/abs/0707.2824).

The algorithm — induced metric and rotation one-form from Cauchy data, Hodge
gauge fixing, uniformization by two-dimensional Ricci flow with Newton
polish, the ℓ=1 eigenfunctions of the round Laplacian, the Möbius generators
in the original chart, and the invariants $\vec J$, $\vec K$, the spin
$J = \sqrt{(A+\sqrt{A^2+4B^2})/2}$, the spin axis, and the axial vector
field — is worked out in detail in [docs/algorithm.tex](docs/algorithm.tex).
A survey of related methods is in
[docs/related_work.tex](docs/related_work.tex).

## Usage

```julia
using KorzyńskiSpin
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

For a horizon given as a radial shape function $h(\theta,\phi)$ about a
centre (as produced by an apparent horizon finder), use
`shape_embedding(h; center=c)`.

All angular derivatives are taken pseudospectrally (spin-weighted spherical
harmonics via
[AbstractSphericalHarmonics.jl](https://github.com/eschnett/AbstractSphericalHarmonics.jl)
/ [SSHT.jl](https://github.com/eschnett/SSHT.jl)); no derivatives of the
Cauchy data are required.

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
  convergence in `lmax`.

## History

The pre-1.0 version of this package (preserved in the git history) was an
earlier, incomplete attempt; it stalled at the construction of the rotation
generators in distorted coordinates, which is solved here by evaluating the
Möbius generators directly in the original chart
($\phi_i{}^A = \mathring\epsilon^{AB}\partial_B\chi_i$,
$\xi_i{}^A = -\mathring q^{AB}\partial_B\chi_i$; Lemma 1 in
docs/algorithm.tex).
