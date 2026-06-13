# KorzyńskiSpin.jl

Quasi-local spin of a (possibly non-axisymmetric) apparent horizon via
Korzyński's conformal decomposition (Korzyński, Class. Quantum Grav. **24**,
5935 (2007), [arXiv:0707.2824](https://arxiv.org/abs/0707.2824)).

The method is worked out in full detail in [The algorithm](@ref "Algorithm") — from the Cauchy data and horizon shape through the
rotation one-form, Hodge gauge fixing, uniformization by Ricci flow, the
Möbius generators, and the invariants ``\vec J``, ``\vec K`` to the spin
``J``, the spin axis, and the axial vector field.  A survey of other
quasi-local spin prescriptions and their implementations is in
[Related work](@ref "Related work").  Typeset PDF versions:
[algorithm.pdf](assets/algorithm.pdf),
[related\_work.pdf](assets/related_work.pdf) (the LaTeX sources in
`docs.tex/` are canonical).

## Using ApparentHorizonFinder

`ApparentHorizonFinder.find_horizon` (v2) returns its shape on the same
`AbstractSphericalHarmonics` grid objects, so the result feeds directly into
`horizon_spin(horizon, metric3, excurv3)` — optionally with a different
`grid=` for spectral resampling.

## API reference

```@autodocs
Modules = [KorzyńskiSpin]
```
