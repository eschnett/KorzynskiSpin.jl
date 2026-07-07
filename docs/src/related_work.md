# Related work



We survey the literature for existing implementations of Korzyński's
conformal-decomposition definition of quasi-local horizon angular momentum
 [1], i.e. the pipeline: uniformization of the horizon
two-metric (e.g. by Ricci flow), construction of the Möbius generators,
and evaluation of the invariants ``\vec{J}`` and ``\vec{K}``.  To the best of our
knowledge *no published paper or public code implements this method
end-to-end*; its citation record is small and consists of theory papers and
of codes that adopted different spin definitions.  What does exist falls into
three groups: a methodologically adjacent (but different) spectral method by
Jasiulek; the approximate-Killing-vector and Killing-transport spin measures
implemented in the SpEC and Einstein Toolkit production codes; and
general-purpose surface-uniformization software from computational conformal
geometry.  Consequently `KorzynskiSpin.jl` would, as far as we can
tell, be the first implementation, and validation must rely on analytic
limits (Kerr, tilted slicings) and on cross-checks against the other spin
definitions on the same data.



## Summary

A search of the citation record of  [1], via Semantic
Scholar and arXiv full-text search (June 2026), finds only a handful of
citing works, none of which implements the method: they are reviews
 [13, 14], theoretical developments
 [10, 11, 12], and numerical-relativity papers that
cite it as an alternative while implementing approximate-Killing-vector
measures instead  [5, 6].  Notably, the follow-up paper
announced in  [1] itself (“in preparation”, meant to
contain the angular momentum flux law and the first law for non-axisymmetric
horizons) appears never to have been published, and we found no numerical
implementation by the author either.

The remainder of these notes describes the closest relatives, ordered by
proximity to the present algorithm.


## Closest relative in spirit: Jasiulek's invariant-integral method

Jasiulek  [2] computes quasi-local spin, mass and higher
multipole moments on marginally trapped surfaces in ``3+1`` simulations with a
pseudospectral, spherical-harmonic-based method that — like the present
algorithm, and unlike Killing-vector approaches — avoids solving the
Killing equation.  The mechanism is however entirely different: spin is
extracted from *surface moments of curvature invariants* (the scalar
2-curvature and the rotational Weyl scalar ``\operatorname{Im}\Psi_2``), e.g.
central moments ``\mu_n(\cdot) = \langle(\langle\cdot\rangle-\cdot)^n\rangle``,
which are related algebraically to the isolated-horizon multipoles and
inverted for the spin.  No conformal decomposition, Ricci flow, or
Laplacian-eigenfunction construction is used (an adapted-coordinates
transformation is introduced only for axisymmetric checks), and
 [1] is not cited.  No public code was released.  For the
`KorzynskiSpin.jl` test suite, the published Kerr and
perturbed-black-hole numbers of  [2] are nevertheless a
useful independent comparison point.

Jasiulek and Korzyński later developed a spectral *embedding flow*
 [3] that isometrically embeds horizon two-metrics into
Euclidean ``\mathbb{R}^3`` (Weyl's embedding problem, via a Weingarten-type
flow).  This is methodologically adjacent — a geometric flow on horizon
data, solved with spherical-harmonic spectral methods — but it solves a
different problem (isometric embedding rather than uniformization) and
computes no angular momentum.


## What production numerical-relativity codes implement instead

All widely used codes evaluate the same surface integral
``J_\phi = -\frac{1}{8\pi G}\oint\omega(\phi)\,\epsilon`` that underlies
 [1], but with a different prescription for the rotation
vector ``\phi``:

- **SpEC: approximate Killing vectors.**  The Cook–Whiting
  construction  [4], further developed by Owen, finds the
  vector field minimizing a Killing-equation residual by solving a
  generalized eigenvalue problem on the horizon.  It is the standard spin
  diagnostic in the SXS collaboration's SpEC code, used e.g. for
  nearly extremal horizons  [6] and for the systematic study
  of *spin direction* measures by Owen et al.  [5].  Both
  papers cite  [1] as an alternative definition but do not
  implement it.
- **Einstein Toolkit: `QuasiLocalMeasures`.**  The
  `QuasiLocalMeasures` thorn  [8] computes the
  Dreyer–Krishnan–Schnetter–Shoemaker isolated-horizon spin
   [7] with Killing vectors obtained by Killing transport, or
  with coordinate rotational vectors.
- **Flat-space conformal Killing vectors.**  Caudill, Cook,
  Grigsby and Pfeiffer  [9] used the rotational Killing
  vectors of the flat conformal background in initial-data spin
  measurements;  [1] notes explicitly that this is
  equivalent to the conformal-decomposition definition in many special
  cases, but not in general (no invariance discussion, no gauge fixing of
  ``\omega``).
- **Rácz's axial vector construction.**  A recent alternative
   [10] constructs axial vector fields from
  “centre-of-mass” unit-sphere reference systems; interestingly it also
  imposes ``\ell=1``-type conditions on reference functions on the sphere,
  conceptually close to the eigenfunction triple ``\chi_i`` of the present
  algorithm.  It cites  [1], is purely analytic, and has no
  accompanying software that we could find.


## Building blocks that exist as software

The individual ingredients of the algorithm are all available, just never
assembled for this purpose:

- **Uniformization.**  Discrete surface Ricci flow and spherical
  conformal parameterization are mature topics in computational conformal
  geometry (Gu, Luo and collaborators  [15]; various public
  MATLAB/Python spherical-conformal-map tools).  These solve exactly the
  “round representative plus conformal factor” problem, but for triangle
  meshes rather than spectrally, and know nothing of ``\omega``, ``\vec{J}``,
  ``\vec{K}``.
- **Spectral horizon machinery.**  The ``\eth``-formalism
   [16] and fast spin-weighted spherical-harmonic transforms
  (e.g. the `FastSphericalHarmonics` Julia package) provide the
  pole-regular derivative and quadrature infrastructure; the pre-1.0
  prototype of this package implemented the curvature computation of
  a horizon two-metric in this framework but did not reach the flow,
  eigenfunction, or invariant stages.
- **The rotation one-form integral.**  The integrand
  ``K_{ij}\phi^i s^j`` is implemented in every isolated-horizon spin
  diagnostic  [7, 8]; only the choice of ``\phi`` differs.


## Consequences for validation

Since no reference implementation of  [1] exists, the test
strategy of the companion document `algorithm.tex` cannot include a
code-to-code comparison of the method itself.  The available external
anchors are:
- analytic limits: Kerr (``J=Ma``, axis along the symmetry axis,
  ``\vec{K}=0`` after gauge fixing), Schwarzschild and gauge-distorted
  Schwarzschild (``J=0``);
- the tilted-slicing Kerr test proposed in  [1] itself,
  which exercises the Hodge gauge fixing;
- cross-checks against *other* spin definitions on identical data:
  approximate-Killing-vector spin (SpEC-style  [4]),
  Killing-transport spin (`QuasiLocalMeasures`  [8]), and
  Jasiulek's published values  [2] — agreement is expected
  for nearly axisymmetric horizons, with quantifiable differences in
  strongly distorted regimes where the definitions genuinely differ.

## References

1. M. Korzyński,
*Quasi-local angular momentum of non-symmetric isolated and dynamical
horizons from the conformal decomposition of the metric*,
Class. Quantum Grav. **24**, 5935 (2007);
[arXiv:0707.2824 [gr-qc]](https://arxiv.org/abs/0707.2824).

2. M. Jasiulek,
*A new method to compute quasi-local spin and other invariants on
marginally trapped surfaces*,
Class. Quantum Grav. **26**, 245008 (2009);
[arXiv:0906.1228 [gr-qc]](https://arxiv.org/abs/0906.1228).

3. M. Jasiulek, M. Korzyński,
*Isometric embeddings of 2-spheres by embedding flow for applications
in numerical relativity*,
Class. Quantum Grav. **29**, 155010 (2012);
[arXiv:1111.6523 [gr-qc]](https://arxiv.org/abs/1111.6523).

4. G. B. Cook, B. F. Whiting,
*Approximate Killing vectors on ``S^2``*,
Phys. Rev. D **76**, 041501(R) (2007);
[arXiv:0706.0199 [gr-qc]](https://arxiv.org/abs/0706.0199).

5. R. Owen, A. S. Fox, J. A. Freiberg, T. Pierre Jacques,
*Black hole spin axis in numerical relativity*,
Phys. Rev. D **99**, 084031 (2019);
[arXiv:1708.07325 [gr-qc]](https://arxiv.org/abs/1708.07325).

6. G. Lovelace et al.,
*Nearly extremal apparent horizons in simulations of merging black
holes*,
Class. Quantum Grav. **32**, 065007 (2015);
[arXiv:1411.7297 [gr-qc]](https://arxiv.org/abs/1411.7297).

7. O. Dreyer, B. Krishnan, E. Schnetter, D. Shoemaker,
*Introduction to isolated horizons in numerical relativity*,
Phys. Rev. D **67**, 024018 (2003);
[arXiv:gr-qc/0206008](https://arxiv.org/abs/gr-qc/0206008).

8. E. Schnetter et al.,
`QuasiLocalMeasures` thorn, Einstein Toolkit;
<https://einsteintoolkit.org/thornguide/EinsteinAnalysis/QuasiLocalMeasures/documentation.html>.

9. M. Caudill, G. B. Cook, J. D. Grigsby, H. P. Pfeiffer,
*Circular orbits and spin in black-hole initial data*,
Phys. Rev. D **74**, 064011 (2006);
[arXiv:gr-qc/0605053](https://arxiv.org/abs/gr-qc/0605053).

10. I. Rácz,
*Quasi-local spin-angular momentum and the construction of axial
vector fields*,
Phys. Rev. D (2025);
[arXiv:2401.14251 [gr-qc]](https://arxiv.org/abs/2401.14251).

11. A. Ashtekar, N. Khera, M. Kolanowski, J. Lewandowski,
*Non-expanding horizons: multipoles and the symmetry group*,
JHEP **01** (2022) 028;
[arXiv:2111.07873 [gr-qc]](https://arxiv.org/abs/2111.07873).

12. I. Booth, S. Fairhurst,
*Extremality conditions for isolated and dynamical horizons*,
Phys. Rev. D **77**, 084005 (2008);
[arXiv:0708.2209 [gr-qc]](https://arxiv.org/abs/0708.2209).

13. J. L. Jaramillo,
*An introduction to local black hole horizons in the 3+1 approach to
general relativity*,
Int. J. Mod. Phys. D **20**, 2169 (2011);
[arXiv:1108.2408 [gr-qc]](https://arxiv.org/abs/1108.2408).

14. L. B. Szabados,
*Quasi-local energy-momentum and angular momentum in general
relativity*,
Living Rev. Relativity **12**, 4 (2009);
[DOI:10.12942/lrr-2009-4](https://doi.org/10.12942/lrr-2009-4).

15. X. D. Gu, S.-T. Yau,
*Computational Conformal Geometry*,
Advanced Lectures in Mathematics 3, International Press (2008);
see also M. Jin, J. Kim, F. Luo, X. Gu,
*Discrete surface Ricci flow*,
IEEE Trans. Vis. Comput. Graphics **14**, 1030 (2008).

16. R. Gómez, L. Lehner, P. Papadopoulos, J. Winicour,
*The eth formalism in numerical relativity*,
Class. Quantum Grav. **14**, 977 (1997);
[arXiv:gr-qc/9702002](https://arxiv.org/abs/gr-qc/9702002).



