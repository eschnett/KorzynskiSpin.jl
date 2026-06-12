# Goal for this package

## Outline

The paper "Quasi-local angular momentum of non-symmetric isolated and
dynamical horizons from the conformal decomposition of the metric" by
Mikolaj Korzynski (Class. Quantum Grav. 24, 5935 (2007), DOI
10.1088/0264-9381/24/23/015; arXiv:0707.2824 [gr-qc]) describes a
method to calculate the spin of a distorted horizon. This package
implements the proposed algorithm using Ricci flow and calculates the
invariants $J_i$ and $K_i$ (eqns. (12) and (13)).

## Inputs

The input to the algorithm are the shape of an apparent horizon, given
as distance function h(\theta, \phi) from a centre point. The distance
function is expanded into spherical harmonics.

The metric, extrinsic curvature, and their derivatives are available
at collocation points.

## Algorithm

The paper implicitly suggests the following algorithm. See the section
"Conformal decomposition and transformations" where $F(\theta, \phi)$
is introduced:

- Choose an arbitrary coordinate system on the sphere (implicitly
  given by how h is provided)
- Calculate the two-metric $q_{ab}(\theta, \phi)$ on the horizon
- Calculate the Ricci scalar $R[q_{ab}]$ as a function of the
  two-metric
- Use Ricci flow to find a conformally round metric $q_0$. This metric
  will be given in the original $(\theta, \phi)$ coordinates, and
  while it will be round, it will not be manifestly so.
- Find a "good" coordinate system for $q_0$, i.e. a coordinate system
  $(\theta_1, \phi_1)$ in which $q_0$ will be manifestly round, i.e.
  $q_0(\theta_1, \phi_1) = F \diag(1, (\sin \theta_1)^2)$. Ideas for
  this "finding" are given below.
- Undo the Ricci flow to find $q(\theta_1, \phi_1)$, the original
  two-metric, now in "good" coordinates.

With this transformation in hand the other quantities, in particular
$J_i$ (12) and $K_i$ (13), can be calculated.

Ideas for finding a "good" coordinate system for a round (but not
manifestly round) sphere:

- Define three scalar function $x$, $y$, and $z$ on the sphere. These
  functions need to satisfy certain conditions.
- They all range from $-1$ to $+1$.
- Their integrals over the sphere vanish.
- They are orthogonal when integrated over the sphere.
- They are properly normalized when square-integrated over the sphere.
- In fact, these functions will be the values of $Y_{lm}$ for $l=1$
  and $m={-1, +1, 0}$, up to normalization and sign conventions.
- Determine $(\theta_1, \phi_1)$ from $x$, $y$, and $z$.

These are linear conditions (except the normalization) on $x$, $y$,
and $z$. With suitable additional conditions to select a north pole
and a zero meridian they can presumable be solved as linear system or
as eigenvalue problem.

## Output

- Spin $J$
- Axial vector field $\phi$

## Other packages

- ApparentHorizonFinder (to obtain the surface)
- FastSphericalHarmonics (for spin-weighted spherical harmonics)
- SpacetimeMetrics (to define metrics)
- KorzyńskiSpin (earlier, incomplete attempt)
