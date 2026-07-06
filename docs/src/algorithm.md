# Algorithm



Korzyński  [1] defines the quasi-local angular momentum of a
non-axisymmetric marginally outer trapped surface (MOTS) through the conformal
decomposition of its two-metric and the action of the Möbius group of the
sphere.  These notes turn that definition into a concrete numerical algorithm:
starting from the shape function ``h(\theta,\varphi)`` of an apparent horizon and
the Cauchy data ``(\gamma_{ij},K_{ij})`` at its collocation points, we construct
the induced two-metric, the rotation one-form, uniformize the metric by a
fast conformal flow, obtain the Möbius generators directly in the
original coordinates from the first eigenfunctions of the round Laplacian, and
evaluate the invariants ``\vec{J}`` and ``\vec{K}`` (eqs. (12) and (13) of
 [1]), the spin ``J``, the spin axis, and the axial vector field
``\phi``.  We work out every formula needed for a pseudospectral implementation
based on spin-weighted spherical harmonics, and we collect the corrections and
simplifications relative to the algorithm skeleton in `GOAL.md`; the
most important one is that no explicit “good” coordinate system needs to be
constructed and the uniformization never needs to be “undone”.




## Notation and conventions

- Spacetime signature ``({-}{+}{+}{+})``; geometric units ``c=1``; we keep
  Newton's constant ``G`` explicit (set ``G=1`` in code).
- Greek indices ``\mu,\nu,\dots`` are four-dimensional, lower-case Latin
  indices ``i,j,k,\dots`` are spatial (three-dimensional), capital Latin indices
  ``A,B,C,\dots`` are two-dimensional indices on the horizon surface
  ``\Delta\simeq S^2``, with coordinates ``(\theta,\varphi)``.
- ``\gamma_{ij}`` is the spatial metric of the Cauchy slice ``\Sigma``,
  ``n^\mu`` its future-pointing unit normal, and the extrinsic curvature uses
  the ADM/numerical-relativity sign convention
  ```math
K_{ij} \;=\; -\tfrac12 \mathcal{L}_n \gamma_{ij}
            \;=\; -\gamma_i{}^\mu \gamma_j{}^\nu \nabla_\mu n_\nu .
\tag{1}
```
  With the opposite (Wald) convention all components of ``\vec{J}`` and ``\vec{K}``, and
  the axial vector ``\phi``, flip sign, while ``J=|\vec{J}|`` is unaffected.
- ``q_{AB}`` is the induced (positive definite) metric on ``\Delta``, ``D_A``
  its Levi-Civita connection, ``\Delta_q`` its Laplace–Beltrami operator,
  ``\epsilon_{AB}`` its area two-form, and ``R[q]`` its Ricci scalar
  (``R=2K_{\mathrm{Gauss}}``; for the unit round sphere ``R=2``).
- The chart ``(\theta,\varphi)`` is assumed right-handed as seen from
  outside the surface, and we orient ``\epsilon_{AB}`` such that
  ``\epsilon_{\theta\varphi}=+\sqrt{\det q}>0``.  Reversing the orientation
  flips ``\vec{J}``, ``\vec{K}`` and ``\phi``.
- A ring denotes quantities of the *unit-round representative*
  ``\mathring{q}_{AB}`` produced by the uniformization (area ``4\pi``, ``R[\mathring{q}]=2``):
  ``\mathring{\epsilon}_{AB}``, ``\mathring{D}_A``, ``\mathring{\Delta}``.  A hat denotes quantities of the
  *unit coordinate sphere* ``\hat q = \mathrm{d}\theta^2 +
  \sin^2\!\theta\,\mathrm{d}\varphi^2`` in the given chart, which serves only as a
  pseudospectral bookkeeping device (the section [Numerical implementation](@ref "Numerical implementation")); ``\hat q`` is
  *not* in general conformal to ``q`` in this chart.
- In two dimensions we use repeatedly, for ``\tilde q = \mathrm{e}^{2u} q``,
  ```math
\tilde\epsilon_{AB} = \mathrm{e}^{2u}\epsilon_{AB},\qquad
    \Delta_{\tilde q} f = \mathrm{e}^{-2u}\Delta_q f,\qquad
    R[\tilde q] = \mathrm{e}^{-2u}\bigl(R[q] - 2\Delta_q u\bigr),
\tag{2}
```
  the identities ``\epsilon_A{}^C \epsilon_C{}^B = -\delta_A^B``,
  ``\epsilon^{AC}\epsilon_{BC} = \delta^A_B``, and the conformal invariance of
  the Hodge star on one-forms, ``(\star\alpha)_A = \epsilon_A{}^B \alpha_B``,
  which is unchanged under ``q \to \mathrm{e}^{2u} q``.


## Korzyński's definition of angular momentum

This section summarizes the parts of  [1] that the algorithm
implements; equation numbers in parentheses refer to that paper.

### MOTS geometry and the rotation one-form

A MOTS ``\Delta`` carries the metric ``q_{AB}``, the area form ``\epsilon``, and the
*rotation one-form*
```math
\omega_A \;=\; -\bigl(\nabla_A l^\mu\bigr) k_\mu
          \;=\; \bigl(\nabla_A k^\mu\bigr) l_\mu ,
  \qquad l^\mu k_\mu = -1,
\tag{3}
```
where ``l^\mu,k^\mu`` are the outgoing and ingoing null normals (paper eq. (1)).
Under a rescaling ``l\to C\,l``, ``k\to C^{-1}k`` (equivalently: a change of the
slicing through ``\Delta``) the rotation form changes by a gradient,
```math
\omega_A \;\to\; \omega_A + \partial_A \ln C .
\tag{4}
```
Every standard definition of quasi-local angular momentum has the form (paper
eq. (2))
```math
J_\phi \;=\; -\frac{1}{8\pi G}\int_\Delta \omega(\phi)\,\epsilon
\tag{5}
```
for some *axial* vector field ``\phi`` on ``\Delta`` (two poles, closed
orbits, affine parameter period ``2\pi``).  The entire problem is the choice of
``\phi`` when ``\Delta`` has no symmetry.

### Conformally spherical coordinates and the Möbius group

On any oriented ``(S^2,q)`` there exist global *conformally spherical
coordinates* (CSCS) ``(\theta,\varphi)`` in which
```math
q \;=\; F(\theta,\varphi)\,\bigl(\mathrm{d}\theta^2 + \sin^2\!\theta\,
  \mathrm{d}\varphi^2\bigr),
  \qquad F>0
\tag{6}
```
(uniformization; constructible by Ricci flow
 [2, 3]).  Such coordinates are unique only up to the
six-parameter Möbius group, the connected component of ``SO(1,3)``, generated
by the three rotations
```math
\begin{aligned}
    \phi_1 &= -\sin\varphi\,\partial_\theta
              - \cot\theta\,\cos\varphi\,\partial_\varphi,\\
    \phi_2 &= \phantom{-}\cos\varphi\,\partial_\theta
              - \cot\theta\,\sin\varphi\,\partial_\varphi,\\
    \phi_3 &= \partial_\varphi,
  \end{aligned}
\tag{7}
```
and the three proper conformal generators
```math
\begin{aligned}
    \xi_1 &= -\cos\theta\,\cos\varphi\,\partial_\theta
             + \frac{\sin\varphi}{\sin\theta}\,\partial_\varphi,\\
    \xi_2 &= -\cos\theta\,\sin\varphi\,\partial_\theta
             - \frac{\cos\varphi}{\sin\theta}\,\partial_\varphi,\\
    \xi_3 &= \phantom{-}\sin\theta\,\partial_\theta,
  \end{aligned}
\tag{8}
```
with commutators ``[\phi_i,\phi_j] = -\epsilon_{ijk}\phi_k``,
``[\xi_i,\xi_j] = +\epsilon_{ijk}\phi_k``,
``[\xi_i,\phi_j] = -\epsilon_{ijk}\xi_k``.
Every unit combination ``n_i\phi_i`` (``n_in_i=1``) is axial; no combination of
the ``\xi_i`` is.

The finite action of the proper conformal transformation generated by
``\lambda\, n_i \xi_i`` (``n_i n_i = 1``, ``\lambda\in\mathbb{R}``) on the
generators themselves is (paper eqs. (10)–(11))
```math
\begin{aligned}
    \tilde\phi_i &= n_i\,(n_k\phi_k)
      + \cosh\lambda\,\bigl(\phi_i - n_i\,(n_k\phi_k)\bigr)
      + \sinh\lambda\;\epsilon_{ijk}\,n_j\,\xi_k,\\
    \tilde\xi_i &= n_i\,(n_k\xi_k)
      + \cosh\lambda\,\bigl(\xi_i - n_i\,(n_k\xi_k)\bigr)
      - \sinh\lambda\;\epsilon_{ijk}\,n_j\,\phi_k .
  \end{aligned}
\tag{9}
```

### The invariants \texorpdfstring{``\vec{J`` and ``\vec{K}``}{J and K}}

Given a CSCS one defines the two triples (paper eqs. (12)–(13))
```math
\boxed{\;
  J_i = -\frac{1}{8\pi G}\int_\Delta \omega(\phi_i)\,\epsilon,
  \qquad
  K_i = -\frac{1}{8\pi G}\int_\Delta \omega(\xi_i)\,\epsilon .
  \;}
\tag{10}
```
Note that ``\epsilon`` is the *physical* area form of ``q``, while ``\phi_i``
and ``\xi_i`` are the round-sphere generators (7)–(8)
of the chosen CSCS.  Under the rotation subgroup ``SO(3)``, ``\vec{J}`` and ``\vec{K}``
rotate as Euclidean 3-vectors.  Under the proper conformal transformation
(9), with
``\vec{\beta} = \tanh\lambda\;\hat{n}``, ``\gamma = (1-\beta^2)^{-1/2}``,
they transform exactly like the electric and magnetic fields under a Lorentz
boost (paper eqs. (14)–(15)):
```math
\begin{aligned}
    \vec{J}' &= \gamma\bigl(\vec{J} + \vec{\beta}\times\vec{K}\bigr)
            - \frac{\gamma^2}{\gamma+1}\,\vec{\beta}\,(\vec{\beta}\cdot\vec{J}),\\
    \vec{K}' &= \gamma\bigl(\vec{K} - \vec{\beta}\times\vec{J}\bigr)
            - \frac{\gamma^2}{\gamma+1}\,\vec{\beta}\,(\vec{\beta}\cdot\vec{K}).
  \end{aligned}
\tag{11}
```
The two ``SO(1,3)`` invariants are (paper eqs. (16)–(17))
```math
A = |\vec{J}|^2 - |\vec{K}|^2, \qquad B = \vec{K}\cdot\vec{J} .
\tag{12}
```

### Spin, axial vector, and gauge fixing

If ``A^2+B^2>0`` there is a proper conformal transformation making ``\vec{J}`` and
``\vec{K}`` parallel (or one of them zero): take
```math
\vec{\beta} = \beta\,\frac{\vec{J}\times\vec{K}}{|\vec{J}\times\vec{K}|},
  \qquad
  \beta^2 - S\,\beta + 1 = 0,\quad
  S = \frac{|\vec{J}|^2+|\vec{K}|^2}{|\vec{J}\times\vec{K}|},
  \qquad
  \beta = \frac{S - \sqrt{S^2-4}}{2}\in(0,1)
\tag{13}
```
(the root with ``0<\beta<1``; note ``S\ge 2`` always, by
``|\vec{J}|^2+|\vec{K}|^2 \ge 2|\vec{J}||\vec{K}| \ge 2|\vec{J}\times\vec{K}|``).  In the resulting
frame the spin and axial vector are defined as (paper Definition and
eq. (18))
```math
J = |\vec{J}'|,
  \qquad
  \phi = \frac{J'_i}{|\vec{J}'|}\,\tilde\phi_i
  \quad (\text{or } \phi = \tfrac{K'_i}{|\vec{K}'|}\,\tilde\phi_i
  \text{ if } \vec{J}'=0).
\tag{14}
```
Equivalently, and more conveniently, ``J`` has the manifestly
``SO(1,3)``-invariant form (paper eq. (21))
```math
\boxed{\;
  J = \sqrt{\frac{A + \sqrt{A^2 + 4B^2}}{2}} .
  \;}
\tag{15}
```
For ``\vec{K}=0`` this reduces to ``J=|\vec{J}|``, as it must.  If ``A=B=0`` (the “plane
wave” case) then ``J=0`` and no axis exists.

Finally, because of the gauge freedom (4) the paper fixes
the normalization of the null normals by requiring that in the Hodge
decomposition
```math
\omega = \star\,\mathrm{d} f + \mathrm{d} g
\tag{16}
```
the exact part vanishes, ``\mathrm{d} g = 0``, i.e. ``\mathrm{d}{\star}\omega = 0`` (paper
eq. (22)).  Since ``\star`` on one-forms is conformally invariant, the
decomposition is the same with respect to ``q`` and ``\mathring{q}``.  In practice one
never rescales the normals: one simply replaces ``\omega`` by its co-exact part
```math
\omega^{\mathrm{inv}} = \star\,\mathrm{d} f = \omega - \mathrm{d} g
\tag{17}
```
in (10).  This gauge fixing is what makes the result independent
of the choice of Cauchy slicing through the horizon (e.g. tilted slicings of
Kerr).


## The algorithm in detail

### Inputs

- A centre ``c^i`` and a shape function ``h(\theta,\varphi)>0``, given by
  spherical-harmonic coefficients, such that the horizon is
  ```math
x^i(\theta,\varphi) = c^i + h(\theta,\varphi)\,\hat r^i(\theta,\varphi),
    \qquad
    \hat r^i = (\sin\theta\cos\varphi,\;\sin\theta\sin\varphi,\;\cos\theta),
\tag{18}
```
  i.e. the surface is star-shaped about ``c^i`` (as produced by
  `ApparentHorizonFinder`).
- The spatial metric ``\gamma_{ij}`` and extrinsic curvature ``K_{ij}``
  evaluated at the collocation points ``x^i(\theta_p,\varphi_p)`` of a
  spin-weighted spherical-harmonic grid.

**Remark.** No spatial *derivatives* of ``\gamma_{ij}`` or ``K_{ij}`` are needed:
all derivatives that enter the algorithm are tangential to the surface and are
taken pseudospectrally on the sphere.  If ``\partial_k\gamma_{ij}`` is available
it can be used for an independent cross-check of the spectral derivatives of
``q_{AB}``.

### Surface geometry

The tangent vectors and induced metric are
```math
e_A{}^i = \partial_A x^i
          = (\partial_A h)\,\hat r^i + h\,\partial_A \hat r^i,
  \qquad
  q_{AB} = \gamma_{ij}\, e_A{}^i e_B{}^j .
\tag{19}
```
The angular derivatives ``\partial_A h`` are evaluated spectrally from the
spherical-harmonic coefficients of ``h``.  The area form and the physical area
are
```math
\epsilon_{AB} = \sqrt{\det q}\;\bigl(\mathrm{d}\theta\wedge\mathrm{d}\varphi\bigr)_{AB},
  \qquad
  \mathcal{A} = \oint_\Delta \epsilon
             = \int_0^\pi\!\!\int_0^{2\pi} \sqrt{\det q}\;
               \mathrm{d}\theta\,\mathrm{d}\varphi .
\tag{20}
```
The outward unit normal of ``\Delta`` within ``\Sigma`` is obtained without any
derivatives of ``\gamma``: the covector
```math
\sigma_i = [ijk]\, e_\theta{}^j e_\varphi{}^k
  \qquad (\text{``[ijk]`` the flat permutation symbol})
\tag{21}
```
annihilates both tangents (this requires no metric), and
```math
s^i = \frac{\gamma^{ij}\sigma_j}
             {\sqrt{\gamma^{kl}\sigma_k\sigma_l}},
  \qquad \text{sign fixed by } s^i\,\bigl(x_i - c_i\bigr) > 0 .
\tag{22}
```

### The rotation one-form from ADM data

Let ``n^\mu`` be the future unit normal of ``\Sigma`` and ``s^\mu`` the outward
unit normal of ``\Delta`` in ``\Sigma``, and set
```math
l^\mu = \tfrac{1}{\sqrt2}\,(n^\mu + s^\mu),
  \qquad
  k^\mu = \tfrac{1}{\sqrt2}\,(n^\mu - s^\mu),
  \qquad l^\mu k_\mu = -1 .
```
Inserting these into (3) and using
``n_\mu \nabla_\nu n^\mu = s_\mu \nabla_\nu s^\mu = 0`` and
``n_\mu \nabla_\nu s^\mu = - s^\mu \nabla_\nu n_\mu`` (from ``n\cdot s=0``),
```math
\omega_A
  = -\tfrac12\, e_A{}^\nu (n-s)_\mu \nabla_\nu (n+s)^\mu
  = e_A{}^\nu s^\mu \nabla_\nu n_\mu .
```
Both ``e_A`` and ``s`` are tangent to ``\Sigma``, so with the convention
(1),
```math
\boxed{\;
  \omega_A = -\,K_{ij}\, e_A{}^i s^j .
  \;}
\tag{25}
```
Consequently (5) becomes
``J_\phi = \frac{1}{8\pi G}\oint K_{ij}\,\phi^i s^j\,\epsilon`` with
``\phi^i = \phi^A e_A{}^i``, which is the familiar isolated-horizon /
Brown–York integrand of  [5].  Note that the constant boost
factor ``1/\sqrt2`` in ``(l,k)`` is pure gauge (4) and drops
out after the Hodge projection below; (25) holds for any
constant normalization.

### Gauge fixing: Hodge decomposition of \texorpdfstring{``\omega``{omega}}

On ``S^2`` there are no harmonic one-forms, so (16) is the complete
Hodge decomposition.  Taking the divergence of (16) and using
``D^A(\epsilon_A{}^B D_B f) = \epsilon^{AB} D_A D_B f = 0``,
```math
\Delta_q\, g = D^A \omega_A
  = \frac{1}{\sqrt{\det q}}\,\partial_A
    \bigl(\sqrt{\det q}\; q^{AB} \omega_B\bigr),
\tag{26}
```
a Poisson equation with zero-mean right-hand side, solved (up to an
irrelevant constant) pseudospectrally, after which
```math
\omega^{\mathrm{inv}}_A = \omega_A - \partial_A g .
```
Because ``\star`` on one-forms and the splitting
exact\,``\oplus``\,co-exact are conformally invariant, solving
(26) with ``q`` or with ``\mathring{q}`` gives the same ``g`` — a useful
consistency check.  The potential ``f`` is not needed.

**Remark.** This step cannot be skipped.  The fields ``\phi_i`` are divergence-free with
respect to the *round* area form ``\mathring{\epsilon}``, but not with respect to the
physical ``\epsilon = F\,\mathring{\epsilon}``; hence
``\oint (\mathrm{d} g)(\phi_i)\,\epsilon = -\oint g\,\phi_i(\ln F)\,\epsilon \ne 0``
in general, and without the projection the ``J_i`` would depend on the slicing
through the horizon.

### Ricci scalar of the two-metric

The uniformization in the section [Uniformization by a fast flow](@ref "Uniformization by a fast flow") needs ``R[q]``.  In any chart,
```math
\Gamma^C_{AB} = \tfrac12\, q^{CD}\bigl(\partial_A q_{DB}
                  + \partial_B q_{AD} - \partial_D q_{AB}\bigr),
  \qquad
  R = q^{AB}\bigl(\partial_C\Gamma^C_{AB} - \partial_A\Gamma^C_{CB}
      + \Gamma^C_{CD}\Gamma^D_{AB} - \Gamma^C_{AD}\Gamma^D_{CB}\bigr).
\tag{28}
```
Coordinate components of ``q_{AB}`` are singular at the poles of the chart, so
in practice (28) is evaluated in the spin-weighted form of
the section [Numerical implementation](@ref "Numerical implementation"): with ``\hat\nabla`` the connection of the unit
coordinate sphere ``\hat q`` and
```math
C^A_{BC} = \tfrac12\, q^{AD}\bigl(\hat\nabla_B q_{DC}
             + \hat\nabla_C q_{BD} - \hat\nabla_D q_{BC}\bigr)
\tag{29}
```
(a globally regular tensor field),
```math
R[q] = q^{AB}\Bigl(\hat q_{AB}
         + \hat\nabla_C C^C_{AB} - \hat\nabla_A C^C_{CB}
         + C^C_{CD} C^D_{AB} - C^C_{AD} C^D_{CB}\Bigr),
\tag{30}
```
where ``\hat R_{AB} = \hat q_{AB}`` is the Ricci tensor of the unit sphere.
Similarly, for any scalar ``f``,
```math
\Delta_q f = q^{AB}\bigl(\hat\nabla_A\hat\nabla_B f
               - C^C_{AB}\,\partial_C f\bigr).
\tag{31}
```

### Uniformization by a fast flow

First rescale to unit area: let
```math
\bar q_{AB} = \frac{4\pi}{\mathcal{A}}\, q_{AB},
  \qquad \oint \bar\epsilon = 4\pi,
```
so that by Gauss–Bonnet the average curvature is
``\bar R_{\mathrm{avg}} = 2``.  In two dimensions ``R_{AB} = \tfrac12 R\,
q_{AB}``, so the normalized Ricci flow
```math
\partial_t\, q_{AB}(t) = \bigl(2 - R(t)\bigr)\, q_{AB}(t),
  \qquad q(0) = \bar q,
\tag{33}
```
preserves the conformal class *and* the total area: writing
``q(t) = \mathrm{e}^{2u(t)}\bar q`` and using (2),
(33) is the single scalar parabolic equation
```math
\boxed{\;
  \partial_t u = 1 - \tfrac12 R(t),
  \qquad
  R(t) = \mathrm{e}^{-2u}\bigl(\bar R - 2\bar\Delta u\bigr),
  \qquad u(0)=0,
  \;}
\tag{34}
```
with ``\bar R = R[\bar q]`` and ``\bar\Delta = \Delta_{\bar q}`` computed once via
(30)–(31).  By Hamilton and Chow
 [2, 3] the flow exists for all ``t`` and converges
exponentially, for *any* initial metric on ``S^2``, to the constant
curvature metric: ``u(t)\to u_\infty`` with
```math
\mathring{q}_{AB} = \mathrm{e}^{2u_\infty}\,\bar q_{AB},
  \qquad R[\mathring{q}] = 2,\qquad \oint\mathring{\epsilon} = 4\pi .
```
This ``\mathring{q}`` is the unit-round representative of the conformal class of ``q``,
expressed in the original chart.  Equivalently, the conformal factor of
(6) is ``F = (\mathcal{A}/4\pi)\,\mathrm{e}^{-2u_\infty}``.
The flow serves here as the existence and uniqueness argument for
``u_\infty``; it is not integrated numerically.

**The Liouville equation.**
The fixed point of (34) satisfies the Liouville-type
elliptic equation
```math
\mathcal{N}[u] \equiv \bar\Delta u - \tfrac12\bar R + \mathrm{e}^{2u} = 0 ,
\tag{36}
```
with Fréchet derivative
``\mathcal{N}'[u]\,\delta u = \bar\Delta\,\delta u + 2\mathrm{e}^{2u}\delta u``.
Note that this nonlinearity is essential: no substitution ``v = F^s`` of the
conformal factor ``F = \mathrm{e}^{2u}`` linearizes (36).  E.g.
``v = 1/F`` gives ``\bar\Delta v + \bar R\,v - |\bar\nabla v|^2/v = 2``, which
misses linearity only by the gradient term, but the term never cancels for
any power (in contrast to the Yamabe problem in ``n\ge3`` dimensions, where
``\tilde q = \psi^{4/(n-2)}q`` removes it identically); see
the puncture method below for a genuinely linear alternative.

One subtlety: ``\mathcal{N}'`` is *not* sign definite, and at the
solution it is singular.  Indeed
``\mathcal{N}'[u_\infty] = \mathrm{e}^{2u_\infty}(\mathring{\Delta} + 2)``, whose kernel is
exactly the ``\ell=1`` eigenspace ``\mathrm{span}\{\chi_i\}`` of
the section [The first eigenfunctions of the round Laplacian](@ref "The first eigenfunctions of the round Laplacian").  This must be so: the solutions of
(36) form a three-parameter family (the Möbius orbit of
round representatives, all equally acceptable), and the kernel directions
are its tangent, ``\delta u = \chi_i``, matching the conformal Killing flow
``\mathcal{L}_{\xi_i}\mathring{q} = 2\chi_i\,\mathring{q}``.  Constants are a positive direction
(``(\mathring{\Delta}+2)\,c = 2c``) and ``\ell\ge2`` modes are negative.  Drift along
the kernel is pure gauge and harmless.

**The fast flow.**
We solve (36) by a fast flow in the style of
Gundlach's pseudo-spectral apparent-horizon finder [9]: an
approximate Newton iteration whose model Jacobian is diagonal in the
spherical-harmonic basis of the chart.  Two exact identities drive it:
pointwise
```math
\mathcal{N}[u] = -\tfrac12\,\mathrm{e}^{2u}\bigl(R(t)-2\bigr),
\qquad
\mathcal{N}'[u] = \bar\Delta + 2\mathrm{e}^{2u} = \mathrm{e}^{2u}\bigl(\Delta_{q(t)} + 2\bigr),
```
with ``q(t) = \mathrm{e}^{2u}\bar q`` and ``R(t) = R[q(t)]`` from (34), using
(2).  The exact Newton step is therefore
``\delta u = (\Delta_{q(t)}+2)^{-1}\,(R(t)-2)/2``.  The fast flow replaces
``\Delta_{q(t)}`` by the unit-sphere Laplacian ``\hat\Delta`` (diagonal,
eigenvalues ``-\ell(\ell+1)``), scaled by a stability constant ``\kappa``:
```math
\delta u_{\ell m}
  = \frac{\bigl[(R(t)-2)/2\bigr]_{\ell m}}{2 - \kappa\,\ell(\ell+1)} ,
```
iterated with ``u(0) = 0`` and the area renormalized to exactly ``4\pi``
after every step (a constant shift of ``u``).  Unlike the operator of the
horizon-finding problem, the model operator here is indefinite (``+2`` at
``\ell=0``, ``0`` at ``\ell=1``, negative for ``\ell\ge2``), so the signed
mode-wise inverse replaces the single-sign damping of [9]; the
vanishing ``\ell=1`` eigenvalue (the Möbius gauge kernel above) is replaced
by the adjacent ``\ell=2`` value ``2-6\kappa``, which is safe because at the
solution the residual has no component along the kernel.  ``\kappa`` is the
Richardson midpoint ``(c_{\min}+c_{\max})/2`` (floored at ``1``) of the
pointwise eigenvalue range of ``\mathrm{e}^{-2u}\bar q^{AB}`` relative to
``\hat q``, which keeps the high-``\ell`` iteration factors ``1 - c/\kappa``
inside ``(-1,1)`` for anisotropic metrics.  The iteration converges linearly;
the contraction per step is set by the chart distortion and anisotropy of
``\bar q`` (the model is diagonal in the *chart's* harmonic basis, while
``\mathring{q}`` is round only up to a diffeomorphism), but not by the resolution.
Convergence is declared when the L² norm of the coefficients of
``(R(t)-2)/2`` reaches the requested tolerance; three consecutive iterations
without a 1% improvement of the best residual signal the resolution's
round-off/aliasing floor, and the best iterate is returned as a best-effort
result.

**A fully linear alternative: flattening through a puncture.**
Although (36) cannot be linearized by a change of variables,
genus-zero uniformization *can* be reduced to linear solves by changing
the route: flatten first, then map to the sphere by inverse stereographic
projection  [6].  Pick an arbitrary puncture point
``p\in\Delta`` and solve the *linear* Poisson equation
```math
\bar\Delta\, w = \tfrac12\bar R - 4\pi\,\delta_p
\tag{37}
```
(solvable on the closed surface because both terms integrate to ``4\pi`` by
Gauss–Bonnet; near ``p``, ``w \simeq -2\ln r``).  Then ``\mathrm{e}^{2w}\bar q`` is flat
on ``\Delta\setminus\{p\}`` and complete — it is the Euclidean plane, with
``p`` at infinity.  Its Cartesian coordinates ``(x,y)`` are a conjugate pair of
``\bar q``-harmonic functions on ``\Delta\setminus\{p\}`` with a simple pole at
``p`` (harmonicity of functions is conformally invariant in two dimensions),
obtained from a second linear solve.  Inverse stereographic projection of
``z = x+iy`` then gives the round representative and the eigenfunction triple
in closed form,
```math
\chi_1 + i\chi_2 = \frac{2z}{1+|z|^2},
  \qquad
  \chi_3 = \frac{|z|^2-1}{|z|^2+1},
  \qquad
  \mathring{q} = \frac{4\,\mathrm{d} z\,\mathrm{d}\bar z}{(1+|z|^2)^2},
```
so this route replaces *both* the fast flow of
the section [Uniformization by a fast flow](@ref "Uniformization by a fast flow") *and* the eigenproblem of
the section [The first eigenfunctions of the round Laplacian](@ref "The first eigenfunctions of the round Laplacian").  The arbitrary choices (the puncture ``p``, the
normalization of the conjugate pair) move the resulting CSCS by exactly a
Möbius transformation, which the invariants of
the section [Invariants, boost, spin, axis, and axial vector field](@ref "Invariants, boost, spin, axis, and axial vector field") are designed to absorb.  The price is the
singular sources: to retain spectral accuracy the logarithm and the pole
must be split off analytically (solve for the smooth remainders), which on
a curved background ``\bar q`` requires local expansions around ``p``.  Since
the fast flow above costs only a short sequence of diagonally preconditioned
smooth iterations anyway, we keep it as the mainline and recommend the
puncture method as an independent cross-check of the uniformization stage.

### The first eigenfunctions of the round Laplacian

On the unit round sphere the lowest nonzero eigenvalue of the
Laplace–Beltrami operator is ``-2``, with the three-dimensional eigenspace
spanned by the ``\ell=1`` harmonics.  We therefore solve
```math
\mathring{\Delta} \chi = -2\,\chi
  \qquad\Longleftrightarrow\qquad
  \bar\Delta \chi = -2\,\mathrm{e}^{2u_\infty}\chi
\tag{39}
```
(using ``\mathring{\Delta} = \mathrm{e}^{-2u_\infty}\bar\Delta``) for the three eigenfunctions
``\chi_i``, ``i=1,2,3`` — these are the functions called ``x,y,z`` in
`GOAL.md`.  The right form for numerics is the *generalized
symmetric eigenproblem* on the right of (39): ``\bar\Delta``
is self-adjoint with respect to the measure ``\bar\epsilon``, and the weight
``\mathrm{e}^{2u_\infty}`` is positive, so the eigenvalues are real and eigenfunctions
belonging to different eigenvalues are orthogonal in
```math
\langle a,b\rangle_{\mathring{}}
  = \oint a\,b\;\mathrm{e}^{2u_\infty}\bar\epsilon
  = \oint a\,b\;\mathring{\epsilon} .
```
In exact arithmetic the eigenvalue ``-2`` is exactly triply degenerate;
numerically one finds a tight cluster of three eigenvalues near ``-2``
separated by a gap from ``0`` above and from ``\approx-6`` (``\ell=2``) below, so a
shift-invert Krylov method targeted at ``-2`` is robust
(the section [Numerical implementation](@ref "Numerical implementation")).

The cluster fixes only the 3-space ``\mathrm{span}\{\chi_i\}``; normalize and
orient as follows:
- **Orthonormalization.**  Compute the Gram matrix
  ``G_{ij} = \frac{3}{4\pi}\oint \chi_i\chi_j\,\mathring{\epsilon}`` and replace
  ``\chi_i \leftarrow (G^{-1/2})_{ij}\,\chi_j`` (symmetric/Löwdin
  orthonormalization), so that
  ```math
\oint \chi_i \chi_j\;\mathring{\epsilon} = \frac{4\pi}{3}\,\delta_{ij},
    \qquad
    \oint \chi_i\;\mathring{\epsilon} = 0
\tag{41}
```
  (the zero mean is automatic: ``\chi_i \perp \text{constants}``).
- **Handedness.**  Compute
  ```math
\mathcal{O} = \oint \chi_1\,
      \mathring{\epsilon}^{AB}\,\partial_A\chi_2\,\partial_B\chi_3\;\mathring{\epsilon} ;
\tag{42}
```
  if ``\mathcal{O}<0``, swap ``\chi_1\leftrightarrow\chi_2`` (or flip the sign
  of one ``\chi_i``).
By the classical rigidity of the first eigenfunctions (Takahashi
 [7]; the map ``p\mapsto\vec\chi(p)`` is an isometric embedding
of ``(\Delta,\mathring{q})`` onto the unit sphere in ``\mathbb{R}^3``), the normalized
triple satisfies *pointwise*
```math
\delta^{ij}\chi_i\chi_j = 1,
  \qquad
  \delta^{ij}\,\partial_A\chi_i\,\partial_B\chi_j = \mathring{q}_{AB},
\tag{43}
```
and, with the correct handedness,
```math
\chi_1\,\mathrm{d}\chi_2\wedge\mathrm{d}\chi_3
  + \chi_2\,\mathrm{d}\chi_3\wedge\mathrm{d}\chi_1
  + \chi_3\,\mathrm{d}\chi_1\wedge\mathrm{d}\chi_2 = \mathring{\epsilon},
  \qquad\text{i.e.}\qquad
  \tfrac12\,\epsilon^{ijk}\chi_i\,
  \mathring{\epsilon}^{AB}\partial_A\chi_j\,\partial_B\chi_k = 1 .
\tag{44}
```
Equation (44) integrated over the sphere shows that the
handedness integral (42) evaluates to
``\mathcal{O} = +\frac{4\pi}{3}`` for a right-handed triple (and
``-\frac{4\pi}{3}`` otherwise), so the test is unambiguous.  The residuals of (39),
(43) and (44) are sharp diagnostics of the
overall accuracy of the uniformization and eigenvalue steps.

The remaining freedom in ``\{\chi_i\}`` is exactly one global rotation
``\chi_i \to \Lambda_i{}^j \chi_j``, ``\Lambda\in SO(3)``.  No “north pole” or
“zero meridian” need be selected: under this freedom ``\vec{J}`` and ``\vec{K}`` rotate
covariantly, the invariants (12), the spin (15) and
the axial field (14) are unchanged.

### The Möbius generators in the original chart

This is the step where the present algorithm departs from (and simplifies)
the skeleton in `GOAL.md`: the generators
(7)–(8) can be written covariantly in terms of
``\mathring{q}`` and ``\chi_i`` and therefore evaluated *directly in the original
coordinates*, without ever constructing the good coordinates
``(\theta_1,\varphi_1)`` and without “undoing” the uniformization.

**Lemma 1.**
Let ``\mathring{q}`` be a unit-round metric on ``S^2`` and ``\chi_i`` a normalized,
right-handed triple of first eigenfunctions as in
the section [The first eigenfunctions of the round Laplacian](@ref "The first eigenfunctions of the round Laplacian").  Then the rotation and proper conformal generators
of the Möbius group of ``\mathring{q}`` are
```math
\boxed{\;
  \phi_i{}^A = \mathring{\epsilon}^{AB}\,\partial_B \chi_i,
  \qquad
  \xi_i{}^A = -\,\mathring{q}^{AB}\,\partial_B \chi_i .
  \;}
\tag{45}
```
Moreover ``\phi_i{}^A = \epsilon_{ijk}\,\chi_j\,\mathring{q}^{AB}\partial_B\chi_k``.

**Proof.** Both sides of each identity are tensorial, so it suffices to verify them in
coordinates ``(\theta,\varphi)`` adapted to ``\mathring{q}``, which exist by
(43): ``\mathring{q} = \mathrm{d}\theta^2+\sin^2\!\theta\,\mathrm{d}\varphi^2`` and
```math
\chi_1 = \sin\theta\cos\varphi,\qquad
  \chi_2 = \sin\theta\sin\varphi,\qquad
  \chi_3 = \cos\theta .
\tag{46}
```
With ``\mathring{\epsilon}_{\theta\varphi} = \sin\theta``, i.e.
``\mathring{\epsilon}^{\theta\varphi} = 1/\sin\theta``:
```math
\begin{aligned}
  \mathring{\epsilon}^{AB}\partial_B\chi_3 &:\quad
  \mathring{\epsilon}^{\theta\varphi}\partial_\varphi\cos\theta = 0,\qquad
  \mathring{\epsilon}^{\varphi\theta}\partial_\theta\cos\theta
  = \Bigl(-\frac{1}{\sin\theta}\Bigr)(-\sin\theta) = 1
  \quad\Longrightarrow\quad \partial_\varphi = \phi_3 . \\
  \mathring{\epsilon}^{AB}\partial_B\chi_1 &:\quad
  \mathring{\epsilon}^{\theta\varphi}\partial_\varphi(\sin\theta\cos\varphi)
  = -\sin\varphi,\qquad
  \mathring{\epsilon}^{\varphi\theta}\partial_\theta(\sin\theta\cos\varphi)
  = -\cot\theta\cos\varphi
  \quad\Longrightarrow\quad \phi_1 ,
\end{aligned}
```
and analogously for ``\phi_2``, reproducing (7).  For the second
family, ``-\mathring{q}^{AB}\partial_B\chi_3`` has components
``(-\partial_\theta\cos\theta,\,0) = (\sin\theta,\,0)``, i.e.
``\sin\theta\,\partial_\theta = \xi_3``, and
``-\mathring{q}^{AB}\partial_B\chi_1 = -\cos\theta\cos\varphi\,\partial_\theta +
\frac{\sin\varphi}{\sin\theta}\,\partial_\varphi = \xi_1``, etc., reproducing
(8).  The alternative formula for ``\phi_i`` follows from
``\epsilon_{ijk}\chi_j\partial_A\chi_k = \mathring{\epsilon}_{AB}\,\mathring{q}^{BC}\partial_C\chi_i``,
which is readily checked on (46) (both sides are the
rotation covector field about the ``i``-axis).
∎

Concretely, at every collocation point:
```math
\mathring{q}_{AB} = \mathrm{e}^{2u_\infty}\,\frac{4\pi}{\mathcal{A}}\,q_{AB},
  \qquad
  \mathring{\epsilon}^{AB}
  = \frac{\pm 1}{\sqrt{\det\mathring{q}}}
  \quad (\mathring{\epsilon}^{\theta\varphi} = +1/\sqrt{\det\mathring{q}}),
  \qquad
  \phi_i{}^A,\ \xi_i{}^A \text{ from (45)}.
\tag{47}
```
The fields (45) are normalization-correct (``2\pi``-periodic
orbits for ``n_i\phi_i``) precisely because ``\mathring{q}`` has area ``4\pi`` and the
``\chi_i`` obey (41)–(43); this is why the area
normalization of the section [Uniformization by a fast flow](@ref "Uniformization by a fast flow") matters.

### Invariants, boost, spin, axis, and axial vector field

With ``\omega^{\mathrm{inv}}`` from the section [Gauge fixing: Hodge decomposition of ω](@ref "Gauge fixing: Hodge decomposition of ω"), the generators
from the section [The Möbius generators in the original chart](@ref "The Möbius generators in the original chart"), and the *physical* area form
``\epsilon`` of ``q``:
```math
J_i = -\frac{1}{8\pi G}\oint_\Delta
        \omega^{\mathrm{inv}}_A\,\phi_i{}^A\;\epsilon,
  \qquad
  K_i = -\frac{1}{8\pi G}\oint_\Delta
        \omega^{\mathrm{inv}}_A\,\xi_i{}^A\;\epsilon .
\tag{48}
```
Then:
- **Spin.**  Compute ``A`` and ``B`` from (12) and
  ```math
J = \sqrt{\frac{A+\sqrt{A^2+4B^2}}{2}} .
```
  This requires no frame change and is the primary output.  If
  ``A^2+B^2 = 0`` (to within tolerance), report ``J=0`` and stop: no axis or
  axial field exists.
- **Boost to the parallel frame.**  If
  ``|\vec{J}\times\vec{K}| > `tol```, compute ``S``, ``\beta``, and ``\hat{n} =
  (\vec{J}\times\vec{K})/|\vec{J}\times\vec{K}|`` from (13), set
  ``\lambda = \operatorname{artanh}\beta``, and obtain ``\vec{J}'`` (and ``\vec{K}'``, as a check:
  ``\vec{J}'\times\vec{K}' \approx 0``) from (11) and the transformed
  rotation generators ``\tilde\phi_i`` from (9), all evaluated
  pointwise in the original chart.  If ``\vec{J}\times\vec{K} \approx 0`` already, set
  ``\tilde\phi_i = \phi_i``, ``\vec{J}' = \vec{J}``.  (Consistency: ``|\vec{J}'|`` must equal
  (15) up to round-off, since in the parallel frame
  ``A = J'^2 - K'^2``, ``B = \pm J'K'`` and (15) returns
  ``J'``.)
- **Axis and axial vector field.**  The spin axis (unit vector in
  the ``\mathbb{R}^3`` of the ``\chi'_i``, i.e. relative to the boosted CSCS
  frame, defined up to the global ``SO(3)`` freedom) is
  ``\hat a_i = J'_i/|\vec{J}'|`` (or ``K'_i/|\vec{K}'|`` if ``\vec{J}'=0``), and the axial
  vector field on ``\Delta`` is
  ```math
\phi^A = \hat a_i\;\tilde\phi_i{}^A ,
```
  an explicit vector field at the collocation points of the *original*
  chart.  Its two zeros are the rotation poles.  For reporting the axis as a
  spatial direction, the poles can be located on the surface and the
  corresponding embedding points ``x^i`` (or the vector between them) quoted;
  for small ``|\vec{K}|`` the axis is simply ``\vec{J}/|\vec{J}|`` in the ``\chi``-frame.

### Optional: explicit good coordinates

The algorithm never needs them, but conformally spherical coordinates can be
read off from the eigenfunctions if desired (e.g. for visualization or for
comparing with (6)):
```math
\theta_1 = \arccos \chi_3,
  \qquad
  \varphi_1 = \operatorname{atan2}(\chi_2,\,\chi_1).
\tag{51}
```
In these coordinates ``q = F\,(\mathrm{d}\theta_1^2 +
\sin^2\!\theta_1\,\mathrm{d}\varphi_1^2)`` with
``F = (\mathcal{A}/4\pi)\,\mathrm{e}^{-2u_\infty}`` expressed as a function of
``(\theta_1,\varphi_1)``.  Verifying this (by interpolating ``F`` and ``q`` onto a
grid in ``(\theta_1,\varphi_1)`` and checking the off-diagonal and trace parts)
is a strong end-to-end test, but as a computational route it is inferior:
the map (51) clusters and dilutes collocation points, and
re-interpolation costs accuracy for no benefit.


## Numerical implementation

### Spin-weighted spherical harmonic representation

All fields live on a spectral grid on the coordinate sphere (equiangular or
Gauss–Legendre in ``\theta``, equispaced in ``\varphi``), with band limit
``\ell_{\max}``, as provided by the packages `FastSphericalHarmonics`
and `Abstract\-Spherical\-Harmonics`.  Tensor components in the
``(\theta,\varphi)`` chart are singular at the poles; the cure is the
``\eth``-formalism  [4]: fix the unit-sphere dyad
```math
m^A = \tfrac{1}{\sqrt2}\,\bigl(1,\; \tfrac{i}{\sin\theta}\bigr),
  \qquad
  \hat q_{AB}\,m^A\bar m^B = 1,\quad m^Am_A = 0,
```
and store any tensor through its spin-weighted scalar components, e.g. for
the metric
```math
q_0 = q_{AB}\,m^A\bar m^B \;(\text{spin } 0,\ \text{real}),
  \qquad
  q_2 = q_{AB}\,m^A m^B \;(\text{spin } 2),
```
and for one-forms ``\omega_1 = \omega_A m^A`` (spin 1).  These are globally
smooth functions expandable in spin-weighted harmonics ``{}_sY_{\ell m}``.
Covariant derivatives ``\hat\nabla_A`` with respect to the *unit
coordinate sphere* act as ``\eth``/``\bar{\eth}`` on spin components and are
diagonal in the ``{}_sY_{\ell m}`` basis; derivatives with respect to ``q`` (or
``\bar q``, ``\mathring{q}``) are then assembled algebraically with the tensor
``C^A_{BC}`` of (29), exactly as in
(30)–(31).  (The pre-1.0 prototype of this
package implemented (30) this way.)

Scalars such as ``h``, ``u``, ``g``, ``\chi_i``, ``R``, and the integrands of
(48) are ordinary spin-0 fields.  Surface integrals are
evaluated by quadrature as
``\oint f\,\epsilon = \int f\,\frac{\sqrt{\det q}}{\sin\theta}\,
\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi``, where the ratio
``\sqrt{\det q}/\sin\theta`` is a smooth strictly positive scalar; with a
spectral transform this is just ``\sqrt{4\pi}`` times the ``\ell=0`` coefficient
of ``f\sqrt{\det q}/\sin\theta``.

**Dealiasing.**  The pipeline is full of pointwise products and of
the nonlinearity ``\mathrm{e}^{2u}``; use a grid with ``\gtrsim 2\ell_{\max}`` points
per dimension (3/2-rule at minimum) and verify spectral tails.  All
quantities derived from ``h`` and the metric are smooth, so coefficients decay
exponentially; the resolution requirement is set by the distortion of the
horizon (the dynamic range of ``F``).

### Elliptic solves and the eigenproblem

The operators ``\Delta_q`` (for (26)) and ``\bar\Delta + 2
\mathrm{e}^{2u_\infty}`` (shift-invert in (39)) are of the
form “unit-sphere Laplacian plus smooth corrections”.  Apply them
matrix-free on spherical-harmonic coefficients (transform ``\to`` pointwise
algebra ``\to`` transform) and solve with preconditioned CG/MINRES, using the
diagonal unit-sphere Laplacian ``-\ell(\ell+1)`` (shifted) as preconditioner;
the fast flow of the section [Uniformization by a fast flow](@ref "Uniformization by a fast flow") needs no linear solves at all.
For moderate resolution (``\ell_{\max}\lesssim 50``, i.e.
``(\ell_{\max}+1)^2 \lesssim 2600`` modes) dense assembly and direct
factorization/eigensolution is perfectly affordable and simplest.

For (39), request the three eigenpairs nearest ``-2`` of
the pencil ``(\bar\Delta,\ \mathrm{e}^{2u_\infty}\!\cdot)``, e.g. with shift-invert
Lanczos/Arnoldi (`Arpack.jl`, as in the pre-1.0 prototype) in the
inner product ``\langle\cdot,\cdot\rangle_{\mathring{}}``, or equivalently the
eigenpairs of the symmetrized operator
``\mathrm{e}^{-u_\infty}\bar\Delta\,\mathrm{e}^{-u_\infty}``.  The constant function
(eigenvalue ``0``) and the ``\ell=2``-like cluster (near ``-6``) are well
separated, so convergence is fast and unambiguous.

### Diagnostics

Cheap, sharp internal checks, in pipeline order:
- spectral tail of ``h``, ``q_0``, ``q_2`` (input resolution);
- Gauss–Bonnet: ``\oint R[q]\,\epsilon = 8\pi``;
- uniformization (fast flow): ``\lVert R[\mathring{q}]-2\rVert_\infty``, area drift;
- eigencluster: ``|\mu_i + 2|`` for the three eigenvalues, residuals
  ``\lVert\bar\Delta\chi_i + 2\mathrm{e}^{2u_\infty}\chi_i\rVert``;
- rigidity: ``\lVert \delta^{ij}\chi_i\chi_j - 1\rVert_\infty`` and
  ``\lVert \delta^{ij}\partial_A\chi_i\partial_B\chi_j -
  \mathring{q}_{AB}\rVert_\infty`` \ (43);
- Killing check: ``\lVert \mathcal{L}_{\phi_i}\mathring{q} \rVert``, conformal check:
  ``\mathcal{L}_{\xi_i}\mathring{q} = 2\chi_i\,\mathring{q}`` (from
  ``\mathring{D}_A\mathring{D}_B\chi_i = -\chi_i\,\mathring{q}_{AB}``);
- Hodge: ``\oint D^A\omega^{\mathrm{inv}}_A\,``-residual; same ``g`` from
  ``q``- and ``\mathring{q}``-divergences;
- invariance: ``|\vec{J}'|`` from the boosted frame vs.
  (15); ``\vec{J}'\times\vec{K}'\approx 0``;
- convergence of ``J``, ``\vec{J}``, ``\vec{K}`` under ``\ell_{\max}`` doubling
  (should be exponential).


## Algorithm summary

- **Geometry.**  From ``h`` and ``\gamma_{ij}``: tangents ``e_A{}^i``,
  metric ``q_{AB}``, area form ``\epsilon``, area ``\mathcal{A}``, outward normal
  ``s^i`` \ (19)–(22).
- **Rotation form.**  ``\omega_A = -K_{ij}e_A{}^i s^j``
  \ (25).
- **Gauge fix.**  Solve ``\Delta_q g = D^A\omega_A``; set
  ``\omega^{\mathrm{inv}} = \omega - \mathrm{d} g`` \ (26).
- **Normalize.**  ``\bar q = (4\pi/\mathcal{A})\,q``; compute
  ``\bar R`` \ (30).
- **Uniformize.**  Fast flow on the Liouville equation
  (36) ``\to u_\infty``,
  ``\mathring{q} = \mathrm{e}^{2u_\infty}\bar q`` with ``R[\mathring{q}]=2``.
- **Eigenfunctions.**  Solve
  ``\bar\Delta\chi = -2\mathrm{e}^{2u_\infty}\chi`` for the triple near ``-2``;
  orthonormalize (41); fix handedness (44).
- **Generators.**  ``\phi_i{}^A = \mathring{\epsilon}^{AB}\partial_B\chi_i``,
  ``\xi_i{}^A = -\mathring{q}^{AB}\partial_B\chi_i`` \ (45).
- **Invariants.**  ``J_i``, ``K_i`` by quadrature (48)
  with the *physical* ``\epsilon``; then ``A``, ``B`` (12) and
  ``J`` (15).
- **Axis and axial field.**  If ``A^2+B^2>0``: boost
  (13), (11), (9); output
  ``J``, axis ``\hat a_i = J'_i/|\vec{J}'|``, and
  ``\phi^A = \hat a_i\tilde\phi_i{}^A`` \ (14).


## Tests

- **Round sphere in flat space / Schwarzschild.**  ``\omega = 0``
  identically; every stage should return its trivial value (``u_\infty``
  constant ``=0``, ``\chi_i`` the ``\ell=1`` harmonics, ``\vec{J}=\vec{K}=0``, ``J = 0``) to
  round-off.
- **Coordinate-distorted Schwarzschild.**  Apply a smooth
  angle-dependent radial coordinate distortion (so that ``h`` is nontrivial
  but the geometry is unchanged): again ``J=0``, and ``\mathring{q}``, ``\chi_i`` must
  reproduce the distortion map.  This tests uniformization + eigenfunctions
  in isolation from ``\omega``.
- **Kerr, standard slicing.**  For a Kerr–Schild (or
  Boyer–Lindquist) slice of Kerr with parameters ``(M,a)``, the horizon
  ``r=r_+`` gives ``\vec{K}=0`` (after gauge fixing), ``\vec{J}`` along the spin axis,
  and ``J = Ma`` to spectral accuracy; the axial field ``\phi`` must agree with
  the axial Killing vector.  The `SpacetimeMetrics` package provides
  the analytic data (`KerrSchild` with `adm\_decompose` and
  `ExtrinsicCurvature`).  Verify ``J_3=+Ma`` (not ``-Ma``)
  to pin down orientation conventions.
- **Rotated/translated Kerr.**  Rotate the spin axis and offset the
  centre ``c^i``: ``J`` invariant, ``\vec{J}`` rotates as a vector, the recovered axis
  follows.
- **Tilted slicing of Kerr.**  Change the slicing through the
  horizon (tilted/waved leaves).  Without the Hodge projection the raw
  ``\vec{J}`` shifts; with it, ``J=Ma`` is recovered — this is the paper's own
  acid test for the gauge fixing  [1].
- **Distorted non-axisymmetric data.**  E.g. Bowen–York or
  superposed-Kerr initial data: check exponential convergence of ``J`` with
  ``\ell_{\max}``, the internal diagnostics of
  the section [Numerical implementation](@ref "Numerical implementation"), and stability of the axis under resolution
  changes.


## Corrections and clarifications to the `GOAL.md skeleton`

- **“Use Ricci flow to find a conformally round metric.”**
  In two dimensions the (normalized) Ricci flow moves only the conformal
  factor, so it does not *find* a conformally round metric — ``q`` is
  already conformally round by uniformization.  What the flow finds is the
  *round representative* ``\mathring{q} = \mathrm{e}^{2u_\infty}\bar q`` of the conformal
  class, in the original chart.  Operationally this is what the skeleton
  intended, but it reduces the PDE problem from a tensor flow to the single
  scalar equation (34).
- **“Undo the Ricci flow to find ``q(\theta_1,\varphi_1)``.”**
  Unnecessary, and best avoided.  The physical metric is known all along in
  the original chart; the invariants need only ``\omega^{\mathrm{inv}}``,
  ``\epsilon``, and the generators ``\phi_i,\xi_i``, and
  Lemma 1 provides the generators directly in the
  original chart.  No quantity is ever transformed to
  ``(\theta_1,\varphi_1)``, and no re-interpolation occurs.  The explicit
  coordinates (51) survive only as an optional diagnostic.
- **Determination of ``x,y,z`` (here ``\chi_{1,2,3**``).}  The
  skeleton's conditions (range ``[-1,1]``, zero mean, orthogonality,
  normalization) do not determine the functions and cannot be “solved as a
  linear system”: infinitely many triples satisfy them.  The missing
  condition is that they span the *lowest nonzero eigenspace of*
  ``\mathring{\Delta}`` (eigenvalue ``-2``), eq. (39) — so the correct
  guess in the skeleton is “eigenvalue problem”.  Zero mean is then
  automatic; orthogonality and normalization fix the Gram matrix
  (41); the range ``[-1,1]`` and indeed
  ``\sum_i\chi_i^2 = 1`` follow by rigidity (43) rather than
  being imposable conditions.
- **Integration measure.**  The orthogonality/normalization
  integrals must use the *round* area form ``\mathring{\epsilon}`` (equivalently weight
  ``\mathrm{e}^{2u_\infty}`` in ``\bar q``-integrals), with
  ``\oint\chi_i\chi_j\,\mathring{\epsilon} = \frac{4\pi}{3}\delta_{ij}``, not the coordinate
  or physical measure.
- **Area normalization.**  The metric must be scaled to area ``4\pi``
  before the flow/eigenproblem; otherwise the target curvature is not ``2``,
  the eigenvalue is not ``-2``, and the generators
  (45) acquire wrong normalizations (orbits would not
  have period ``2\pi``).  The physical area form still appears in the final
  integrals (48).
- **North pole and zero meridian.**  No such choice is needed: the
  residual freedom is a global ``SO(3)`` under which all outputs are
  covariant.  What *is* needed instead is a handedness fix
  (44), which the skeleton omits; with the wrong
  orientation, ``\vec{J} \to -\vec{J}``.
- **Gauge fixing of ``\omega``.**  The skeleton omits the Hodge
  projection (17) entirely.  It is essential: without it the
  ``J_i`` depend on the choice of null normals / slicing through the horizon
  (the section [Gauge fixing: Hodge decomposition of ω](@ref "Gauge fixing: Hodge decomposition of ω")), and e.g. a tilted slice of Kerr gives a wrong
  spin.
- **``K_i`` and the boost.**  The skeleton computes only the
  ``J_i``-side implicitly.  The ``K_i`` are needed both for the invariant value
  of the spin (15) and for the axial vector field, which is
  defined in the boosted frame where ``\vec{J}\parallel\vec{K}``
  (13); only for ``\vec{K}\times\vec{J} = 0`` (e.g. exact
  axisymmetry) can the boost be skipped.
- **Inputs.**  Derivatives of ``\gamma_{ij}`` and ``K_{ij}`` are not
  required (the section [Inputs](@ref "Inputs")); listing them among the inputs is
  harmless but unnecessary.

## Two-dimensional identities used

For ``\tilde q = \mathrm{e}^{2u}q`` on a two-manifold:
``\tilde q^{AB} = \mathrm{e}^{-2u}q^{AB}``;
``\tilde\epsilon_{AB} = \mathrm{e}^{2u}\epsilon_{AB}``,
``\tilde\epsilon^{AB} = \mathrm{e}^{-2u}\epsilon^{AB}``;
``\tilde\epsilon_A{}^B = \epsilon_A{}^B`` (conformal invariance of the Hodge
star on one-forms);
``\Delta_{\tilde q}f = \mathrm{e}^{-2u}\Delta_q f``;
``R[\tilde q] = \mathrm{e}^{-2u}(R[q] - 2\Delta_q u)``.
Levi-Civita identities: ``\epsilon_A{}^C\epsilon_C{}^B = -\delta_A^B``,
``\epsilon^{AC}\epsilon_{BC} = \delta^A_B``,
``\epsilon^{AB}D_AD_Bf = 0``.
Hodge: ``\star\star = -1`` on one-forms;
for ``\omega = \star\mathrm{d} f + \mathrm{d} g``:\ \
``D^A\omega_A = \Delta g``, \ \
``\epsilon^{AB}D_A\omega_B = -\Delta f``.
Gauss–Bonnet: ``\oint R\,\epsilon = 8\pi`` on ``S^2``.
First-eigenfunction (Obata-type) identity on the unit round sphere:
``\mathring{D}_A\mathring{D}_B\chi_i = -\chi_i\,\mathring{q}_{AB}``, whence ``\mathring{\Delta}\chi_i = -2\chi_i`` and,
for ``\xi_i{}^A = -\mathring{q}^{AB}\partial_B\chi_i``,
```math
\mathcal{L}_{\xi_i}\mathring{q}_{AB} = -2\,\mathring{D}_A\mathring{D}_B\chi_i = 2\chi_i\,\mathring{q}_{AB},
  \qquad
  \mathring{D}_A\,\xi_i{}^A = 2\chi_i ,
```
the conformal Killing equation with divergence ``2\chi_i``.

## References

1. M. Korzyński,
*Quasi-local angular momentum of non-symmetric isolated and dynamical
horizons from the conformal decomposition of the metric*,
Class. Quantum Grav. **24**, 5935 (2007);
[arXiv:0707.2824 [gr-qc]](https://arxiv.org/abs/0707.2824).

2. R. S. Hamilton,
*The Ricci flow on surfaces*,
in *Mathematics and General Relativity*, Contemp. Math. **71**,
237–262 (AMS, 1988).

3. B. Chow,
*The Ricci flow on the 2-sphere*,
J. Differential Geom. **33**, 325–334 (1991).

4. R. Gómez, L. Lehner, P. Papadopoulos, J. Winicour,
*The eth formalism in numerical relativity*,
Class. Quantum Grav. **14**, 977 (1997);
[arXiv:gr-qc/9702002](https://arxiv.org/abs/gr-qc/9702002).

5. O. Dreyer, B. Krishnan, E. Schnetter, D. Shoemaker,
*Introduction to isolated horizons in numerical relativity*,
Phys. Rev. D **67**, 024018 (2003);
[arXiv:gr-qc/0206008](https://arxiv.org/abs/gr-qc/0206008).

6. S. Haker, S. Angenent, A. Tannenbaum, R. Kikinis, G. Sapiro, M. Halle,
*Conformal surface parameterization for texture mapping*,
IEEE Trans. Vis. Comput. Graphics **6**, 181 (2000).

7. T. Takahashi,
*Minimal immersions of Riemannian manifolds*,
J. Math. Soc. Japan **18**, 380–385 (1966).

8. G. B. Cook, B. F. Whiting,
*Approximate Killing vectors on ``S^2``*,
Phys. Rev. D **76**, 041501(R) (2007);
[arXiv:0706.0199 [gr-qc]](https://arxiv.org/abs/0706.0199).

9. C. Gundlach,
*Pseudo-spectral apparent horizon finders: an efficient new algorithm*,
Phys. Rev. D **57**, 863 (1998);
[arXiv:gr-qc/9707050](https://arxiv.org/abs/gr-qc/9707050).



