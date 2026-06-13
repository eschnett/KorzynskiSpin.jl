# Visualization scripts

Experimentation scripts for rendering `KorzyńskiSpin.horizon_spin` output as
3D figures.  Not part of the package — a sandbox to iterate on.

## Run

```sh
cd viz
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=. horizon_viz.jl                       # → viz/figures/*.png
```

Backend (set the env var):

- `GLMakie` (default) — true 3D OpenGL, interactive windows, renders to PNG
  headless. Best for exploring.
- `CairoMakie` — vector-quality stills for papers (software 3D, no lighting).

```sh
KSPIN_VIZ_BACKEND=CairoMakie julia --project=. horizon_viz.jl
```

The plotting code is identical for both (Makie is backend-agnostic); only the
`using` line at the top branches.

## Figures produced

- `gauss_curvature.png` — surface colored by the intrinsic Gauss curvature
  `K = R[q]/2` (coordinate-invariant shape).
- `rotation_portrait_{1,2,3}.png` — the spin portrait from three camera
  angles: surface colored by the rotation scalar `Ω = ⋆dω` (∝ horizon
  vorticity / 2 Im Ψ₂), overlaid with the axial vector field `φ` and the
  spin axis.
- `angular_momentum_density.png` — surface colored by the integrand `ω(φ)` of
  `J = −1/8π ∮ ω(φ) ε`, with the rotation one-form `ω` as arrows.
- `conformal_factor.png` — the uniformization exponent `u` (departure from
  round).

## Saving specific camera views

`save_views(fig, ax, name; angles=[(azimuth, elevation), ...])` sets the
`Axis3` camera and writes one PNG per angle at `px_per_unit=2`.  Edit the
`angles` argument (radians) for publication framing.

## Using your own horizon

Replace `sample_result()` (a tilted Kerr horizon) with your data. To use an
[ApparentHorizonFinder](https://github.com/eschnett/ApparentHorizonFinder)
result:

```julia
hor = find_horizon(admvars, origin, EquiangularGrid(lmax), r)
result = horizon_spin(hor, metric3, excurv3)   # shares the grid; no interpolation
```

The derived-field helpers (`gauss_curvature`, `rotation_scalar`,
`angular_momentum_density`, `conformal_exponent`, `cartesian_field`) and the
`GridMesh` builder take any `SpinResult` / `SurfaceGeometry`, so they are easy
to lift into a package extension or a Documenter example once you settle on
the views you want.
