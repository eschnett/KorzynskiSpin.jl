# Visualize the output of KorzyńskiSpin.horizon_spin as 3D figures.
#
# Run with (from this directory):
#     julia --project=. horizon_viz.jl
# Backend (3D OpenGL vs. paper-quality vector output):
#     KSPIN_VIZ_BACKEND=GLMakie   julia --project=. horizon_viz.jl   # default
#     KSPIN_VIZ_BACKEND=CairoMakie julia --project=. horizon_viz.jl
# Both write PNGs into viz/figures/.  The plotting code is identical for the
# two backends (Makie is backend-agnostic); GLMakie gives true 3D shading and
# interactive windows, CairoMakie gives crisp vector-style output for papers
# (it renders 3D in software — fine for stills, no real lighting).
#
# This is an experimentation script, not part of the package: it recomputes a
# few derived fields (Gauss curvature, rotation scalar, angular-momentum
# density) from the `SpinResult` using the exported API, builds a mesh from
# the collocation grid, and renders several views.  Tweak the `FIELDS`,
# colormaps, camera angles, and the sample horizon at the bottom.

const BACKEND = get(ENV, "KSPIN_VIZ_BACKEND", "GLMakie")
if BACKEND == "CairoMakie"
    using CairoMakie
else
    using GLMakie
end

using AbstractSphericalHarmonics
using GeometryBasics: Point3f, Vec3f, TriangleFace, Mesh
using KorzyńskiSpin
using LinearAlgebra
using SpacetimeMetrics
using StaticArrays
using Statistics: mean

const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

################################################################################
# Derived scalar/vector fields on the horizon, from a `SpinResult`.

"Gauss curvature K = R[q]/2 of the induced 2-metric (coordinate-invariant)."
function gauss_curvature(result::SpinResult)
    ops = MetricOps(result.geometry.q)
    R = real_part(scalar_curvature(ops))
    return real.(grid_values(R)) ./ 2
end

"""
Rotation scalar Ω = ⋆dω = ε_q^{AB} D_A ω_B of the gauge-fixed rotation
one-form (the horizon "vorticity", ∝ the pullback of 2 Im Ψ₂).  Computed from
the unit-sphere curl of the dyad components divided by the area-density ratio
√(det q)/sinθ, which converts the curl to the physical area form.
"""
function rotation_scalar(result::SpinResult)
    ω = result.ωinv
    G = grad(ω)                      # G[a,b] = ∇̂_b ω_a  (unit-sphere connection)
    sdq = real.(grid_values(result.geometry.sqrtdetq))
    # orthonormal-frame curl ε̂^{ab} ∇̂_a ω_b = ∇̂_θ̂ ω_φ̂ − ∇̂_φ̂ ω_θ̂
    curl̂ = [real(G.values[ij][2, 1] - G.values[ij][1, 2]) for ij in CartesianIndices(G.values)]
    return curl̂ ./ sdq
end

"Angular-momentum density ω^inv(φ) (the integrand of J = −1/8π ∮ ω(φ) ε)."
function angular_momentum_density(result::SpinResult)
    ω = result.ωinv
    φ = result.axial
    return [real(sum(ω.values[ij][a] * φ.values[ij][a] for a in 1:2)) for ij in CartesianIndices(ω.values)]
end

"Conformal exponent u of the uniformization (q̊ = e^{2u} q̄; 0 ⇔ already round)."
conformal_exponent(result::SpinResult) = real.(grid_values(result.uniformization.u))

"""
Push a dyad-component field (`Tensor{1}`, components on the orthonormal
unit-sphere dyad eθ̂, eϕ̂) to Cartesian 3-vectors using the surface tangents
`E` (which are stored in that same dyad frame: E[a,:] = eₐ̂ in ℝ³).
"""
function cartesian_field(field, geom)
    return [SVector{3,Float64}(sum(real(field.values[ij][a]) * geom.E[ij][a, :] for a in 1:2)) for ij in CartesianIndices(field.values)]
end

################################################################################
# Mesh from the (θ, φ) collocation grid.  Assumes an `EquiangularGrid`, whose
# point matrix is laid out (nθ, nφ) with θ along the first axis and no points
# at the poles (θ midpoints), so we close the φ seam and fan two polar caps.

struct GridMesh
    points::Vector{Point3f}
    faces::Vector{TriangleFace{Int}}
    normals::Vector{Vec3f}
    idx::Matrix{Int}        # idx[i,j] -> vertex number
    itop::Int               # north cap apex vertex
    ibot::Int               # south cap apex vertex
    nθ::Int
    nφ::Int
end

function GridMesh(geom)
    pts = geom.x
    nrm = geom.s
    nθ, nφ = size(pts)
    points = Point3f[]
    normals = Vec3f[]
    idx = zeros(Int, nθ, nφ)
    for j in 1:nφ, i in 1:nθ
        push!(points, Point3f(pts[i, j]...))
        push!(normals, Vec3f(nrm[i, j]...))
        idx[i, j] = length(points)
    end
    faces = TriangleFace{Int}[]
    for j in 1:nφ
        jn = j == nφ ? 1 : j + 1            # wrap the φ seam
        for i in 1:(nθ - 1)
            a, b = idx[i, j], idx[i, jn]
            c, d = idx[i + 1, jn], idx[i + 1, j]
            push!(faces, TriangleFace(a, b, c))
            push!(faces, TriangleFace(a, c, d))
        end
    end
    # polar caps (the grid has no pole point): fan from the ring mean
    top = Point3f(mean(SVector{3,Float64}[pts[1, j] for j in 1:nφ])...)
    push!(points, top)
    push!(normals, Vec3f(mean(SVector{3,Float64}[nrm[1, j] for j in 1:nφ])...))
    itop = length(points)
    bot = Point3f(mean(SVector{3,Float64}[pts[nθ, j] for j in 1:nφ])...)
    push!(points, bot)
    push!(normals, Vec3f(mean(SVector{3,Float64}[nrm[nθ, j] for j in 1:nφ])...))
    ibot = length(points)
    for j in 1:nφ
        jn = j == nφ ? 1 : j + 1
        push!(faces, TriangleFace(itop, idx[1, jn], idx[1, j]))
        push!(faces, TriangleFace(ibot, idx[nθ, j], idx[nθ, jn]))
    end
    return GridMesh(points, faces, normals, idx, itop, ibot, nθ, nφ)
end

makie_mesh(m::GridMesh) = Mesh(m.points, m.faces)

"Per-vertex colors from a scalar field on the (nθ, nφ) grid (caps = ring mean)."
function vertex_colors(m::GridMesh, scalar::AbstractMatrix)
    cols = Vector{Float64}(undef, length(m.points))
    for j in 1:(m.nφ), i in 1:(m.nθ)
        cols[m.idx[i, j]] = scalar[i, j]
    end
    cols[m.itop] = mean(scalar[1, :])
    cols[m.ibot] = mean(scalar[m.nθ, :])
    return cols
end

################################################################################
# Plot helpers

"Symmetric color range about zero for signed fields."
function symrange(scalar)
    a = maximum(abs, scalar)
    return (-a, a)
end

"""
    surface_figure(geom, m, scalar; title, colormap, colorrange, ...) -> (fig, ax)

Render the horizon mesh colored by a vertex scalar.  Optionally overlay a
Cartesian vector field as arrows (`vectors`), the spin axis (`axis`/`center`),
and the rotation poles.
"""
function surface_figure(
    geom,
    m::GridMesh,
    scalar;
    title="",
    colormap=:viridis,
    colorrange=extrema(scalar),
    colorlabel="",
    vectors=nothing,
    vstride=3,
    vcolor=:black,
    axis=nothing,
    center=nothing,
    shading=NoShading,
    azimuth=1.1π,
    elevation=π / 8,
)
    fig = Figure(; size=(950, 820))
    ax = Axis3(fig[1, 1]; aspect=:data, azimuth, elevation, title, protrusions=30)
    hidedecorations!(ax)
    cols = vertex_colors(m, scalar)
    plt = mesh!(ax, makie_mesh(m); color=cols, colormap, colorrange, shading)
    Colorbar(fig[1, 2], plt; label=colorlabel)

    if vectors !== nothing
        nθ, nφ = size(vectors)
        ps = Point3f[]
        ns = Vec3f[]
        for j in 1:vstride:nφ, i in 1:vstride:nθ
            push!(ps, Point3f(geom.x[i, j]...))
            push!(ns, Vec3f(vectors[i, j]...))
        end
        L = maximum(norm, ns)
        r = meshradius(m)
        scale = L == 0 ? 1.0 : 0.22 * r / L
        arrows3d!(ax, ps, ns; lengthscale=scale, shaftradius=0.006r,
                  tipradius=0.013r, tiplength=0.028r, color=vcolor)
    end

    if axis !== nothing && center !== nothing
        R = 1.35 * meshradius(m)
        n = normalize(SVector{3,Float64}(axis))
        c = SVector{3,Float64}(center)
        lines!(ax, [Point3f((c - R * n)...), Point3f((c + R * n)...)]; color=:black, linewidth=3)
        scatter!(ax, [Point3f((c + R * n)...), Point3f((c - R * n)...)]; color=:black, markersize=12)
    end

    return fig, ax
end

meshradius(m::GridMesh) = maximum(norm(p - mean(m.points)) for p in m.points)

"Save the current figure, optionally from several camera angles."
function save_views(fig, ax, name; angles=[(1.1π, π / 8)])
    paths = String[]
    for (k, (az, el)) in enumerate(angles)
        ax.azimuth[] = az
        ax.elevation[] = el
        path = joinpath(FIGDIR, length(angles) == 1 ? "$name.png" : "$(name)_$(k).png")
        save(path, fig; px_per_unit=2)
        push!(paths, path)
    end
    return paths
end

################################################################################
# A sample distorted horizon: Kerr (a = 0.7) with the spin axis tilted 30°
# from ẑ, in the Kerr–Schild slicing.  Swap this block for your own data —
# e.g. an ApparentHorizonFinder result fed through
#   horizon_spin(horizon_points(hor), slice_metric, slice_excurv; grid=hor.grid).

function sample_result(; M=1.0, a=0.7, tilt=π / 6, lmax=24)
    ks = rotate(KerrSchild(M, a), 0.0, tilt, 0.0)      # spin axis R_y(tilt)·ẑ
    metric3(x) = adm_decompose(ks, SVector(0.0, x...))[3]
    excurv3(x) = ExtrinsicCurvature(ks, SVector(0.0, x...))
    r₊ = M + sqrt(M^2 - a^2)
    sα, cα = sincos(tilt)
    function embedding(θ, φ)
        # tilted oblate spheroid: rotate the ray back by R_y(−tilt), intersect
        cθ0 = sα * sin(θ) * cos(φ) + cα * cos(θ)
        h = 1 / sqrt((1 - cθ0^2) / (r₊^2 + a^2) + cθ0^2 / r₊^2)
        return h * SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
    end
    return horizon_spin(embedding, metric3, excurv3; lmax)
end

function main()
    @info "backend" BACKEND
    result = sample_result()
    @info "spin" J = result.J axis = result.axis_embedding area = result.area
    geom = result.geometry
    m = GridMesh(geom)
    center = mean(geom.x)

    # 1. Intrinsic Gauss curvature — the coordinate-invariant shape.
    K = gauss_curvature(result)
    fig, ax = surface_figure(geom, m, K; title="Gauss curvature K = R[q]/2",
                             colormap=:viridis, colorlabel="K")
    save_views(fig, ax, "gauss_curvature")

    # 2. Rotation scalar Ω + axial vector field φ + spin axis — the spin portrait.
    Ω = rotation_scalar(result)
    φcart = cartesian_field(result.axial, geom)
    fig, ax = surface_figure(geom, m, Ω; title="rotation scalar Ω = ⋆dω  +  axial field φ",
                             colormap=:balance, colorrange=symrange(Ω), colorlabel="Ω",
                             vectors=φcart, vstride=3, vcolor=(:black, 0.6),
                             axis=result.axis_embedding, center=center)
    save_views(fig, ax, "rotation_portrait";
               angles=[(1.1π, π / 8), (0.4π, π / 6), (1.1π, 0.46π)])

    # 3. Angular-momentum density ω(φ) + the rotation one-form ω^inv as arrows.
    jdens = angular_momentum_density(result)
    ωcart = cartesian_field(result.ωinv, geom)
    fig, ax = surface_figure(geom, m, jdens; title="angular-momentum density ω(φ)  +  ω",
                             colormap=:balance, colorrange=symrange(jdens), colorlabel="ω(φ)",
                             vectors=ωcart, vstride=3, vcolor=(:black, 0.6))
    save_views(fig, ax, "angular_momentum_density")

    # 4. Conformal exponent u — where the metric departs from round.
    u = conformal_exponent(result)
    fig, ax = surface_figure(geom, m, u; title="conformal exponent u  (q̊ = e^{2u} q̄)",
                             colormap=:balance, colorrange=symrange(u), colorlabel="u")
    save_views(fig, ax, "conformal_factor")

    @info "wrote figures" dir = FIGDIR files = readdir(FIGDIR)
    return result
end

main()
