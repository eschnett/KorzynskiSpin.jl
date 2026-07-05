# Surface geometry and the rotation one-form (docs/algorithm.tex §3.2–3.3).
#
# Inputs are three callables:
#   embedding(θ, ϕ) :: SVector{3}  — surface point in Cartesian slice coordinates
#   metric3, excurv3               — the Cauchy data γ_ij and K_ij (extrinsic
#     curvature convention K_ij = −(1/2) £_n γ_ij).  These use the *batched*
#     interface: a callable
#         Xs::AbstractArray{SVector{3,Float64}} -> AbstractArray{SMatrix{3,3}}
#     that receives **all** queried points at once (a grid-shaped matrix) and
#     returns the value at each, in an array of the **same shape**.  Evaluating
#     all points together lets the caller parallelize the metric evaluation
#     (threads, pmap, GPU, batched autodiff, …).  For backward compatibility a
#     per-point callable x::SVector{3} -> SMatrix{3,3} is still accepted: one
#     whose argument is annotated ::SVector{3} is detected and wrapped
#     automatically, and a bare (untyped) per-point closure can be wrapped
#     explicitly with `pointwise`.
# The embedding is differentiated *spectrally*, so no analytic derivatives
# are required.

struct SurfaceGeometry
    grid::SphereGrid
    "embedding points x^i"
    x::Matrix{SVector{3,Float64}}
    "tangents E[a,i] = (∇̂_a x^i): dyad index a, Cartesian index i"
    E::Matrix{SMatrix{2,3,Float64,6}}
    "outward unit normal s^i"
    s::Matrix{SVector{3,Float64}}
    "induced metric q_ab (dyad components, real symmetric)"
    q::Tensor{2}
    "√(det q_ab) (dyad components; area density relative to ε̂)"
    sqrtdetq::Tensor{0}
    "physical area"
    area::Float64
    "rotation one-form ω_a = −K_ij e_a^i s^j (dyad components)"
    ω::Tensor{1}
end

"Embedding map x(θ,ϕ) = c + h(θ,ϕ) r̂(θ,ϕ) from a radial shape function"
function shape_embedding(h; center::SVector{3,Float64}=SVector(0.0, 0.0, 0.0))
    return function (θ, ϕ)
        r̂ = SVector(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))
        return center + h(θ, ϕ) * r̂
    end
end

# Adapt a user field provider to the batched interface
#     batched:  Xs::AbstractArray{<:SVector{3}} -> AbstractArray
# A pointwise provider (x::SVector{3} -> value) is NOT applicable to an array of
# points and is wrapped via `map`. Detection is type-based (`applicable`), so it
# is cheap and independent of the array's shape or contents. A single `SVector`
# is itself a 1-D `AbstractArray`, so the probe uses a `Matrix` (never mistaken
# for one point) to avoid false positives.
function _batched(f)
    sample = Matrix{SVector{3,Float64}}(undef, 0, 0)
    return applicable(f, sample) ? f : Xs -> map(f, Xs)
end

"""
    pointwise(f) -> batched callable

Wrap a per-point Cauchy-data provider `f(x::SVector{3,Float64}) -> SMatrix{3,3}`
so it can be passed to [`horizon_spin`](@ref) / [`surface_geometry`](@ref) under
the batched API (see [`surface_geometry`](@ref)). This is only needed for bare
(untyped) closures that auto-detection cannot classify; a function whose
argument is annotated `::SVector{3}` is detected and wrapped automatically.
"""
pointwise(f) = Xs -> map(f, Xs)

"""
    surface_geometry(embedding, metric3, excurv3, grid) -> SurfaceGeometry
    surface_geometry(x::AbstractMatrix{SVector{3,Float64}}, metric3, excurv3, grid)

Construct the induced geometry and rotation one-form of the surface.  The
first form evaluates the callable `embedding(θ, ϕ)::SVector{3}` at the
collocation points of `grid`; the second form accepts the surface points
directly (size `ash_grid_size(grid)`), e.g. from
`ApparentHorizonFinder.horizon_points`.

`metric3` and `excurv3` supply the Cauchy data through the batched interface
`Xs::AbstractArray{SVector{3}} -> AbstractArray{SMatrix{3,3}}` (see the module
header).  A per-point callable annotated `x::SVector{3} -> SMatrix{3,3}` is
detected and wrapped automatically; wrap a bare untyped closure with
[`pointwise`](@ref).
"""
function surface_geometry(embedding, metric3, excurv3, grid::SphereGrid)
    coords = grid_coords(grid)
    x = [SVector{3,Float64}(embedding(θϕ[1], θϕ[2])) for θϕ in coords]
    return surface_geometry(x, metric3, excurv3, grid)
end

function surface_geometry(x::AbstractMatrix{SVector{3,Float64}}, metric3, excurv3, grid::SphereGrid)
    sz = ash_grid_size(grid)
    size(x) == sz || throw(DimensionMismatch("surface points have size $(size(x)), expected $(sz) for this grid"))

    # Cauchy data at every surface point, evaluated in a single batched call.
    γ = SMatrix{3,3,Float64}.(_batched(metric3)(x))
    K = SMatrix{3,3,Float64}.(_batched(excurv3)(x))

    # Tangents E[a,i] = ∇̂_a x^i, computed spectrally component by component
    dx = ntuple(i -> grad(make_scalar(map(v -> v[i], x), grid)), 3)
    for i in 1:3
        @assert imag_norm(dx[i]) < 1.0e-8 * (1 + maximum(v -> norm(v), x))
    end
    E = [SMatrix{2,3,Float64}(real(dx[i].values[ij][a]) for a in 1:2, i in 1:3) for ij in CartesianIndices(sz)]

    # Induced metric q_ab = γ_ij E_a^i E_b^j
    qvals = [SMatrix{2,2,ComplexF64}(E[ij] * γ[ij] * transpose(E[ij])) for ij in CartesianIndices(sz)]
    q = Tensor{2}(qvals, grid)

    # Outward unit normal: σ_i = [ijk] E_1^j E_2^k annihilates the tangents;
    # normalize with γ and orient outward from the centroid.
    center = sum(x) / length(x)
    s = similar(x)
    for ij in CartesianIndices(sz)
        σ = cross(E[ij][1, :], E[ij][2, :])
        sv = γ[ij] \ σ
        sv /= sqrt(dot(sv, γ[ij] * sv))
        if dot(sv, x[ij] - center) < 0
            sv = -sv
        end
        s[ij] = sv
    end

    # Area density and area
    sqrtdetq = make_scalar([sqrt(abs(det(real.(qvals[ij])))) for ij in CartesianIndices(sz)], grid)
    area = integrate_unit(sqrtdetq)

    # Rotation one-form ω_a = −K_ij e_a^i s^j
    ωvals = [SVector{2,ComplexF64}(-(E[ij] * K[ij] * s[ij])) for ij in CartesianIndices(sz)]
    ω = Tensor{1}(ωvals, grid)

    return SurfaceGeometry(grid, x, E, s, q, sqrtdetq, area, ω)
end

"∮ f ε_q with the physical area form of q"
integrate(g::SurfaceGeometry, f::Tensor{0}) = integrate_unit(map_fields((a, b) -> a .* b, f, g.sqrtdetq))
