# Surface geometry and the rotation one-form (docs/algorithm.tex §3.2–3.3).
#
# Inputs are three callables:
#   embedding(θ, ϕ) :: SVector{3}  — surface point in Cartesian slice coordinates
#   metric3(x)      :: SMatrix{3,3} — spatial metric γ_ij at x
#   excurv3(x)      :: SMatrix{3,3} — extrinsic curvature K_ij at x,
#                                     convention K_ij = −(1/2) £_n γ_ij
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

function surface_geometry(embedding, metric3, excurv3, grid::SphereGrid)
    sz = ash_grid_size(grid)
    coords = grid_coords(grid)

    # Embedding points and Cauchy data at the points
    x = [SVector{3,Float64}(embedding(θϕ[1], θϕ[2])) for θϕ in coords]
    γ = [SMatrix{3,3,Float64}(metric3(xi)) for xi in x]
    K = [SMatrix{3,3,Float64}(excurv3(xi)) for xi in x]

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
