# Kerr initial data for tests, provided by SpacetimeMetrics (Kerr–Schild
# Cartesian coordinates; spin +Ma about the +z axis).
#
# SpacetimeMetrics uses the same extrinsic-curvature convention as
# docs/algorithm.tex: K_ij = −(1/(2α))(∂_t γ_ij − D_i β_j − D_j β_i)
# = −(1/2) £_n γ_ij.
#
# The closed-form balanced-frame horizon multipoles of Kerr at the end are
# from Gourgoulhon, Le Tiec & Casals, arXiv:2602.05823.

using SpacetimeMetrics
using StaticArrays

# The `::SVector{3}` annotations mark these as per-point providers so
# KorzynskiSpin auto-wraps them for the batched Cauchy-data interface.

"Spatial metric γ_ij of the t = 0 slice as a function of x⃗"
slice_metric(m::AbstractMetric) = (x::SVector{3}) -> adm_decompose(m, SVector(0.0, x...))[3]

"Extrinsic curvature K_ij of the t = 0 slice as a function of x⃗"
slice_excurv(m::AbstractMetric) = (x::SVector{3}) -> ExtrinsicCurvature(m, SVector(0.0, x...))

"Embedding of the Kerr horizon r = r₊ = M + √(M² − a²) (a coordinate spheroid)"
function ks_horizon_embedding(M, a)
    r₊ = M + sqrt(M^2 - a^2)
    ρ₊ = sqrt(r₊^2 + a^2)
    return (θ, ϕ) -> SVector(ρ₊ * sin(θ) * cos(ϕ), ρ₊ * sin(θ) * sin(ϕ), r₊ * cos(θ))
end

"Analytic horizon area of Kerr: A = 8π M r₊"
ks_horizon_area(M, a) = 8π * M * (M + sqrt(M^2 - a^2))

"ADMVars adapter for ApparentHorizonFinder from a SpacetimeMetrics metric (t = 0 slice)"
function slice_admvars(m::AbstractMetric)
    return function (x::SVector{3})
        p = SVector(0.0, x...)
        g, ∂g = dmetric(m, p)
        K = ExtrinsicCurvature(m, p)
        γ = SMatrix{3,3}(g[i, j] for i in 2:4, j in 2:4)
        ∂γ = SArray{Tuple{3,3,3}}(∂g[i, j, k] for i in 2:4, j in 2:4, k in 2:4)
        return ADMVars(γ, ∂γ, K)
    end
end

"Gauss–Legendre nodes and weights on [−1, 1] (Golub–Welsch)"
function gauss_legendre(n)
    ev = eigen(SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]))
    return ev.values, 2 .* ev.vectors[1, :] .^ 2
end

"Legendre polynomial P_l(x) by the three-term recurrence"
function legendre_p(l, x)
    l == 0 && return one(x)
    p0, p1 = one(x), x
    for k in 1:(l - 1)
        p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
    end
    return p1
end

"Balanced ℓ=1 function of the Kerr horizon, z(ζ) with ζ = cos θ (Gourgoulhon et al., arXiv:2602.05823, eq. 6.30)"
function kerr_balanced_z(M, a, ζ)
    r₊ = M + sqrt(M^2 - a^2)
    return tanh(atanh(ζ) - a^2 / (r₊^2 + a^2) * ζ)
end

"Kerr horizon multipoles I_l + i L_l (m = 0) in the balanced frame (arXiv:2602.05823, eq. 6.42)"
function kerr_multipole(M, a, l; n=400)
    r₊ = M + sqrt(M^2 - a^2)
    â = a / r₊
    ζ, w = gauss_legendre(n)
    return (1 + â^2)^2 / 2 * sqrt((2l + 1) * π) * sum(@. w * legendre_p(l, kerr_balanced_z(M, a, ζ)) / (1 - im * â * ζ)^3)
end
