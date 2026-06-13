# Kerr initial data for tests, provided by SpacetimeMetrics (Kerr–Schild
# Cartesian coordinates; spin +Ma about the +z axis).
#
# SpacetimeMetrics uses the same extrinsic-curvature convention as
# docs/algorithm.tex: K_ij = −(1/(2α))(∂_t γ_ij − D_i β_j − D_j β_i)
# = −(1/2) £_n γ_ij.

using SpacetimeMetrics
using StaticArrays

"Spatial metric γ_ij of the t = 0 slice as a function of x⃗"
slice_metric(m::AbstractMetric) = x -> adm_decompose(m, SVector(0.0, x...))[3]

"Extrinsic curvature K_ij of the t = 0 slice as a function of x⃗"
slice_excurv(m::AbstractMetric) = x -> ExtrinsicCurvature(m, SVector(0.0, x...))

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
