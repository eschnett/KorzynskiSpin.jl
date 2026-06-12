# Hodge gauge fixing, the invariants J⃗ and K⃗, the spin, and the axial
# vector field (docs/algorithm.tex §3.4, §3.9).  Units: G = 1.

"""
    hodge_fix(ops, ω; Δmat) -> (ωinv, g)

Project out the exact part of ω: solve Δ_q g = D^a ω_a and return
ω^inv = ω − dg (eq. 22 of Korzyński 2007: d⋆ω = 0).
"""
function hodge_fix(ops::MetricOps, ω::Tensor{1}; Δmat::Union{Nothing,Matrix{ComplexF64}}=nothing)
    grid = ops.grid
    n = ash_nmodes(grid)[1]
    L = Δmat === nothing ? operator_matrix(f -> laplacian(ops, f), grid) : Δmat
    ρ = scalar_coeffs(divergence(ops, ω))
    # Pin the constant mode (Δ kernel): solve (L + p₀p₀ᵀ) g = ρ
    i0 = LinearIndices((n,))[ash_mode_index(grid, 0, 0, 0)]
    L̂ = copy(L)
    L̂[i0, i0] += 1
    gc = L̂ \ ρ
    gc[i0] = 0
    g = coeffs_scalar(gc, grid)
    dg = differential(g)
    ωinv = Tensor{1}([SVector{2}(ω[1] - dg[1], ω[2] - dg[2]) for (ω, dg) in zip(ω.values, dg.values)], grid)
    return ωinv, g
end

################################################################################
# Invariants and spin

"J_i or K_i: −(1/8π) ∮ ω_a v^a ε_q  (dyad-frame contraction)"
function momentum_integral(geom::SurfaceGeometry, ω::Tensor{1}, v::Tensor{1})
    grid = geom.grid
    vals = [real(sum(ω[a] * v[a] for a in 1:2)) for (ω, v) in zip(ω.values, v.values)]
    return -1 / 8π * integrate(geom, make_scalar(vals, grid))
end

"Spin from the SO(1,3) invariants (eq. 21 of Korzyński 2007)"
function spin_from_invariants(Jvec::SVector{3,Float64}, Kvec::SVector{3,Float64})
    A = dot(Jvec, Jvec) - dot(Kvec, Kvec)
    B = dot(Kvec, Jvec)
    J = sqrt((A + sqrt(A^2 + 4B^2)) / 2)
    return J, A, B
end

"Möbius boost of (J⃗, K⃗) with velocity β⃗ (eqs. 14–15 of Korzyński 2007)"
function mobius_boost(Jvec::SVector{3}, Kvec::SVector{3}, β⃗::SVector{3})
    β² = dot(β⃗, β⃗)
    γ = 1 / sqrt(1 - β²)
    J′ = γ * (Jvec + cross(β⃗, Kvec)) - γ^2 / (γ + 1) * β⃗ * dot(β⃗, Jvec)
    K′ = γ * (Kvec - cross(β⃗, Jvec)) - γ^2 / (γ + 1) * β⃗ * dot(β⃗, Kvec)
    return J′, K′
end

"""
    parallel_frame_boost(Jvec, Kvec; tol) -> β⃗

Boost velocity that makes J⃗′ ∥ K⃗′: β⃗ = β (J⃗×K⃗)/|J⃗×K⃗| with β ∈ (0,1) the
root of β² − Sβ + 1 = 0, S = (|J⃗|²+|K⃗|²)/|J⃗×K⃗|.  Returns zero when J⃗×K⃗
vanishes (already parallel) or both invariants vanish.
"""
function parallel_frame_boost(Jvec::SVector{3,Float64}, Kvec::SVector{3,Float64}; tol::Float64=1.0e-10)
    JxK = cross(Jvec, Kvec)
    scale = norm(Jvec)^2 + norm(Kvec)^2
    if norm(JxK) ≤ tol * max(scale, eps())
        return SVector(0.0, 0.0, 0.0)
    end
    S = scale / norm(JxK)
    β = (S - sqrt(S^2 - 4)) / 2
    return β * JxK / norm(JxK)
end

"""
    boosted_rotation_generators(gen, β⃗) -> (φ̃₁, φ̃₂, φ̃₃)

Finite proper conformal transformation of the rotation generators
(eq. 10 of Korzyński 2007) with β⃗ = tanh(λ) n̂.
"""
function boosted_rotation_generators(gen::MobiusGenerators, β⃗::SVector{3,Float64})
    β = norm(β⃗)
    β == 0 && return gen.φ
    n̂ = β⃗ / β
    λ = atanh(β)
    chλ, shλ = cosh(λ), sinh(λ)
    grid = gen.φ[1].grid
    ε(i, j, k) = (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)) ? 1 : ((i, j, k) in ((3, 2, 1), (1, 3, 2), (2, 1, 3)) ? -1 : 0)
    φ̃ = ntuple(3) do i
        vals = [
            begin
                nφ = sum(n̂[k] * gen.φ[k].values[ij] for k in 1:3)
                v = n̂[i] * nφ + chλ * (gen.φ[i].values[ij] - n̂[i] * nφ)
                for j in 1:3, k in 1:3
                    if ε(i, j, k) ≠ 0
                        v += shλ * ε(i, j, k) * n̂[j] * gen.ξ[k].values[ij]
                    end
                end
                v
            end for ij in CartesianIndices(ash_grid_size(grid))
        ]
        Tensor{1}(vals, grid)
    end
    return φ̃
end

################################################################################
# Result and driver

struct SpinResult
    grid::SphereGrid
    "physical area of the surface"
    area::Float64
    "the spin J (eq. 21, invariant)"
    J::Float64
    "J⃗ and K⃗ in the constructed conformally spherical frame"
    Jvec::SVector{3,Float64}
    Kvec::SVector{3,Float64}
    "SO(1,3) invariants"
    A::Float64
    B::Float64
    "boost velocity to the J⃗′ ∥ K⃗′ frame"
    β⃗::SVector{3,Float64}
    "J⃗′ in the boosted frame (|J⃗′| = J)"
    Jvec′::SVector{3,Float64}
    Kvec′::SVector{3,Float64}
    "spin axis (unit vector in the boosted CSCS frame), zero if J = 0"
    axis::SVector{3,Float64}
    "spin axis as a spatial direction: normalized ∮ x⃗ × v⃗ ε_q with v the
     axial flow field pushed to the embedding; zero if J = 0"
    axis_embedding::SVector{3,Float64}
    "axial vector field φ^a (dyad components), zero if J = 0"
    axial::Tensor{1}
    "surface geometry, uniformization, eigenfunctions, generators"
    geometry::SurfaceGeometry
    uniformization::Uniformization
    eigenfunctions::SphereEigenfunctions
    generators::MobiusGenerators
    "gauge-invariant rotation one-form"
    ωinv::Tensor{1}
    "diagnostics (residuals of internal identities)"
    diagnostics::Dict{Symbol,Float64}
end

"""
    horizon_spin(embedding, metric3, excurv3; lmax, kwargs...) -> SpinResult

Compute the Korzyński quasi-local spin of the surface
x(θ,ϕ) = `embedding(θ,ϕ)::SVector{3}` in the Cauchy slice with data
`metric3(x)::SMatrix{3,3}` (γ_ij) and `excurv3(x)::SMatrix{3,3}` (K_ij,
convention K_ij = −(1/2)£_n γ_ij).
"""
function horizon_spin(
    embedding,
    metric3,
    excurv3;
    lmax::Int=24,
    grid::SphereGrid=EquiangularGrid(lmax),
    flow_tol::Float64=1.0e-3,
    flow_maxiter::Int=10_000,
    newton_tol::Float64=1.0e-13,
    use_newton::Bool=true,
)
    diagnostics = Dict{Symbol,Float64}()

    # §3.2–3.3: geometry and rotation one-form
    geom = surface_geometry(embedding, metric3, excurv3, grid)
    diagnostics[:q_imag] = imag_norm(geom.q)

    # Operators of the physical metric; Δ_q matrix is reused throughout
    ops = MetricOps(geom.q)
    Δmat = operator_matrix(f -> laplacian(ops, f), grid)

    # §3.4: Hodge gauge fixing
    ωinv, _ = hodge_fix(ops, geom.ω; Δmat=Δmat)
    diagnostics[:hodge_residual] = maximum(abs.(real.(grid_values(divergence(ops, ωinv))))) * geom.area / 4π

    # §3.5: curvature; Gauss–Bonnet check
    R = scalar_curvature(ops)
    diagnostics[:R_imag] = imag_norm(R)
    R = real_part(R)
    diagnostics[:gauss_bonnet] = integrate(geom, R) - 8π

    # §3.6: normalize area and uniformize.  q̄ = (4π/𝒜) q, so
    # Δ̄ = (𝒜/4π) Δ_q and R̄ = (𝒜/4π) R[q].
    s = geom.area / 4π
    ops̄ = MetricOps(map_fields(v -> v ./ s, geom.q))
    Δ̄mat = s .* Δmat
    R̄ = make_scalar(s .* grid_values(R), grid)
    unif = uniformize(ops̄, R̄; Δ̄mat=Δ̄mat, flow_tol=flow_tol, flow_maxiter=flow_maxiter, newton_tol=newton_tol, use_newton=use_newton)
    diagnostics[:round_residual] = unif.residual

    # §3.7: eigenfunctions
    eig = sphere_eigenfunctions(ops̄, unif.u; Δ̄mat=Δ̄mat)
    diagnostics[:eigenvalue_offset] = maximum(abs.(eig.eigenvalues .+ 2))
    diagnostics[:takahashi_residual] = eig.norm_residual

    # §3.8: generators
    gen = mobius_generators(ops̄, unif.u, eig)

    # §3.9: invariants
    Jvec = SVector{3}(momentum_integral(geom, ωinv, gen.φ[i]) for i in 1:3)
    Kvec = SVector{3}(momentum_integral(geom, ωinv, gen.ξ[i]) for i in 1:3)
    J, A, B = spin_from_invariants(Jvec, Kvec)

    β⃗ = parallel_frame_boost(Jvec, Kvec)
    Jvec′, Kvec′ = norm(β⃗) == 0 ? (Jvec, Kvec) : mobius_boost(Jvec, Kvec, β⃗)
    diagnostics[:boost_consistency] = abs(norm(Jvec′) - J)

    if J > 1.0e-12 * max(geom.area / 4π, 1.0) && norm(Jvec′) > 0
        axis = Jvec′ / norm(Jvec′)
        φ̃ = boosted_rotation_generators(gen, β⃗)
        axial = Tensor{1}(
            [sum(axis[i] * φ̃[i].values[ij] for i in 1:3) for ij in CartesianIndices(ash_grid_size(grid))], grid
        )
        # Spatial axis: angular-momentum direction of the axial flow field
        # v^i = φ^a e_a^i in the embedding (exactly the rotation axis for a
        # rigid rotation; a well-defined reporting proxy in general)
        m⃗vals = [
            cross(geom.x[ij], SVector{3}(sum(real(axial.values[ij][a]) * geom.E[ij][a, i] for a in 1:2) for i in 1:3)) for
            ij in CartesianIndices(ash_grid_size(grid))
        ]
        m⃗ = SVector{3}(integrate(geom, make_scalar([m[i] for m in m⃗vals], grid)) for i in 1:3)
        axis_embedding = m⃗ / norm(m⃗)
    else
        axis = SVector(0.0, 0.0, 0.0)
        axis_embedding = SVector(0.0, 0.0, 0.0)
        axial = Tensor{1}([zero(SVector{2,ComplexF64}) for _ in CartesianIndices(ash_grid_size(grid))], grid)
    end

    return SpinResult(
        grid, geom.area, J, Jvec, Kvec, A, B, β⃗, Jvec′, Kvec′, axis, axis_embedding, axial, geom, unif, eig, gen, ωinv,
        diagnostics,
    )
end
