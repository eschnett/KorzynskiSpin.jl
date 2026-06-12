# First eigenfunctions of the round Laplacian and the Möbius generators
# (docs/algorithm.tex §3.7–3.8).
#
# The triple χ_i solves the generalized eigenproblem
#     Δ̄ χ = −2 e^{2u} χ
# (eigenvalue −2 of Δ̊ = e^{−2u} Δ̄, triply degenerate).  After Löwdin
# orthonormalization w.r.t. the round measure ε̊ and a handedness fix, the
# Möbius generators in the original chart are
#     φ_i^a = ε̊^{ab} ∂_b χ_i,    ξ_i^a = −q̊^{ab} ∂_b χ_i .

struct SphereEigenfunctions
    "the triple χ_i (real scalar fields)"
    χ::NTuple{3,Tensor{0}}
    "their differentials ∂_a χ_i"
    dχ::NTuple{3,Tensor{1}}
    "eigenvalues (should be ≈ −2)"
    eigenvalues::SVector{3,Float64}
    "‖Σ_i χ_i² − 1‖∞ (Takahashi rigidity check)"
    norm_residual::Float64
end

"""
    sphere_eigenfunctions(ops̄, u; Δ̄mat) -> SphereEigenfunctions

`ops̄` are the operators of the area-normalized metric q̄, `u` the
uniformizing conformal exponent.
"""
function sphere_eigenfunctions(ops̄::MetricOps, u::Tensor{0}; Δ̄mat::Union{Nothing,Matrix{ComplexF64}}=nothing)
    lmax = ops̄.lmax
    n = ash_nmodes(lmax)[1]

    e2u = exp.(2 .* real.(grid_values(u)))
    A = Δ̄mat === nothing ? operator_matrix(f -> laplacian(ops̄, f), lmax) : Δ̄mat
    B = multiplication_matrix(make_scalar(e2u, lmax))

    # Dense generalized eigenproblem; take the cluster nearest −2
    ev = eigen(A, B)
    idx = sortperm(abs.(ev.values .+ 2))[1:3]
    eigenvalues = SVector{3}(real.(ev.values[idx]))

    # The eigenvectors may mix arbitrarily (complex phases included) within
    # the near-degenerate cluster.  Recover a real orthonormal triple from
    # the real and imaginary parts of the eigenfields.
    candidates = Vector{Matrix{Float64}}()
    for j in idx
        f = grid_values(coeffs_scalar(ev.vectors[:, j], lmax))
        push!(candidates, real.(f))
        push!(candidates, imag.(f))
    end
    # Round measure relative to ε̂
    μ̊ = e2u .* real.(grid_values(ops̄.sqrtdetq))
    inner(f, g) = integrate_unit(make_scalar(f .* g .* μ̊, lmax))
    G = [inner(f, g) for f in candidates, g in candidates]
    evG = eigen(Symmetric(G))
    # Top three directions span the eigenspace
    χs = Vector{Matrix{Float64}}()
    for k in 0:2
        v = evG.vectors[:, end - k]
        f = sum(v[m] .* candidates[m] for m in eachindex(candidates))
        push!(χs, f ./ sqrt(max(evG.values[end - k], eps())))
    end

    # Löwdin-orthonormalize so that ∮ χ_i χ_j ε̊ = (4π/3) δ_ij
    G3 = Symmetric([3 / 4π * inner(χs[i], χs[j]) for i in 1:3, j in 1:3])
    W = inv(sqrt(G3))
    χs = [sum(W[i, m] .* χs[m] for m in 1:3) for i in 1:3]

    χ = ntuple(i -> make_scalar(χs[i], lmax), 3)
    dχ = ntuple(i -> real_part(differential(χ[i])), 3)

    # Handedness: 𝒪 = ∮ χ₁ ε̊^{ab} ∂_aχ₂ ∂_bχ₃ ε̊ = +4π/3 for a right-handed
    # triple.  The dyad-frame expression is metric-free (conformal invariance
    # of the Hodge star on one-forms).
    cross12 = [
        real(d2[1] * d3[2] - d2[2] * d3[1]) for (d2, d3) in zip(dχ[2].values, dχ[3].values)
    ]
    𝒪 = integrate_unit(make_scalar(χs[1] .* cross12, lmax))
    if 𝒪 < 0
        χ = (χ[2], χ[1], χ[3])
        dχ = (dχ[2], dχ[1], dχ[3])
        χs = [χs[2], χs[1], χs[3]]
    end

    norm_residual = maximum(abs.(χs[1] .^ 2 .+ χs[2] .^ 2 .+ χs[3] .^ 2 .- 1))
    return SphereEigenfunctions(χ, dχ, eigenvalues, norm_residual)
end

################################################################################
# Möbius generators

struct MobiusGenerators
    "rotation generators φ_i^a (dyad components)"
    φ::NTuple{3,Tensor{1}}
    "proper conformal generators ξ_i^a (dyad components)"
    ξ::NTuple{3,Tensor{1}}
end

"""
    mobius_generators(ops̄, u, eigfns) -> MobiusGenerators

Round metric q̊ = e^{2u} q̄; generators per Lemma 1 of docs/algorithm.tex:
φ_i^a = ε̊^{ab} ∂_b χ_i, ξ_i^a = −q̊^{ab} ∂_b χ_i.
"""
function mobius_generators(ops̄::MetricOps, u::Tensor{0}, eig::SphereEigenfunctions)
    lmax = ops̄.lmax
    sz = ash_grid_size(lmax)
    e2u = exp.(2 .* real.(grid_values(u)))

    φs = Vector{Tensor{1}}()
    ξs = Vector{Tensor{1}}()
    for i in 1:3
        dχ = eig.dχ[i]
        φvals = Matrix{SVector{2,ComplexF64}}(undef, sz...)
        ξvals = Matrix{SVector{2,ComplexF64}}(undef, sz...)
        for ij in CartesianIndices(sz)
            q̊ = e2u[ij] * real.(ops̄.q.values[ij])
            q̊u = inv(q̊)
            sd = sqrt(abs(det(q̊)))
            d = real.(dχ.values[ij])
            # ε̊^{12} = +1/√det q̊ in the (right-handed) dyad frame
            φvals[ij] = SVector{2,ComplexF64}(d[2] / sd, -d[1] / sd)
            ξvals[ij] = SVector{2,ComplexF64}(-(q̊u * d))
        end
        push!(φs, Tensor{1}(φvals, lmax))
        push!(ξs, Tensor{1}(ξvals, lmax))
    end
    return MobiusGenerators((φs[1], φs[2], φs[3]), (ξs[1], ξs[2], ξs[3]))
end
