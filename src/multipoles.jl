# Balanced round metric and horizon multipoles (docs.tex/algorithm.tex,
# "Balanced round metric and horizon multipoles").
#
# The unit round metrics conformal to q form one Möbius orbit.  Acting with
# Λ ∈ SO⁺(1,3) on X = (1, χ) maps the ℓ=1 triple and the conformal exponent
# pointwise,
#     χ′ = (ΛX)_{1:3} / (ΛX)_0,    u′ = u − log (ΛX)_0 ,
# so no further eigenproblem is needed.  Following Ashtekar, Khera, Kolanowski
# & Lewandowski (arXiv:2111.07873, eq. 2.15 and App. A) the canonical member
# is the unique one with vanishing area dipole, ∮ χ′_i ε_q = 0.  In that frame
# the shape and current multipoles are (Gourgoulhon, Le Tiec & Casals,
# arXiv:2602.05823, eqs. 3.9–3.10 and 5.10)
#     I_lm = −∮ Re Ψ₂ Y̊_lm ε = ¼ ∮ R[q] Y̊_lm ε,
#     L_lm = −∮ Im Ψ₂ Y̊_lm ε = −½ ∮ Y̊_lm dω = ½ ∮ dY̊_lm ∧ ω .

################################################################################
# Möbius action on the ℓ=1 triple

"Lorentz boost with velocity b⃗ (|b⃗| < 1) acting on (t, x⃗)"
function boost_matrix(b⃗::SVector{3,Float64})
    b = norm(b⃗)
    b == 0 && return SMatrix{4,4,Float64}(I)
    γ = 1 / sqrt(1 - b^2)
    n̂ = b⃗ / b
    S = SMatrix{3,3,Float64}(I) + (γ - 1) * n̂ * n̂'
    return SMatrix{4,4,Float64}([γ γ*b⃗'; γ*b⃗ S])
end

"Spatial rotation R ∈ SO(3) acting on (t, x⃗)"
function rotation_matrix4(R::SMatrix{3,3,Float64})
    return SMatrix{4,4,Float64}([1 zero(SVector{3})'; zero(SVector{3}) R])
end

"Möbius image of the point χ⃗ ∈ S²: (χ⃗′, Ω) with q̊′ = Ω² q̊"
function mobius_point(Λ::SMatrix{4,4,Float64}, χ⃗::SVector{3,Float64})
    Y = Λ * SVector(1.0, χ⃗[1], χ⃗[2], χ⃗[3])
    return SVector(Y[2], Y[3], Y[4]) / Y[1], 1 / Y[1]
end

"Pointwise triple (SVector{3} per grid point) of three real scalar fields"
point_triple(χ::NTuple{3,Tensor{0}}) = [SVector{3}(real(v[]) for v in vs) for vs in zip((c.values for c in χ)...)]

"Complex ∮ f ε̂ over the unit coordinate sphere (`integrate_unit` keeps only the real part)"
function cintegrate_unit(vals::AbstractMatrix{<:Number}, grid::SphereGrid)
    return integrate_unit(make_scalar(real.(vals), grid)) + im * integrate_unit(make_scalar(imag.(vals), grid))
end

"Dyad-frame 2-form α ∧ β in units of ε̂ (metric-free, cf. the handedness integral)"
wedge(α::SVector{2}, β::SVector{2}) = α[1] * β[2] - α[2] * β[1]

"""
    balance_frame(geom, u, χ; tol=1e-13, maxiter=50) -> NamedTuple

Find the Möbius transformation Λ ∈ SO⁺(1,3) whose image of the unit round
metric `q̊ = e^{2u} q̄` (with ℓ=1 triple `χ`) has vanishing area dipole
`d⃗ = ∮ χ′ ε_q = 0` (AKKL, arXiv:2111.07873, eq. 2.15; unique by App. A).

Damped Newton iteration on the conformal barycenter: in the current frame the
Jacobian of d⃗ with respect to a boost velocity b⃗ is `H = ∮ (1 − χχᵀ) ε_q`
(positive definite), so the step is `b⃗ = −H⁻¹ d⃗`, halved until |d⃗| decreases.

Returns `(; Λ, u, χ, iters, residual, converged)` where `χ` and `u` are the
balanced triple and conformal exponent and `residual = |d⃗|/𝒜`.
"""
function balance_frame(geom::SurfaceGeometry, u::Tensor{0}, χ::NTuple{3,Tensor{0}}; tol::Float64=1.0e-13, maxiter::Int=50)
    grid = geom.grid
    χ0 = point_triple(χ)
    μvals = real.(grid_values(geom.sqrtdetq))
    ∮(f) = integrate_unit(make_scalar(f .* μvals, grid))

    dipole(χs) = SVector{3}(∮(map(c -> c[i], χs)) for i in 1:3)
    apply(Λ) = map(c -> mobius_point(Λ, c)[1], χ0)

    Λ = SMatrix{4,4,Float64}(I)
    χs = χ0
    d⃗ = dipole(χs)
    residual = norm(d⃗) / geom.area
    iters = 0
    while residual > tol && iters < maxiter
        iters += 1
        H = SMatrix{3,3}(∮(map(c -> (i == j) - c[i] * c[j], χs)) for i in 1:3, j in 1:3)
        b⃗ = -(H \ d⃗)
        norm(b⃗) > 0.5 && (b⃗ *= 0.5 / norm(b⃗))
        improved = false
        for _ in 1:30
            Λ′ = boost_matrix(b⃗) * Λ
            χs′ = apply(Λ′)
            d⃗′ = dipole(χs′)
            if norm(d⃗′) < norm(d⃗)
                Λ, χs, d⃗ = Λ′, χs′, d⃗′
                improved = true
                break
            end
            b⃗ /= 2
        end
        residual = norm(d⃗) / geom.area
        improved || break   # round-off floor
    end

    uvals = real.(grid_values(u)) .+ map(c -> log(mobius_point(Λ, c)[2]), χ0)
    χ′ = ntuple(i -> make_scalar(map(c -> c[i], χs), grid), 3)
    return (; Λ, u=make_scalar(uvals, grid), χ=χ′, iters, residual, converged=residual ≤ tol)
end

################################################################################
# Multipoles

"""
    HorizonMultipoles

Result of [`horizon_multipoles`](@ref): shape and current multipoles
`I[(l, m)]`, `L[(l, m)]` in the area-dipole balanced, oriented frame, the
balanced round metric (`u`, `χ`), the Möbius map `Λ` that produced it, and
diagnostics.
"""
struct HorizonMultipoles
    "largest degree ℓ computed"
    lmax::Int
    "shape (mass) multipoles I_lm = ¼ ∮ R[q] Y̊_lm ε, keyed by (l, m)"
    I::Dict{Tuple{Int,Int},ComplexF64}
    "current (spin) multipoles L_lm = ½ ∮ dY̊_lm ∧ ω, keyed by (l, m)"
    L::Dict{Tuple{Int,Int},ComplexF64}
    "Möbius map from `SpinResult.eigenfunctions.χ` to the balanced, oriented triple"
    Λ::SMatrix{4,4,Float64,16}
    "net boost velocity of Λ"
    β⃗::SVector{3,Float64}
    "balanced conformal exponent: q̊ = e^{2u} q̄ (AKKL's ψ² = (4π/𝒜) e^{2u})"
    u::Tensor{0}
    "balanced, oriented ℓ=1 triple (Cartesian coordinates of the canonical round sphere)"
    χ::NTuple{3,Tensor{0}}
    "real current dipole ½ ∮ dχ_i ∧ ω in the final frame (along +z unless ≈ 0)"
    L1vec::SVector{3,Float64}
    "Korzyński's J⃗ and K⃗ evaluated in the balanced frame"
    Jvec::SVector{3,Float64}
    Kvec::SVector{3,Float64}
    "diagnostics (residuals of internal identities)"
    diagnostics::Dict{Symbol,Float64}
end

"Unit vector v̂ with sign chosen so that v̂ ⋅ ref ≥ 0"
aligned(v, ref) = (v̂=normalize(v); dot(v̂, ref) < 0 ? -v̂ : v̂)

"""
    orient_frame(χs, dχ, ω, Rvals, geom) -> R::SMatrix{3,3}

Rotation R (rows = new axes in the current frame) fixing the SO(3) freedom of
a balanced triple: ẑ along the real current dipole L⃗ = ½ ∮ dχ ∧ ω, and x̂
along the principal axis (largest eigenvalue) of the mass quadrupole
Q_ij = ∮ R χ_i χ_j ε_q in the plane ⊥ ẑ.  Without a current dipole, ẑ is the
principal axis of Q whose eigenvalue lies farthest from the mean.  Degenerate
directions keep the current axes; eigenvector signs are chosen to overlap
positively with the current axes (a rotation by π about ẑ, X_lm → (−1)^m X_lm,
is left undetermined for reflection-symmetric shapes).
"""
function orient_frame(χs, dχ, ω::Tensor{1}, Rvals, geom::SurfaceGeometry; tol::Float64=1.0e-10)
    grid = geom.grid
    μvals = real.(grid_values(geom.sqrtdetq))
    L⃗ = SVector{3}(
        integrate_unit(make_scalar([real(wedge(d, w)) for (d, w) in zip(dχ[i].values, ω.values)], grid)) / 2 for i in 1:3
    )
    Q = Symmetric(
        Matrix(SMatrix{3,3}(integrate_unit(make_scalar(map(c -> c[i] * c[j], χs) .* Rvals .* μvals, grid)) for i in 1:3, j in 1:3))
    )
    e = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
    scaleQ = max(opnorm(Q), eps())

    if norm(L⃗) > tol
        ẑ = normalize(L⃗)
    else
        ev = eigen(Q)
        λ̄ = sum(ev.values) / 3
        k = argmax(abs.(ev.values .- λ̄))
        ẑ = abs(ev.values[k] - λ̄) > tol * scaleQ ? aligned(SVector{3}(ev.vectors[:, k]), e[3]) : e[3]
    end
    # current x̂ projected into the plane ⊥ ẑ (fall back to ŷ if nearly parallel)
    x0 = e[1] - dot(e[1], ẑ) * ẑ
    norm(x0) < 0.1 && (x0 = e[2] - dot(e[2], ẑ) * ẑ)
    e1 = normalize(x0)
    e2 = cross(ẑ, e1)
    Q2 = Symmetric([dot(a, Q * b) for a in (e1, e2), b in (e1, e2)])
    ev2 = eigen(Q2)
    x̂ = if abs(ev2.values[2] - ev2.values[1]) > tol * scaleQ
        v = ev2.vectors[:, 2]
        aligned(v[1] * e1 + v[2] * e2, e1)
    else
        e1
    end
    ŷ = cross(ẑ, x̂)
    return SMatrix{3,3}(x̂[1], ŷ[1], ẑ[1], x̂[2], ŷ[2], ẑ[2], x̂[3], ŷ[3], ẑ[3])
end

"""
    horizon_multipoles(res::SpinResult; lmax, balance_tol=1e-13) -> HorizonMultipoles

Shape and current multipoles of the surface in the canonical (area-dipole
balanced) unit round metric of Ashtekar, Khera, Kolanowski & Lewandowski
(arXiv:2111.07873):

    I_lm = −∮ Re Ψ₂ Y̊_lm ε = ¼ ∮ R[q] Y̊_lm ε,
    L_lm = −∮ Im Ψ₂ Y̊_lm ε = ½ ∮ dY̊_lm ∧ ω,

with `Y̊_lm` the standard (Condon–Shortley) spherical harmonics of the balanced
round sphere, not complex-conjugated.  Monopoles are universal: `I₀₀ = √π`,
`L₀₀ = 0`.  On a non-expanding horizon these are the NEH multipoles; on a
general marginally trapped surface they are the corresponding geometric
moments of R[q] and dω.

The global SO(3) left by the balancing is fixed by pointing ẑ along the
current dipole and x̂ along the principal axis of the mass quadrupole
(see [`orient_frame`](@ref)).  Default `lmax = min(8, ash_lmax(grid) ÷ 2)`.
"""
function horizon_multipoles(res::SpinResult; lmax::Int=min(8, ash_lmax(res.grid) ÷ 2), balance_tol::Float64=1.0e-13)
    geom = res.geometry
    grid = geom.grid
    diagnostics = Dict{Symbol,Float64}()

    Rvals = real.(grid_values(real_part(scalar_curvature(MetricOps(geom.q)))))
    ω = res.ωinv

    # Balance, then orient
    bal = balance_frame(geom, res.uniformization.u, res.eigenfunctions.χ; tol=balance_tol)
    diagnostics[:dipole_residual] = bal.residual
    diagnostics[:balance_iters] = bal.iters
    χb = point_triple(bal.χ)
    dχb = ntuple(i -> real_part(differential(bal.χ[i])), 3)
    Rot = orient_frame(χb, dχb, ω, Rvals, geom)
    Λ = rotation_matrix4(Rot) * bal.Λ
    β⃗ = SVector(Λ[2, 1], Λ[3, 1], Λ[4, 1]) / Λ[1, 1]

    χs = map(c -> Rot * c, χb)
    χ = ntuple(i -> make_scalar(map(c -> c[i], χs), grid), 3)
    dχ = ntuple(i -> real_part(differential(χ[i])), 3)
    u = bal.u
    diagnostics[:takahashi_residual] = maximum(c -> abs(dot(c, c) - 1), χs)
    L1vec = SVector{3}(
        integrate_unit(make_scalar([real(wedge(d, w)) for (d, w) in zip(dχ[i].values, ω.values)], grid)) / 2 for i in 1:3
    )

    # Multipoles
    μvals = real.(grid_values(geom.sqrtdetq))
    θϕ = map(c -> (acos(clamp(c[3] / norm(c), -1.0, 1.0)), atan(c[2], c[1])), χs)
    Is = Dict{Tuple{Int,Int},ComplexF64}()
    Ls = Dict{Tuple{Int,Int},ComplexF64}()
    for l in 0:lmax, m in (-l):l
        Yvals = map(p -> sYlm(0, l, m, p[1], p[2]), θϕ)
        Is[(l, m)] = cintegrate_unit(Rvals .* Yvals .* μvals, grid) / 4
        dY = differential(make_scalar(Yvals, grid))
        Ls[(l, m)] = cintegrate_unit([wedge(d, w) for (d, w) in zip(dY.values, ω.values)], grid) / 2
    end
    diagnostics[:I00_offset] = abs(Is[(0, 0)] - sqrt(π))
    diagnostics[:L00] = abs(Ls[(0, 0)])
    diagnostics[:conjugation] = maximum(abs(X[(l, -m)] - (-1)^m * conj(X[(l, m)])) for X in (Is, Ls) for l in 0:lmax for m in 0:l)

    # Consistency: Korzyński's invariants in the balanced frame
    s = geom.area / 4π
    ops̄ = MetricOps(map_fields(v -> v ./ s, geom.q))
    eig = SphereEigenfunctions(χ, dχ, res.eigenfunctions.eigenvalues, diagnostics[:takahashi_residual])
    gen = mobius_generators(ops̄, u, eig)
    Jvec = SVector{3}(momentum_integral(geom, ω, gen.φ[i]) for i in 1:3)
    Kvec = SVector{3}(momentum_integral(geom, ω, gen.ξ[i]) for i in 1:3)
    _, A, B = spin_from_invariants(Jvec, Kvec)
    diagnostics[:invariant_A] = abs(A - res.A)
    diagnostics[:invariant_B] = abs(B - res.B)

    return HorizonMultipoles(lmax, Is, Ls, Λ, β⃗, u, χ, L1vec, Jvec, Kvec, diagnostics)
end
