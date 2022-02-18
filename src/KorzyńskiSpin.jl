module KorzyńskiSpin

using AbstractSphericalHarmonics
using Arpack
using ForwardDiff
using LinearAlgebra
using LinearOperators
using StaticArrays
using StatsPlots

include("kerr_schild.jl")

################################################################################
# General helpers

bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))

flatten(xss::SVector{N,<:SVector{M}}) where {N,M} = SVector{N * M}(xss[n][m] for m in 1:M, n in 1:N)

Base.chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
Base.chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))
Base.chop(x::SArray) = chop.(x)

################################################################################

Base.:+(A::LinearOperator, B::UniformScaling) = A + Diagonal(fill(B.λ, size(A, 1)))
Base.:-(A::LinearOperator, B::UniformScaling) = A + -B

################################################################################
# Accuracy

# The discretization error decreases exponentially.
# The floating-point errors increase linearly.
atol(lmax) = 1.0e+6 * max(exp(-lmax), lmax * eps())

################################################################################
# Tensor helpers

function nmodes2lmax(nmodes::Int)
    lmax0 = 0
    nmodes0 = ash_nmodes(lmax0)
    @assert nmodes0 ≤ nmodes
    lmax1 = 100
    nmodes1 = ash_nmodes(lmax1)
    while nmodes1 < nmodes
        lmax1 *= 2
        nmodes1 = ash_nmodes(lmax1)
    end
    @assert nmodes1 ≥ nmodes
    while lmax0 < lmax1
        range_old = lmax1 - lmax0
        lmax′ = (lmax0 + lmax1) ÷ 2
        nmodes′ = ash_nmodes(lmax′)
        nmodes′ == nmodes && return lmax′
        if nmodes′ < nmodes
            lmax0 = lmax′ + 1
            nmodes0 = ash_nmodes(lmax0)
        else
            lmax1 = lmax′ - 1
            nmodes1 = ash_nmodes(lmax1)
        end
        range_new = lmax1 - lmax0
        @assert 0 ≤ range_new < range_old
    end
    @assert nmodes0 == nmodes
    return lmax0
end

function mode_norm(t::SpinTensor{2}, l::Int)
    lmax = t.lmax
    @assert 0 ≤ l ≤ lmax
    norms = real(eltype(t))[]
    l ≥ 2 && push!(norms, norm([t.coeffs[1, 1][ash_mode_index(+2, l, m, lmax)] for m in (-l):l]))
    push!(norms, norm([t.coeffs[2, 1][ash_mode_index(0, l, m, lmax)] for m in (-l):l]))
    push!(norms, norm([t.coeffs[1, 2][ash_mode_index(0, l, m, lmax)] for m in (-l):l]))
    l ≥ 2 && push!(norms, norm([t.coeffs[2, 2][ash_mode_index(-2, l, m, lmax)] for m in (-l):l]))
    return norm(norms)
end

function impose_symmetry(f, t::Tensor{D}; check::Bool=true) where {D}
    if !(norm(t.values, Inf) < Inf)
        @show t
    end
    @assert norm(t.values, Inf) < Inf
    if check
        if !(norm(t.values - f.(t.values), Inf) ≤ atol(t.lmax))
            @show t.values
        end
        @assert norm(t.values - f.(t.values), Inf) ≤ atol(t.lmax)
    end
    return Tensor{D}(f.(t.values), t.lmax)
end

make_real(t::Tensor; check::Bool=true) = impose_symmetry(x -> map(a -> Complex(real(a)), x), t; check=check)
make_symmetric(t::Tensor{2}; check::Bool=true) = impose_symmetry(x -> (x + x') / 2, t; check=check)
function make_symmetric12(t::Tensor{3})
    s12 = x -> SArray{Tuple{2,2,2}}((x[a, b, c] + conj(x[b, a, c])) / 2 for a in 1:2, b in 1:2, c in 1:2)
    return impose_symmetry(s12, t)
end
function make_symmetric12(t::Tensor{4})
    s12 = x -> SArray{Tuple{2,2,2,2}}((x[a, b, c, d] + conj(x[b, a, c, d])) / 2 for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    return impose_symmetry(s12, t)
end
function make_symmetric1234(t::Tensor{4})
    s12 = x -> SArray{Tuple{2,2,2,2}}((x[a, b, c, d] + conj(x[b, a, c, d])) / 2 for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    s34 = x -> SArray{Tuple{2,2,2,2}}((x[a, b, c, d] + conj(x[a, b, d, c])) / 2 for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    return impose_symmetry(s34 ∘ s12, t)
end
function make_riemann_symmetry(t::Tensor{4})
    a12 = x -> SArray{Tuple{2,2,2,2}}((x[a, b, c, d] - conj(x[b, a, c, d])) / 2 for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    a34 = x -> SArray{Tuple{2,2,2,2}}((x[a, b, c, d] - conj(x[a, b, d, c])) / 2 for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    s1234 = x -> SArray{Tuple{2,2,2,2}}((x[a, b, c, d] + conj(x[c, d, a, b])) / 2 for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    return impose_symmetry(s1234 ∘ a34 ∘ a12, t)
end

################################################################################
# Coordinate transformations

"X^a(x^b)"
function xform_distort(x::SVector{2})
    α1 = 0.1
    α2 = 0                      # cannot be nonzero
    α3 = 0.1
    β1 = 0                      # cannot be nonzero
    β2 = 0.1
    β3 = 0                      # cannot be nonzero

    θ, ϕ = x
    Q = θ + α1 * sin(θ) + α2 * sin(θ) * sin(ϕ) + α3 * sin(θ)^2 * sin(ϕ)
    F = ϕ + β1 * sin(θ) + β2 * sin(θ)^2 + β3 * sin(θ)^2 * sin(ϕ)
    X = SVector{2}(Q, F)
    return X
end

"X^a(x^b)"
function xform_3d(θϕ::SVector{2})
    dx = 0.1
    dy = 0.1
    dz = 0.1

    θ, ϕ = θϕ
    r = 1
    x = r * sin(θ) * cos(ϕ)
    y = r * sin(θ) * sin(ϕ)
    z = r * cos(θ)
    X = x + dx
    Y = y + dy
    Z = z + dz
    R = sqrt(X^2 + Y^2 + Z^2)
    Q = acos(Z / R)
    F = atan(Y, X)
    QF = SVector{2}(Q, F)
    return QF
end

"J^a_b = ∂X^a/∂x_b(x^c)   [unscaled]"
Jac0(xform, x::SVector{2}) = ForwardDiff.jacobian(xform, x)

"dJ^a_bc = ∂²X^a/∂x_b∂x_c(x^d)   [unscaled]"
dJac0(xform, x::SVector{2}) = reshape(ForwardDiff.jacobian(y -> Jac0(xform, y), x), (2, 2, 2))

"J^a_b = ∂X^a/∂x_b(x^c)"
function Jac(xform, x::SVector{2})
    θ, ϕ = x
    Q, F = xform(x)
    F = SVector(1, sin(Q))
    f = SVector(1, 1 / sin(θ))
    J0 = Jac0(xform, x)
    J = SMatrix{2,2}(J0[a, b] * F[a] * f[b] for a in 1:2, b in 1:2)
    return J
end

"dJ^a_bc = ∂²X^a/∂x_b∂x_c(x^d)"
function dJac(xform, x::SVector{2})
    θ, ϕ = x
    Q, F = xform(x)
    F = SVector(1, sin(Q))
    f = SVector(1, 1 / sin(θ))
    dJ0 = dJac0(xform, x)
    dJ = SArray{Tuple{2,2,2}}(dJ0[a, b, c] * F[a] * f[b] * f[c] for a in 1:2, b in 1:2, c in 1:2)
    return dJ
end

################################################################################
# Round metric

# q_ab = (1, 0, 0, 1)
make_q(x::SVector{2,T}) where {T} = SMatrix{2,2,Complex{T}}(1, 0, 0, 1)

function make_q(lmax::Int)
    sz = ash_grid_size(lmax)
    return Tensor{2}([
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            make_q(x)
        end for ij in CartesianIndices(sz)
    ], lmax)
end

################################################################################
# Physical metric

# g_ab = J_a^c J_b^d q_cd
function make_g(xform, x::SVector{2,T}) where {T}
    X = xform(x)
    q = make_q(X)
    J = Jac(xform, x)
    g = SMatrix{2,2}(sum(q[c, d] * J[c, a] * J[d, b] for c in 1:2, d in 1:2) for a in 1:2, b in 1:2)
    return g
end

function make_g(xform, lmax::Int)
    sz = ash_grid_size(lmax)
    return Tensor{2}([
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            make_g(xform, x)
        end for ij in CartesianIndices(sz)
    ], lmax)
end

function check_g(xform, g::Tensor{2})
    lmax = g.lmax
    g₀ = Tensor{2}([
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            θ, ϕ = x
            X = xform(x)
            Q, F = X
            J = Jac0(xform, x)

            q, f = x
            dQdq = J[1, 1]
            dQdf = J[1, 2]
            dFdq = J[2, 1]
            dFdf = J[2, 2]

            Cos = cos
            Cot = cot
            Csc = csc
            Sin = sin
            Power = (^)

            gθθ = Power(Sin(Q), 2) * Power(dFdq, 2) + Power(dQdq, 2)
            gθϕ = Power(Sin(Q), 2) * dFdf * dFdq + dQdf * dQdq
            gϕϕ = Power(Sin(Q), 2) * Power(dFdf, 2) + Power(dQdf, 2)

            gθθ /= 1
            gθϕ /= sin(θ)
            gϕϕ /= sin(θ)^2

            SMatrix{2,2}(gθθ, gθϕ, gθϕ, gϕϕ)
        end for ij in CartesianIndices(g.values)
    ], lmax)
    if !(norm(g.values - g₀.values, Inf) ≤ atol(lmax))
        println("lmax: ", lmax, "   atol: ", atol(lmax))
        for a in 1:2, b in a:2
            println("g₀[$a,$b]: ", g₀.values[1, 1][a, b])
            println("g[$a,$b]: ", g.values[1, 1][a, b])
            println("|(g-g₀)[$a,$b]|∞: ", norm(map(x -> x[a, b], g.values - g₀.values), Inf))
        end
        @assert false
    end
    return nothing
end

make_gu(g::Tensor{2}) = Tensor{2}([inv(g) for g in g.values], g.lmax)

################################################################################
# Connection

function make_Γ(gu::Tensor{2}, dg::Tensor{3})
    return Tensor{3}(
        [
            SArray{Tuple{2,2,2}}(
                sum(gu[a, d] * (dg[d, c, b] + dg[b, d, c] - dg[b, c, d]) / 2 for d in 1:2) for a in 1:2, b in 1:2, c in 1:2
            ) for (gu, dg) in zip(gu.values, dg.values)
        ],
        gu.lmax,
    )
end

################################################################################
# Curvature

function make_Rm(
    gu::SMatrix{2,2},
    v1::SVector{2},
    v2::SVector{2},
    v3::SVector{2},
    DDv1::SArray{Tuple{2,2,2}},
    DDv2::SArray{Tuple{2,2,2}},
    DDv3::SArray{Tuple{2,2,2}},
) where {T}
    vs = SVector(v1, v2, v3)
    DDvs = SVector(DDv1, DDv2, DDv3)
    vus = SVector{3}(SVector{2}(sum(gu[a, x] * v[x] for x in 1:2) for a in 1:2) for v in vs)
    lhs = flatten(SVector{3}(SVector{2}(vu[2], vu[1]) for vu in vus))
    rhs = flatten(SVector{3}(SVector{2}(DDv[1, 2, 1] - DDv[1, 1, 2], DDv[2, 1, 2] - DDv[2, 2, 1]) for DDv in DDvs))
    Rm1212 = lhs \ rhs          # uses least squares
    Rm = SArray{Tuple{2,2,2,2}}(0, 0, 0, 0, 0, Rm1212, -Rm1212, 0, 0, -Rm1212, Rm1212, 0, 0, 0, 0, 0)
    return Rm
end

function make_Riemann(g::Tensor{2}, gu::Tensor{2}, Γ::Tensor{3})
    lmax = g.lmax
    # Define 3 vector fields (x⃗, y⃗, z⃗), consider their covariant Hessians:
    #     (D_c D_b v_a - D_b D_c v_a) = A_x R^xabc
    s1 = Tensor{0}([
        begin
            θ, ϕ = SVector{2}(ash_point_coord(ij, lmax))
            x = sin(θ) * cos(ϕ)
            Scalar(x)
        end for ij in CartesianIndices(g.values)
    ], lmax)
    s2 = Tensor{0}([
        begin
            θ, ϕ = SVector{2}(ash_point_coord(ij, lmax))
            y = sin(θ) * sin(ϕ)
            Scalar(y)
        end for ij in CartesianIndices(g.values)
    ], lmax)
    s3 = Tensor{0}([
        begin
            θ, ϕ = SVector{2}(ash_point_coord(ij, lmax))
            z = cos(θ)
            Scalar(z)
        end for ij in CartesianIndices(g.values)
    ], lmax)

    ss = [s1, s2, s3]
    s̃s = SpinTensor.(ss)
    ṽs = tensor_gradient.(s̃s)
    vs = Tensor.(ṽs)

    dṽs = tensor_gradient.(ṽs)
    dvs = Tensor.(dṽs)
    Dvs = [
        Tensor{2}(
            [
                SArray{Tuple{2,2}}(dv[a, b] - sum(Γ[x, a, b] * v[x] for x in 1:2) for a in 1:2, b in 1:2) for
                (v, dv, Γ) in zip(v.values, dv.values, Γ.values)
            ],
            lmax,
        ) for (v, dv) in zip(vs, dvs)
    ]
    Dṽs = SpinTensor.(Dvs)

    dDṽs = tensor_gradient.(Dṽs)
    dDvs = Tensor.(dDṽs)
    DDvs = [
        Tensor{3}(
            [
                SArray{Tuple{2,2,2}}(
                    dDv[a, b, c] - sum(Γ[x, a, c] * Dv[x, b] + Γ[x, b, c] * Dv[a, x] for x in 1:2) for a in 1:2, b in 1:2, c in 1:2
                ) for (Dv, dDv, Γ) in zip(Dv.values, dDv.values, Γ.values)
            ],
            lmax,
        ) for (Dv, dDv) in zip(Dvs, dDvs)
    ]

    Rm = Tensor{4}(
        [
            begin
                vs = SVector(v1, v2, v3)
                DDvs = SVector(DDv1, DDv2, DDv3)
                vus = SVector{3}(SVector{2}(sum(gu[a, x] * v[x] for x in 1:2) for a in 1:2) for v in vs)
                lhs = flatten(SVector{3}(SVector{2}(vu[2], vu[1]) for vu in vus))
                rhs = flatten(SVector{3}(SVector{2}(DDv[1, 2, 1] - DDv[1, 1, 2], DDv[2, 1, 2] - DDv[2, 2, 1]) for DDv in DDvs))
                Rm1212 = lhs \ rhs          # uses least squares
                SArray{Tuple{2,2,2,2}}(0, 0, 0, 0, 0, Rm1212, -Rm1212, 0, 0, -Rm1212, Rm1212, 0, 0, 0, 0, 0)
            end for (gu, v1, v2, v3, DDv1, DDv2, DDv3) in
            zip(gu.values, vs[1].values, vs[2].values, vs[3].values, DDvs[1].values, DDvs[2].values, DDvs[3].values)
        ],
        lmax,
    )

    return Rm
end

# Rm₀ = Tensor{4}(
#     [
#         begin
#             x = SVector{2}(ash_point_coord(ij, lmax))
#             θ, ϕ = x
#             X = x2X(x)
#             Q, F = X
#             J = Jac0(x)
# 
#             q, f = x
#             dQdq = J[1, 1]
#             dQdf = J[1, 2]
#             dFdq = J[2, 1]
#             dFdf = J[2, 2]
# 
#             Cos = cos
#             Cot = cot
#             Csc = csc
#             Sin = sin
#             Power = (^)
# 
#             Rm1212 = Power(Sin(Q), 2) * Power(dQdf * dFdq - dFdf * dQdq, 2)
# 
#             Rm1212 /= sin(θ)^2
# 
#             SArray{Tuple{2,2,2,2}}(0, 0, 0, 0, 0, Rm1212, -Rm1212, 0, 0, -Rm1212, Rm1212, 0, 0, 0, 0, 0)
#         end for ij in CartesianIndices(g.values)
#     ],
#     lmax,
# )
# if !(norm(Rm.values - Rm₀.values, Inf) ≤ 10atol)
#     for a in 1:2, b in (a + 1):2, c in 1:2, d in (c + 1):2
#         if (a, b) ≤ (c, d)
#             println("Rm₀[$a,$b,$c,$d]: ", Rm₀.values[1, 1][a, b, c, d])
#             println("Rm[$a,$b,$c,$d]: ", Rm.values[1, 1][a, b, c, d])
#             println("|(Rm-Rm₀)[$a,$b,$c,$d]|∞: ", norm(map(x -> x[a, b, c, d], Rm.values - Rm₀.values), Inf))
#         end
#     end
#     println("10atol: ", 10atol)
# end
# @assert norm(Rm.values - Rm₀.values, Inf) ≤ 10atol

function make_Ricci(gu::Tensor{2}, Rm::Tensor{4})
    return Tensor{2}(
        [
            SMatrix{2,2}(-sum(gu[c, d] * Rm[c, a, b, d] for c in 1:2, d in 1:2) for a in 1:2, b in 1:2) for
            (gu, Rm) in zip(gu.values, Rm.values)
        ],
        gu.lmax,
    )
end

# Rc₀ = Tensor{2}([
#     begin
#         x = SVector{2}(ash_point_coord(ij, lmax))
#         θ, ϕ = x
#         X = x2X(x)
#         Q, F = X
#         J = Jac0(x)
# 
#         q, f = x
#         dQdq = J[1, 1]
#         dQdf = J[1, 2]
#         dFdq = J[2, 1]
#         dFdf = J[2, 2]
# 
#         Cos = cos
#         Cot = cot
#         Csc = csc
#         Sin = sin
#         Power = (^)
# 
#         Rcθθ = Power(Sin(Q), 2) * Power(dFdq, 2) + Power(dQdq, 2)
#         Rcθϕ = Power(Sin(Q), 2) * dFdf * dFdq + dQdf * dQdq
#         Rcϕϕ = Power(Sin(Q), 2) * Power(dFdf, 2) + Power(dQdf, 2)
# 
#         Rcθθ /= 1
#         Rcθϕ /= sin(θ)
#         Rcϕϕ /= sin(θ)^2
# 
#         SMatrix{2,2}(Rcθθ, Rcθϕ, Rcθϕ, Rcϕϕ)
#     end for ij in CartesianIndices(Rc.values)
# ], lmax)
# if !(norm(Rc.values - Rc₀.values, Inf) ≤ 100atol)
#     for a in 1:2, b in a:2
#         println("Rc₀[$a,$b]: ", Rc₀.values[1, 1][a, b])
#         println("Rc[$a,$b]: ", Rc.values[1, 1][a, b])
#         println("|(Rc-Rc₀)[$a,$b]|∞: ", norm(map(x -> x[a, b], Rc.values - Rc₀.values), Inf))
#     end
#     println("100atol: ", 100atol)
# end
# @assert norm(Rc.values - Rc₀.values, Inf) ≤ 100atol

function make_Rsc(gu::Tensor{2}, Rc::Tensor{2})
    return Tensor{0}([Scalar(sum(gu[a, b] * Rc[a, b] for a in 1:2, b in 1:2)) for (gu, Rc) in zip(gu.values, Rc.values)], gu.lmax)
end

function test_ricci(; lmax::Int=30, visualize::Bool=false)
    xform = identity

    g = make_g(xform, lmax)
    check_g(xform, g)

    # Project g_ab into our function space
    g̃ = SpinTensor(g)
    g̃ = filter_modes(g̃)
    g = Tensor(g̃)
    g = make_real(g)
    g = make_symmetric(g)
    check_g(xform, g)

    # gu^ab
    gu = make_gu(g)
    gu = make_real(gu)
    gu = make_symmetric(gu)

    # dg_abc = g_ab,c
    g̃ = SpinTensor(g)
    dg̃ = tensor_gradient(g̃)
    dg̃ = filter_modes(dg̃)
    dg = Tensor(dg̃)
    dg = make_real(dg)
    dg = make_symmetric12(dg)
    # TODO: Check dg

    # Γ^a_bc = g^ad (g_dc,b + g_bd,c - g_bc,d) / 2
    Γ = make_Γ(gu, dg)
    # TODO: Check Γ^a_bc

    Rm = make_Riemann(g, gu, Γ)
    Rm = make_real(Rm)
    Rm = make_riemann_symmetry(Rm)
    # TODO: Check Rm_abcd

    Rc = make_Ricci(gu, Rm)
    Rc = make_real(Rc)
    Rc = make_symmetric(Rc)
    # TODO: Check Rc_ab

    Rsc = make_Rsc(gu, Rc)
    Rsc = make_real(Rsc)

    if visualize
        plt = plot(; aspect_ratio=1, legend=false)
        heatmap!(
            plt,
            ash_phis(lmax),
            ash_thetas(lmax),
            real(transpose(map(x -> x[], ash_grid_as_phi_theta(Rsc.values))));
            colorbar=true,
            clims=(1.5, 2.5),
            title="Rsc",
        )
        display(plt)
    end

    return nothing
end

################################################################################
# Main program

function invent_metric(lmax::Int)
    sz = ash_grid_size(lmax)
    g = Tensor{2}([one(SMatrix{2,2,Complex{Float64}}) for ij in CartesianIndices(sz)], lmax)
    g̃ = SpinTensor(g)
    g̃.coeffs[1, 1][ash_mode_index(2, 2, 1, lmax)] += 1 # tensor mode
    g̃.coeffs[1, 2][ash_mode_index(0, 1, 0, lmax)] += 0 # scalar mode
    g = Tensor(g̃)
    g = make_real(g; check=false)
    g = make_symmetric(g; check=false)
    return g
end

function calc_ricci(g::Tensor{2})
    g = make_real(g)
    g = make_symmetric(g)

    # gu^ab
    gu = make_gu(g)
    gu = make_real(gu)
    gu = make_symmetric(gu)

    # dg_abc = g_ab,c
    g̃ = SpinTensor(g)
    dg̃ = tensor_gradient(g̃)
    dg̃ = filter_modes(dg̃)
    dg = Tensor(dg̃)
    dg = make_real(dg)
    dg = make_symmetric12(dg)

    # Γ^a_bc = g^ad (g_dc,b + g_bd,c - g_bc,d) / 2
    Γ = make_Γ(gu, dg)

    Rm = make_Riemann(g, gu, Γ)
    Rm = make_real(Rm)
    Rm = make_riemann_symmetry(Rm)

    Rc = make_Ricci(gu, Rm)
    Rc = make_real(Rc)
    Rc = make_symmetric(Rc)

    Rsc = make_Rsc(gu, Rc)
    Rsc = make_real(Rsc)

    return Rsc
end

function ricci_flow(g₀::Tensor{2})
    # In 2D we have
    #     R_ab = 1/2 R g_ab
    # because
    #     R := g^ab R_ab = g^ab 1/2 R g_ab = 1/2 δ^a_a R = R
    # We represent the metric with a conformal factor ϕ:
    #     g_ab = ϕ g₀_ab
    # Then we also have
    #     R = 1/ϕ R₀ - 2/ϕ Δ 1/2 log ϕ
    #       = 1/ϕ R₀ - 1/ϕ^2 Δ ϕ
    #       = 1/ϕ R₀ - 1/ϕ^2 (g₀^ab D_b D_a ϕ + g₀^ab Γ₀^c_ab D_c ϕ)
    # Normalized Ricci flow:
    #     ∂ₜg_ab = - 2 R_ab + (∫ R[g] / 4π) g_ab
    # in terms on an unphysical time t.
    # Stationary end state:
    #     -2 R_ab + (∫ R[g] / 4π) g_ab = 0
    # With conformal factor:
    #     -R g_ab + (∫ R[g] / 4π) g_ab = 0
    #     -(1/ϕ R₀ - 1/ϕ^2 (g₀^ab D_b D_a ϕ + g₀^ab Γ₀^c_ab D_c ϕ)) ϕ g₀_ab + (∫ R[g] / 4π) ϕ g₀_ab = 0
    #     - R₀ ϕ + g₀^ab D_b D_a ϕ + g₀^ab Γ₀^c_ab D_c ϕ + (∫ R[g] / 4π) ϕ^2 = 0
    # This is an elliptic problem, similar to the apparent horizon equation.
    # With a modified normalization:
    #     g₀^ab D_b D_a ϕ + g₀^ab Γ₀^c_ab D_c ϕ - R₀ ϕ + ∫f(ϕ) ϕ^2 = 0

    lmax = g₀.lmax
    nmodes = prod(Tuple(ash_nmodes(lmax)))

    g₀ = make_real(g₀)
    g₀ = make_symmetric(g₀)

    ϕ̃ = SpinTensor{0}(Scalar(zeros(Complex{Float64}, nmodes)), lmax)
    ϕ̃.coeffs[][ash_mode_index(0, 0, 0, lmax)] = sqrt(4π)
    # ϕ = Tensor(ϕ̃)
    # @assert ϕ.values[1, 1][] ≈ 1

    # function tensor2vector!(v::AbstractVector{T}, ϕ̃::SpinTensor{0,Complex{T}}) where {T}
    #     @assert length(v) == nmodes
    #     @assert length(ϕ̃.coeffs[]) == nmodes
    #     ϕ = Tensor(ϕ̃)
    #     ϕ = make_real(ϕ)
    #     idx = 0
    #     for l in 0:lmax, m in 0:l
    #         c = ϕ̃.coeffs[][ash_mode_index(0, l, m, lmax)]
    #         v[idx += 1] = real(c)
    #         if m > 0
    #             v[idx += 1] = imag(c)
    #         end
    #     end
    #     @assert idx == nmodes
    #     ϕ̃′ = vector2tensor(v)
    #     @assert ϕ̃′.coeffs[] ≈ ϕ̃.coeffs[]
    #     return v
    # end
    # tensor2vector(ϕ̃::SpinTensor{0,Complex{T}}) where {T} = tensor2vector!(zeros(T, nmodes), ϕ̃)
    # 
    # function vector2tensor!(ϕ̃::SpinTensor{0,Complex{T}}, v::AbstractVector{T}) where {T}
    #     @assert length(v) == nmodes
    #     @assert length(ϕ̃.coeffs[]) == nmodes
    #     idx = 0
    #     for l in 0:lmax, m in 0:l
    #         c = zero(Complex{T})
    #         c += v[idx += 1]
    #         if m > 0
    #             c += im * v[idx += 1]
    #         end
    #         ϕ̃.coeffs[][ash_mode_index(0, l, m, lmax)] = c
    #         if m > 0
    #             ϕ̃.coeffs[][ash_mode_index(0, l, -m, lmax)] = bitsign(m) * c'
    #         end
    #     end
    #     @assert idx == nmodes
    #     ϕ = Tensor(ϕ̃)
    #     ϕ = make_real(ϕ)
    #     return ϕ̃
    # end
    # vector2tensor(v::AbstractVector{T}) where {T} = vector2tensor!(SpinTensor{0}(zeros(Complex{T}, nmodes), lmax), v)
    # 
    # function prod!(res, v, α, β)
    #     if !(eltype(res) == Float64)
    #         @show eltype(res) eltype(v) typeof(res) typeof(v) typeof(α) typeof(β)
    #     end
    #     @assert eltype(res) == Float64
    #     @assert eltype(v) == Float64
    #     local ϕ̃ = vector2tensor(v)
    #     local ϕ = Tensor(ϕ̃)
    #     local g = Tensor{2}([SMatrix{2,2}(ϕ[] * g₀[a, b] for a in 1:2, b in 1:2) for (ϕ, g₀) in zip(ϕ.values, g₀.values)], lmax)
    #     local R = calc_ricci(g)
    #     R = make_real(R)
    #     local R̃ = SpinTensor(R)
    #     local avgR = real(R̃.coeffs[][ash_mode_index(0, 0, 0, lmax)]) / sqrt(4π)
    #     local ∂ₜϕ = Tensor{0}([Scalar(-R[] * ϕ[] + 0 * avgR) for (ϕ, R) in zip(ϕ.values, R.values)], lmax)
    #     local ∂ₜϕ̃ = SpinTensor(∂ₜϕ)
    #     ∂ₜϕ̃.coeffs[][ash_mode_index(0, 0, 0, lmax)] += 1
    #     if β == zero(β)
    #         res .= α * tensor2vector(∂ₜϕ̃)
    #     else
    #         res .= α * tensor2vector(∂ₜϕ̃) + β * res
    #     end
    #     return res
    # end
    # L = LinearOperator(Float64, nmodes, nmodes, false, false, prod!, nothing, nothing)
    # 
    # # ∂ₜϕ̃ = vector2tensor(L * tensor2vector(ϕ̃))
    # # ∂ₜϕ = Tensor(∂ₜϕ̃)
    # # @info chop.(real.(ϕ̃.coeffs[]))
    # # @info chop.(real.(∂ₜϕ̃.coeffs[]))
    # # ϕ = ∂ₜϕ
    # 
    # # rhs = SpinTensor{0}(Scalar(zeros(Complex{Float64}, nmodes)), lmax)
    # # rhs.coeffs[][ash_mode_index(0, 0, 0, lmax)] = sqrt(4π)
    # # 
    # # # L * ϕ = rhs
    # # res = zeros(Float64, nmodes)
    # # _, history = bicgstabl!(res, L, tensor2vector(rhs); log=true, max_mv_products=0)
    # # @info history
    # # ϕ̃ = vector2tensor(res)
    # # @info chop.(ϕ̃.coeffs[])
    # # ϕ = Tensor(ϕ̃)
    # # ϕ = make_real(ϕ)

    Δt = 0.25 / (lmax + 1)^2
    ΔR_max = 1.0e-8
    ΔR_old = Inf
    for iter in 1:10000
        # Reconstruct metric
        ϕ = Tensor(ϕ̃)
        ϕ = make_real(ϕ)
        g = Tensor{2}([SMatrix{2,2}(ϕ[] * g₀[a, b] for a in 1:2, b in 1:2) for (ϕ, g₀) in zip(ϕ.values, g₀.values)], lmax)
        # Calculate Ricci tensor
        R = calc_ricci(g)
        R = make_real(R)
        # Calculate average Ricci tensor
        vol_R = Tensor{0}([Scalar(sqrt(det(g)) * R[]) for (g, R) in zip(g.values, R.values)], lmax)
        vol_1 = Tensor{0}([Scalar(sqrt(det(g))) for g in g.values], lmax)
        vol_R̃ = SpinTensor(vol_R)
        vol_1̃ = SpinTensor(vol_1)
        R_avg = real(vol_R̃.coeffs[][ash_mode_index(0, 0, 0, lmax)]) / real(vol_1̃.coeffs[][ash_mode_index(0, 0, 0, lmax)])
        # Examine progress
        ϕ_norm = norm(abs.(ϕ̃.coeffs[]), Inf)
        R_min, R_max = extrema(real(map(x -> x[], R.values)))
        ΔR = R_max - R_min
        # Decide
        did_succeed = ΔR ≤ ΔR_max
        did_succeed && println("[succeeded]")
        did_fail = ΔR ≥ ΔR_old
        did_fail && println("[failed]")
        do_exit = did_succeed || did_fail
        # Output
        if iter % 10 == 0 || do_exit
            # println("$iter:   ϕ_norm: $ϕ_norm   ΔR: $ΔR   R_avg: $R_avg   R_min: $R_min   R_max: $R_max")
            println("$iter:   ϕ_norm: $ϕ_norm   ΔR: $ΔR   R_avg: $R_avg")
        end
        # Take the step
        if !did_fail
            # ∂ₜϕ = Tensor{0}([Scalar(-R[] * ϕ[] + R_avg * ϕ[]) for (ϕ, R) in zip(ϕ.values, R.values)], lmax)
            # ∂ₜϕ̃ = SpinTensor(∂ₜϕ)
            # ϕ̃ += Δt * ∂ₜϕ̃
            # Fast flow (Gundlach)
            α = 1
            β = 1 / 2
            A = α / (lmax * (lmax + 1)) + β
            B = β / α
            ρ = 1
            Δϕ = Tensor{0}([Scalar(ρ * (-R[] * ϕ[] + R_avg * ϕ[])) for (ϕ, R) in zip(ϕ.values, R.values)], lmax)
            Δϕ̃ = SpinTensor(Δϕ)
            for l in 0:lmax, m in (-l):l
                Δϕ̃.coeffs[][ash_mode_index(0, l, m, lmax)] *= A / (1 + B * l * (l + 1)) * ρ
            end
            ϕ̃ += Δϕ̃
            # Filter: Set topmost 4 modes to 0
            # (Filtering seems to help converge faster.)
            for l in max(0, lmax - 3):lmax, m in (-l):l
                ϕ̃.coeffs[][ash_mode_index(0, l, m, lmax)] = 0
            end
        end
        # Control
        do_exit && break
        # Iterate
        ΔR_old = ΔR
    end
    # @info chop.(ϕ̃.coeffs[])

    ϕ = Tensor(ϕ̃)
    ϕ = make_real(ϕ)

    return ϕ::Tensor{0}
end

function find_coordinates_flow(g₀::Tensor{2}, ϕ::Tensor{0})
    lmax = g₀.lmax
    @assert ϕ.lmax == lmax
    nmodes = ash_nmodes(lmax)

    g = Tensor{2}([SMatrix{2,2}(ϕ[] * g₀[a, b] for a in 1:2, b in 1:2) for (ϕ, g₀) in zip(ϕ.values, g₀.values)], lmax)
    g = make_real(g)
    g = make_symmetric(g)

    # gu^ab
    gu = make_gu(g)
    gu = make_real(gu)
    gu = make_symmetric(gu)

    # dg_abc = g_ab,c
    g̃ = SpinTensor(g)
    dg̃ = tensor_gradient(g̃)
    dg̃ = filter_modes(dg̃)
    dg = Tensor(dg̃)
    dg = make_real(dg)
    dg = make_symmetric12(dg)

    # Γ^a_bc = g^ad (g_dc,b + g_bd,c - g_bc,d) / 2
    Γ = make_Γ(gu, dg)

    # Coordinates are eigenfunctions of the Laplacian for l=1:
    # Definition of eigenmodes:
    #     Δx = -2x
    #     Δy = -2y
    #     Δz = -2z
    # Mode 0 is not present:
    #     ∫x = ∫y = ∫z = 0
    # The modes are normalized:
    #     x⋅x = y⋅y = z⋅z = 1
    # The modes are orthogonal:
    #     x⋅y = x⋅z = y⋅z = 0

    # In spectral space:
    #     -l(l+1)x - Γx = -2x
    #     (2 - l(l+1)) x = Γx

    # Define a flow method:
    #     ∂ₜx = Δx + 2x

    # Orthonormality:
    #     ∂ₜx = Δx + 2x - (x⋅y)y - (x⋅z)z

    c̃ = SpinTensor{0}(Scalar(zeros(Complex{Float64}, nmodes)), lmax)
    x̃ = SpinTensor{0}(Scalar(zeros(Complex{Float64}, nmodes)), lmax)
    ỹ = SpinTensor{0}(Scalar(zeros(Complex{Float64}, nmodes)), lmax)
    z̃ = SpinTensor{0}(Scalar(zeros(Complex{Float64}, nmodes)), lmax)
    c̃.coeffs[][ash_mode_index(0, 0, 0, lmax)] = sqrt(4π)
    x̃.coeffs[][ash_mode_index(0, 1, +1, lmax)] = -sqrt(2π)
    x̃.coeffs[][ash_mode_index(0, 1, -1, lmax)] = +sqrt(2π)
    ỹ.coeffs[][ash_mode_index(0, 1, +1, lmax)] = +sqrt(2π) * im
    ỹ.coeffs[][ash_mode_index(0, 1, -1, lmax)] = -sqrt(2π) * im'
    z̃.coeffs[][ash_mode_index(0, 1, 0, lmax)] = -sqrt(4π)

    Δ_old = Inf
    for iter in 1:10000
        c = Tensor(c̃)
        x = Tensor(x̃)
        y = Tensor(ỹ)
        z = Tensor(z̃)
        c = make_real(c)
        x = make_real(x)
        y = make_real(y)
        z = make_real(z)
        V = Tensor{0}([Scalar(sqrt(det(g))) for g in g.values], lmax)
        cc = Tensor{0}([Scalar(sqrt(det(g)) * c[]^2) for (g, c) in zip(g.values, c.values)], lmax)
        xx = Tensor{0}([Scalar(sqrt(det(g)) * x[]^2) for (g, x) in zip(g.values, x.values)], lmax)
        yy = Tensor{0}([Scalar(sqrt(det(g)) * y[]^2) for (g, y) in zip(g.values, y.values)], lmax)
        zz = Tensor{0}([Scalar(sqrt(det(g)) * z[]^2) for (g, z) in zip(g.values, z.values)], lmax)
        cx = Tensor{0}([Scalar(sqrt(det(g)) * c[] * x[]) for (g, c, x) in zip(g.values, c.values, x.values)], lmax)
        cy = Tensor{0}([Scalar(sqrt(det(g)) * c[] * y[]) for (g, c, y) in zip(g.values, c.values, y.values)], lmax)
        cz = Tensor{0}([Scalar(sqrt(det(g)) * c[] * z[]) for (g, c, z) in zip(g.values, c.values, z.values)], lmax)
        xy = Tensor{0}([Scalar(sqrt(det(g)) * x[] * y[]) for (g, x, y) in zip(g.values, x.values, y.values)], lmax)
        xz = Tensor{0}([Scalar(sqrt(det(g)) * x[] * z[]) for (g, x, z) in zip(g.values, x.values, z.values)], lmax)
        yz = Tensor{0}([Scalar(sqrt(det(g)) * y[] * z[]) for (g, y, z) in zip(g.values, y.values, z.values)], lmax)
        Ṽ = SpinTensor(V)
        int(f::Tensor{0}) = real(SpinTensor(f).coeffs[][ash_mode_index(0, 0, 0, lmax)] / Ṽ.coeffs[][ash_mode_index(0, 0, 0, lmax)])
        int_c = int(c)
        int_x = int(x)
        int_y = int(y)
        int_z = int(z)
        int_cc = int(cc)
        int_xx = int(xx)
        int_yy = int(yy)
        int_zz = int(zz)
        int_cx = int(cx)
        int_cy = int(cy)
        int_cz = int(cz)
        int_xy = int(xy)
        int_xz = int(xz)
        int_yz = int(yz)
        # These hold only for a manifestly round metric
        # @assert int_c ≈ 1
        # @assert int_x + 1 ≈ 1
        # @assert int_y + 1 ≈ 1
        # @assert int_z + 1 ≈ 1
        # @assert int_cc ≈ 1
        # @assert int_xx ≈ 1
        # @assert int_yy ≈ 1
        # @assert int_zz ≈ 1
        # @assert int_cx + 1 ≈ 1
        # @assert int_cy + 1 ≈ 1
        # @assert int_cz + 1 ≈ 1
        # @assert int_xy + 1 ≈ 1
        # @assert int_xz + 1 ≈ 1
        # @assert int_yz + 1 ≈ 1

        # Orthonormalize via Gram-Schmidt
        error("TODO: do this properly, otherwise unstable")
        x̃ -= int_cx * c̃

        dc̃ = tensor_gradient(c̃)
        dx̃ = tensor_gradient(x̃)
        dỹ = tensor_gradient(ỹ)
        dz̃ = tensor_gradient(z̃)
        ddc̃ = tensor_gradient(dc̃)
        ddx̃ = tensor_gradient(dx̃)
        ddỹ = tensor_gradient(dỹ)
        ddz̃ = tensor_gradient(dz̃)
        dc = Tensor(dc̃)
        dx = Tensor(dx̃)
        dy = Tensor(dỹ)
        dz = Tensor(dz̃)
        ddc = Tensor(ddc̃)
        ddx = Tensor(ddx̃)
        ddy = Tensor(ddỹ)
        ddz = Tensor(ddz̃)
        Lc = Tensor{0}(
            [
                Scalar(sum(gu[a, b] * (ddc[a, b] - sum(Γ[c, a, b] * dc[c] for c in 1:2)) for a in 1:2, b in 1:2)) for
                (gu, Γ, dc, ddc) in zip(gu.values, Γ.values, dc.values, ddc.values)
            ],
            lmax,
        )
        Lx = Tensor{0}(
            [
                Scalar(sum(gu[a, b] * (ddx[a, b] - sum(Γ[c, a, b] * dx[c] for c in 1:2)) for a in 1:2, b in 1:2)) for
                (gu, Γ, dx, ddx) in zip(gu.values, Γ.values, dx.values, ddx.values)
            ],
            lmax,
        )
        Ly = Tensor{0}(
            [
                Scalar(sum(gu[a, b] * (ddy[a, b] - sum(Γ[c, a, b] * dy[c] for c in 1:2)) for a in 1:2, b in 1:2)) for
                (gu, Γ, dy, ddy) in zip(gu.values, Γ.values, dy.values, ddy.values)
            ],
            lmax,
        )
        Lz = Tensor{0}(
            [
                Scalar(sum(gu[a, b] * (ddz[a, b] - sum(Γ[c, a, b] * dz[c] for c in 1:2)) for a in 1:2, b in 1:2)) for
                (gu, Γ, dz, ddz) in zip(gu.values, Γ.values, dz.values, ddz.values)
            ],
            lmax,
        )
        Lc̃ = SpinTensor(Lc)
        Lx̃ = SpinTensor(Lx)
        Lỹ = SpinTensor(Ly)
        Lz̃ = SpinTensor(Lz)

        ∂ₜc = Tensor{0}([Scalar(Lc[] - 0 * (int_c - 1)) for (Lc, c) in zip(Lc.values, c.values)], lmax)
        ∂ₜx = Tensor{0}(
            [
                Scalar(Lx[] + 2 * x[] - 0 * int_cx * c[] - 0 * (int_xx - 1) * x[] - 0 * int_xy * y[] - 0 * int_xz * z[]) for
                (Lx, c, x, y, z) in zip(Lx.values, c.values, x.values, y.values, z.values)
            ],
            lmax,
        )
        ∂ₜy = Tensor{0}(
            [
                Scalar(Ly[] + 2 * y[] - int_xy * x[] - (int_yy - 1) * y[] - int_yz * z[]) for
                (Ly, x, y, z) in zip(Ly.values, x.values, y.values, z.values)
            ],
            lmax,
        )
        ∂ₜz = Tensor{0}(
            [
                Scalar(Lz[] + 2 * z[] - int_xz * x[] - int_yz * y[] - (int_zz - 1) * z[]) for
                (Lz, x, y, z) in zip(Lz.values, x.values, y.values, z.values)
            ],
            lmax,
        )

        ∂ₜc̃ = SpinTensor(∂ₜc)
        ∂ₜx̃ = SpinTensor(∂ₜx)
        ∂ₜỹ = SpinTensor(∂ₜy)
        ∂ₜz̃ = SpinTensor(∂ₜz)

        Δ = norm(∂ₜc̃.coeffs[], Inf) + norm(∂ₜx̃.coeffs[], Inf) # + norm(∂ₜỹ.coeffs[], Inf) + norm(∂ₜz̃.coeffs[], Inf)
        did_fail = Δ ≥ Δ_old
        do_exit = did_fail && iter ≥ 10
        if iter % 10 == 0 || do_exit
            # println("$iter:   Δ: $Δ   |x|=$(norm(x̃.coeffs[], Inf))   int_xx=$int_xx   int_xy=$int_xy   int_xz=$int_xz")
            # println(
            #     "$iter:   Δ: $Δ   |x|=$(norm(x̃.coeffs[], Inf))   int_x=$int_x   int_xx=$int_xx   x̃[0]=$(chop(real(x̃.coeffs[][ash_mode_index(0,0,0,lmax)])))",
            # )
            println("$iter:   Δ: $Δ")
            println("   |c|=$(norm(c̃.coeffs[], Inf))   |Lc|=$(norm(Lc̃.coeffs[], Inf))   int_c=$int_c")
            println("   |x|=$(norm(x̃.coeffs[], Inf))   |Lx|=$(norm(Lx̃.coeffs[], Inf))   int_x=$int_x   int_cx=$int_cx")
        end

        if true || !did_fail
            Δt = 0.5 / (lmax + 1)^2
            # @show chop.(x̃.coeffs[])
            # @show chop.(∂ₜx̃.coeffs[])
            c̃ += Δt * ∂ₜc̃
            x̃ += Δt * ∂ₜx̃
            #TODO ỹ += Δt * ∂ₜỹ
            #TODO z̃ += Δt * ∂ₜz̃
        end

        do_exit && break

        c = Tensor(c̃)
        x = Tensor(x̃)
        c = make_real(c)
        x = make_real(x)
        c̃ = SpinTensor(c)
        x̃ = SpinTensor(x)

        Δ_old = Δ
    end

    c = Tensor(c̃)
    x = Tensor(x̃)
    y = Tensor(ỹ)
    z = Tensor(z̃)
    c = make_real(c)
    x = make_real(x)
    y = make_real(y)
    z = make_real(z)

    return c, x, y, z
end

function find_coordinates(g₀::Tensor{2}, ϕ::Tensor{0})
    lmax = g₀.lmax
    @assert ϕ.lmax == lmax
    nmodes = prod(ash_nmodes(lmax))

    g = Tensor{2}([SMatrix{2,2}(ϕ[] * g₀[a, b] for a in 1:2, b in 1:2) for (ϕ, g₀) in zip(ϕ.values, g₀.values)], lmax)
    g = make_real(g)
    g = make_symmetric(g)

    # gu^ab
    gu = make_gu(g)
    gu = make_real(gu)
    gu = make_symmetric(gu)

    # dg_abc = g_ab,c
    g̃ = SpinTensor(g)
    dg̃ = tensor_gradient(g̃)
    dg̃ = filter_modes(dg̃)
    dg = Tensor(dg̃)
    dg = make_real(dg)
    dg = make_symmetric12(dg)

    # Γ^a_bc = g^ad (g_dc,b + g_bd,c - g_bc,d) / 2
    Γ = make_Γ(gu, dg)

    # Coordinates are eigenfunctions of the Laplacian for l=1:
    # Definition of eigenmodes:
    #     Δx = -2x
    #     Δy = -2y
    #     Δz = -2z

    # function tensor2vector!(v::AbstractVector{T}, t̃::SpinTensor{0,Complex{T}}) where {T}
    #     @assert length(v) == nmodes
    #     @assert length(t̃.coeffs[]) == nmodes
    #     t = Tensor(t̃)
    #     t = make_real(t)
    #     idx = 0
    #     for l in 0:lmax, m in 0:l
    #         c = t̃.coeffs[][ash_mode_index(0, l, m, lmax)]
    #         v[idx += 1] = real(c)
    #         if m > 0
    #             v[idx += 1] = imag(c)
    #         end
    #     end
    #     @assert idx == nmodes
    #     return v
    # end
    # tensor2vector(t̃::SpinTensor{0,Complex{T}}) where {T} = tensor2vector!(zeros(T, nmodes), t̃)
    # 
    # function vector2tensor!(t̃::SpinTensor{0,Complex{T}}, v::AbstractVector{T}) where {T}
    #     @assert length(v) == nmodes
    #     @assert length(t̃.coeffs[]) == nmodes
    #     idx = 0
    #     for l in 0:lmax, m in 0:l
    #         c = zero(Complex{T})
    #         c += v[idx += 1]
    #         if m > 0
    #             c += im * v[idx += 1]
    #         end
    #         t̃.coeffs[][ash_mode_index(0, l, m, lmax)] = c
    #         if m > 0
    #             t̃.coeffs[][ash_mode_index(0, l, -m, lmax)] = bitsign(m) * c'
    #         end
    #     end
    #     @assert idx == nmodes
    #     return t̃
    # end
    # vector2tensor(v::AbstractVector{T}) where {T} = vector2tensor!(SpinTensor{0}(zeros(Complex{T}, nmodes), lmax), v)

    tensor2vector!(v::AbstractVector, t̃::SpinTensor{0}) = (v .= t̃.coeffs[]; v)
    tensor2vector(t̃::SpinTensor{0,T}) where {T} = tensor2vector!(zeros(T, nmodes), t̃)

    vector2tensor!(t̃::SpinTensor{0}, v::AbstractVector) = (t̃.coeffs[] .= v; t̃)
    vector2tensor(v::AbstractVector{T}) where {T} = vector2tensor!(SpinTensor{0}(zeros(T, nmodes), lmax), v)

    function prod!(res, v, α, β)
        # @assert eltype(res) == Float64
        # @assert eltype(v) == Float64
        local t̃ = vector2tensor(v)
        local dt̃ = tensor_gradient(t̃)
        local ddt̃ = tensor_gradient(dt̃)
        local t = Tensor(t̃)
        local dt = Tensor(dt̃)
        local ddt = Tensor(ddt̃)
        local Lt = Tensor{0}(
            [
                Scalar(sum(gu[a, b] * (ddt[a, b] - sum(Γ[c, a, b] * dt[c] for c in 1:2)) for a in 1:2, b in 1:2)) for
                (gu, Γ, dt, ddt) in zip(gu.values, Γ.values, dt.values, ddt.values)
            ],
            lmax,
        )
        Lt̃ = SpinTensor(Lt)
        if β == zero(β)
            res .= α * tensor2vector(Lt̃)
        else
            res .= α * tensor2vector(Lt̃̃) + β * res
        end
        return res
    end

    println("Calling eigs...")
    # L = LinearOperator(Float64, nmodes, nmodes, false, false, prod!, nothing, nothing)
    # λ, v, nconv, niter, nmult, resid = eigs(L; nev=9, ritzvec=true, tol=eps()^(7/8), v0=ones(nmodes), which=:LR)
    L = LinearOperator(Complex{Float64}, nmodes, nmodes, false, false, prod!, nothing, nothing)
    λ, v, nconv, niter, nmult, resid = eigs(L; nev=9, ritzvec=true, tol=eps()^(7 / 8), v0=ones(Complex{Float64}, nmodes), which=:LR)
    println("    nconv=$nconv   niter=$niter   nmult=$nmult   |resid|=$(norm(resid))")
    for (i, λ) in enumerate(λ)
        println("    λ[$i]=$(chop(λ))   v[1,$i]=$(chop(v[1,i]))")
    end
    # @show resid
    # @show v[1,:]

    @assert isapprox(λ[1], 0; atol=sqrt(sqrt(eps())))
    @assert isapprox(λ[2], -2; atol=sqrt(sqrt(eps())))
    @assert isapprox(λ[3], -2; atol=sqrt(sqrt(eps())))
    @assert isapprox(λ[4], -2; atol=sqrt(sqrt(eps())))

    x̃ = SpinTensor{0}(Scalar(@view v[:, 2]), lmax)
    ỹ = SpinTensor{0}(Scalar(@view v[:, 3]), lmax)
    z̃ = SpinTensor{0}(Scalar(@view v[:, 4]), lmax)
    x = Tensor(x̃)
    y = Tensor(ỹ)
    z = Tensor(z̃)
    # @show norm(imag.(map(x -> x[], x.values)))
    # @show norm(imag.(map(x -> x[], y.values)))
    # @show norm(imag.(map(x -> x[], z.values)))
    x = make_real(x)
    y = make_real(y)
    z = make_real(z)

    # Orthonormalize
    V = Tensor{0}([Scalar(sqrt(det(g))) for g in g.values], lmax)
    Ṽ = SpinTensor(V)
    int(f::Tensor{0}) = real(SpinTensor(f).coeffs[][ash_mode_index(0, 0, 0, lmax)] / Ṽ.coeffs[][ash_mode_index(0, 0, 0, lmax)])
    function dot(x::Tensor{0}, y::Tensor{0})
        return int(Tensor{0}([Scalar(sqrt(det(g)) * x[] * y[]) for (g, x, y) in zip(g.values, x.values, y.values)], lmax))
    end

    x.values ./= sqrt(dot(x, x))

    y.values ./= sqrt(dot(y, y))
    y.values .-= dot(y, x) * x.values
    y.values ./= sqrt(dot(y, y))

    z.values ./= sqrt(dot(z, z))
    z.values .-= dot(z, x) * x.values
    z.values ./= sqrt(dot(z, z))
    z.values .-= dot(z, y) * y.values
    z.values ./= sqrt(dot(z, z))

    println("Eigenvalue accuracy:")
    x̃ = SpinTensor(x)
    ỹ = SpinTensor(y)
    z̃ = SpinTensor(z)
    println("    x: $(norm(L*tensor2vector(x̃)+2*tensor2vector(x̃)))")
    println("    y: $(norm(L*tensor2vector(ỹ)+2*tensor2vector(ỹ)))")
    println("    z: $(norm(L*tensor2vector(z̃)+2*tensor2vector(z̃)))")

    return x, y, z
end

function kerr_schild_omega(θ, ϕ)
    # Cook, sec. 3.3.1
    M = 1
    a = 0.5
    Q = 0
    r_EH = M + sqrt(M^2 - a^2 - Q^2)
    # @show M a Q r_EH
    x = SVector(0, r_EH, θ, ϕ)
    g, l, k = null_normals(M, a, Q, x)
    gu = inv(g)
    # Christoffel symbols
    dg = SArray{Tuple{4,4,4}}(ForwardDiff.jacobian(x -> null_normals(M, a, Q, x)[1], x)...)
    Γ = SArray{Tuple{4,4,4}}(
        sum(gu[a, d] * (dg[d, c, b] + dg[b, d, c] - dg[b, c, d]) / 2 for d in 1:4) for a in 1:4, b in 1:4, c in 1:4
    )
    # Gradients of null normals
    dl = SMatrix{4,4}(ForwardDiff.jacobian(x -> null_normals(M, a, Q, x)[2], x)...)
    Dl = SMatrix{4,4}(dl[a, b] - sum(Γ[c, a, b] * l[c] for c in 1:4) for a in 1:4, b in 1:4)

    # 2-metric on horizon
    q = SMatrix{4,4}(g[a, b] + l[a] * k[b] + k[a] * l[b] for a in 1:4, b in 1:4)
    # @show chop(norm(q' - q, Inf))
    q = (q + q') / 2
    # @show chop.(eigvals(q))

    # rotation one-form ω
    ω = SVector{4}(-sum(q[a, b] * gu[b, c] * Dl[c, d] * gu[d, e] * k[e] for b in 1:4, c in 1:4, d in 1:4, e in 1:4) for a in 1:4)
    # @show chop.(ω)
    ω2 = ω' * gu * ω
    # @show chop(ω2)

    return q, ω
end

################################################################################

function main(; lmax::Int=30, visualize::Bool=false)
    # Set up the metric
    # g = make_g(identity, lmax)
    # g = make_g(xform_3d, lmax)
    g = invent_metric(lmax)
    Rsc = calc_ricci(g)
    println("Rsc:   min: ", minimum(real(map(x -> x[], Rsc.values))), "   max: ", maximum(real(map(x -> x[], Rsc.values))))

    # Find conformal transformation
    ϕ = ricci_flow(g)
    ϕ_min, ϕ_max = extrema(real(map(x -> x[], ϕ.values)))
    println("ϕ:   min: ", ϕ_min, "   max: ", ϕ_max)
    g′ = Tensor{2}([SMatrix{2,2}(ϕ[] * g[a, b] for a in 1:2, b in 1:2) for (ϕ, g) in zip(ϕ.values, g.values)], lmax)
    Rsc′ = calc_ricci(g′)
    println("Rsc′:   min: ", minimum(real(map(x -> x[], Rsc′.values))), "   max: ", maximum(real(map(x -> x[], Rsc′.values))))

    # Rescale sphere
    # Why doesn't the Ricci flow do this automatically?
    avg_Rsc′ = (minimum(real(map(x -> x[], Rsc′.values))) + maximum(real(map(x -> x[], Rsc′.values)))) / 2
    ϕ.values .*= avg_Rsc′ / 2
    g′ = Tensor{2}([SMatrix{2,2}(ϕ[] * g[a, b] for a in 1:2, b in 1:2) for (ϕ, g) in zip(ϕ.values, g.values)], lmax)
    Rsc′ = calc_ricci(g′)
    println("Rsc′:   min: ", minimum(real(map(x -> x[], Rsc′.values))), "   max: ", maximum(real(map(x -> x[], Rsc′.values))))

    # Find good coordinates
    x, y, z = find_coordinates(g, ϕ)
    xs = [x, y, z]

    # Calculate vector fields ϕᵢ and ξᵢ
    delta = SMatrix{2,2}(i == j for i in 1:2, j in 1:2)
    epsilon = SMatrix{2,2}(if (i, j) == (1, 2)
        +1
    elseif (i, j) == (2, 1)
        -1
    else
        0
    end for i in 1:2, j in 1:2)
    delta3 = SMatrix{3,3}(i == j for i in 1:3, j in 1:3)
    epsilon3 = SArray{Tuple{3,3,3}}(if (i, j, k) ∈ ((1, 2, 3), (2, 3, 1), (3, 1, 2))
        +1
    elseif (i, j, k) ∈ ((1, 3, 2), (2, 1, 3), (3, 2, 1))
        -1
    else
        0
    end for i in 1:3, j in 1:3, k in 1:3)
    # ϕᵢ = curl xᵢ
    x̃ = SpinTensor(x)
    ỹ = SpinTensor(y)
    z̃ = SpinTensor(z)
    dx̃ = tensor_gradient(x̃)
    dỹ = tensor_gradient(ỹ)
    dz̃ = tensor_gradient(z̃)
    dx = Tensor(dx̃)
    dy = Tensor(dỹ)
    dz = Tensor(dz̃)
    ϕx = Tensor{1}([SVector{2}(sum(epsilon[a, b] * dx[b] for b in 1:2) for a in 1:2) for dx in dx.values], lmax)
    ϕy = Tensor{1}([SVector{2}(sum(epsilon[a, b] * dy[b] for b in 1:2) for a in 1:2) for dy in dy.values], lmax)
    ϕz = Tensor{1}([SVector{2}(sum(epsilon[a, b] * dz[b] for b in 1:2) for a in 1:2) for dz in dz.values], lmax)
    # ξᵢ = grad xᵢ
    ξx = dx
    ξy = dy
    ξz = dz

    # Set up rotation one-form ω
    sz = ash_grid_size(lmax)
    ω = Tensor{1}([
        begin
            θ, ϕ = ash_point_coord(ij, lmax)
            q, ω = kerr_schild_omega(θ, ϕ)
            SVector(ω[3], ω[4])
        end for ij in CartesianIndices(sz)
    ], lmax)

    # Calculate spin integrals

    if visualize
        plt = plot(; aspect_ratio=1, legend=false)

        # heatmap!(
        #     plt,
        #     ash_phis(lmax),
        #     ash_thetas(lmax),
        #     real(transpose(map(x -> x[], ash_grid_as_phi_theta(Rsc.values))));
        #     colorbar=true,
        #     clims=(1.5, 2.5),
        #     title="Rsc",
        # )

        # heatmap!(
        #     plt,
        #     ash_phis(lmax),
        #     ash_thetas(lmax),
        #     real(transpose(map(x -> x[], ash_grid_as_phi_theta(ϕ.values))));
        #     colorbar=true,
        #     clims=(0.9, 1.1),
        #     title="ϕ",
        # )

        heatmap!(
            plt,
            ash_phis(lmax),
            ash_thetas(lmax),
            real(transpose(map(x -> x[], ash_grid_as_phi_theta(x.values))));
            colorbar=true,
            clims=(-1.1, +1.1),
            title="x",
        )
        heatmap!(
            plt,
            ash_phis(lmax),
            ash_thetas(lmax),
            real(transpose(map(x -> x[], ash_grid_as_phi_theta(y.values))));
            colorbar=true,
            clims=(-1.1, +1.1),
            title="y",
        )
        heatmap!(
            plt,
            ash_phis(lmax),
            ash_thetas(lmax),
            real(transpose(map(x -> x[], ash_grid_as_phi_theta(z.values))));
            colorbar=true,
            clims=(-1.1, +1.1),
            title="z",
        )

        display(plt)
    end

    return nothing
end

end
