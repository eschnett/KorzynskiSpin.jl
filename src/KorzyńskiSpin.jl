module KorzyńskiSpin

using AbstractSphericalHarmonics
using ForwardDiff
using LinearAlgebra
using StaticArrays

################################################################################
# General helpers

bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))

flatten(xss::SVector{N,<:SVector{M}}) where {N,M} = SVector{N * M}(xss[n][m] for m in 1:M, n in 1:N)

Base.chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
Base.chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))
Base.chop(x::SArray) = chop.(x)

################################################################################
# Accuracy

# The discretization error decreases exponentially.
# The floating-point errors increase linearly.
atol(lmax) = 1.0e+6 * max(exp(-lmax), lmax * eps())

################################################################################
# Tensor helpers

function impose_symmetry(f, t::Tensor{D}) where {D}
    @assert norm(t.values, Inf) < Inf
    @assert norm(t.values - f.(t.values), Inf) ≤ atol(t.lmax)
    return Tensor{D}(f.(t.values), t.lmax)
end

make_real(t::Tensor) = impose_symmetry(x -> Complex.(real.(x)), t)
make_symmetric(t::Tensor{2}) = impose_symmetry(x -> (x + x') / 2, t)
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

################################################################################
# Main program

function main(lmax::Int)
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

    println("Rsc:   min: ", minimum(real(map(x -> x[], Rsc.values))), "   max: ", maximum(real(map(x -> x[], Rsc.values))))

    return nothing
end

end
