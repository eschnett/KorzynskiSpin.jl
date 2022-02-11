using AbstractSphericalHarmonics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Test

# Global settings

lmax = 30

nmodes = ash_nmodes(lmax)
sz = ash_grid_size(lmax)

# lmax=10   atol=1e-4
# lmax=20   atol=1e-10
# lmax=30   atol=1e-15

# lmax=40   atol=1e-12
# lmax=100   atol=1e-11
# lmax=200   atol=1e-10
# lmax=300   atol=1e-10
# lmax=400   atol=1e-10

# The discretization error decreases exponentially.
# The floating-point errors increase linearly.
atol = 1.0e+6 * max(exp(-lmax), lmax * eps())

# Helper functions

bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))

flatten(xss::SVector{N,<:SVector{M}}) where {N,M} = SVector{N * M}(xss[n][m] for m in 1:M, n in 1:N)

Base.chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
Base.chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))
Base.chop(x::SArray) = chop.(x)

function impose_symmetry(f, t::Tensor{D}) where {D}
    @assert norm(t.values, Inf) < Inf
    @assert norm(t.values - f.(t.values), Inf) ≤ atol
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

# Computational ("bad") coordinates:
#     x = (θ, ϕ)
# Ideal ("good") coordinates:
#     X = (Q, F)

# Coordinate transformation:

if false
    α1 = 0.1
    α2 = 0                      # cannot be nonzero
    α3 = 0.1
    β1 = 0                      # cannot be nonzero
    β2 = 0.1
    β3 = 0                      # cannot be nonzero
    "X^a(x^b)"
    function x2X(x::SVector{2})
        θ, ϕ = x
        Q = θ + α1 * sin(θ) + α2 * sin(θ) * sin(ϕ) + α3 * sin(θ)^2 * sin(ϕ)
        F = ϕ + β1 * sin(θ) + β2 * sin(θ)^2 + β3 * sin(θ)^2 * sin(ϕ)
        X = SVector{2}(Q, F)
        return X
    end

else
    dx = 0.1
    dy = 0.1
    dz = 0.1
    "X^a(x^b)"
    function x2X(θϕ::SVector{2})
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
end

"J^a_b = ∂X^a/∂x_b(x^c)   [unscaled]"
Jac0(x::SVector{2}) = ForwardDiff.jacobian(x2X, x)

"dJ^a_bc = ∂²X^a/∂x_b∂x_c(x^d)   [unscaled]"
dJac0(x::SVector{2}) = reshape(ForwardDiff.jacobian(Jac0, x), (2, 2, 2))

"ddJ^a_bcd = ∂³X^a/∂x_b∂x_c∂x_d(x^e)   [unscaled]"
ddJac0(x::SVector{2}) = reshape(ForwardDiff.jacobian(dJac0, x), (2, 2, 2, 2))

"J^a_b = ∂X^a/∂x_b(x^c)"
function Jac(x::SVector{2})
    θ, ϕ = x
    Q, F = x2X(x)
    F = SVector(1, sin(Q))
    f = SVector(1, 1 / sin(θ))
    J0 = Jac0(x)
    J = SMatrix{2,2}(J0[a, b] * F[a] * f[b] for a in 1:2, b in 1:2)
    return J
end

"dJ^a_bc = ∂²X^a/∂x_b∂x_c(x^d)"
function dJac(x::SVector{2})
    θ, ϕ = x
    Q, F = x2X(x)
    F = SVector(1, sin(Q))
    f = SVector(1, 1 / sin(θ))
    dJ0 = dJac0(x)
    dJ = SArray{Tuple{2,2,2}}(dJ0[a, b, c] * F[a] * f[b] * f[c] for a in 1:2, b in 1:2, c in 1:2)
    return dJ
end

"ddJ^a_bcd = ∂³X^a/∂x_b∂x_c∂x_d(x^e)"
function ddJac(x::SVector{2})
    θ, ϕ = x
    Q, F = x2X(x)
    F = SVector(1, sin(Q))
    f = SVector(1, 1 / sin(θ))
    ddJ0 = ddJac0(x)
    ddJ = SArray{Tuple{2,2,2,2}}(ddJ0[a, b, c, d] * F[a] * f[b] * f[c] * f[d] for a in 1:2, b in 1:2, c in 1:2, d in 1:2)
    return ddJ
end

# Round metric:

# q_ab = (1, 0, 0, 1)
make_q(x::SVector{2}) = SMatrix{2,2,Complex{Float64}}(1, 0, 0, 1)
q = Tensor{2}([SMatrix{2,2,Complex{Float64}}(1, 0, 0, 1) for ij in CartesianIndices(sz)], lmax)
q̃ = SpinTensor(q)

# Test q_ab,c = 0
dq̃ = tensor_gradient(q̃)
@assert norm(dq̃.coeffs, Inf) ≤ atol

# Physical metric:

# g_ab = J_a^c J_b^d q_cd
function make_g(ij::CartesianIndex{2})
    x = SVector{2}(ash_point_coord(ij, lmax))
    X = x2X(x)
    q = make_q(X)
    J = Jac(x)
    g = SMatrix{2,2}(sum(q[c, d] * J[c, a] * J[d, b] for c in 1:2, d in 1:2) for a in 1:2, b in 1:2)
    return g
end

g = Tensor{2}([make_g(ij) for ij in CartesianIndices(q.values)], lmax)

g₀ = Tensor{2}([
    begin
        x = SVector{2}(ash_point_coord(ij, lmax))
        θ, ϕ = x
        X = x2X(x)
        Q, F = X
        J = Jac0(x)

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
if !(norm(g.values - g₀.values, Inf) ≤ atol)
    for a in 1:2, b in a:2
        println("g₀[$a,$b]: ", g₀.values[1, 1][a, b])
        println("g[$a,$b]: ", g.values[1, 1][a, b])
        println("|(g-g₀)[$a,$b]|∞: ", norm(map(x -> x[a, b], g.values - g₀.values), Inf))
    end
    println("atol: ", atol)
end
@assert norm(g.values - g₀.values, Inf) ≤ atol

g̃ = SpinTensor(g)

# g̃.coeffs[1, 1][5] += 0   # 6
# g̃.coeffs[1, 1][6] += 0   # 6
# g̃.coeffs[1, 1][7] += 0   # 4
# g̃.coeffs[1, 2][2] += 0   # 2   # x
# g̃.coeffs[1, 2][3] += 0   # 2   # z
# 
# # This does not work???
# # # Ensure q is symmetric
# # g̃.coeffs[1, 2] .= real(g̃.coeffs[1, 2])
# # 
# # # Ensure q is real
# # g̃.coeffs[2, 1] .= conj(g̃.coeffs[1, 2])
# # g̃.coeffs[2, 2] .= conj(g̃.coeffs[1, 1])

# Project g into our function space
g̃ = filter_modes(g̃)
g = Tensor(g̃)
g = make_real(g)
g = make_symmetric(g)
if !(norm(g.values - g₀.values, Inf) ≤ atol)
    for a in 1:2, b in a:2
        println("g₀[$a,$b]: ", g₀.values[1, 1][a, b])
        println("g[$a,$b]: ", g.values[1, 1][a, b])
        println("|(g-g₀)[$a,$b]|∞: ", norm(map(x -> x[a, b], g.values - g₀.values), Inf))
    end
    println("atol: ", atol)
end
@assert norm(g.values - g₀.values, Inf) ≤ atol

# Calculate curvature

# gu^ab
gu = Tensor{2}([inv(g) for g in g.values], lmax)
gu = make_real(gu)
gu = make_symmetric(gu)

# g_ab,c
g̃ = SpinTensor(g)
dg̃ = tensor_gradient(g̃)
dg̃ = filter_modes(dg̃)
dg = Tensor(dg̃)
dg = make_real(dg)
dg = make_symmetric12(dg)

dg₀ = Tensor{3}(
    [
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            θ, ϕ = x
            X = x2X(x)
            Q, F = X
            J = Jac0(x)
            dJ = dJac0(x)

            q, f = x
            dQdq = J[1, 1]
            dQdf = J[1, 2]
            dFdq = J[2, 1]
            dFdf = J[2, 2]
            ddQdqdq = dJ[1, 1, 1]
            ddQdqdf = dJ[1, 1, 2]
            ddQdfdf = dJ[1, 2, 2]
            ddFdqdq = dJ[2, 1, 1]
            ddFdqdf = dJ[2, 1, 2]
            ddFdfdf = dJ[2, 2, 2]

            Cos = cos
            Cot = cot
            Csc = csc
            Sin = sin
            Power = (^)

            dgθθθ = 2 * (Cos(Q) * Sin(Q) * Power(dFdq, 2) * dQdq + Power(Sin(Q), 2) * dFdq * ddFdqdq + dQdq * ddQdqdq)
            dgθθϕ =
                2 * (
                    -(Cot(q) * Power(Sin(Q), 2) * dFdf * dFdq) +
                    dQdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - Cot(q) * dQdq) +
                    Power(Sin(Q), 2) * dFdq * ddFdqdf +
                    dQdq * ddQdqdf
                )
            dgθϕθ =
                Power(Sin(Q), 2) * dFdq * ddFdqdf +
                dQdq * ddQdqdf +
                Sin(Q) * dFdf * (dFdq * (-(Cot(q) * Sin(Q)) + 2 * Cos(Q) * dQdq) + Sin(Q) * ddFdqdq) +
                dQdf * (-(Cot(q) * dQdq) + ddQdqdq)
            dgθϕϕ =
                -(Cot(q) * Power(Sin(Q), 2) * Power(dFdf, 2)) - Cot(q) * Power(dQdf, 2) +
                Power(Sin(Q), 2) * ddFdfdf * dFdq +
                Cos(q) * Sin(q) * Power(Sin(Q), 2) * Power(dFdq, 2) +
                ddQdfdf * dQdq +
                Cos(q) * Sin(q) * Power(dQdq, 2) +
                Sin(Q) * dFdf * (2 * Cos(Q) * dQdf * dFdq + Sin(Q) * ddFdqdf) +
                dQdf * ddQdqdf
            dgϕϕθ =
                2 * (
                    Sin(Q) * Power(dFdf, 2) * (-(Cot(q) * Sin(Q)) + Cos(Q) * dQdq) +
                    Power(Sin(Q), 2) * dFdf * ddFdqdf +
                    dQdf * (-(Cot(q) * dQdf) + ddQdqdf)
                )
            dgϕϕϕ =
                2 * (
                    Cos(Q) * Sin(Q) * Power(dFdf, 2) * dQdf +
                    Power(Sin(Q), 2) * dFdf * (ddFdfdf + Cos(q) * Sin(q) * dFdq) +
                    dQdf * (ddQdfdf + Cos(q) * Sin(q) * dQdq)
                )

            dgθθθ /= 1
            dgθθϕ /= sin(θ)
            dgθϕθ /= sin(θ)
            dgθϕϕ /= sin(θ)^2
            dgϕϕθ /= sin(θ)^2
            dgϕϕϕ /= sin(θ)^3

            SArray{Tuple{2,2,2}}(dgθθθ, dgθϕθ, dgθϕθ, dgϕϕθ, dgθθϕ, dgθϕϕ, dgθϕϕ, dgϕϕϕ)
        end for ij in CartesianIndices(g.values)
    ],
    lmax,
)
if !(norm(dg.values - dg₀.values, Inf) ≤ atol)
    for a in 1:2, b in a:2, c in 1:2
        println("dg₀[$a,$b,$c]: ", dg₀.values[1, 1][a, b, c])
        println("dg[$a,$b,$c]: ", dg.values[1, 1][a, b, c])
        println("|(dg-dg₀)[$a,$b,$c]|∞: ", norm(map(x -> x[a, b, c], dg.values - dg₀.values), Inf))
    end
    println("atol: ", atol)
end
@assert norm(dg.values - dg₀.values, Inf) ≤ atol

# Γ^a_bc = g^ad (g_dc,b + g_bd,c - g_bc,d) / 2
Γ = Tensor{3}(
    [
        SArray{Tuple{2,2,2}}(
            sum(gu[a, d] * (dg[d, c, b] + dg[b, d, c] - dg[b, c, d]) / 2 for d in 1:2) for a in 1:2, b in 1:2, c in 1:2
        ) for (gu, dg) in zip(gu.values, dg.values)
    ],
    lmax,
)

Γ₀ = Tensor{3}(
    [
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            θ, ϕ = x
            X = x2X(x)
            Q, F = X
            J = Jac(x)
            dJ = dJac(x)

            q, f = x
            dQdq = J[1, 1]
            dQdf = J[1, 2] * sin(q)
            dFdq = J[2, 1] / sin(Q)
            dFdf = J[2, 2] / sin(Q) * sin(q)
            ddQdqdq = dJ[1, 1, 1]
            ddQdqdf = dJ[1, 1, 2] * sin(q)
            ddQdfdf = dJ[1, 2, 2] * sin(q)^2
            ddFdqdq = dJ[2, 1, 1] / sin(Q)
            ddFdqdf = dJ[2, 1, 2] / sin(Q) * sin(q)
            ddFdfdf = dJ[2, 2, 2] / sin(Q) * sin(q)^2

            Cos = cos
            Cot = cot
            Csc = csc
            Sin = sin
            Power = (^)

            Γθθθ =
                (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) /
                (dQdf * dFdq - dFdf * dQdq)
            Γθθϕ =
                (
                    Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                    dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                    dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                ) / (dQdf * dFdq - dFdf * dQdq)
            Γθϕϕ =
                (
                    Cos(Q) * Sin(Q) * Power(dFdf, 3) +
                    dQdf * (ddFdfdf + Cos(q) * Sin(q) * dFdq) +
                    dFdf * (2 * Cot(Q) * Power(dQdf, 2) - ddQdfdf - Cos(q) * Sin(q) * dQdq)
                ) / (dQdf * dFdq - dFdf * dQdq)
            Γϕθθ =
                (-(Cos(Q) * Sin(Q) * Power(dFdq, 3)) - dQdq * ddFdqdq + dFdq * (-2 * Cot(Q) * Power(dQdq, 2) + ddQdqdq)) /
                (dQdf * dFdq - dFdf * dQdq)
            Γϕθϕ = -(
                (
                    dQdf * dFdq * (Cot(q) + Cot(Q) * dQdq) +
                    dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - Cot(q) * dQdq + Cot(Q) * Power(dQdq, 2)) +
                    dQdq * ddFdqdf - dFdq * ddQdqdf
                ) / (dQdf * dFdq - dFdf * dQdq)
            )
            Γϕϕϕ = -(
                (Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq + 2 * Cot(Q) * dFdf * dQdf * dQdq + ddFdfdf * dQdq) /
                (dQdf * dFdq - dFdf * dQdq)
            )

            Γθθθ *= 1
            Γθθϕ /= sin(θ)
            Γθϕϕ /= sin(θ)^2
            Γϕθθ *= sin(θ)
            Γϕθϕ *= 1
            Γϕϕϕ /= sin(θ)

            SArray{Tuple{2,2,2}}(Γθθθ, Γϕθθ, Γθθϕ, Γϕθϕ, Γθθϕ, Γϕθϕ, Γθϕϕ, Γϕϕϕ)
        end for ij in CartesianIndices(g.values)
    ],
    lmax,
)
if !(norm(Γ.values - Γ₀.values, Inf) ≤ atol)
    for a in 1:2, b in 1:2, c in b:2
        println("Γ₀[$a,$b,$c]: ", Γ₀.values[1, 1][a, b, c])
        println("Γ[$a,$b,$c]: ", Γ.values[1, 1][a, b, c])
        println("|(Γ-Γ₀)[$a,$b,$c]|∞: ", norm(map(x -> x[a, b, c], Γ.values - Γ₀.values), Inf))
    end
    println("atol: ", atol)
end
@assert norm(Γ.values - Γ₀.values, Inf) ≤ atol

# Define 3 vector fields (x⃗, y⃗, z⃗), consider their covariant Hessians:
#
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

Rm = Tensor{4}(
    [
        make_Rm(gu, v1, v2, v3, DDv1, DDv2, DDv3) for (gu, v1, v2, v3, DDv1, DDv2, DDv3) in
        zip(gu.values, vs[1].values, vs[2].values, vs[3].values, DDvs[1].values, DDvs[2].values, DDvs[3].values)
    ],
    lmax,
)
Rm = make_real(Rm)
Rm = make_riemann_symmetry(Rm)

Rm₀ = Tensor{4}(
    [
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            θ, ϕ = x
            X = x2X(x)
            Q, F = X
            J = Jac0(x)

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

            Rm1212 = Power(Sin(Q), 2) * Power(dQdf * dFdq - dFdf * dQdq, 2)

            Rm1212 /= sin(θ)^2

            SArray{Tuple{2,2,2,2}}(0, 0, 0, 0, 0, Rm1212, -Rm1212, 0, 0, -Rm1212, Rm1212, 0, 0, 0, 0, 0)
        end for ij in CartesianIndices(g.values)
    ],
    lmax,
)
if !(norm(Rm.values - Rm₀.values, Inf) ≤ 10atol)
    for a in 1:2, b in (a + 1):2, c in 1:2, d in (c + 1):2
        if (a, b) ≤ (c, d)
            println("Rm₀[$a,$b,$c,$d]: ", Rm₀.values[1, 1][a, b, c, d])
            println("Rm[$a,$b,$c,$d]: ", Rm.values[1, 1][a, b, c, d])
            println("|(Rm-Rm₀)[$a,$b,$c,$d]|∞: ", norm(map(x -> x[a, b, c, d], Rm.values - Rm₀.values), Inf))
        end
    end
    println("10atol: ", 10atol)
end
@assert norm(Rm.values - Rm₀.values, Inf) ≤ 10atol

# Ricci
Rc = Tensor{2}(
    [
        SMatrix{2,2}(-sum(gu[c, d] * Rm[c, a, b, d] for c in 1:2, d in 1:2) for a in 1:2, b in 1:2) for
        (gu, Rm) in zip(gu.values, Rm.values)
    ],
    lmax,
)
Rc = make_real(Rc)
Rc = make_symmetric(Rc)

Rc₀ = Tensor{2}([
    begin
        x = SVector{2}(ash_point_coord(ij, lmax))
        θ, ϕ = x
        X = x2X(x)
        Q, F = X
        J = Jac0(x)

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

        Rcθθ = Power(Sin(Q), 2) * Power(dFdq, 2) + Power(dQdq, 2)
        Rcθϕ = Power(Sin(Q), 2) * dFdf * dFdq + dQdf * dQdq
        Rcϕϕ = Power(Sin(Q), 2) * Power(dFdf, 2) + Power(dQdf, 2)

        Rcθθ /= 1
        Rcθϕ /= sin(θ)
        Rcϕϕ /= sin(θ)^2

        SMatrix{2,2}(Rcθθ, Rcθϕ, Rcθϕ, Rcϕϕ)
    end for ij in CartesianIndices(Rc.values)
], lmax)
if !(norm(Rc.values - Rc₀.values, Inf) ≤ 100atol)
    for a in 1:2, b in a:2
        println("Rc₀[$a,$b]: ", Rc₀.values[1, 1][a, b])
        println("Rc[$a,$b]: ", Rc.values[1, 1][a, b])
        println("|(Rc-Rc₀)[$a,$b]|∞: ", norm(map(x -> x[a, b], Rc.values - Rc₀.values), Inf))
    end
    println("100atol: ", 100atol)
end
@assert norm(Rc.values - Rc₀.values, Inf) ≤ 100atol

Rsc = Tensor{0}([Scalar(sum(gu[a, b] * Rc[a, b] for a in 1:2, b in 1:2)) for (gu, Rc) in zip(gu.values, Rc.values)], lmax)
Rsc = make_real(Rsc)

# println("q:")
# display(map(x -> chop.(real(x)), q.values))
# println()
# # println("det q:")
# # display(map(x -> chop(det(real(x))), q.values))
# # println()
# # println("Γ:")
# # display(map(x -> chop.(real(x)), Γ.values))
# # println()
# println("R:")
# display(map(x -> chop.(real(x)), R.values))
# println()
# println("Rsc:")
# display(map(x -> chop(real(x[])), Rsc.values))
# println()

################################################################################

using StatsPlots

trace(A::SMatrix{N,N,T}) where {N,T} = sum(SVector{N,T}(A[i] for i in 1:N))

# plt = plot(; aspect_ratio=1, legend=false)
# heatmap!(plt, ash_phis(lmax), ash_thetas(lmax), det.(real(transpose(ash_grid_as_phi_theta(q.values)))))
# for ij in CartesianIndices(ash_grid_size(lmax))
#     # iseven(ij[2]) && continue
#     theta, phi = ash_point_coord(ij, lmax)
#     dtheta, dphi = ash_point_delta(ij, lmax)
#     qij = q.values[ij]
#     for i in 0:1
#         covellipse!(plt, SVector(phi + i * dphi / 2, theta), dtheta * dphi / 25 * real(qij); color=:green)
#     end
# end
# plt

for a in 1:2, b in (a + 1):2, c in 1:2, d in (c + 1):2
    if (c, d) ≥ (a, b)
        println(
            "Rm[$a,$b,$c,$d]:   min: ",
            minimum(real(map(x -> x[a, b, c, d], Rm.values))),
            "   max: ",
            maximum(real(map(x -> x[a, b, c, d], Rm.values))),
        )
    end
end
# plt = plot(; aspect_ratio=1, legend=false)
# heatmap!(
#     plt,
#     ash_phis(lmax),
#     ash_thetas(lmax),
#     real(transpose(map(x -> x[1, 2, 1, 2], ash_grid_as_phi_theta(Rm.values))));
#     colorbar=true,
#     clims=(-1, 1),
#     title="Rm[1,2,1,2]",
# )
# display(plt)

for a in 1:2, b in a:2
    println(
        "Rc[$a,$b]:   min: ", minimum(real(map(x -> x[a, b], Rc.values))), "   max: ", maximum(real(map(x -> x[a, b], Rc.values)))
    )
end
# plt = plot(; aspect_ratio=1, legend=false)
# heatmap!(
#     plt,
#     ash_phis(lmax),
#     ash_thetas(lmax),
#     real(transpose(map(x -> x[1, 1], ash_grid_as_phi_theta(Rc.values))));
#     colorbar=true,
#     clims=(-2, -0),
#     title="Rc[1,1]:",
# )
# display(plt)

println("Rsc:   min: ", minimum(real(map(x -> x[], Rsc.values))), "   max: ", maximum(real(map(x -> x[], Rsc.values))))
plt = plot(; aspect_ratio=1, legend=false)
heatmap!(
    plt,
    ash_phis(lmax),
    ash_thetas(lmax),
    real(transpose(map(x -> x[], ash_grid_as_phi_theta(Rsc.values))));
    colorbar=true,
    clims=(1.9, 2.1),
    title="Rsc:",
)
display(plt)

nothing
