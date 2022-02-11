using AbstractSphericalHarmonics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Test

# Global settings

lmax = 40

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

Base.chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
Base.chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))
Base.chop(x::SArray) = chop.(x)

function impose_symmetry(f, t::Tensor{D}) where {D}
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
    α2 = 0                          # cannot be nonzero
    α3 = 0.1
    β1 = 0                          # cannot be nonzero
    β2 = 0.1
    β3 = 0                          # cannot be nonzero
    "X^a(x^b)"
    function x2X(x::SVector{2})
        θ, ϕ = x
        Q = θ + α1 * sin(θ) + α2 * sin(θ) * sin(ϕ) + α3 * sin(θ)^2 * sin(ϕ)
        F = ϕ + β1 * sin(θ) + β2 * sin(θ)^2 + β3 * sin(θ)^2 * sin(ϕ)
        X = SVector{2}(Q, F)
        return X
    end

    "J^a_b = ∂X^a/∂x_b(x^c)"
    function Jac(x::SVector{2})
        θ, ϕ = x
        Q, F = x2X(x)
        #
        ∂Q∂θ = 1 + cos(θ) * (α1 + sin(ϕ) * (α2 + 2 * α3 * sin(θ)))
        ∂Q∂ϕ = cos(ϕ) * sin(θ) * (α2 + α3 * sin(θ))
        ∂F∂θ = cos(θ) * (β1 + 2 * (β2 + β3 * sin(ϕ)) * sin(θ))
        ∂F∂ϕ = 1 + β3 * cos(ϕ) * sin(θ)^2
        #
        ∂Q∂θ *= 1
        ∂Q∂ϕ *= 1 / sin(θ)
        ∂F∂θ *= sin(Q)
        ∂F∂ϕ *= sin(Q) / sin(θ)
        #
        J = SMatrix{2,2}(∂Q∂θ, ∂F∂θ, ∂Q∂ϕ, ∂F∂ϕ)
        return J
    end

else
    dx = 0.1
    dy = 0
    dz = 0
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

    "J^a_b = ∂X^a/∂x_b(x^c)"
    Jac0(x::SVector{2}) = ForwardDiff.jacobian(x2X, x)

    "dJ^a_bc = ∂²X^a/∂x_b∂x_c(x^d)"
    dJac0(x::SVector{2}) = reshape(ForwardDiff.jacobian(Jac0, x), (2, 2, 2))

    "ddJ^a_bcd = ∂³X^a/∂x_b∂x_c∂x_d(x^e)"
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

    "J^a_b = ∂X^a/∂x_b(x^c)"
    function Jac_old(x::SVector{2})
        θ, ϕ = x
        Q, F = x2X(x)
        #
        Cos = cos
        Cot = cot
        Csc = csc
        Power = (^)
        Sin = sin
        Sqrt = sqrt
        q = θ
        f = ϕ
        #
        ∂Q∂θ =
            (
                dx * Cos(f) * (3 + 2 * dz * Cos(q) - Cos(2 * q)) +
                dy * (3 + 2 * dz * Cos(q) - Cos(2 * q)) * Sin(f) +
                2 * (1 + Power(dx, 2) + Power(dy, 2) + dz * Cos(q)) * Sin(q)
            ) / (
                2 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 1.5) *
                Sqrt(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                )
            )
        ∂Q∂ϕ =
            ((dz + Cos(q)) * (dy * Cos(f) - dx * Sin(f)) * Sin(q)) / (
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 1.5) * Sqrt(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                )
            )
        ∂F∂θ =
            (2 * Cos(q) * (-(dy * Cos(f)) + dx * Sin(f))) / (
                1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Power(Cos(q), 2) +
                4 * dx * Cos(f) * Sin(q) +
                4 * dy * Sin(f) * Sin(q) +
                Power(Sin(q), 2)
            )
        ∂F∂ϕ =
            (2 * Sin(q) * (dx * Cos(f) + dy * Sin(f) + Sin(q))) / (
                1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Power(Cos(q), 2) +
                4 * dx * Cos(f) * Sin(q) +
                4 * dy * Sin(f) * Sin(q) +
                Power(Sin(q), 2)
            )
        #
        ∂Q∂θ *= 1
        ∂Q∂ϕ *= 1 / sin(θ)
        ∂F∂θ *= sin(Q)
        ∂F∂ϕ *= sin(Q) / sin(θ)
        #
        J = SMatrix{2,2}(∂Q∂θ, ∂F∂θ, ∂Q∂ϕ, ∂F∂ϕ)
        return J
    end

    "dJ^a_bc = ∂²X^a/∂x_b∂x_c(x^d)"
    function dJac_old(x::SVector{2})
        θ, ϕ = x
        Q, F = x2X(x)
        #
        Cos = cos
        Cot = cot
        Csc = csc
        Power = (^)
        Sin = sin
        Sqrt = sqrt
        q = θ
        f = ϕ
        #
        ∂∂Q∂θ∂θ =
            (
                -(
                    (
                        (dz + Cos(q)) * Power(
                            dx * Cos(f) * (3 + 2 * dz * Cos(q) - Cos(2 * q)) +
                            dy * (3 + 2 * dz * Cos(q) - Cos(2 * q)) * Sin(f) +
                            2 * (1 + Power(dx, 2) + Power(dy, 2) + dz * Cos(q)) * Sin(q),
                            2,
                        )
                    ) / (
                        1 +
                        Power(dx, 2) +
                        Power(dy, 2) +
                        Power(dz, 2) +
                        2 * dz * Cos(q) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q)
                    )
                ) +
                2 *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (
                    2 * dz * Power(Cos(q), 2) - 2 * dz * Sin(q) * (dx * Cos(f) + dy * Sin(f) + Sin(q)) +
                    2 * Cos(q) * (1 + Power(dx, 2) + Power(dy, 2) + 2 * dx * Cos(f) * Sin(q) + 2 * dy * Sin(f) * Sin(q))
                ) -
                6 *
                (dx * Cos(f) * Cos(q) + dy * Cos(q) * Sin(f) - dz * Sin(q)) *
                (
                    dx * Cos(f) * (3 + 2 * dz * Cos(q) - Cos(2 * q)) +
                    dy * (3 + 2 * dz * Cos(q) - Cos(2 * q)) * Sin(f) +
                    2 * (1 + Power(dx, 2) + Power(dy, 2) + dz * Cos(q)) * Sin(q)
                ) *
                (
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                )
            ) / (
                4 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 2.5) *
                Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    1.5,
                )
            )
        ∂∂Q∂θ∂ϕ =
            (
                (dy * Cos(f) - dx * Sin(f)) * (
                    -(
                        Power(dz + Cos(q), 2) *
                        Sin(q) *
                        (
                            dx * Cos(f) * (3 + 2 * dz * Cos(q) - Cos(2 * q)) +
                            dy * (3 + 2 * dz * Cos(q) - Cos(2 * q)) * Sin(f) +
                            2 * (1 + Power(dx, 2) + Power(dy, 2) + dz * Cos(q)) * Sin(q)
                        )
                    ) -
                    3 *
                    Sin(q) *
                    (
                        dx * Cos(f) * (3 + 2 * dz * Cos(q) - Cos(2 * q)) +
                        dy * (3 + 2 * dz * Cos(q) - Cos(2 * q)) * Sin(f) +
                        2 * (1 + Power(dx, 2) + Power(dy, 2) + dz * Cos(q)) * Sin(q)
                    ) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) +
                    (3 + 2 * dz * Cos(q) - Cos(2 * q)) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                )
            ) / (
                2 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 3.5) *
                Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    1.5,
                )
            )
        ∂∂Q∂ϕ∂ϕ =
            (
                (dz + Cos(q)) *
                Sin(q) *
                (
                    -2 * Power(dz + Cos(q), 2) * Power(dy * Cos(f) - dx * Sin(f), 2) * Sin(q) -
                    6 *
                    Power(dy * Cos(f) - dx * Sin(f), 2) *
                    Sin(q) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) +
                    2 *
                    (-(dx * Cos(f)) - dy * Sin(f)) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                )
            ) / (
                2 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 3.5) *
                Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    1.5,
                )
            )
        ∂∂F∂θ∂θ =
            (
                (dy * Cos(f) - dx * Sin(f)) *
                (8 * dx * Cos(f) + 8 * dy * Sin(f) + 2 * (3 + 2 * Power(dx, 2) + 2 * Power(dy, 2) + Cos(2 * q)) * Sin(q))
            ) / Power(
                1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Power(Cos(q), 2) +
                4 * dx * Cos(f) * Sin(q) +
                4 * dy * Sin(f) * Sin(q) +
                Power(Sin(q), 2),
                2,
            )
        ∂∂F∂θ∂ϕ =
            (
                2 *
                Cos(q) *
                (
                    dx * Cos(f) * (1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Cos(2 * q)) +
                    dy * (1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Cos(2 * q)) * Sin(f) +
                    4 * (Power(dx, 2) + Power(dy, 2)) * Sin(q)
                )
            ) / Power(
                1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Power(Cos(q), 2) +
                4 * dx * Cos(f) * Sin(q) +
                4 * dy * Sin(f) * Sin(q) +
                Power(Sin(q), 2),
                2,
            )
        ∂∂F∂ϕ∂ϕ =
            (2 * (-1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) + Cos(2 * q)) * (dy * Cos(f) - dx * Sin(f)) * Sin(q)) / Power(
                1 + 2 * Power(dx, 2) + 2 * Power(dy, 2) - Power(Cos(q), 2) +
                4 * dx * Cos(f) * Sin(q) +
                4 * dy * Sin(f) * Sin(q) +
                Power(Sin(q), 2),
                2,
            )
        #
        ∂∂Q∂θ∂θ *= 1
        ∂∂Q∂θ∂ϕ *= 1 / sin(θ)
        ∂∂Q∂ϕ∂ϕ *= 1 / sin(θ)^2
        ∂∂F∂θ∂θ *= sin(Q)
        ∂∂F∂θ∂ϕ *= sin(Q) / sin(θ)
        ∂∂F∂ϕ∂ϕ *= sin(Q) / sin(θ)^2
        #
        dJ = SArray{Tuple{2,2,2}}(∂∂Q∂θ∂θ, ∂∂F∂θ∂θ, ∂∂Q∂θ∂ϕ, ∂∂F∂θ∂ϕ, ∂∂Q∂θ∂ϕ, ∂∂F∂θ∂ϕ, ∂∂Q∂ϕ∂ϕ, ∂∂F∂ϕ∂ϕ)
        return dJ
    end

    "ddJ^a_bcd = ∂³X^a/∂x_b∂x_c∂x_d(x^e)"
    function ddJac_old(x::SVector{2})
        θ, ϕ = x
        Q, F = x2X(x)
        #
        Cos = cos
        Cot = cot
        Csc = csc
        Power = (^)
        Sin = sin
        Sqrt = sqrt
        q = θ
        f = ϕ
        #
        ddJQq1111 =
            (
                -5 *
                (
                    -2 * (dz + Cos(q)) * Sin(q) +
                    2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                    2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                ) *
                (
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                ) *
                (
                    (
                        -4 *
                        (dz + Cos(q)) *
                        Power(
                            dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                            dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                            Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                            2,
                        )
                    ) / (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        Power(dz, 2) +
                        2 * dz * Cos(q) +
                        Power(Cos(q), 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) +
                    4 *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Cos(q) * (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                    ) -
                    3 *
                    (
                        -2 * (dz + Cos(q)) * Sin(q) +
                        2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                        2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                    ) *
                    (
                        1 -
                        Power(dz + Cos(q), 2) /
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                ) -
                (
                    3 *
                    (dz + Cos(q)) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) *
                    (
                        (
                            -4 *
                            (dz + Cos(q)) *
                            Power(
                                dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                                Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                                Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                                dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                                Sin(q) *
                                (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                                2,
                            )
                        ) / (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) +
                        4 *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) *
                        (
                            dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                            Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                            Cos(q) * (
                                Power(dx, 2) +
                                Power(dy, 2) +
                                2 * dx * Cos(f) * Sin(q) +
                                2 * dy * Sin(f) * Sin(q) +
                                Power(Cos(f), 2) * Power(Sin(q), 2) +
                                Power(Sin(f), 2) * Power(Sin(q), 2)
                            ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                        ) -
                        3 *
                        (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) *
                        (
                            1 -
                            Power(dz + Cos(q), 2) /
                            (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                        ) *
                        (
                            (dz + Cos(q)) * (
                                -2 * (dz + Cos(q)) * Sin(q) +
                                2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                                2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                            ) +
                            2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                        )
                    )
                ) / (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) +
                2 *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (
                    -4 *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        Power(Cos(q), 2) * Sin(f) * (-2 * dy + Sin(f) * Sin(q)) +
                        dz * Cos(q) * Sin(f) * (dy + 4 * Sin(f) * Sin(q)) +
                        Power(Cos(f), 2) * Sin(q) * (4 * dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                        dx * Cos(f) * (dz * Cos(q) - 2 * Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                        Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2))
                    ) +
                    (
                        4 *
                        Sin(q) *
                        Power(
                            dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                            dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                            Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                            2,
                        )
                    ) / (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) +
                    (
                        24 *
                        (dz + Cos(q)) *
                        (
                            -(dx * Cos(f) * Cos(q)) + dz * Sin(q) - Power(Cos(f), 2) * Cos(q) * Sin(q) +
                            Cos(q) * (-(dy * Sin(f)) + Sin(q) - Power(Sin(f), 2) * Sin(q))
                        ) *
                        Power(
                            dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                            dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                            Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                            2,
                        )
                    ) / Power(
                        Power(dx, 2) +
                        Power(dy, 2) +
                        Power(dz, 2) +
                        2 * dz * Cos(q) +
                        Power(Cos(q), 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2),
                        2,
                    ) +
                    (
                        8 *
                        (dz + Cos(q)) *
                        (
                            dx * Cos(f) * Cos(q) - dz * Sin(q) +
                            Power(Cos(f), 2) * Cos(q) * Sin(q) +
                            Cos(q) * (dy * Sin(f) - Sin(q) + Power(Sin(f), 2) * Sin(q))
                        ) *
                        Power(
                            dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                            dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                            Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                            2,
                        )
                    ) / Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 2) +
                    8 *
                    Cos(q) *
                    (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q))) *
                    (
                        dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Cos(q) * (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                    ) -
                    (
                        8 *
                        (dz + Cos(q)) *
                        (
                            dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                            Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                            dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                            Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2))
                        ) *
                        (
                            dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                            Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                            Cos(q) * (
                                Power(dx, 2) +
                                Power(dy, 2) +
                                2 * dx * Cos(f) * Sin(q) +
                                2 * dy * Sin(f) * Sin(q) +
                                Power(Cos(f), 2) * Power(Sin(q), 2) +
                                Power(Sin(f), 2) * Power(Sin(q), 2)
                            ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                        )
                    ) / (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        Power(dz, 2) +
                        2 * dz * Cos(q) +
                        Power(Cos(q), 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) -
                    6 *
                    (
                        -2 * (dz + Cos(q)) * Sin(q) +
                        2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                        2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                    ) *
                    (
                        dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Cos(q) * (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                    ) *
                    (
                        1 -
                        Power(dz + Cos(q), 2) /
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    3 *
                    (
                        -2 * dz * Cos(q) + 2 * Power(Cos(q), 2) * (-1 + Power(Cos(f), 2) + Power(Sin(f), 2)) -
                        2 * Sin(q) * (dx * Cos(f) + dy * Sin(f) - Sin(q) + Power(Cos(f), 2) * Sin(q) + Power(Sin(f), 2) * Sin(q))
                    ) *
                    (
                        1 -
                        Power(dz + Cos(q), 2) /
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                )
            ) / (
                8 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 3.5) *
                Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    2.5,
                )
            )

        ddJQq1112 =
            -0.125 * (
                6 *
                Power(dz + Cos(q), 2) *
                (-(dy * Cos(f)) + dx * Sin(f)) *
                Sin(q) *
                (
                    4 *
                    (dz + Cos(q)) *
                    Power(
                        dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                        dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                        Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                        2,
                    ) -
                    4 *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (
                        dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Cos(q) * (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                    ) -
                    3 *
                    (
                        2 * (dz + Cos(q)) * Sin(q) - 2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) -
                        2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                    ) *
                    (Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                ) +
                10 *
                (-(dy * Cos(f)) + dx * Sin(f)) *
                Sin(q) *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (
                    4 *
                    (dz + Cos(q)) *
                    Power(
                        dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                        dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                        Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                        2,
                    ) -
                    4 *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (
                        dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Cos(q) * (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                    ) -
                    3 *
                    (
                        2 * (dz + Cos(q)) * Sin(q) - 2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) -
                        2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                    ) *
                    (Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                ) +
                2 *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (
                    -6 *
                    (dy * Cos(f) - dx * Sin(f)) *
                    (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) *
                    (
                        2 * (dz + Cos(q)) * Sin(q) - 2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) -
                        2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                    ) *
                    (Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) +
                    4 *
                    (dz - 2 * Cos(q)) *
                    (dy * Cos(f) - dx * Sin(f)) *
                    Sin(q) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 2) +
                    8 *
                    (dz + Cos(q)) *
                    (dy * Cos(f) - dx * Sin(f)) *
                    (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        Power(dz, 2) +
                        2 * dz * Cos(q) +
                        Power(Cos(q), 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                        dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                        Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2))
                    ) -
                    8 *
                    (dz + Cos(q)) *
                    (dy * Cos(f) - dx * Sin(f)) *
                    Sin(q) *
                    Power(
                        dz * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        Power(Cos(f), 2) * Sin(q) * (dz * Cos(q) + Power(Cos(q), 2) + Power(Sin(q), 2)) +
                        dx * Cos(f) * (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) +
                        Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)),
                        2,
                    ) -
                    8 *
                    (dy * Cos(f) - dx * Sin(f)) *
                    Sin(q) *
                    Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 2) *
                    (
                        dz * Power(Cos(q), 2) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Power(Cos(q), 3) * (Power(Cos(f), 2) + Power(Sin(f), 2)) +
                        Cos(q) * (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) - dz * Sin(q) * (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q)))
                    ) +
                    6 *
                    Power(dz + Cos(q), 2) *
                    (-(dy * Cos(f)) + dx * Sin(f)) *
                    Sin(q) *
                    (
                        2 * (dz + Cos(q)) * Sin(q) - 2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) -
                        2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                    ) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    6 *
                    Cos(q) *
                    (-(dy * Cos(f)) + dx * Sin(f)) *
                    (Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                )
            ) / (
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 5.5) * Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    2.5,
                )
            )

        ddJQq1122 =
            (
                -4 *
                (dy * Cos(f) - dx * Sin(f)) *
                (-(dy * Cos(f)) + dx * Sin(f)) *
                Sin(q) *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                (
                    3 * dz * Power(Cos(q), 3) +
                    Power(Cos(q), 4) +
                    Power(Cos(q), 2) * (
                        Power(dx, 2) + Power(dy, 2) + 3 * Power(dz, 2) - 4 * dx * Cos(f) * Sin(q) - 4 * dy * Sin(f) * Sin(q) +
                        2 * Power(Sin(q), 2) - 5 * Power(Cos(f), 2) * Power(Sin(q), 2) - 5 * Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) +
                    dz *
                    Cos(q) *
                    (
                        Power(dx, 2) + Power(dy, 2) + Power(dz, 2) - 4 * dx * Cos(f) * Sin(q) - 4 * dy * Sin(f) * Sin(q) +
                        4 * Power(Sin(q), 2) - 5 * Power(Cos(f), 2) * Power(Sin(q), 2) - 5 * Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) -
                    2 *
                    Power(Sin(q), 2) *
                    (
                        2 * Power(dx, 2) + 2 * Power(dy, 2) - Power(dz, 2) +
                        4 * dx * Cos(f) * Sin(q) +
                        4 * dy * Sin(f) * Sin(q) +
                        2 * Power(Cos(f), 2) * Power(Sin(q), 2) +
                        2 * Power(Sin(f), 2) * Power(Sin(q), 2)
                    )
                ) -
                6 *
                Power(dz + Cos(q), 2) *
                Power(dy * Cos(f) - dx * Sin(f), 2) *
                Sin(q) *
                (
                    2 *
                    (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) -
                    Power(dz + Cos(q), 2) *
                    Sin(q) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    3 *
                    Sin(q) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                ) -
                14 *
                Power(dy * Cos(f) - dx * Sin(f), 2) *
                Sin(q) *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (
                    2 *
                    (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) -
                    Power(dz + Cos(q), 2) *
                    Sin(q) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    3 *
                    Sin(q) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                ) +
                2 *
                (-(dx * Cos(f)) - dy * Sin(f)) *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                (
                    2 *
                    (dz * Cos(q) + Power(Cos(q), 2) + 2 * Power(Sin(q), 2)) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) -
                    Power(dz + Cos(q), 2) *
                    Sin(q) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    3 *
                    Sin(q) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        (dz + Cos(q)) * (
                            -2 * (dz + Cos(q)) * Sin(q) +
                            2 * Cos(f) * Cos(q) * (dx + Cos(f) * Sin(q)) +
                            2 * Cos(q) * Sin(f) * (dy + Sin(f) * Sin(q))
                        ) + 2 * Sin(q) * (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                )
            ) / (
                4 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 5.5) *
                Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    2.5,
                )
            )

        ddJQq1222 =
            (
                (dz + Cos(q)) *
                Sin(q) *
                (
                    4 *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) *
                    (
                        2 * Power(dz + Cos(q), 2) * (dy * Cos(f) - dx * Sin(f)) * (dx * Cos(f) + dy * Sin(f)) * Sin(q) -
                        6 * Power(dy * Cos(f) - dx * Sin(f), 3) * Power(Sin(q), 2) +
                        6 *
                        (dy * Cos(f) - dx * Sin(f)) *
                        (dx * Cos(f) + dy * Sin(f)) *
                        Sin(q) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) +
                        2 *
                        (-(dy * Cos(f)) + dx * Sin(f)) *
                        (dx * Cos(f) + dy * Sin(f)) *
                        Sin(q) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) +
                        2 *
                        (dy * Cos(f) - dx * Sin(f)) *
                        (-(dx * Cos(f)) - dy * Sin(f)) *
                        Sin(q) *
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)) +
                        (-(dy * Cos(f)) + dx * Sin(f)) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) *
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    6 *
                    Power(dz + Cos(q), 2) *
                    (dy * Cos(f) - dx * Sin(f)) *
                    Sin(q) *
                    (
                        -2 * Power(dz + Cos(q), 2) * Power(dy * Cos(f) - dx * Sin(f), 2) * Sin(q) -
                        6 *
                        Power(dy * Cos(f) - dx * Sin(f), 2) *
                        Sin(q) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) +
                        2 *
                        (-(dx * Cos(f)) - dy * Sin(f)) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) *
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    ) -
                    14 *
                    (dy * Cos(f) - dx * Sin(f)) *
                    Sin(q) *
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) *
                    (
                        -2 * Power(dz + Cos(q), 2) * Power(dy * Cos(f) - dx * Sin(f), 2) * Sin(q) -
                        6 *
                        Power(dy * Cos(f) - dx * Sin(f), 2) *
                        Sin(q) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) +
                        2 *
                        (-(dx * Cos(f)) - dy * Sin(f)) *
                        (
                            Power(dx, 2) +
                            Power(dy, 2) +
                            2 * dx * Cos(f) * Sin(q) +
                            2 * dy * Sin(f) * Sin(q) +
                            Power(Cos(f), 2) * Power(Sin(q), 2) +
                            Power(Sin(f), 2) * Power(Sin(q), 2)
                        ) *
                        (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2))
                    )
                )
            ) / (
                4 *
                Power(Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2), 5.5) *
                Power(
                    1 -
                    Power(dz + Cos(q), 2) /
                    (Power(dz + Cos(q), 2) + Power(dx + Cos(f) * Sin(q), 2) + Power(dy + Sin(f) * Sin(q), 2)),
                    2.5,
                )
            )

        ddJQq2111 =
            (
                Cos(q) *
                (dy * Cos(f) - dx * Sin(f)) *
                (
                    (
                        Power(dx, 2) +
                        Power(dy, 2) +
                        2 * dx * Cos(f) * Sin(q) +
                        2 * dy * Sin(f) * Sin(q) +
                        Power(Cos(f), 2) * Power(Sin(q), 2) +
                        Power(Sin(f), 2) * Power(Sin(q), 2)
                    ) * (
                        Power(dx, 2) + Power(dy, 2) + 2 * Power(Cos(q), 2) * Power(Sin(f), 2) -
                        Power(Sin(f), 2) * Power(Sin(q), 2) + Power(Cos(f), 2) * (2 * Power(Cos(q), 2) - Power(Sin(q), 2))
                    ) -
                    4 *
                    (dx * Cos(f) + Power(Cos(f), 2) * Sin(q) + Sin(f) * (dy + Sin(f) * Sin(q))) *
                    (
                        2 * Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                        2 * dx * Cos(f) * (Power(Cos(q), 2) + Power(Sin(q), 2)) +
                        Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)) +
                        Power(Cos(f), 2) * (2 * Power(Cos(q), 2) * Sin(q) + Power(Sin(q), 3))
                    )
                )
            ) / Power(
                Power(dx, 2) +
                Power(dy, 2) +
                2 * dx * Cos(f) * Sin(q) +
                2 * dy * Sin(f) * Sin(q) +
                Power(Cos(f), 2) * Power(Sin(q), 2) +
                Power(Sin(f), 2) * Power(Sin(q), 2),
                3,
            )

        ddJQq2112 =
            (
                2 *
                Power(dy * Cos(f) - dx * Sin(f), 2) *
                (Power(Cos(q), 2) + Power(Sin(q), 2)) *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) -
                4 *
                Power(dy * Cos(f) - dx * Sin(f), 2) *
                Sin(q) *
                (
                    2 * Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                    2 * dx * Cos(f) * (Power(Cos(q), 2) + Power(Sin(q), 2)) +
                    Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)) +
                    Power(Cos(f), 2) * (2 * Power(Cos(q), 2) * Sin(q) + Power(Sin(q), 3))
                ) +
                (-(dx * Cos(f)) - dy * Sin(f)) *
                (
                    Power(dx, 2) +
                    Power(dy, 2) +
                    2 * dx * Cos(f) * Sin(q) +
                    2 * dy * Sin(f) * Sin(q) +
                    Power(Cos(f), 2) * Power(Sin(q), 2) +
                    Power(Sin(f), 2) * Power(Sin(q), 2)
                ) *
                (
                    2 * Power(Cos(q), 2) * Sin(f) * (dy + Sin(f) * Sin(q)) +
                    2 * dx * Cos(f) * (Power(Cos(q), 2) + Power(Sin(q), 2)) +
                    Sin(q) * (Power(dx, 2) + Power(dy, 2) + 2 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)) +
                    Power(Cos(f), 2) * (2 * Power(Cos(q), 2) * Sin(q) + Power(Sin(q), 3))
                )
            ) / Power(
                Power(dx, 2) +
                Power(dy, 2) +
                2 * dx * Cos(f) * Sin(q) +
                2 * dy * Sin(f) * Sin(q) +
                Power(Cos(f), 2) * Power(Sin(q), 2) +
                Power(Sin(f), 2) * Power(Sin(q), 2),
                3,
            )

        ddJQq2122 =
            (
                Cos(q) *
                (dy * Cos(f) - dx * Sin(f)) *
                (
                    Power(Power(dx, 2) + Power(dy, 2), 2) - 2 * dy * (Power(dx, 2) + Power(dy, 2)) * Sin(f) * Sin(q) -
                    6 * (Power(dx, 2) + Power(dy, 2)) * Power(Sin(f), 2) * Power(Sin(q), 2) -
                    2 * dx * Power(Cos(f), 3) * Power(Sin(q), 3) - 2 * dy * Power(Sin(f), 3) * Power(Sin(q), 3) +
                    Power(Cos(f), 4) * Power(Sin(q), 4) +
                    Power(Sin(f), 4) * Power(Sin(q), 4) -
                    2 *
                    Power(Cos(f), 2) *
                    Power(Sin(q), 2) *
                    (3 * (Power(dx, 2) + Power(dy, 2)) + dy * Sin(f) * Sin(q) - Power(Sin(f), 2) * Power(Sin(q), 2)) -
                    2 * dx * Cos(f) * Sin(q) * (Power(dx, 2) + Power(dy, 2) + Power(Sin(f), 2) * Power(Sin(q), 2))
                )
            ) / Power(
                Power(dx, 2) +
                Power(dy, 2) +
                2 * dx * Cos(f) * Sin(q) +
                2 * dy * Sin(f) * Sin(q) +
                Power(Cos(f), 2) * Power(Sin(q), 2) +
                Power(Sin(f), 2) * Power(Sin(q), 2),
                3,
            )

        ddJQq2222 =
            (
                Sin(q) *
                (-Power(dx, 2) - Power(dy, 2) + Power(Cos(f), 2) * Power(Sin(q), 2) + Power(Sin(f), 2) * Power(Sin(q), 2)) *
                (
                    dx * Power(Cos(f), 3) * Power(Sin(q), 2) +
                    Power(Cos(f), 2) * Sin(q) * (2 * Power(dx, 2) + 4 * Power(dy, 2) + dy * Sin(f) * Sin(q)) +
                    dx * Cos(f) * (Power(dx, 2) + Power(dy, 2) - 4 * dy * Sin(f) * Sin(q) + Power(Sin(f), 2) * Power(Sin(q), 2)) +
                    Sin(f) * (
                        dy * (Power(dx, 2) + Power(dy, 2)) +
                        2 * (2 * Power(dx, 2) + Power(dy, 2)) * Sin(f) * Sin(q) +
                        dy * Power(Sin(f), 2) * Power(Sin(q), 2)
                    )
                )
            ) / Power(
                Power(dx, 2) +
                Power(dy, 2) +
                2 * dx * Cos(f) * Sin(q) +
                2 * dy * Sin(f) * Sin(q) +
                Power(Cos(f), 2) * Power(Sin(q), 2) +
                Power(Sin(f), 2) * Power(Sin(q), 2),
                3,
            )
        #
        ∂∂∂Q∂θ∂θ∂θ = ddJQq1111
        ∂∂∂Q∂θ∂θ∂ϕ = ddJQq1112
        ∂∂∂Q∂θ∂ϕ∂ϕ = ddJQq1122
        ∂∂∂Q∂ϕ∂ϕ∂ϕ = ddJQq1222
        ∂∂∂F∂θ∂θ∂θ = ddJQq2111
        ∂∂∂F∂θ∂θ∂ϕ = ddJQq2112
        ∂∂∂F∂θ∂ϕ∂ϕ = ddJQq2122
        ∂∂∂F∂ϕ∂ϕ∂ϕ = ddJQq2222
        #
        ∂∂∂Q∂θ∂θ∂θ *= 1
        ∂∂∂Q∂θ∂θ∂ϕ *= 1 / sin(θ)
        ∂∂∂Q∂θ∂ϕ∂ϕ *= 1 / sin(θ)^2
        ∂∂∂Q∂ϕ∂ϕ∂ϕ *= 1 / sin(θ)^3
        ∂∂∂F∂θ∂θ∂θ *= sin(Q)
        ∂∂∂F∂θ∂θ∂ϕ *= sin(Q) / sin(θ)
        ∂∂∂F∂θ∂ϕ∂ϕ *= sin(Q) / sin(θ)^2
        ∂∂∂F∂ϕ∂ϕ∂ϕ *= sin(Q) / sin(θ)^3
        #
        ddJ = SArray{Tuple{2,2,2,2}}(
            ∂∂∂Q∂θ∂θ∂θ,
            ∂∂∂F∂θ∂θ∂θ,
            ∂∂∂Q∂θ∂θ∂ϕ,
            ∂∂∂F∂θ∂θ∂ϕ,
            ∂∂∂Q∂θ∂θ∂ϕ,
            ∂∂∂F∂θ∂θ∂ϕ,
            ∂∂∂Q∂θ∂ϕ∂ϕ,
            ∂∂∂F∂θ∂ϕ∂ϕ,
            ∂∂∂Q∂θ∂θ∂ϕ,
            ∂∂∂F∂θ∂θ∂ϕ,
            ∂∂∂Q∂θ∂ϕ∂ϕ,
            ∂∂∂F∂θ∂ϕ∂ϕ,
            ∂∂∂Q∂θ∂ϕ∂ϕ,
            ∂∂∂F∂θ∂ϕ∂ϕ,
            ∂∂∂Q∂ϕ∂ϕ∂ϕ,
            ∂∂∂F∂ϕ∂ϕ∂ϕ,
        )
        return ddJ
    end
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

# D_c q_ab = q_ab,c - Γ^d_ac q_db - Γ^d_bc q_ad
Dq = Tensor{3}(
    [
        SArray{Tuple{2,2,2}}(sum(-Γ[d, a, c] * q[d, b] - Γ[d, b, c] * q[a, d] for d in 1:2) for a in 1:2, b in 1:2, c in 1:2) for
        (q, Γ) in zip(q.values, Γ.values)
    ],
    lmax,
)
Dq̃ = SpinTensor{3}(Dq)
Dq̃ = filter_modes(Dq̃)

#  Dq̃′ = SpinTensor{3}(SArray{Tuple{2,2,2}}(sum(-Γ̃.coeffs[d, a, c] .* q̃.coeffs[d, b] - Γ̃.coeffs[d, b, c] .* q̃.coeffs[a, d]
#                                                for d in 1:2) for a in 1:2, b in 1:2, c in 1:2), lmax)
#  @show maximum(map(x -> maximum(abs.(x)) , Dq̃.coeffs - Dq̃′.coeffs))

# D_d q_ab;c = q_ab;c,d - Γ^e_ad q_eb;c - Γ^e_bd q_ae;c - Γ^e_cd q_ab;e
dDq̃ = tensor_gradient(Dq̃)
dDq = Tensor{4}(dDq̃)
dDq = make_real(dDq)
dDq = make_symmetric12(dDq)
DDq = Tensor{4}(
    [
        SArray{Tuple{2,2,2,2}}(
            dDq[a, b, c, d] + sum(-Γ[e, a, d] * Dq[e, b, c] - Γ[e, b, d] * Dq[a, e, c] - Γ[e, c, d] * Dq[a, b, e] for e in 1:2) for
            a in 1:2, b in 1:2, c in 1:2, d in 1:2
        ) for (dDq, Dq, Γ) in zip(dDq.values, Dq.values, Γ.values)
    ],
    lmax,
)
DDq = make_real(DDq)
DDq = make_symmetric12(DDq)

DDq₀ = Tensor{4}(
    [
        begin
            x = SVector{2}(ash_point_coord(ij, lmax))
            θ, ϕ = x
            X = x2X(x)
            Q, F = X
            J = Jac(x)
            dJ = dJac(x)
            ddJ = ddJac(x)

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
            dddQdqdqdq = ddJ[1, 1, 1, 1]
            dddQdqdqdf = ddJ[1, 1, 1, 2] * sin(q)
            dddQdqdfdf = ddJ[1, 1, 2, 2] * sin(q)^2
            dddQdfdfdf = ddJ[1, 2, 2, 2] * sin(q)^3
            dddFdqdqdq = ddJ[2, 1, 1, 1] / sin(Q)
            dddFdqdqdf = ddJ[2, 1, 1, 2] / sin(Q) * sin(q)
            dddFdqdfdf = ddJ[2, 1, 2, 2] / sin(Q) * sin(q)^2
            dddFdfdfdf = ddJ[2, 2, 2, 2] / sin(Q) * sin(q)^3

            Cos = cos
            Cot = cot
            Csc = csc
            Sin = sin
            Power = (^)

            DDq1111 =
                (
                    2 * (
                        3 * Power(
                            dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq), 2
                        ) +
                        (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) *
                        (-(dQdq * ddFdqdf) + dFdq * ddQdqdf + dQdf * ddFdqdq - dFdf * ddQdqdq) +
                        3 *
                        (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) *
                        (-(Cos(Q) * Sin(Q) * Power(dFdq, 3)) - dQdq * ddFdqdq + dFdq * (-2 * Cot(Q) * Power(dQdq, 2) + ddQdqdq)) -
                        (dQdf * dFdq - dFdf * dQdq) * (
                            ddQdqdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) +
                            ddFdqdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq) +
                            dQdf * (
                                2 * Cot(Q) * dQdq * ddFdqdq +
                                dFdq * (-2 * Power(Csc(Q), 2) * Power(dQdq, 2) + 2 * Cot(Q) * ddQdqdq) +
                                dddFdqdqdq
                            ) +
                            dFdf * (Cos(2 * Q) * Power(dFdq, 2) * dQdq + Sin(2 * Q) * dFdq * ddFdqdq - dddQdqdqdq)
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq1112 =
                (
                    2 * (
                        Power(
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf),
                            2,
                        ) -
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) * (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) -
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) * (
                            -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                            dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                            Power(Sin(Q), 2) *
                            (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                        ) +
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) * (
                            -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                            dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                            Power(Sin(Q), 2) *
                            (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                        ) +
                        (ddQdfdf * dFdq - ddFdfdf * dQdq + dQdf * ddFdqdf - dFdf * ddQdqdf) *
                        (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) -
                        (dQdf * dFdq - dFdf * dQdq) * (
                            ddQdfdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) +
                            ddFdfdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq) +
                            dQdf * (
                                -2 * Power(Csc(Q), 2) * dQdf * dFdq * dQdq +
                                2 * Cot(Q) * dQdq * ddFdqdf +
                                2 * Cot(Q) * dFdq * ddQdqdf +
                                dddFdqdqdf
                            ) +
                            dFdf * (Cos(2 * Q) * dQdf * Power(dFdq, 2) + Sin(2 * Q) * dFdq * ddFdqdf - dddQdqdqdf)
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq1121 =
                (
                    2 * (
                        3 *
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) *
                        (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) -
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) * (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) +
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) * (-(dQdq * ddFdqdf) + dFdq * ddQdqdf + dQdf * ddFdqdq - dFdf * ddQdqdq) +
                        2 *
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) *
                        (-(Cos(Q) * Sin(Q) * Power(dFdq, 3)) - dQdq * ddFdqdq + dFdq * (-2 * Cot(Q) * Power(dQdq, 2) + ddQdqdq)) -
                        (dQdf * dFdq - dFdf * dQdq) * (
                            Power(dFdf, 2) * (Cos(2 * Q) * dFdq * dQdq + Cos(Q) * Sin(Q) * ddFdqdq) +
                            dQdf * (
                                Cot(Q) * dQdq * ddFdqdf +
                                2 * Cot(Q) * dFdq * ddQdqdf +
                                dQdf * (-(Power(Csc(Q), 2) * dFdq * dQdq) + Cot(Q) * ddFdqdq) +
                                dddFdqdqdf
                            ) +
                            dFdf * (
                                Sin(2 * Q) * dFdq * ddFdqdf +
                                Cot(Q) * dQdq * ddQdqdf +
                                dQdf * (-(Power(Csc(Q), 2) * Power(dQdq, 2)) + Cot(Q) * ddQdqdq) - dddQdqdqdf
                            )
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq1122 =
                (
                    2 * (
                        (Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * ddFdfdf + dFdf * (2 * Cot(Q) * Power(dQdf, 2) - ddQdfdf)) * (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) -
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                            2 * Cot(Q) * dFdf * dQdf * dQdq +
                            ddFdfdf * dQdq
                        ) * (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) +
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) * (ddQdfdf * dFdq - ddFdfdf * dQdq + dQdf * ddFdqdf - dFdf * ddQdqdf) +
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) * (
                            Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                            dFdf * (
                                -2 * Cot(Q) * Power(dQdf, 2) +
                                ddQdfdf +
                                Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                            ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                        ) -
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) * (
                            Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                            dFdf * (
                                -2 * Cot(Q) * Power(dQdf, 2) +
                                ddQdfdf +
                                Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                            ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                        ) -
                        (dQdf * dFdq - dFdf * dQdq) * (
                            -(Power(Csc(Q), 2) * Power(dQdf, 3) * dFdq) +
                            Cot(Q) * Power(dQdf, 2) * ddFdqdf +
                            ddQdfdf * ddFdqdf +
                            Power(dFdf, 2) * (Cos(2 * Q) * dQdf * dFdq + Cos(Q) * Sin(Q) * ddFdqdf) - ddFdfdf * ddQdqdf +
                            dQdf * (2 * Cot(Q) * ddQdfdf * dFdq + Cot(Q) * ddFdfdf * dQdq + dddFdqdfdf) +
                            dFdf * (
                                Sin(2 * Q) * ddFdfdf * dFdq - Power(Csc(Q), 2) * Power(dQdf, 2) * dQdq +
                                Cot(Q) * ddQdfdf * dQdq +
                                Cot(Q) * dQdf * ddQdqdf - dddQdqdfdf
                            )
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq1211 =
                (
                    2 *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                    ) *
                    (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) -
                    2 *
                    (
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf
                    ) *
                    (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) -
                    2 *
                    (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                        dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) *
                    (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) -
                    (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                        dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) * (-(dQdq * ddFdqdf) + dFdq * ddQdqdf + dQdf * ddFdqdq - dFdf * ddQdqdq) -
                    2 *
                    (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                        dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) *
                    (-(Cos(Q) * Sin(Q) * Power(dFdq, 3)) - dQdq * ddFdqdq + dFdq * (-2 * Cot(Q) * Power(dQdq, 2) + ddQdqdq)) +
                    (dQdf * dFdq - dFdf * dQdq) * (
                        Power(dQdf, 2) * (Power(Csc(Q), 2) * dFdq * dQdq - Cot(Q) * ddFdqdq) -
                        (Power(dFdf, 2) * (2 * Cos(2 * Q) * dFdq * dQdq + Sin(2 * Q) * ddFdqdq)) / 2.0 -
                        dQdf * (Cot(Q) * dQdq * ddFdqdf + 2 * Cot(Q) * dFdq * ddQdqdf + dddFdqdqdf) +
                        dFdf * (
                            -(Sin(2 * Q) * dFdq * ddFdqdf) - Cot(Q) * dQdq * ddQdqdf +
                            dQdf * (Power(Csc(Q), 2) * Power(dQdq, 2) - Cot(Q) * ddQdqdq) +
                            dddQdqdqdf
                        ) +
                        Power(Sin(Q), 2) * (
                            Cos(2 * Q) * Power(dFdq, 3) * dQdq +
                            3 * Cos(Q) * Sin(Q) * Power(dFdq, 2) * ddFdqdq +
                            dQdq * (2 * Cot(Q) * dQdq * ddFdqdq + dddFdqdqdq) -
                            dFdq * (2 * Power(Csc(Q), 2) * Power(dQdq, 3) - 4 * Cot(Q) * dQdq * ddQdqdq + dddQdqdqdq)
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq1212 =
                (
                    -2 *
                    Power(Sin(Q), 2) *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                    ) *
                    (
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf
                    ) +
                    2 *
                    Power(Sin(Q), 2) *
                    Power(
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf,
                        2,
                    ) +
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                    ) * (
                        Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) -
                    (
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf
                    ) * (
                        Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) +
                    (Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * ddFdfdf + dFdf * (2 * Cot(Q) * Power(dQdf, 2) - ddQdfdf)) * (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        Cot(Q) * Power(dQdf, 2) * dFdq +
                        dQdf * ddFdqdf +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf) -
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) -
                    (Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq + 2 * Cot(Q) * dFdf * dQdf * dQdq + ddFdfdf * dQdq) *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        Cot(Q) * Power(dQdf, 2) * dFdq +
                        dQdf * ddFdqdf +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf) -
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) +
                    (ddQdfdf * dFdq - ddFdfdf * dQdq + dQdf * ddFdqdf - dFdf * ddQdqdf) * (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        Cot(Q) * Power(dQdf, 2) * dFdq +
                        dQdf * ddFdqdf +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf) -
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) -
                    (dQdf * dFdq - dFdf * dQdq) * (
                        -(Power(Csc(Q), 2) * Power(dQdf, 3) * dFdq) + Cot(Q) * Power(dQdf, 2) * ddFdqdf + ddQdfdf * ddFdqdf -
                        3 * Cos(Q) * Power(Sin(Q), 2) * Sin(Q) * Power(dFdq, 2) * ddFdqdf -
                        2 * Cot(Q) * Power(Sin(Q), 2) * Power(dQdq, 2) * ddFdqdf +
                        Power(dFdf, 2) * (Cos(2 * Q) * dQdf * dFdq + Cos(Q) * Sin(Q) * ddFdqdf) - ddFdfdf * ddQdqdf -
                        4 * Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq * ddQdqdf +
                        dQdf * (
                            2 * Cot(Q) * ddQdfdf * dFdq - Cos(2 * Q) * Power(Sin(Q), 2) * Power(dFdq, 3) +
                            Cot(Q) * ddFdfdf * dQdq +
                            2 * Power(Csc(Q), 2) * Power(Sin(Q), 2) * dFdq * Power(dQdq, 2) +
                            dddFdqdfdf
                        ) +
                        dFdf * (
                            Sin(2 * Q) * ddFdfdf * dFdq - Power(Csc(Q), 2) * Power(dQdf, 2) * dQdq +
                            Cot(Q) * ddQdfdf * dQdq +
                            Cot(Q) * dQdf * ddQdqdf - dddQdqdfdf
                        ) - Power(Sin(Q), 2) * ddQdqdf * ddFdqdq + Power(Sin(Q), 2) * ddFdqdf * ddQdqdq -
                        Power(Sin(Q), 2) * dQdq * dddFdqdqdf + Power(Sin(Q), 2) * dFdq * dddQdqdqdf
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq1221 = -(
                (
                    -2 * Power(
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf),
                        2,
                    ) +
                    2 *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                    ) *
                    (
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf
                    ) +
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                    ) * (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                        dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) -
                    (
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf
                    ) * (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                        dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                        Power(Sin(Q), 2) *
                        (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                    ) +
                    (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 3)) +
                        dQdf * (-ddFdfdf + Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) +
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) +
                        Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) * (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) +
                    (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 3)) +
                        dQdf * (-ddFdfdf + Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) +
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) +
                        Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) * (-(dQdq * ddFdqdf) + dFdq * ddQdqdf + dQdf * ddFdqdq - dFdf * ddQdqdq) +
                    (
                        -(Cos(Q) * Sin(Q) * Power(dFdf, 3)) +
                        dQdf * (-ddFdfdf + Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) +
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) +
                        Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) * (-(Cos(Q) * Sin(Q) * Power(dFdq, 3)) - dQdq * ddFdqdq + dFdq * (-2 * Cot(Q) * Power(dQdq, 2) + ddQdqdq)) -
                    (dQdf * dFdq - dFdf * dQdq) * (
                        -(Power(Cos(Q), 2) * Power(dFdf, 3) * dQdq) + Power(Sin(Q), 2) * Power(dFdf, 3) * dQdq -
                        3 * Cos(Q) * Sin(Q) * Power(dFdf, 2) * ddFdqdf +
                        (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) * ddFdqdf +
                        (-ddFdfdf + Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) * ddQdqdf +
                        dQdf * (
                            -dddFdqdfdf +
                            Cot(Q) * Power(Sin(Q), 2) * dQdq * ddFdqdq +
                            Power(Sin(Q), 2) * dFdq * (-(Power(Csc(Q), 2) * Power(dQdq, 2)) + Cot(Q) * ddQdqdq)
                        ) +
                        dFdf * (
                            2 * Power(Csc(Q), 2) * Power(dQdf, 2) * dQdq - 4 * Cot(Q) * dQdf * ddQdqdf +
                            dddQdqdfdf +
                            Power(Sin(Q), 2) * (
                                Cos(2 * Q) * Power(dFdq, 2) * dQdq - Power(Csc(Q), 2) * Power(dQdq, 3) +
                                Sin(2 * Q) * dFdq * ddFdqdq +
                                2 * Cot(Q) * dQdq * ddQdqdq
                            )
                        ) +
                        Power(Sin(Q), 2) * (-(ddQdqdf * ddFdqdq) + ddFdqdf * ddQdqdq + dQdq * dddFdqdqdf - dFdq * dddQdqdqdf)
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)
            )

            DDq1222 =
                (
                    -2 *
                    Power(Sin(Q), 2) *
                    (Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq + 2 * Cot(Q) * dFdf * dQdf * dQdq + ddFdfdf * dQdq) *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                        dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                        dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                    ) +
                    2 *
                    Power(Sin(Q), 2) *
                    (Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq + 2 * Cot(Q) * dFdf * dQdf * dQdq + ddFdfdf * dQdq) *
                    (
                        Cot(Q) * dQdf * dFdq * dQdq +
                        dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                        dQdq * ddFdqdf - dFdq * ddQdqdf
                    ) +
                    2 *
                    (Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * ddFdfdf + dFdf * (2 * Cot(Q) * Power(dQdf, 2) - ddQdfdf)) *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) -
                    2 *
                    (Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq + 2 * Cot(Q) * dFdf * dQdf * dQdq + ddFdfdf * dQdq) *
                    (
                        Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) +
                    (ddQdfdf * dFdq - ddFdfdf * dQdq + dQdf * ddFdqdf - dFdf * ddQdqdf) * (
                        Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * (ddFdfdf - Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) -
                        dFdf * (
                            -2 * Cot(Q) * Power(dQdf, 2) +
                            ddQdfdf +
                            Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                        ) - Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                    ) -
                    (dQdf * dFdq - dFdf * dQdq) * (
                        Cos(2 * Q) * Power(dFdf, 3) * dQdf +
                        3 * Cos(Q) * Sin(Q) * Power(dFdf, 2) * ddFdfdf +
                        Power(dQdf, 2) * (2 * Cot(Q) * ddFdfdf + Power(Csc(Q), 2) * Power(Sin(Q), 2) * dFdq * dQdq) +
                        dQdf * (dddFdfdfdf - Cot(Q) * Power(Sin(Q), 2) * (dQdq * ddFdqdf + dFdq * ddQdqdf)) -
                        dFdf * (
                            2 * Power(Csc(Q), 2) * Power(dQdf, 3) +
                            dddQdfdfdf +
                            dQdf * (
                                -4 * Cot(Q) * ddQdfdf +
                                Power(Sin(Q), 2) * (Cos(2 * Q) * Power(dFdq, 2) - Power(Csc(Q), 2) * Power(dQdq, 2))
                            ) +
                            Power(Sin(Q), 2) * (Sin(2 * Q) * dFdq * ddFdqdf + 2 * Cot(Q) * dQdq * ddQdqdf)
                        ) -
                        Power(Sin(Q), 2) * (
                            Cot(Q) * ddQdfdf * dFdq * dQdq +
                            ddFdfdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * dddFdqdfdf - dFdq * dddQdqdfdf
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq2211 =
                (
                    2 * (
                        -(
                            (
                                Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                                dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                                dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                            ) * (
                                -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                                dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                                Power(Sin(Q), 2) *
                                (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                            )
                        ) +
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) * (
                            -(Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq) - Cot(Q) * Power(dQdf, 2) * dFdq - dQdf * ddFdqdf +
                            dFdf * (-(Cot(Q) * dQdf * dQdq) + ddQdqdf) +
                            Power(Sin(Q), 2) *
                            (Cos(Q) * Sin(Q) * Power(dFdq, 3) + dQdq * ddFdqdq + dFdq * (2 * Cot(Q) * Power(dQdq, 2) - ddQdqdq))
                        ) -
                        Power(Sin(Q), 2) *
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) *
                        (dQdf * (2 * Cot(Q) * dFdq * dQdq + ddFdqdq) + dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) - ddQdqdq)) -
                        Power(Sin(Q), 2) *
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) *
                        (-(dQdq * ddFdqdf) + dFdq * ddQdqdf + dQdf * ddFdqdq - dFdf * ddQdqdq) -
                        Power(Sin(Q), 2) *
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) *
                        (-(Cos(Q) * Sin(Q) * Power(dFdq, 3)) - dQdq * ddFdqdq + dFdq * (-2 * Cot(Q) * Power(dQdq, 2) + ddQdqdq)) +
                        Power(Sin(Q), 2) *
                        (dQdf * dFdq - dFdf * dQdq) *
                        (
                            Cos(Q) * Sin(Q) * Power(dFdq, 2) * ddFdqdf +
                            Cot(Q) * Power(dQdq, 2) * ddFdqdf +
                            Cot(Q) * dFdq * dQdq * ddQdqdf - ddQdqdf * ddFdqdq +
                            ddFdqdf * ddQdqdq +
                            dFdf * (
                                Cos(2 * Q) * Power(dFdq, 2) * dQdq - Power(Csc(Q), 2) * Power(dQdq, 3) +
                                Sin(2 * Q) * dFdq * ddFdqdq +
                                2 * Cot(Q) * dQdq * ddQdqdq
                            ) +
                            dQdf * (Cot(Q) * dQdq * ddFdqdq + dFdq * (-(Power(Csc(Q), 2) * Power(dQdq, 2)) + Cot(Q) * ddQdqdq)) +
                            dQdq * dddFdqdqdf - dFdq * dddQdqdqdf
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq2212 =
                (
                    2 *
                    Power(Sin(Q), 2) *
                    (
                        -(
                            (
                                Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                                2 * Cot(Q) * dFdf * dQdf * dQdq +
                                ddFdfdf * dQdq
                            ) * (
                                Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                                dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                                dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                            )
                        ) -
                        2 *
                        (Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * ddFdfdf + dFdf * (2 * Cot(Q) * Power(dQdf, 2) - ddQdfdf)) *
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) +
                        3 *
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                            2 * Cot(Q) * dFdf * dQdf * dQdq +
                            ddFdfdf * dQdq
                        ) *
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) -
                        (ddQdfdf * dFdq - ddFdfdf * dQdq + dQdf * ddFdqdf - dFdf * ddQdqdf) * (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) -
                        (dQdf * dFdq - dFdf * dQdq) * (
                            Power(Csc(Q), 2) * Power(dQdf, 2) * dFdq * dQdq - Cot(Q) * ddQdfdf * dFdq * dQdq -
                            ddFdfdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) -
                            Cot(Q) * dQdf * dQdq * ddFdqdf - Cot(Q) * dQdf * dFdq * ddQdqdf -
                            dFdf * (
                                dQdf * (Cos(2 * Q) * Power(dFdq, 2) - Power(Csc(Q), 2) * Power(dQdq, 2)) +
                                Sin(2 * Q) * dFdq * ddFdqdf +
                                2 * Cot(Q) * dQdq * ddQdqdf
                            ) - dQdq * dddFdqdfdf + dFdq * dddQdqdfdf
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq2221 =
                (
                    2 * (
                        -(
                            Power(Sin(Q), 2) *
                            (
                                Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                                dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                                dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                            ) *
                            (
                                Cot(Q) * dQdf * dFdq * dQdq +
                                dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                                dQdq * ddFdqdf - dFdq * ddQdqdf
                            )
                        ) +
                        Power(Sin(Q), 2) * Power(
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf,
                            2,
                        ) -
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq +
                            dQdf * (Cot(Q) * dQdf * dFdq + ddFdqdf) +
                            dFdf * (Cot(Q) * dQdf * dQdq - ddQdqdf)
                        ) * (
                            -(Cos(Q) * Sin(Q) * Power(dFdf, 3)) +
                            dQdf * (-ddFdfdf + Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) +
                            dFdf * (
                                -2 * Cot(Q) * Power(dQdf, 2) +
                                ddQdfdf +
                                Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                            ) +
                            Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                        ) +
                        (
                            Cot(Q) * dQdf * dFdq * dQdq +
                            dFdf * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2)) +
                            dQdq * ddFdqdf - dFdq * ddQdqdf
                        ) * (
                            -(Cos(Q) * Sin(Q) * Power(dFdf, 3)) +
                            dQdf * (-ddFdfdf + Cot(Q) * Power(Sin(Q), 2) * dFdq * dQdq) +
                            dFdf * (
                                -2 * Cot(Q) * Power(dQdf, 2) +
                                ddQdfdf +
                                Power(Sin(Q), 2) * (Cos(Q) * Sin(Q) * Power(dFdq, 2) + Cot(Q) * Power(dQdq, 2))
                            ) +
                            Power(Sin(Q), 2) * (dQdq * ddFdqdf - dFdq * ddQdqdf)
                        ) +
                        Power(Sin(Q), 2) *
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                            2 * Cot(Q) * dFdf * dQdf * dQdq +
                            ddFdfdf * dQdq
                        ) *
                        (dQdq * ddFdqdf - dFdq * ddQdqdf - dQdf * ddFdqdq + dFdf * ddQdqdq) +
                        Power(Sin(Q), 2) *
                        (dQdf * dFdq - dFdf * dQdq) *
                        (
                            2 * Cot(Q) * dQdf * dQdq * ddFdqdf + dQdq * dddFdqdfdf - dFdq * dddQdqdfdf - ddQdfdf * ddFdqdq +
                            Power(dFdf, 2) * (Cos(2 * Q) * dFdq * dQdq + Cos(Q) * Sin(Q) * ddFdqdq) +
                            ddFdfdf * ddQdqdq +
                            2 *
                            dFdf *
                            (
                                Cos(Q) * Sin(Q) * dFdq * ddFdqdf +
                                Cot(Q) * dQdq * ddQdqdf +
                                dQdf * (-(Power(Csc(Q), 2) * Power(dQdq, 2)) + Cot(Q) * ddQdqdq)
                            )
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDq2222 =
                (
                    2 *
                    Power(Sin(Q), 2) *
                    (
                        -3 *
                        (Cos(Q) * Sin(Q) * Power(dFdf, 3) + dQdf * ddFdfdf + dFdf * (2 * Cot(Q) * Power(dQdf, 2) - ddQdfdf)) *
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                            2 * Cot(Q) * dFdf * dQdf * dQdq +
                            ddFdfdf * dQdq
                        ) +
                        3 * Power(
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                            2 * Cot(Q) * dFdf * dQdf * dQdq +
                            ddFdfdf * dQdq,
                            2,
                        ) +
                        (
                            Cos(Q) * Sin(Q) * Power(dFdf, 2) * dFdq - ddQdfdf * dFdq +
                            2 * Cot(Q) * dFdf * dQdf * dQdq +
                            ddFdfdf * dQdq
                        ) * (-(ddQdfdf * dFdq) + ddFdfdf * dQdq - dQdf * ddFdqdf + dFdf * ddQdqdf) +
                        (dQdf * dFdq - dFdf * dQdq) * (
                            -(dddQdfdfdf * dFdq) + 2 * Cot(Q) * dQdf * ddFdfdf * dQdq + dddFdfdfdf * dQdq - ddQdfdf * ddFdqdf +
                            Power(dFdf, 2) * (Cos(2 * Q) * dQdf * dFdq + Cos(Q) * Sin(Q) * ddFdqdf) +
                            ddFdfdf * ddQdqdf +
                            2 *
                            dFdf *
                            (
                                Cos(Q) * Sin(Q) * ddFdfdf * dFdq - Power(Csc(Q), 2) * Power(dQdf, 2) * dQdq +
                                Cot(Q) * ddQdfdf * dQdq +
                                Cot(Q) * dQdf * ddQdqdf
                            )
                        )
                    )
                ) / Power(dQdf * dFdq - dFdf * dQdq, 2)

            DDqθθθθ = DDq1111
            DDqθθθϕ = DDq1112
            DDqθθϕθ = DDq1121
            DDqθθϕϕ = DDq1122

            DDqθϕθθ = DDq1211
            DDqθϕθϕ = DDq1212
            DDqθϕϕθ = DDq1221
            DDqθϕϕϕ = DDq1222

            DDqϕϕθθ = DDq2211
            DDqϕϕθϕ = DDq2212
            DDqϕϕϕθ = DDq2221
            DDqϕϕϕϕ = DDq2222

            DDqθθθθ /= 1
            DDqθθθϕ /= sin(θ)
            DDqθθϕθ /= sin(θ)
            DDqθθϕϕ /= sin(θ)^2

            DDqθϕθθ /= sin(θ)
            DDqθϕθϕ /= sin(θ)^2
            DDqθϕϕθ /= sin(θ)^2
            DDqθϕϕϕ /= sin(θ)^3

            DDqϕϕθθ /= sin(θ)^2
            DDqϕϕθϕ /= sin(θ)^3
            DDqϕϕϕθ /= sin(θ)^3
            DDqϕϕϕϕ /= sin(θ)^4

            SArray{Tuple{2,2,2,2}}(
                DDqθθθθ,
                DDqθϕθθ,
                DDqθϕθθ,
                DDqϕϕθθ,
                DDqθθθϕ,
                DDqθϕθϕ,
                DDqθϕθϕ,
                DDqϕϕθϕ,
                DDqθθθϕ,
                DDqθϕθϕ,
                DDqθϕθϕ,
                DDqϕϕθϕ,
                DDqθθϕϕ,
                DDqθϕϕϕ,
                DDqθϕϕϕ,
                DDqϕϕϕϕ,
            )
        end for ij in CartesianIndices(g.values)
    ],
    lmax,
)
if !(norm(DDq.values - DDq₀.values, Inf) ≤ atol)
    for a in 1:2, b in a:2, c in 1:2, d in 1:2
        println("DDq₀[$a,$b,$c,$d]: ", DDq₀.values[1, 1][a, b, c, d])
        println("DDq[$a,$b,$c,$d]: ", DDq.values[1, 1][a, b, c, d])
        println("|(DDq-DDq₀)[$a,$b,$c,$d]|∞: ", norm(map(x -> x[a, b, c, d], DDq.values - DDq₀.values), Inf))
    end
    println("atol: ", atol)
end
@assert norm(DDq.values - DDq₀.values, Inf) ≤ atol

#     (D_a D_b - D_b D_a) q_cd = g^ef (R_abcf q_ed + R_abdf q_ce)
#                              = g^ef (R_abxf δ^x_c q_ed + R_abxf δ^x_d q_ce)
#                              = g^ef (δ^x_c q_ed + δ^x_d q_ce) R_abxf
#                              = g^ef g^xy (g_yc q_ed + g_yd q_ce) R_abxf
#     A[c,d,x,f] := g^ef (δ^x_c q_ed + δ^x_d q_ce)
#     B[a,b,c,d] := (D_a D_b - D_b D_a) q_cd
#     B[a,b,c,d] = A[c,d,x,f] Rm[a,b,x,f]
#
# <https://en.wikipedia.org/wiki/Riemann_curvature_tensor> states:
#     (D_a D_b - D_b D_a) q_cd = - R^e_cab q_ed - R^e_dab q_ce
#                              = - g^ef (R_fcab q_ed + R_fdab q_ce)
#                              = g^ef (R_abcf q_ed + R_abdf q_ce)
# This confirms (25) above.

#     (D_a D_b - D_b D_a) q_cd = g^ef (R_abcf q_ed + R_abdf q_ce)
#
#     q_ab: round sphere metric
#     Γ^c_ab = q^cd (∂_a q_db + ∂_b q_ad - ∂_d q_ab) / 2
#
#     g_ab: physical metric
#     C^c_ab = g^cd (ð_a g_db + ð_b g_ad - ð_d g_ab) / 2
#
#     ð_b X_a = ∂_b X_a - Γ^c_ab X_c   implemented by ð
#     D_b X_a = ∂_b X_a - C^c_ab X_c
#             = ð_b X_a + Γ^c_ab X_c - C^c_ab X_c

function make_Rm(ij::CartesianIndex{2}, gu::SMatrix{2,2,T}, q::SMatrix{2,2,T}, DDq::SArray{Tuple{2,2,2,2},T}) where {T}
    A = SArray{Tuple{2,2,2,2}}(
        sum(gu[e, f] * ((x == c) * q[e, d] + (x == d) * q[c, e]) for e in 1:2) for c in 1:2, d in 1:2, x in 1:2, f in 1:2
    )
    # thistime = ij[1] == lmax ÷ 2 && ij[2] == 1
    # thistime = ij[1] == 1 && ij[2] == 1
    thistime = false
    if thistime
        for c in 1:2, d in 1:2, x in 1:2, f in 1:2
            if chop(real(A[c, d, x, f])) ≠ 0
                println("A[$c,$d,$x,$f]: ", chop(real(A[c, d, x, f])))
            end
        end
    end
    @assert all(A[c, d, x, f] ≈ A[d, c, x, f] for c in 1:2, d in 1:2, x in 1:2, f in 1:2)
    # @assert all(A[c, d, x, f] ≈ -A[c, d, f, x] for c in 1:2, d in 1:2, x in 1:2, f in 1:2)
    Rm = zero(MArray{Tuple{2,2,2,2},T})
    for a in 1:2, b in 1:2
        if thistime
            println("[$a,$b]:")
        end
        B = SMatrix{2,2}(DDq[c, d, b, a] - DDq[c, d, a, b] for c in 1:2, d in 1:2)
        if thistime
            for c in 1:2, d in 1:2
                if chop(real(B[c, d])) ≠ 0
                    println("B[$a,$b,$c,$d]: ", chop(real(B[c, d])))
                end
            end
        end
        @assert all(B[c, d] ≈ B[d, c] for c in 1:2, d in 1:2)
        A′ = reshape(A, (4, 4))
        B′ = reshape(B, 4)
        # Rm1′ = A′ \ B′
        A″ = A′[:, SVector(2, 3)]
        B″ = B′
        Rm1″ = A″ \ B″
        # Rm1′ = SVector(0, Rm1″[1], Rm1″[2], 0)
        Rm1′ = SVector(0, (Rm1″[1] - Rm1″[2]) / 2, (Rm1″[2] - Rm1″[1]) / 2, 0)
        Rm1 = reshape(Rm1′, (2, 2))
        if thistime
            for x in 1:2, f in 1:2
                if chop(real(Rm1[x, f])) ≠ 0
                    println("Rm1[$a,$b,$x,$f]: ", chop(real(Rm1[x, f])))
                end
            end
        end

        BB = SMatrix{2,2}(sum(A[c, d, x, f] * Rm1[x, f] for x in 1:2, f in 1:2) for c in 1:2, d in 1:2)
        if thistime
            for c in 1:2, d in 1:2
                if chop(real(BB[c, d])) ≠ 0
                    println("BB[$a,$b,$c,$d]: ", chop(real(BB[c, d])))
                end
            end
        end

        #DISABLED @assert isapprox(B, BB; atol=atol)
        for x in 1:2, f in 1:2
            Rm[a, b, x, f] = Rm1[x, f]
        end
    end
    # TODO: Why is this sign necessary?
    return -SArray(Rm)
end

Rm = Tensor{4}(
    [make_Rm(ij, gu, q, DDq) for (ij, gu, q, DDq) in zip(CartesianIndices(gu.values), gu.values, q.values, DDq.values)], lmax
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
            J = Jac(x)

            q, f = x
            dQdq = J[1, 1]
            dQdf = J[1, 2] * sin(q)
            dFdq = J[2, 1] / sin(Q)
            dFdf = J[2, 2] / sin(Q) * sin(q)

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
if !(norm(Rm.values - Rm₀.values, Inf) ≤ atol)
    for a in 1:2, b in (a + 1):2, c in 1:2, d in (c + 1):2
        if (a, b) ≤ (c, d)
            println("Rm₀[$a,$b,$c,$d]: ", Rm₀.values[1, 1][a, b, c, d])
            println("Rm[$a,$b,$c,$d]: ", Rm.values[1, 1][a, b, c, d])
            println("|(Rm-Rm₀)[$a,$b,$c,$d]|∞: ", norm(map(x -> x[a, b, c, d], Rm.values - Rm₀.values), Inf))
        end
    end
    println("atol: ", atol)
end
@assert norm(Rm.values - Rm₀.values, Inf) ≤ atol

# Ricci
Rc = Tensor{2}(
    [
        SMatrix{2,2}(sum(gu[c, d] * Rm[c, a, b, d] for c in 1:2, d in 1:2) for a in 1:2, b in 1:2) for
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
        J = Jac(x)

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
