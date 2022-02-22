function kerr_schild(M, a, Q, x::SVector{4})
    # Cook, section 3.3.1:

    t, r, θ, ϕ = x

    # (79)
    ρ = sqrt(r^2 + a^2 * cos(θ)^2)

    # (86)
    α = 1 / sqrt(1 + (2 * M * r - Q^2) / ρ^2)
    # (87)
    βr = α^2 * (2 * M * r - Q^2) / ρ^2
    # (88)
    γrr = 1 + (2 * M * r - Q^2) / ρ^2
    # (89)
    γrϕ = -(1 + (2 * M * r - Q^2) / ρ^2) * a * sin(θ)^2
    # (90)
    γθθ = ρ^2
    # (91)
    γϕϕ = (r^2 + a^2 + (2 * M * r - Q^2) / ρ^2 * a^2 * sin(θ)^2) * sin(θ)^2

    # below (83)
    r_EH = M + sqrt(M^2 - a^2 - Q^2)
    r_CH = M - sqrt(M^2 - a^2 - Q^2)

    β = SVector(βr, 0, 0)
    γ = SMatrix{3,3}((γrr, 0, γrϕ)..., (0, γθθ, 0)..., (γrϕ, 0, γϕϕ)...)

    g00 = -α^2 + β' * γ * β
    g0i = γ * β
    gij = γ

    # sq = sin(θ)
    sq = 1

    g = SMatrix{4,4}(
        (g00, g0i[1], g0i[2], g0i[3] / sq)...,
        (g0i[1], gij[1, 1], gij[1, 2], gij[1, 3] / sq)...,
        (g0i[2], gij[2, 1], gij[2, 2], gij[2, 3] / sq)...,
        (g0i[3] / sq, gij[3, 1] / sq, gij[3, 2] / sq, gij[3, 3] / sq^2)...,
    )

    return g
end

function null_normals(M, a, Q, x::SVector{4}; debug=false)
    g = kerr_schild(M, a, Q, x)
    debug && @show chop.(eigvals(g))
    gu = inv(g)
    debug && @show chop.(eigvals(gu))
    # timelike normal to t=const hypersurface
    n = SVector(-1, 0, 0, 0)
    n /= sqrt(-n' * gu * n)
    debug && @show chop.(n)
    n2 = n' * gu * n
    debug && @show chop(n2)
    # Tangent vectors to surface
    eϕ = SVector(0, 0, 0, 1)
    eϕ /= (eϕ' * g * eϕ)
    debug && @show chop.(eϕ)
    eθ = SVector(0, 0, 1, 0)
    # eθ /= (eθ' * g * eθ)
    eθ -= (eθ' * g * eϕ) * eϕ
    eθ /= (eθ' * g * eθ)
    debug && @show chop.(eθ)
    # spacelike normal to r_EH=const hypersurface, within t=const hypersurface
    s = SVector(0, 1, 0, 0)
    s += (s' * gu * n) * n
    s /= sqrt(s' * gu * s)
    debug && @show chop.(s)
    s2 = s' * gu * s
    sn = s' * gu * n
    seθ = s' * eθ
    seϕ = s' * eϕ
    debug && @show chop(s2) chop(sn) chop(seθ) chop(seϕ)
    # outgoing and ingoing null normals
    l = (n + s) / sqrt(2)
    k = (n - s) / sqrt(2)
    debug && @show chop.(l) chop.(k)
    debug && @show chop.(gu * l) chop.(gu * k)
    l2 = l' * gu * l
    k2 = k' * gu * k
    lk = l' * gu * k
    debug && @show chop(l2) chop(k2) chop(lk)
    return g, l, k
end

function kerr_schild_mma(M, a, Q, x::SVector{4})
    # t, r, θ, ϕ = x
    _, r, θ, _ = x

    Cos = cos
    Cot = cot
    Csc = csc
    Power = (^)
    Sin = sin
    Sqrt = sqrt

    # The Q-related terms below are all wrong, but they are correct for Q=0
    @assert Q == 0

    g11 =
        -(1 / (1 + (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)))) +
        Power(-Q + 2 * M * r, 2) / (
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
            (1 + (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)))
        )
    g12 = (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
    g13 = 0
    g14 =
        (a * (-Q + 2 * M * r) * (-1 - (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) * Power(Sin(θ), 2)) /
        ((Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * (1 + (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))))

    g22 = 1 + (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
    g23 = 0
    g24 = a * (-1 - (-Q + 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) * Power(Sin(θ), 2)

    g33 = Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)
    g34 = 0

    g44 =
        Power(Sin(θ), 2) * (
            Power(a, 2) +
            Power(r, 2) +
            (Power(a, 2) * (-Q + 2 * M * r) * Power(Sin(θ), 2)) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
        )

    g = SMatrix{4,4}((g11, g12, g13, g14)..., (g12, g22, g23, g24)..., (g13, g23, g33, g34)..., (g14, g24, g34, g44)...)

    n1 = 1 / Sqrt((-Q + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)))

    n = SVector{4}(n1, 0, 0, 0)

    s1 =
        (Q - 2 * M * r) / (
            (Q - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                (
                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Q - 2 * M * r) * Power(Sin(θ), 2)
                ) / (
                    (Q - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                ),
            )
        )
    s2 =
        1 / Sqrt(
            (
                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                Power(a, 2) * (Q - 2 * M * r) * Power(Sin(θ), 2)
            ) /
            ((Q - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))),
        )

    s = SVector{4}(s1, s2, 0, 0)

    l = (n + s) / sqrt(2)
    k = (n - s) / sqrt(2)

    dl11 = 0
    dl12 =
        (
            (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) / (
                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5)
            ) -
            (2 * M) / (
                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                    (
                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ),
                )
            ) -
            (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                    (
                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ),
                )
            ) +
            (
                (Power(Q, 2) - 2 * M * r) * (
                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                    Power(Cos(θ), 2) * (
                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                        Power(a, 6) * M * Power(Sin(θ), 4)
                    ) +
                    r * (
                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                        Power(a, 2) *
                        (
                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                        ) *
                        Power(Sin(θ), 2) +
                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                    )
                )
            ) / (
                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                Power(
                    (
                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                    1.5,
                )
            )
        ) / Sqrt(2)
    dl13 =
        (
            Power(a, 2) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Sin(θ) *
            (
                1 / (
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5)
                ) -
                2 / (
                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                ) -
                (
                    (Power(a, 2) + Power(r, 2)) * (
                        Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                        Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                    ) +
                    2 * Power(a, 2) * (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) * Power(Cos(θ), 2) +
                    Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                    2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                    Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
                ) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                    Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                    Power(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                        1.5,
                    )
                )
            )
        ) / Sqrt(2)
    dl14 = 0

    dl21 = 0
    dl22 =
        (
            Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
            Power(Cos(θ), 2) * (
                Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                Power(a, 6) * M * Power(Sin(θ), 4)
            ) +
            r * (
                (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                Power(a, 2) *
                (
                    Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                    Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                ) *
                Power(Sin(θ), 2) +
                Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
            )
        ) / (
            Sqrt(2) *
            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            ) *
            (
                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
            )
        )
    dl23 =
        (
            Power(a, 2) *
            Cos(θ) *
            Sin(θ) *
            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) *
            (
                (Power(a, 2) + Power(r, 2)) * (
                    Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                    Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                ) +
                2 * Power(a, 2) * (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) * Power(Cos(θ), 2) +
                Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
            )
        ) / (
            Sqrt(2) *
            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
            Power(
                (
                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) + Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    )
                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                1.5,
            )
        )
    dl24 = 0

    dl31 = 0
    dl32 = 0
    dl33 = 0
    dl34 = 0

    dl41 = 0
    dl42 = 0
    dl43 = 0
    dl44 = 0

    dl = SMatrix{4,4}(
        (dl11, dl21, dl31, dl41)..., (dl12, dl22, dl32, dl42)..., (dl13, dl23, dl33, dl43)..., (dl14, dl24, dl34, dl44)...
    )

    Dl11 =
        (
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) -
                (Power(Q, 2) - 2 * M * r) * (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
    Dl12 =
        (
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                Power(a, 2) * (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * Power(Sin(θ), 2)
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            )
        ) -
        (
            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
        (
            (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) / (
                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5)
            ) -
            (2 * M) / (
                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                    (
                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ),
                )
            ) -
            (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                    (
                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ),
                )
            ) +
            (
                (Power(Q, 2) - 2 * M * r) * (
                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                    Power(Cos(θ), 2) * (
                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                        Power(a, 6) * M * Power(Sin(θ), 4)
                    ) +
                    r * (
                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                        Power(a, 2) *
                        (
                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                        ) *
                        Power(Sin(θ), 2) +
                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                    )
                )
            ) / (
                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                Power(
                    (
                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                    1.5,
                )
            )
        ) / Sqrt(2)
    Dl13 =
        -(
            (Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Cos(θ) * Sin(θ) * (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))) / (
                Sqrt(2) *
                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                Sqrt(
                    -(
                        (
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                    ),
                )
            )
        ) -
        (
            Power(a, 2) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Sin(θ) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)) +
        (
            Power(a, 2) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Sin(θ) *
            (
                1 / (
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5)
                ) -
                2 / (
                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                ) -
                (
                    (Power(a, 2) + Power(r, 2)) * (
                        Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                        Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                    ) +
                    2 * Power(a, 2) * (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) * Power(Cos(θ), 2) +
                    Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                    2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                    Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
                ) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                    Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                    Power(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                        1.5,
                    )
                )
            )
        ) / Sqrt(2)
    Dl14 =
        (
            a *
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
            Power(Sin(θ), 2) *
            (
                (
                    -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) * (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))

    Dl21 =
        (
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                (
                    (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                    Power(a, 2) * (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) -
                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
    Dl22 =
        (
            (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                Power(Cos(θ), 2) * (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            )
        ) -
        (
            (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
        (
            Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
            Power(Cos(θ), 2) * (
                Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                Power(a, 6) * M * Power(Sin(θ), 4)
            ) +
            r * (
                (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                Power(a, 2) *
                (
                    Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                    Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                ) *
                Power(Sin(θ), 2) +
                Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
            )
        ) / (
            Sqrt(2) *
            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            ) *
            (
                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
            )
        )
    Dl23 =
        (
            Power(a, 2) *
            Cos(θ) *
            Sin(θ) *
            (
                Power(a, 2) * Power(Q, 2) - 2 * Power(a, 2) * M * r +
                Power(r, 4) +
                Power(a, 2) * (-Power(Q, 2) + 2 * r * (M + r)) * Power(Cos(θ), 2) +
                Power(a, 4) * Power(Cos(θ), 4) - Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            )
        ) +
        (
            Power(a, 2) *
            Cos(θ) *
            Sin(θ) *
            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) *
            (
                (Power(a, 2) + Power(r, 2)) * (
                    Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                    Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                ) +
                2 * Power(a, 2) * (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) * Power(Cos(θ), 2) +
                Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
            )
        ) / (
            Sqrt(2) *
            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
            Power(
                (
                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) + Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    )
                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                1.5,
            )
        ) -
        (
            Power(a, 2) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Sin(θ) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))
    Dl24 =
        -(
            (
                a *
                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                Power(Sin(θ), 2) *
                (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
        ) -
        (
            a *
            Power(Sin(θ), 2) *
            (
                Power(a, 6) * r * Power(Cos(θ), 6) +
                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                Power(a, 2) *
                Power(Cos(θ), 2) *
                (
                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                ) +
                r * (
                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                    Power(r, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                    Power(a, 2) * (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) * Power(Sin(θ), 2)
                )
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            )
        )

    Dl31 =
        (
            Power(a, 2) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Sin(θ) *
            (
                -(
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) -
                (Power(a, 2) * (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (-Power(Q, 2) + 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))
    Dl32 =
        (
            Power(a, 2) *
            Cos(θ) *
            Sin(θ) *
            (
                Power(a, 2) * Power(Q, 2) - 2 * Power(a, 2) * M * r +
                Power(r, 4) +
                Power(a, 2) * (-Power(Q, 2) + 2 * r * (M + r)) * Power(Cos(θ), 2) +
                Power(a, 4) * Power(Cos(θ), 4) - Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            )
        ) -
        (
            Power(a, 2) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Sin(θ) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))
    Dl33 =
        (
            r * (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) -
                (Power(Q, 2) - 2 * M * r) * (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)))
    Dl34 =
        (
            Power(a, 3) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Power(Sin(θ), 3) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(a, 2) * (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))

    Dl41 =
        (
            a *
            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
            Power(Sin(θ), 2) *
            (
                (
                    -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) * (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
    Dl42 =
        -(
            (
                a *
                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                Power(Sin(θ), 2) *
                (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
        ) -
        (
            a *
            Power(Sin(θ), 2) *
            (
                Power(a, 6) * r * Power(Cos(θ), 6) +
                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                Power(a, 2) *
                Power(Cos(θ), 2) *
                (
                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                ) +
                r * (
                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                    Power(r, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                    Power(a, 2) * (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) * Power(Sin(θ), 2)
                )
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
            Sqrt(
                -(
                    (
                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                ),
            )
        )
    Dl43 =
        (
            Power(a, 3) *
            (Power(Q, 2) - 2 * M * r) *
            Cos(θ) *
            Power(Sin(θ), 3) *
            (
                1 / Sqrt(
                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(a, 2) * (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) +
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))
    Dl44 =
        (
            Power(Sin(θ), 2) *
            (
                Power(r, 5) +
                Power(a, 4) * r * Power(Cos(θ), 4) +
                Power(a, 2) * r * (Power(Q, 2) - M * r) * Power(Sin(θ), 2) +
                Power(Cos(θ), 2) * (2 * Power(a, 2) * Power(r, 3) + Power(a, 4) * M * Power(Sin(θ), 2))
            ) *
            (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) -
                (Power(Q, 2) - 2 * M * r) * (
                    1 / Sqrt(
                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) /
                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))

    Dl = SMatrix{4,4}(
        (Dl11, Dl21, Dl31, Dl41)..., (Dl12, Dl22, Dl32, Dl42)..., (Dl13, Dl23, Dl33, Dl43)..., (Dl14, Dl24, Dl34, Dl44)...
    )

    ω1 =
        (
            (
                a * (
                    (
                        Power(Sin(θ), 2) *
                        (
                            Power(r, 5) +
                            Power(a, 4) * r * Power(Cos(θ), 4) +
                            Power(a, 2) * r * (Power(Q, 2) - M * r) * Power(Sin(θ), 2) +
                            Power(Cos(θ), 2) * (2 * Power(a, 2) * Power(r, 3) + Power(a, 4) * M * Power(Sin(θ), 2))
                        ) *
                        (
                            (a * (Power(Q, 2) - 2 * M * r) * Power(Csc(θ), 2)) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                            ) -
                            (
                                Power(a, 3) *
                                (Power(Q, 2) - 2 * M * r) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        a *
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        Power(Sin(θ), 2) *
                        (
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) +
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) +
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) -
                        (Power(a, 2) * Power(Power(Q, 2) - 2 * M * r, 3) * Power(Sin(θ), 2)) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) -
                        (
                            Power(a, 2) *
                            (Power(Q, 2) - 2 * M * r) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        -(
                            (
                                a *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                        ) -
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 6) * r * Power(Cos(θ), 6) +
                                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                Power(a, 2) *
                                Power(Cos(θ), 2) *
                                (
                                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                                ) +
                                r * (
                                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                    Power(r, 2) *
                                    (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                    Power(a, 2) *
                                    (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                    Power(Sin(θ), 2)
                                )
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    )
                )
            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
            (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) * (
                    (
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        (
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) +
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) *
                                (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (a * (Power(Q, 2) - 2 * M * r) * Power(Csc(θ), 2)) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                        ) -
                        (
                            Power(a, 3) *
                            (Power(Q, 2) - 2 * M * r) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2)
                        ) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        -(
                            (
                                a *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                        ) -
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 6) * r * Power(Cos(θ), 6) +
                                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                Power(a, 2) *
                                Power(Cos(θ), 2) *
                                (
                                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                                ) +
                                r * (
                                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                    Power(r, 2) *
                                    (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                    Power(a, 2) *
                                    (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                    Power(Sin(θ), 2)
                                )
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    ) +
                    (
                        (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) -
                        (Power(a, 2) * Power(Power(Q, 2) - 2 * M * r, 3) * Power(Sin(θ), 2)) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) -
                        (
                            Power(a, 2) *
                            (Power(Q, 2) - 2 * M * r) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        (
                            (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                                Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                                Power(Cos(θ), 2) * (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        ) -
                        (
                            (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                            Power(Cos(θ), 2) * (
                                Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                Power(a, 6) * M * Power(Sin(θ), 4)
                            ) +
                            r * (
                                (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                Power(a, 2) *
                                (
                                    Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                    Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                ) *
                                Power(Sin(θ), 2) +
                                Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                            )
                        ) / (
                            Sqrt(2) *
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            ) *
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    )
                )
            ) / ((Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))) -
            (
                (Power(Q, 2) - 2 * M * r) * (
                    (
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        (
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) +
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        a *
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        Power(Sin(θ), 2) *
                        (
                            (a * (Power(Q, 2) - 2 * M * r) * Power(Csc(θ), 2)) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                            ) -
                            (
                                Power(a, 3) *
                                (Power(Q, 2) - 2 * M * r) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) +
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) -
                        (Power(a, 2) * Power(Power(Q, 2) - 2 * M * r, 3) * Power(Sin(θ), 2)) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) -
                        (
                            Power(a, 2) *
                            (Power(Q, 2) - 2 * M * r) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        (
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) *
                                (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        ) -
                        (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            ((M + r) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) /
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5) -
                            r / Sqrt(
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (2 * M) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) -
                            (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) +
                            (
                                (Power(Q, 2) - 2 * M * r) * (
                                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                    Power(Cos(θ), 2) * (
                                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                        Power(a, 6) * M * Power(Sin(θ), 4)
                                    ) +
                                    r * (
                                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                        Power(a, 2) *
                                        (
                                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                        ) *
                                        Power(Sin(θ), 2) +
                                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                    )
                                )
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                Power(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    1.5,
                                )
                            )
                        ) / Sqrt(2)
                    )
                )
            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
        ) / (
            Sqrt(2) * Sqrt(
                (
                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                ),
            )
        ) -
        (
            (
                -(
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) -
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            ) * (
                -(
                    (
                        (Power(Q, 2) - 2 * M * r) * (
                            (
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                (
                                    (
                                        Power(a, 2) *
                                        Power(Power(Q, 2) - 2 * M * r, 2) *
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / (
                                        Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) +
                                    (
                                        Power(a, 2) *
                                        Power(Power(Q, 2) - 2 * M * r, 2) *
                                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / (
                                        Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    )
                                ) *
                                (
                                    (
                                        (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) *
                                        (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / Sqrt(
                                        (
                                            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    ) -
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                                        -(
                                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                        ) +
                                        (Power(Q, 2) - 2 * M * r) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                (
                                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                ) / (
                                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                ),
                                            )
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                            (
                                (a * (Power(Q, 2) - 2 * M * r) * Power(Csc(θ), 2)) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                                ) -
                                (
                                    Power(a, 3) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) * (
                                -(
                                    (
                                        a *
                                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2) *
                                        (
                                            -(
                                                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                                Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                            ) +
                                            (Power(Q, 2) - 2 * M * r) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                    (
                                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                    ) / (
                                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                    ),
                                                )
                                            )
                                        )
                                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                                ) -
                                (
                                    a *
                                    Power(Sin(θ), 2) *
                                    (
                                        Power(a, 6) * r * Power(Cos(θ), 6) +
                                        Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                        Power(a, 2) *
                                        Power(Cos(θ), 2) *
                                        (
                                            Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                            Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                            Power(a, 2) *
                                            (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) *
                                            Power(Sin(θ), 2)
                                        ) +
                                        r * (
                                            Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                            Power(r, 2) *
                                            (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                            Power(a, 2) *
                                            (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                            Power(Sin(θ), 2)
                                        )
                                    )
                                ) / (
                                    Sqrt(2) *
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    )
                                )
                            ) +
                            (
                                (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                ) -
                                (Power(a, 2) * Power(Power(Q, 2) - 2 * M * r, 3) * Power(Sin(θ), 2)) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) -
                                (
                                    Power(a, 2) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2) *
                                    (
                                        (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) * (
                                (
                                    (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                        (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                                        Power(Cos(θ), 2) *
                                        (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                                    )
                                ) / (
                                    Sqrt(2) *
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    )
                                ) -
                                (
                                    (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                                    (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                    (
                                        -(
                                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                        ) +
                                        (Power(Q, 2) - 2 * M * r) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                (
                                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                ) / (
                                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                ),
                                            )
                                        )
                                    )
                                ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                                (
                                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                    Power(Cos(θ), 2) * (
                                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                        Power(a, 6) * M * Power(Sin(θ), 4)
                                    ) +
                                    r * (
                                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                        Power(a, 2) *
                                        (
                                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                        ) *
                                        Power(Sin(θ), 2) +
                                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                    )
                                ) / (
                                    Sqrt(2) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    ) *
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            )
                        )
                    ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) +
                (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                        (
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                (
                                    Power(a, 2) *
                                    Power(Power(Q, 2) - 2 * M * r, 2) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) +
                                (
                                    Power(a, 2) *
                                    Power(Power(Q, 2) - 2 * M * r, 2) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) *
                            (
                                (
                                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / Sqrt(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                ) -
                                (Power(Q, 2) - 2 * M * r) * (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            a *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2) *
                            (
                                (a * (Power(Q, 2) - 2 * M * r) * Power(Csc(θ), 2)) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                                ) -
                                (
                                    Power(a, 3) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) *
                            (
                                (
                                    -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / Sqrt(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                ) +
                                (Power(Q, 2) - 2 * M * r) * (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ) -
                            (Power(a, 2) * Power(Power(Q, 2) - 2 * M * r, 3) * Power(Sin(θ), 2)) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) -
                            (
                                Power(a, 2) *
                                (Power(Q, 2) - 2 * M * r) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) * (
                            (
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                    (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) *
                                    (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                )
                            ) / (
                                Sqrt(2) *
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                Sqrt(
                                    -(
                                        (
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                    ),
                                )
                            ) -
                            (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                            (
                                ((M + r) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) /
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5) -
                                r / Sqrt(
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                                ) -
                                (2 * M) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                ) -
                                (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                ) +
                                (
                                    (Power(Q, 2) - 2 * M * r) * (
                                        Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                        Power(Cos(θ), 2) * (
                                            Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                            2 *
                                            (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) *
                                            Power(Sin(θ), 2) + Power(a, 6) * M * Power(Sin(θ), 4)
                                        ) +
                                        r * (
                                            (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                            Power(a, 2) *
                                            (
                                                Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) -
                                                2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                                Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                            ) *
                                            Power(Sin(θ), 2) +
                                            Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                        )
                                    )
                                ) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    Power(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                        1.5,
                                    )
                                )
                            ) / Sqrt(2)
                        )
                    )
                ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
            )
        ) / Sqrt(2)
    ω2 =
        (
            (
                a * (
                    (
                        Power(Sin(θ), 2) *
                        (
                            Power(r, 5) +
                            Power(a, 4) * r * Power(Cos(θ), 4) +
                            Power(a, 2) * r * (Power(Q, 2) - M * r) * Power(Sin(θ), 2) +
                            Power(Cos(θ), 2) * (2 * Power(a, 2) * Power(r, 3) + Power(a, 4) * M * Power(Sin(θ), 2))
                        ) *
                        (
                            (
                                a *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Csc(θ), 2)
                            ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2)) +
                            (
                                Power(a, 3) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(Sin(θ), 2)
                            ) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        a *
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        Power(Sin(θ), 2) *
                        (
                            -(
                                (
                                    Power(a, 2) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) -
                            (
                                Power(a, 2) *
                                (Power(Q, 2) - 2 * M * r) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) +
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (
                            Power(a, 2) *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2)
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                        (
                            Power(a, 2) *
                            Power(Power(Q, 2) - 2 * M * r, 2) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2)
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) +
                        (
                            Power(a, 2) *
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        -(
                            (
                                a *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                        ) -
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 6) * r * Power(Cos(θ), 6) +
                                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                Power(a, 2) *
                                Power(Cos(θ), 2) *
                                (
                                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                                ) +
                                r * (
                                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                    Power(r, 2) *
                                    (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                    Power(a, 2) *
                                    (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                    Power(Sin(θ), 2)
                                )
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    )
                )
            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
            (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) * (
                    (
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        (
                            -(
                                (
                                    Power(a, 2) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) -
                            (
                                Power(a, 2) *
                                (Power(Q, 2) - 2 * M * r) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) *
                                (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (a * (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) * Power(Csc(θ), 2)) /
                        (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2)) +
                        (
                            Power(a, 3) *
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            Power(Sin(θ), 2)
                        ) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        -(
                            (
                                a *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                        ) -
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 6) * r * Power(Cos(θ), 6) +
                                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                Power(a, 2) *
                                Power(Cos(θ), 2) *
                                (
                                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                                ) +
                                r * (
                                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                    Power(r, 2) *
                                    (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                    Power(a, 2) *
                                    (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                    Power(Sin(θ), 2)
                                )
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    ) +
                    (
                        (
                            Power(a, 2) *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2)
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                        (
                            Power(a, 2) *
                            Power(Power(Q, 2) - 2 * M * r, 2) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2)
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) +
                        (
                            Power(a, 2) *
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        (
                            (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                                Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                                Power(Cos(θ), 2) * (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        ) -
                        (
                            (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                            Power(Cos(θ), 2) * (
                                Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                Power(a, 6) * M * Power(Sin(θ), 4)
                            ) +
                            r * (
                                (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                Power(a, 2) *
                                (
                                    Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                    Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                ) *
                                Power(Sin(θ), 2) +
                                Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                            )
                        ) / (
                            Sqrt(2) *
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            ) *
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    )
                )
            ) / ((Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))) -
            (
                (Power(Q, 2) - 2 * M * r) * (
                    (
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        (
                            -(
                                (
                                    Power(a, 2) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) -
                            (
                                Power(a, 2) *
                                (Power(Q, 2) - 2 * M * r) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        a *
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        Power(Sin(θ), 2) *
                        (
                            (
                                a *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Csc(θ), 2)
                            ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2)) +
                            (
                                Power(a, 3) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(Sin(θ), 2)
                            ) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) *
                        (
                            (
                                -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) +
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (
                            Power(a, 2) *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2)
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                        (
                            Power(a, 2) *
                            Power(Power(Q, 2) - 2 * M * r, 2) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2)
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) +
                        (
                            Power(a, 2) *
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    ) * (
                        (
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) *
                                (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        ) -
                        (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            ((M + r) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) /
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5) -
                            r / Sqrt(
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (2 * M) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) -
                            (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) +
                            (
                                (Power(Q, 2) - 2 * M * r) * (
                                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                    Power(Cos(θ), 2) * (
                                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                        Power(a, 6) * M * Power(Sin(θ), 4)
                                    ) +
                                    r * (
                                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                        Power(a, 2) *
                                        (
                                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                        ) *
                                        Power(Sin(θ), 2) +
                                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                    )
                                )
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                Power(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    1.5,
                                )
                            )
                        ) / Sqrt(2)
                    )
                )
            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
        ) / (
            Sqrt(2) * Sqrt(
                (
                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                ),
            )
        ) -
        (
            (
                -(
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) -
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            ) * (
                -(
                    (
                        (Power(Q, 2) - 2 * M * r) * (
                            (
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                (
                                    -(
                                        (
                                            Power(a, 2) *
                                            (Power(Q, 2) - 2 * M * r) *
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                            Power(Sin(θ), 2)
                                        ) / (
                                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        )
                                    ) -
                                    (
                                        Power(a, 2) *
                                        (Power(Q, 2) - 2 * M * r) *
                                        Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                        Power(Sin(θ), 2)
                                    ) / (
                                        Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    )
                                ) *
                                (
                                    (
                                        (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) *
                                        (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / Sqrt(
                                        (
                                            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    ) -
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                                        -(
                                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                        ) +
                                        (Power(Q, 2) - 2 * M * r) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                (
                                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                ) / (
                                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                ),
                                            )
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                            (
                                (
                                    a *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Csc(θ), 2)
                                ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2)) +
                                (
                                    Power(a, 3) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) * (
                                -(
                                    (
                                        a *
                                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2) *
                                        (
                                            -(
                                                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                                Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                            ) +
                                            (Power(Q, 2) - 2 * M * r) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                    (
                                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                    ) / (
                                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                    ),
                                                )
                                            )
                                        )
                                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                                ) -
                                (
                                    a *
                                    Power(Sin(θ), 2) *
                                    (
                                        Power(a, 6) * r * Power(Cos(θ), 6) +
                                        Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                        Power(a, 2) *
                                        Power(Cos(θ), 2) *
                                        (
                                            Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                            Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                            Power(a, 2) *
                                            (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) *
                                            Power(Sin(θ), 2)
                                        ) +
                                        r * (
                                            Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                            Power(r, 2) *
                                            (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                            Power(a, 2) *
                                            (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                            Power(Sin(θ), 2)
                                        )
                                    )
                                ) / (
                                    Sqrt(2) *
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    )
                                )
                            ) +
                            (
                                (
                                    Power(a, 2) *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Sin(θ), 2)
                                ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                                (
                                    Power(a, 2) *
                                    Power(Power(Q, 2) - 2 * M * r, 2) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) +
                                (
                                    Power(a, 2) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Power(Sin(θ), 2) *
                                    (
                                        (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) * (
                                (
                                    (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                        (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                                        Power(Cos(θ), 2) *
                                        (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                                    )
                                ) / (
                                    Sqrt(2) *
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    )
                                ) -
                                (
                                    (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                                    (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                    (
                                        -(
                                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                        ) +
                                        (Power(Q, 2) - 2 * M * r) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                (
                                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                ) / (
                                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                ),
                                            )
                                        )
                                    )
                                ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                                (
                                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                    Power(Cos(θ), 2) * (
                                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                        Power(a, 6) * M * Power(Sin(θ), 4)
                                    ) +
                                    r * (
                                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                        Power(a, 2) *
                                        (
                                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                        ) *
                                        Power(Sin(θ), 2) +
                                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                    )
                                ) / (
                                    Sqrt(2) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    ) *
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            )
                        )
                    ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) +
                (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                        (
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    (
                                        Power(a, 2) *
                                        (Power(Q, 2) - 2 * M * r) *
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / (
                                        Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    )
                                ) -
                                (
                                    Power(a, 2) *
                                    (Power(Q, 2) - 2 * M * r) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) *
                            (
                                (
                                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / Sqrt(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                ) -
                                (Power(Q, 2) - 2 * M * r) * (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            a *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2) *
                            (
                                (
                                    a *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Csc(θ), 2)
                                ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2)) +
                                (
                                    Power(a, 3) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Power(Sin(θ), 2)
                                ) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            ) *
                            (
                                (
                                    -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / Sqrt(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                ) +
                                (Power(Q, 2) - 2 * M * r) * (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            (
                                Power(a, 2) *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2)
                            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                            (
                                Power(a, 2) *
                                Power(Power(Q, 2) - 2 * M * r, 2) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) +
                            (
                                Power(a, 2) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(Sin(θ), 2) *
                                (
                                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) / (
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            )
                        ) * (
                            (
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                    (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) *
                                    (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                )
                            ) / (
                                Sqrt(2) *
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                Sqrt(
                                    -(
                                        (
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                    ),
                                )
                            ) -
                            (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                            (
                                ((M + r) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) /
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5) -
                                r / Sqrt(
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                                ) -
                                (2 * M) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                ) -
                                (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                ) +
                                (
                                    (Power(Q, 2) - 2 * M * r) * (
                                        Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                        Power(Cos(θ), 2) * (
                                            Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                            2 *
                                            (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) *
                                            Power(Sin(θ), 2) + Power(a, 6) * M * Power(Sin(θ), 4)
                                        ) +
                                        r * (
                                            (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                            Power(a, 2) *
                                            (
                                                Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) -
                                                2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                                Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                            ) *
                                            Power(Sin(θ), 2) +
                                            Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                        )
                                    )
                                ) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    Power(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                        1.5,
                                    )
                                )
                            ) / Sqrt(2)
                        )
                    )
                ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
            )
        ) / Sqrt(2)
    ω3 =
        (
            (
                Power(a, 4) *
                (Power(Q, 2) - 2 * M * r) *
                Cos(θ) *
                Power(Sin(θ), 3) *
                (
                    -(
                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                    ) +
                    (Power(a, 2) * (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))) / Sqrt(
                        (
                            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            ) / (
                Sqrt(2) *
                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
            ) +
            (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) * (
                    (
                        Power(a, 2) *
                        Cos(θ) *
                        Sin(θ) *
                        (
                            Power(a, 2) * Power(Q, 2) - 2 * Power(a, 2) * M * r +
                            Power(r, 4) +
                            Power(a, 2) * (-Power(Q, 2) + 2 * r * (M + r)) * Power(Cos(θ), 2) +
                            Power(a, 4) * Power(Cos(θ), 4) - Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (
                        Sqrt(2) *
                        Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                        Sqrt(
                            -(
                                (
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                            ),
                        )
                    ) +
                    (
                        Power(a, 2) *
                        Cos(θ) *
                        Sin(θ) *
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) *
                        (
                            (Power(a, 2) + Power(r, 2)) * (
                                Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                                Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                            ) +
                            2 * Power(a, 2) * (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) * Power(Cos(θ), 2) +
                            Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                            2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                            Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
                        )
                    ) / (
                        Sqrt(2) *
                        Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                        Power(
                            (
                                (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            1.5,
                        )
                    ) -
                    (
                        Power(a, 2) *
                        (Power(Q, 2) - 2 * M * r) *
                        Cos(θ) *
                        Sin(θ) *
                        (
                            -(
                                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            ) +
                            (Power(Q, 2) - 2 * M * r) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))
                )
            ) / ((Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))) -
            (
                (Power(Q, 2) - 2 * M * r) * (
                    -(
                        (Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Cos(θ) * Sin(θ) * (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))) /
                        (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    ) -
                    (
                        Power(a, 2) *
                        (Power(Q, 2) - 2 * M * r) *
                        Cos(θ) *
                        Sin(θ) *
                        (
                            -(
                                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            ) +
                            (Power(Q, 2) - 2 * M * r) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)) +
                    (
                        -(
                            (Power(a, 2) * Cos(θ) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * Sin(θ)) /
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5)
                        ) +
                        (Power(a, 2) * Cos(θ) * Sin(θ)) / Sqrt(
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                        ) -
                        (2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Cos(θ) * Sin(θ)) / (
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                (
                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                ),
                            )
                        ) -
                        (
                            Power(a, 2) *
                            (Power(Q, 2) - 2 * M * r) *
                            Cos(θ) *
                            Sin(θ) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (
                                    Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                                ) +
                                2 *
                                Power(a, 2) *
                                (Power(a, 2) + Power(r, 2)) *
                                (-Power(Q, 2) + r * (2 * M + r)) *
                                Power(Cos(θ), 2) +
                                Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                                2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                                Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
                            )
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                            Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                            Power(
                                (
                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                1.5,
                            )
                        )
                    ) / Sqrt(2)
                )
            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
        ) / (
            Sqrt(2) * Sqrt(
                (
                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                ),
            )
        ) -
        (
            (
                -(
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) -
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            ) * (
                -(
                    (
                        (Power(Q, 2) - 2 * M * r) * (
                            (
                                Power(a, 2) *
                                Cos(θ) *
                                Sin(θ) *
                                (
                                    Power(a, 2) * Power(Q, 2) - 2 * Power(a, 2) * M * r +
                                    Power(r, 4) +
                                    Power(a, 2) * (-Power(Q, 2) + 2 * r * (M + r)) * Power(Cos(θ), 2) +
                                    Power(a, 4) * Power(Cos(θ), 4) - Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) / (
                                Sqrt(2) *
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Sqrt(
                                    -(
                                        (
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                    ),
                                )
                            ) +
                            (
                                Power(a, 2) *
                                Cos(θ) *
                                Sin(θ) *
                                (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) *
                                (
                                    (Power(a, 2) + Power(r, 2)) * (
                                        Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                                        Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                                    ) +
                                    2 *
                                    Power(a, 2) *
                                    (Power(a, 2) + Power(r, 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r)) *
                                    Power(Cos(θ), 2) +
                                    Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                                    2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                                    Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
                                )
                            ) / (
                                Sqrt(2) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Power(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    1.5,
                                )
                            ) -
                            (
                                Power(a, 2) *
                                (Power(Q, 2) - 2 * M * r) *
                                Cos(θ) *
                                Sin(θ) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2))
                        )
                    ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) +
                (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                        -(
                            (
                                Power(a, 4) *
                                (Power(Q, 2) - 2 * M * r) *
                                Cos(θ) *
                                Sin(θ) *
                                (-1 + Power(Cos(θ), 2) + Power(Sin(θ), 2))
                            ) / (
                                Sqrt(2) *
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Sqrt(
                                    -(
                                        (
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                    ),
                                )
                            )
                        ) -
                        (
                            Power(a, 2) *
                            (Power(Q, 2) - 2 * M * r) *
                            Cos(θ) *
                            Sin(θ) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)) +
                        (
                            -(
                                (Power(a, 2) * Cos(θ) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * Sin(θ)) /
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5)
                            ) +
                            (Power(a, 2) * Cos(θ) * Sin(θ)) / Sqrt(
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Cos(θ) * Sin(θ)) / (
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) -
                            (
                                Power(a, 2) *
                                (Power(Q, 2) - 2 * M * r) *
                                Cos(θ) *
                                Sin(θ) *
                                (
                                    (Power(a, 2) + Power(r, 2)) * (
                                        Power(Q, 4) - Power(Q, 2) * r * (4 * M + r) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) +
                                        Power(r, 2) * (4 * Power(M, 2) + 2 * M * r + Power(r, 2))
                                    ) +
                                    2 *
                                    Power(a, 2) *
                                    (Power(a, 2) + Power(r, 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r)) *
                                    Power(Cos(θ), 2) +
                                    Power(a, 4) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 4) -
                                    2 * Power(a, 2) * (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) * Power(Sin(θ), 2) +
                                    Power(a, 4) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 4)
                                )
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                Power(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    1.5,
                                )
                            )
                        ) / Sqrt(2)
                    )
                ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
            )
        ) / Sqrt(2)
    ω4 =
        (
            (
                a * (
                    (
                        Power(Sin(θ), 2) *
                        (
                            Power(r, 5) +
                            Power(a, 4) * r * Power(Cos(θ), 4) +
                            Power(a, 2) * r * (Power(Q, 2) - M * r) * Power(Sin(θ), 2) +
                            Power(Cos(θ), 2) * (2 * Power(a, 2) * Power(r, 3) + Power(a, 4) * M * Power(Sin(θ), 2))
                        ) *
                        (
                            (
                                Power(a, 2) *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2)
                            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                            (
                                Power(Csc(θ), 2) * (
                                    Power(a, 2) + Power(r, 2) -
                                    (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                )
                            ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                        ) *
                        (
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        a *
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        Power(Sin(θ), 2) *
                        (
                            (
                                a *
                                (Power(Q, 2) - 2 * M * r) *
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) -
                            (
                                a *
                                (Power(Q, 2) - 2 * M * r) *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2)
                            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                        ) *
                        (
                            (
                                -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) +
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        -(
                            (a * Power(Power(Q, 2) - 2 * M * r, 2) * Power(Sin(θ), 2)) /
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)
                        ) +
                        (
                            a *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) +
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 2) + Power(r, 2) -
                                (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            )
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ) * (
                        -(
                            (
                                a *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                        ) -
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 6) * r * Power(Cos(θ), 6) +
                                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                Power(a, 2) *
                                Power(Cos(θ), 2) *
                                (
                                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                                ) +
                                r * (
                                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                    Power(r, 2) *
                                    (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                    Power(a, 2) *
                                    (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                    Power(Sin(θ), 2)
                                )
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    )
                )
            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
            (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) * (
                    (
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        (
                            (
                                a *
                                (Power(Q, 2) - 2 * M * r) *
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) -
                            (
                                a *
                                (Power(Q, 2) - 2 * M * r) *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2)
                            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                        ) *
                        (
                            (
                                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) *
                                (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        (
                            Power(a, 2) *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2)
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                        (
                            Power(Csc(θ), 2) * (
                                Power(a, 2) + Power(r, 2) -
                                (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            )
                        ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                    ) * (
                        -(
                            (
                                a *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                        ) -
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 6) * r * Power(Cos(θ), 6) +
                                Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                Power(a, 2) *
                                Power(Cos(θ), 2) *
                                (
                                    Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                    Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                    Power(a, 2) * (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) * Power(Sin(θ), 2)
                                ) +
                                r * (
                                    Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                    Power(r, 2) *
                                    (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                    Power(a, 2) *
                                    (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                    Power(Sin(θ), 2)
                                )
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        )
                    ) +
                    (
                        -(
                            (a * Power(Power(Q, 2) - 2 * M * r, 2) * Power(Sin(θ), 2)) /
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)
                        ) +
                        (
                            a *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) +
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 2) + Power(r, 2) -
                                (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            )
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ) * (
                        (
                            (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                                Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                                Power(Cos(θ), 2) * (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        ) -
                        (
                            (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                            Power(Cos(θ), 2) * (
                                Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                Power(a, 6) * M * Power(Sin(θ), 4)
                            ) +
                            r * (
                                (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                Power(a, 2) *
                                (
                                    Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                    Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                ) *
                                Power(Sin(θ), 2) +
                                Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                            )
                        ) / (
                            Sqrt(2) *
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            ) *
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        )
                    )
                )
            ) / ((Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))) -
            (
                (Power(Q, 2) - 2 * M * r) * (
                    (
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        (
                            (
                                a *
                                (Power(Q, 2) - 2 * M * r) *
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            ) / Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) -
                            (
                                a *
                                (Power(Q, 2) - 2 * M * r) *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2)
                            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                        ) *
                        (
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        a *
                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                        Power(Sin(θ), 2) *
                        (
                            (
                                Power(a, 2) *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2)
                            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                            (
                                Power(Csc(θ), 2) * (
                                    Power(a, 2) + Power(r, 2) -
                                    (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                )
                            ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                        ) *
                        (
                            (
                                -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / Sqrt(
                                (
                                    (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                        Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                            ) +
                            (Power(Q, 2) - 2 * M * r) * (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        )
                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                    (
                        -(
                            (a * Power(Power(Q, 2) - 2 * M * r, 2) * Power(Sin(θ), 2)) /
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)
                        ) +
                        (
                            a *
                            (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                            Power(Sin(θ), 2) *
                            (
                                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (
                            (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) +
                        (
                            a *
                            Power(Sin(θ), 2) *
                            (
                                Power(a, 2) + Power(r, 2) -
                                (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            )
                        ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                    ) * (
                        (
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) *
                                (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(Sin(θ), 2)
                            )
                        ) / (
                            Sqrt(2) *
                            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                            Sqrt(
                                -(
                                    (
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                ),
                            )
                        ) -
                        (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                -(
                                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) +
                                (Power(Q, 2) - 2 * M * r) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            ((M + r) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) /
                            Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5) -
                            r / Sqrt(
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                            ) -
                            (2 * M) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) -
                            (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                    ),
                                )
                            ) +
                            (
                                (Power(Q, 2) - 2 * M * r) * (
                                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                    Power(Cos(θ), 2) * (
                                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                        Power(a, 6) * M * Power(Sin(θ), 4)
                                    ) +
                                    r * (
                                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                        Power(a, 2) *
                                        (
                                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                        ) *
                                        Power(Sin(θ), 2) +
                                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                    )
                                )
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                Power(
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    1.5,
                                )
                            )
                        ) / Sqrt(2)
                    )
                )
            ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
        ) / (
            Sqrt(2) * Sqrt(
                (
                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                ),
            )
        ) -
        (
            (
                -(
                    Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                    Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) -
                (Power(Q, 2) - 2 * M * r) / (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                        (
                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        ) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ),
                    )
                )
            ) * (
                -(
                    (
                        (Power(Q, 2) - 2 * M * r) * (
                            (
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                (
                                    (
                                        a *
                                        (Power(Q, 2) - 2 * M * r) *
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) -
                                    (
                                        a *
                                        (Power(Q, 2) - 2 * M * r) *
                                        (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                        Power(Sin(θ), 2)
                                    ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                ) *
                                (
                                    (
                                        (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                        Power(a, 2) *
                                        (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2)
                                    ) / Sqrt(
                                        (
                                            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                    ) -
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                                        -(
                                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                        ) +
                                        (Power(Q, 2) - 2 * M * r) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                (
                                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                ) / (
                                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                ),
                                            )
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                            (
                                (
                                    Power(a, 2) *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Sin(θ), 2)
                                ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                                (
                                    Power(Csc(θ), 2) * (
                                        Power(a, 2) + Power(r, 2) -
                                        (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    )
                                ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                            ) * (
                                -(
                                    (
                                        a *
                                        (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)) *
                                        (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                        Power(Sin(θ), 2) *
                                        (
                                            -(
                                                Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                                Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                            ) +
                                            (Power(Q, 2) - 2 * M * r) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                    (
                                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                    ) / (
                                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                        (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                    ),
                                                )
                                            )
                                        )
                                    ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3))
                                ) -
                                (
                                    a *
                                    Power(Sin(θ), 2) *
                                    (
                                        Power(a, 6) * r * Power(Cos(θ), 6) +
                                        Power(Cos(θ), 4) * (3 * Power(a, 4) * Power(r, 3) + Power(a, 6) * M * Power(Sin(θ), 2)) +
                                        Power(a, 2) *
                                        Power(Cos(θ), 2) *
                                        (
                                            Power(a, 2) * M * (Power(Q, 2) - 2 * M * r) +
                                            Power(r, 2) * (M * Power(Q, 2) - 2 * Power(M, 2) * r + 3 * Power(r, 3)) +
                                            Power(a, 2) *
                                            (-(M * Power(Q, 2)) + 2 * Power(M, 2) * r + Power(Q, 2) * r) *
                                            Power(Sin(θ), 2)
                                        ) +
                                        r * (
                                            Power(a, 2) * (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2)) +
                                            Power(r, 2) *
                                            (Power(Q, 4) - 3 * M * Power(Q, 2) * r + 2 * Power(M, 2) * Power(r, 2) + Power(r, 4)) -
                                            Power(a, 2) *
                                            (Power(Q, 4) + M * Power(r, 2) * (2 * M + r) - Power(Q, 2) * r * (3 * M + r)) *
                                            Power(Sin(θ), 2)
                                        )
                                    )
                                ) / (
                                    Sqrt(2) *
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    )
                                )
                            ) +
                            (
                                -(
                                    (a * Power(Power(Q, 2) - 2 * M * r, 2) * Power(Sin(θ), 2)) /
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)
                                ) +
                                (
                                    a *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Sin(θ), 2) *
                                    (
                                        (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                ) / (
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                ) +
                                (
                                    a *
                                    Power(Sin(θ), 2) *
                                    (
                                        Power(a, 2) + Power(r, 2) -
                                        (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    )
                                ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ) * (
                                (
                                    (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                        (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                                        Power(Cos(θ), 2) *
                                        (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                                    )
                                ) / (
                                    Sqrt(2) *
                                    Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    )
                                ) -
                                (
                                    (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                                    (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                    (
                                        -(
                                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                        ) +
                                        (Power(Q, 2) - 2 * M * r) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                                (
                                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                ) / (
                                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                                ),
                                            )
                                        )
                                    )
                                ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                                (
                                    Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                    Power(Cos(θ), 2) * (
                                        Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                        2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                                        Power(a, 6) * M * Power(Sin(θ), 4)
                                    ) +
                                    r * (
                                        (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                        Power(a, 2) *
                                        (
                                            Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                            Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                        ) *
                                        Power(Sin(θ), 2) +
                                        Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                    )
                                ) / (
                                    Sqrt(2) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Sqrt(
                                        -(
                                            (
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                    Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                                )
                                            ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                        ),
                                    ) *
                                    (
                                        -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                    )
                                )
                            )
                        )
                    ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                ) +
                (
                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * (
                        (
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            (
                                (
                                    a *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                ) / Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2) -
                                (
                                    a *
                                    (Power(Q, 2) - 2 * M * r) *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Sin(θ), 2)
                                ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                            ) *
                            (
                                (
                                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / Sqrt(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                ) -
                                (Power(Q, 2) - 2 * M * r) * (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            a *
                            (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                            Power(Sin(θ), 2) *
                            (
                                (
                                    Power(a, 2) *
                                    (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                    Power(Sin(θ), 2)
                                ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) +
                                (
                                    Power(Csc(θ), 2) * (
                                        Power(a, 2) + Power(r, 2) -
                                        (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                        (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    )
                                ) / (-Power(a, 2) + (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2))
                            ) *
                            (
                                (
                                    -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / Sqrt(
                                    (
                                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        )
                                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                ) +
                                (Power(Q, 2) - 2 * M * r) * (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            )
                        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                        (
                            -(
                                (a * Power(Power(Q, 2) - 2 * M * r, 2) * Power(Sin(θ), 2)) /
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)
                            ) +
                            (
                                a *
                                (-1 + (Power(Q, 2) - 2 * M * r) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) *
                                Power(Sin(θ), 2) *
                                (
                                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                )
                            ) / (
                                (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ) +
                            (
                                a *
                                Power(Sin(θ), 2) *
                                (
                                    Power(a, 2) + Power(r, 2) -
                                    (Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)) /
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                )
                            ) / (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                        ) * (
                            (
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) * (
                                    (Power(Q, 2) - 2 * M * r) * (Power(a, 2) + Power(r, 2)) +
                                    Power(a, 2) *
                                    (-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(Sin(θ), 2)
                                )
                            ) / (
                                Sqrt(2) *
                                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                                Sqrt(
                                    -(
                                        (
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            )
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                                    ),
                                )
                            ) -
                            (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                                (
                                    -(
                                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                                    ) +
                                    (Power(Q, 2) - 2 * M * r) / (
                                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                            (
                                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                            ) / (
                                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                            ),
                                        )
                                    )
                                )
                            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
                            (
                                ((M + r) * Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))) /
                                Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 1.5) -
                                r / Sqrt(
                                    (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) *
                                    (-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2)),
                                ) -
                                (2 * M) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                ) -
                                (2 * (M + r) * (-Power(Q, 2) + 2 * M * r)) / (
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) * Sqrt(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (
                                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                        ),
                                    )
                                ) +
                                (
                                    (Power(Q, 2) - 2 * M * r) * (
                                        Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                                        Power(Cos(θ), 2) * (
                                            Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                                            2 *
                                            (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) *
                                            Power(Sin(θ), 2) + Power(a, 6) * M * Power(Sin(θ), 4)
                                        ) +
                                        r * (
                                            (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                                            Power(a, 2) *
                                            (
                                                Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) -
                                                2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                                                Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                                            ) *
                                            Power(Sin(θ), 2) +
                                            Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                                        )
                                    )
                                ) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    Power(-Power(Q, 2) + r * (2 * M + r) + Power(a, 2) * Power(Cos(θ), 2), 2) *
                                    Sqrt(Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) *
                                    Power(
                                        (
                                            -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                                        1.5,
                                    )
                                )
                            ) / Sqrt(2)
                        )
                    )
                ) / (Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
            )
        ) / Sqrt(2)

    ω = SVector{4}(ω1, ω2, ω3, ω4)

    Θ =
        (
            r * (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) -
                (Power(Q, 2) - 2 * M * r) * (
                    -(
                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)) +
        (
            (Power(a, 2) * Power(Cot(θ), 2) * Power(Csc(θ), 2) + Power(r, 2) * Power(Csc(θ), 4)) *
            Power(Sin(θ), 2) *
            (
                Power(r, 5) +
                Power(a, 4) * r * Power(Cos(θ), 4) +
                Power(a, 2) * r * (Power(Q, 2) - M * r) * Power(Sin(θ), 2) +
                Power(Cos(θ), 2) * (2 * Power(a, 2) * Power(r, 3) + Power(a, 4) * M * Power(Sin(θ), 2))
            ) *
            (
                (
                    (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                ) / Sqrt(
                    (
                        (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                            Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                            Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                            Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                        )
                    ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                ) -
                (Power(Q, 2) - 2 * M * r) * (
                    -(
                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            )
        ) / (
            Sqrt(2) *
            Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
            (
                -(Power(a, 2) * (Power(Q, 2) - 2 * M * r)) +
                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cot(θ), 2) +
                Power(r, 2) * (Power(a, 2) + Power(r, 2)) * Power(Csc(θ), 2)
            )
        ) +
        (
            (
                (r * (-Power(Q, 2) + M * r) - Power(a, 2) * M * Power(Cos(θ), 2)) * (
                    (Power(a, 2) + Power(r, 2)) * (-Power(Q, 2) + r * (2 * M + r)) +
                    Power(a, 2) * (Power(Q, 2) - 2 * r * (M + r)) * Power(Sin(θ), 2) +
                    Power(Cos(θ), 2) * (Power(a, 4) + Power(a, 2) * Power(r, 2) - 2 * Power(a, 4) * Power(Sin(θ), 2))
                )
            ) / (
                Sqrt(2) *
                Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3) *
                Sqrt(
                    -(
                        (
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                    ),
                )
            ) -
            (
                (Power(Q, 2) - 2 * r * (M + r) - 2 * Power(a, 2) * Power(Cos(θ), 2)) *
                (r * (Power(Q, 2) - M * r) + Power(a, 2) * M * Power(Cos(θ), 2)) *
                (
                    -(
                        Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                        Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                    ) +
                    (Power(Q, 2) - 2 * M * r) / (
                        (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                            (
                                -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            ) / (
                                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                            ),
                        )
                    )
                )
            ) / (Sqrt(2) * Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 3)) +
            (
                Power(a, 6) * r * Power(Cos(θ), 4) * Power(Sin(θ), 2) +
                Power(Cos(θ), 2) * (
                    Power(a, 2) * M * Power(Power(a, 2) + Power(r, 2), 2) -
                    2 * (Power(a, 6) * M + Power(a, 4) * r * (Power(Q, 2) - r * (M + r))) * Power(Sin(θ), 2) +
                    Power(a, 6) * M * Power(Sin(θ), 4)
                ) +
                r * (
                    (Power(Q, 2) - M * r) * Power(Power(a, 2) + Power(r, 2), 2) +
                    Power(a, 2) *
                    (
                        Power(Q, 4) - 4 * Power(Q, 2) * r * (M + r) - 2 * Power(a, 2) * (Power(Q, 2) - M * r) +
                        Power(r, 2) * (4 * Power(M, 2) + 6 * M * r + Power(r, 2))
                    ) *
                    Power(Sin(θ), 2) +
                    Power(a, 4) * (Power(Q, 2) - M * r) * Power(Sin(θ), 4)
                )
            ) / (
                Sqrt(2) *
                (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                Sqrt(
                    -(
                        (
                            (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2)) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2))
                    ),
                ) *
                (
                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) - Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                )
            )
        ) * (
            (
                (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
            ) / ((Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) * (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))) +
            (
                (
                    (
                        (Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2)) +
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / Sqrt(
                        (
                            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                    ) -
                    (Power(Q, 2) - 2 * M * r) * (
                        -(
                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                        ) +
                        (Power(Q, 2) - 2 * M * r) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                (
                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                ),
                            )
                        )
                    )
                ) * (
                    (
                        -((Power(a, 2) + Power(r, 2)) * (Power(Q, 2) - 2 * M * r + Power(r, 2))) -
                        Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                        Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                    ) / Sqrt(
                        (
                            (-Power(a, 2) - Power(r, 2) + Power(a, 2) * Power(Sin(θ), 2)) * (
                                Power(r, 2) * (Power(a, 2) + Power(r, 2)) +
                                Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) -
                                Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                            )
                        ) / (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)),
                    ) -
                    (Power(Q, 2) - 2 * M * r) * (
                        -(
                            Sqrt(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2)) /
                            Sqrt(-Power(Q, 2) + 2 * M * r + Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2))
                        ) +
                        (-Power(Q, 2) + 2 * M * r) / (
                            (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) * Sqrt(
                                (
                                    -(Power(r, 2) * (Power(a, 2) + Power(r, 2))) -
                                    Power(a, 2) * (Power(a, 2) + Power(r, 2)) * Power(Cos(θ), 2) +
                                    Power(a, 2) * (Power(Q, 2) - 2 * M * r) * Power(Sin(θ), 2)
                                ) / (
                                    (Power(Q, 2) - r * (2 * M + r) - Power(a, 2) * Power(Cos(θ), 2)) *
                                    (Power(a, 2) + Power(r, 2) - Power(a, 2) * Power(Sin(θ), 2))
                                ),
                            )
                        )
                    )
                )
            ) / Power(Power(r, 2) + Power(a, 2) * Power(Cos(θ), 2), 2)
        )

    rEH = M + Sqrt(-a^2 + M^2 - Q^2)

    return g, l, k, dl, Dl, ω, Θ, rEH
end
