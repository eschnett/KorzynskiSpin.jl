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
    n = SVector(1, 0, 0, 0)
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
