# Differential operators of a 2-metric q via the difference-tensor method
# (docs/algorithm.tex §3.5, eqs. 28–31).
#
# All quantities are in unit-sphere dyad components.  ∇̂ is the connection of
# the unit coordinate sphere q̂ (whose dyad components are δ_ab), provided by
# `grad`.  The difference tensor is
#     C^a_bc = q^{ad} (∇̂_b q_dc + ∇̂_c q_bd − ∇̂_d q_bc) / 2 ,
# and
#     R[q]  = q^{ab} (q̂_ab + ∇̂_c C^c_ab − ∇̂_a C^c_cb + C^c_cd C^d_ab − C^c_ad C^d_cb)
#     Δ_q f = q^{ab} (∇̂_a ∇̂_b f − C^c_ab ∂_c f)
#     D^a ω_a = q^{ab} (∇̂_b ω_a − C^c_ab ω_c)

"Cached differential-operator data for a 2-metric"
struct MetricOps
    lmax::Int
    "metric q_ab (dyad components)"
    q::Tensor{2}
    "inverse metric q^{ab}"
    qu::Tensor{2}
    "difference tensor C^a_bc"
    C::Tensor{3}
    "q^{ab} C^c_ab (used by Laplacian and divergence)"
    trC::Tensor{1}
    "√(det q_ab)"
    sqrtdetq::Tensor{0}
end

function MetricOps(q::Tensor{2})
    lmax = q.lmax
    q = symmetrize(real_part(q))
    qu = symmetrize(real_part(Tensor{2}(map(inv, q.values), lmax)))
    dq = grad_filtered(q)
    C = Tensor{3}(
        [
            SArray{Tuple{2,2,2}}(
                sum(qu[a, d] * (dq[d, c, b] + dq[b, d, c] - dq[b, c, d]) / 2 for d in 1:2) for a in 1:2, b in 1:2, c in 1:2
            ) for (qu, dq) in zip(qu.values, dq.values)
        ],
        lmax,
    )
    trC = Tensor{1}(
        [SVector{2}(sum(qu[a, b] * C[c, a, b] for a in 1:2, b in 1:2) for c in 1:2) for (qu, C) in zip(qu.values, C.values)],
        lmax,
    )
    sqrtdetq = make_scalar([sqrt(abs(det(real.(qv)))) for qv in q.values], lmax)
    return MetricOps(lmax, q, qu, C, trC, sqrtdetq)
end

"Ricci scalar R[q]"
function scalar_curvature(ops::MetricOps)
    dC = grad_filtered(ops.C)
    R = [
        begin
            # q̂_ab in dyad components is δ_ab
            sum(
                qu[a, b] * (
                    (a == b ? 1 : 0) +
                    sum(dC[c, a, b, c] - dC[c, c, b, a] for c in 1:2) +
                    sum(C[c, c, d] * C[d, a, b] - C[c, a, d] * C[d, c, b] for c in 1:2, d in 1:2)
                ) for a in 1:2, b in 1:2
            )
        end for (qu, C, dC) in zip(ops.qu.values, ops.C.values, dC.values)
    ]
    return make_scalar(R, ops.lmax)
end

"Laplace–Beltrami operator Δ_q applied to a scalar"
function laplacian(ops::MetricOps, f::Tensor{0})
    df = grad(f)
    ddf = grad(df)
    vals = [
        sum(qu[a, b] * (ddf[a, b] - sum(C[c, a, b] * df[c] for c in 1:2)) for a in 1:2, b in 1:2) for
        (qu, C, df, ddf) in zip(ops.qu.values, ops.C.values, df.values, ddf.values)
    ]
    return make_scalar(vals, ops.lmax)
end

"Divergence D^a ω_a of a one-form"
function divergence(ops::MetricOps, ω::Tensor{1})
    dω = grad(ω)
    vals = [
        sum(qu[a, b] * (dω[a, b] - sum(C[c, a, b] * ω[c] for c in 1:2)) for a in 1:2, b in 1:2) for
        (qu, C, ω, dω) in zip(ops.qu.values, ops.C.values, ω.values, dω.values)
    ]
    return make_scalar(vals, ops.lmax)
end

"Gradient of a scalar as a one-form in dyad components (∂_a f)"
function differential(f::Tensor{0})
    df = grad(f)
    return Tensor{1}([SVector{2}(df[1], df[2]) for df in df.values], f.lmax)
end

################################################################################
# Dense operator matrices in spherical-harmonic coefficient space

"""
    operator_matrix(op, lmax) -> Matrix{ComplexF64}

Dense matrix of a linear scalar operator `op :: Tensor{0} -> Tensor{0}` in the
spin-0 spherical-harmonic coefficient basis.
"""
function operator_matrix(op, lmax::Int)
    n = ash_nmodes(lmax)[1]
    M = zeros(ComplexF64, n, n)
    c = zeros(ComplexF64, n)
    for j in 1:n
        c .= 0
        c[j] = 1
        M[:, j] = scalar_coeffs(op(coeffs_scalar(c, lmax)))
    end
    return M
end

"Dense matrix of pointwise multiplication by the scalar field w"
function multiplication_matrix(w::Tensor{0})
    lmax = w.lmax
    wv = grid_values(w)
    return operator_matrix(f -> make_scalar(wv .* grid_values(f), lmax), lmax)
end
