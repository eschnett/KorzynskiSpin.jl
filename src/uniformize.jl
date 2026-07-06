# Uniformization of the (area-normalized) 2-metric by a fast flow
# (docs/src/algorithm.md, "Uniformization by a fast flow"), patterned after the
# pseudo-spectral fast flow of Gundlach, arXiv:gr-qc/9707050, as used in
# ApparentHorizonFinder.find_horizon.
#
# With q̄ scaled to area 4π and q(t) = e^{2u(t)} q̄, the round representative
# q̊ = e^{2u} q̄ satisfies the Liouville equation
#     N[u] = Δ̄u − R̄/2 + e^{2u} = 0.
# Two exact identities drive the iteration:
#     N[u]  = −e^{2u} (R − 2)/2,        R = e^{−2u} (R̄ − 2 Δ̄u),
#     N'[u] = Δ̄ + 2e^{2u} = e^{2u} (Δ_{q(t)} + 2),
# so the exact Newton step is δu = (Δ_{q(t)} + 2)⁻¹ (R − 2)/2.  The fast flow
# replaces Δ_{q(t)} by the round model Laplacian, diagonal in the chart's
# harmonic basis with eigenvalues −l(l+1):
#     δu_lm = [(R − 2)/2]_lm / (2 − κ l(l+1)).
# The model operator is indefinite (+2 at l=0, 0 at l=1, negative for l≥2).
# The l=1 modes are the Möbius gauge kernel: at the solution the residual has
# no component along the kernel, so their vanishing model eigenvalue is
# replaced by the adjacent l=2 value (any bounded negative divisor converges
# there; drift along the kernel is pure gauge and harmless).  κ is the
# Richardson midpoint of the pointwise eigenvalue range of e^{−2u} q̄^{ab}
# (relative to q̂), which keeps the high-l iteration factors 1 − c/κ inside
# (−1, 1) for anisotropic metrics.  The area is renormalized to exactly 4π
# after every step (a constant shift of u).
#
# The iteration converges linearly with a resolution-independent contraction
# set by the chart distortion/anisotropy; a stalled residual (three
# consecutive iterations without 1% improvement of the best residual norm)
# signals the round-off/aliasing floor.

struct Uniformization
    "conformal factor exponent: q̊ = e^{2u} q̄ is the unit-round representative"
    u::Tensor{0}
    "R[q̊] on the grid"
    R::Tensor{0}
    "‖R[q̊] − 2‖∞"
    residual::Float64
    "number of fast-flow iterations"
    iters::Int
    "whether the fast flow converged to `tol` (`false` ⇒ `u` is the
     best-effort iterate, stalled at the resolution's aliasing floor)"
    converged::Bool
end

"""
    uniformize(ops̄, R̄; tol=1.0e-13, maxiter=1000) -> Uniformization

Find the conformal factor exponent `u` such that `q̊ = e^{2u} q̄` is the
unit-round representative (area `4π`, `R[q̊] = 2`) of the conformal class of
the unit-area metric `q̄`, by solving the Liouville equation
`N[u] = Δ̄u − R̄/2 + e^{2u} = 0` with a Gundlach-style fast flow
(arXiv:gr-qc/9707050): each step applies the mode-diagonal inverse of the
round model linearization `Δ̊ + 2` to the weighted residual
`e^{−2u} N[u] = −(R − 2)/2`, with the vanishing `l = 1` (Möbius gauge kernel)
model eigenvalue replaced by the adjacent `l = 2` value.

Converges linearly and monotonically; the contraction per iteration is set by
the chart distortion/anisotropy of `q̄` but not by the resolution.  Success
means the L² norm of the residual coefficients reached `tol · n` (`n` the
number of modes); three consecutive iterations without 1% improvement of the
best residual first signal the resolution's round-off/aliasing floor, and the
best iterate is returned with `converged = false` (increase the resolution to
converge).
"""
function uniformize(ops̄::MetricOps, R̄::Tensor{0}; tol::Float64=1.0e-13, maxiter::Int=1000)
    grid = ops̄.grid
    n = ash_nmodes(grid)[1]
    ls = [ash_mode_numbers(grid, 0, i)[1] for i in 1:n]
    # model spectrum 2 − κ l(l+1), with the l=1 Möbius kernel (eigenvalue 0)
    # replaced by the adjacent l=2 value: at the solution the residual has no
    # component along the kernel, so any bounded negative divisor is safe
    # there, and −(6κ−2) ≤ −4 keeps the step finite for all κ ≥ 1
    lls = [l == 1 ? 6 : l * (l + 1) for l in ls]

    R̄vals = real.(grid_values(R̄))
    sdq̄ = real.(grid_values(ops̄.sqrtdetq))
    ucoeffs = zeros(ComplexF64, n)
    i00 = LinearIndices((n,))[ash_mode_index(grid, 0, 0, 0)]

    ubest = copy(ucoeffs)
    Fbest = Inf
    stall = 0
    iters = 0
    converged = false
    for iter in 0:maxiter
        iters = iter
        u = coeffs_scalar(ucoeffs, grid)
        uvals = real.(grid_values(u))
        Δu = laplacian(ops̄, u)
        Rvals = real.(exp.(-2 .* uvals) .* (R̄vals .- 2 .* grid_values(Δu)))
        # weighted Liouville residual e^{−2u} N[u] = −(R − 2)/2; the step and
        # the control norm use the same coefficients, so the fixed point of
        # the iteration is exactly the discrete solution
        Fc = scalar_coeffs(make_scalar((Rvals .- 2) ./ 2, grid))
        Fnorm = norm(Fc)

        # Stall detection: the flow converges linearly, so three consecutive
        # iterations that fail to improve the best residual by at least 1%
        # signal the round-off (or truncation/aliasing) floor.
        isfinite(Fnorm) && Fnorm < Fbest && (ubest .= ucoeffs)
        stall = Fnorm < 0.99 * Fbest ? 0 : stall + 1
        Fbest = min(Fbest, Fnorm)
        if Fnorm ≤ tol * n
            converged = true
            break
        end
        (stall ≥ 3 || iter == maxiter) && break

        # stability constant: Richardson midpoint of the pointwise eigenvalue
        # range of e^{−2u} q̄^{ab} w.r.t. q̂ (2κ ≥ cmin + cmax > cmax keeps
        # the high-l iteration factors inside (−1, 1) unconditionally)
        cs = (ij -> exp(-2 * uvals[ij]) .* abs.(eigvals(Hermitian(real.(ops̄.qu.values[ij]))))).(
            CartesianIndices(size(uvals))
        )
        κ = max(1.0, (minimum(minimum.(cs)) + maximum(maximum.(cs))) / 2)
        δu = Fc ./ (2 .- κ .* lls)
        ucoeffs .+= δu
        # exact area renormalization (constant shift of u)
        u = coeffs_scalar(ucoeffs, grid)
        uvals = real.(grid_values(u))
        area = integrate_unit(make_scalar(exp.(2 .* uvals) .* sdq̄, grid))
        ucoeffs[i00] -= sqrt(4π) * log(area / 4π) / 2
    end
    converged || (ucoeffs .= ubest)

    # Re-normalize the area of q̊ to exactly 4π
    u = coeffs_scalar(ucoeffs, grid)
    uvals = real.(grid_values(u))
    areå = integrate_unit(make_scalar(exp.(2 .* uvals) .* sdq̄, grid))
    uvals .-= log(areå / 4π) / 2
    u = make_scalar(uvals, grid)

    Δu = laplacian(ops̄, u)
    Rvals = real.(exp.(-2 .* uvals) .* (R̄vals .- 2 .* grid_values(Δu)))
    R = make_scalar(Rvals, grid)
    resid = maximum(abs.(Rvals .- 2))
    return Uniformization(u, R, resid, iters, converged)
end
