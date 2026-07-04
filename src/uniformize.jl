# Uniformization of the (area-normalized) 2-metric by normalized Ricci flow
# with optional Newton polish (docs/algorithm.tex §3.6).
#
# With q̄ scaled to area 4π and q(t) = e^{2u(t)} q̄, the flow is the scalar PDE
#     ∂_t u = 1 − R(t)/2,    R(t) = e^{−2u} (R̄ − 2 Δ̄ u),    u(0) = 0,
# whose fixed point satisfies the Liouville equation
#     N[u] = Δ̄u − R̄/2 + e^{2u} = 0.
# The flow steps are preconditioned mode-wise with 1/(1 + κ l(l+1)) (an
# implicit treatment of the stiff Laplacian); Newton uses an SVD-regularized
# dense solve because the Jacobian Δ̄ + 2e^{2u} is singular at the solution
# (its kernel is the ℓ=1 Möbius gauge freedom).

struct Uniformization
    "conformal factor exponent: q̊ = e^{2u} q̄ is the unit-round representative"
    u::Tensor{0}
    "R[q̊] on the grid"
    R::Tensor{0}
    "‖R[q̊] − 2‖∞"
    residual::Float64
    "number of flow iterations"
    flow_iters::Int
    "number of Newton iterations"
    newton_iters::Int
    "whether the Newton polish converged to `newton_tol` (`false` ⇒ `u` is the
     best-effort iterate, stalled at the resolution's aliasing floor)"
    converged::Bool
end

"R[e^{2u} q̄] given R̄ and Δ̄u"
function conformal_curvature(R̄::Tensor{0}, Δu::Tensor{0}, u::Tensor{0})
    grid = u.grid
    vals = exp.(-2 .* grid_values(u)) .* (grid_values(R̄) - 2 .* grid_values(Δu))
    return make_scalar(vals, grid)
end

function uniformize(
    ops̄::MetricOps,
    R̄::Tensor{0};
    Δ̄mat::Union{Nothing,Matrix{ComplexF64}}=nothing,
    flow_tol::Float64=1.0e-3,
    flow_maxiter::Int=10_000,
    newton_tol::Float64=1.0e-13,
    newton_maxiter::Int=50,
    use_newton::Bool=true,
)
    grid = ops̄.grid
    n = ash_nmodes(grid)[1]
    ls = [ash_mode_numbers(grid, 0, i)[1] for i in 1:n]

    R̄vals = real.(grid_values(R̄))
    sdq̄ = real.(grid_values(ops̄.sqrtdetq))
    ucoeffs = zeros(ComplexF64, n)
    i00 = LinearIndices((n,))[ash_mode_index(grid, 0, 0, 0)]

    residual(Rvals) = maximum(abs.(Rvals .- 2))

    # --- normalized Ricci flow ---
    # ∂_t u = (r̄(t) − R(t))/2 with r̄ = 8π/A(t), mode-preconditioned by
    # 1/(1 + κ dt l(l+1)) (implicit treatment of the stiff Laplacian), with
    # exact area renormalization after every step.  κ is scaled with the
    # largest pointwise eigenvalue of e^{−2u} q̄^{ab} (relative to q̂), so
    # the high-l damping stays stable for anisotropic metrics; the step is
    # halved and reverted when the residual increases.
    dt = 0.5
    flow_iters = 0
    res_prev = Inf
    ucoeffs_prev = copy(ucoeffs)
    for iter in 1:flow_maxiter
        u = coeffs_scalar(ucoeffs, grid)
        uvals = real.(grid_values(u))
        Δu = laplacian(ops̄, u)
        Rvals = real.(exp.(-2 .* uvals) .* (R̄vals .- 2 .* grid_values(Δu)))
        res = residual(Rvals)
        if !isfinite(res) || res > 1.2 * res_prev
            # revert and retry with a smaller step
            ucoeffs .= ucoeffs_prev
            dt /= 2
            # step-size underflow: the flow cannot make further progress from
            # this iterate (typically an under-resolved, strongly distorted
            # surface).  Stop the flow gracefully and hand the best iterate so
            # far to the Newton polish; convergence is judged there.
            dt < 1.0e-6 && break
            continue
        end
        res ≤ flow_tol && break
        ucoeffs_prev .= ucoeffs
        res < 0.5 * res_prev && (dt = min(2dt, 1.0))
        res_prev = min(res, res_prev)
        # stability constant: max eigenvalue of e^{−2u} q̄^{ab} w.r.t. q̂
        cmax = maximum(
            (ij -> exp(-2 * uvals[ij]) * max(abs.(eigvals(Hermitian(real.(inv(ops̄.q.values[ij])))))...)).(
                CartesianIndices(size(uvals))
            ),
        )
        κ = max(1.0, 1.5 * cmax)
        area = integrate_unit(make_scalar(exp.(2 .* uvals) .* sdq̄, grid))
        r̄avg = 8π / area
        δu = scalar_coeffs(make_scalar(dt .* (r̄avg .- Rvals) ./ 2, grid))
        @. δu = δu / (1 + κ * dt * ls * (ls + 1))
        ucoeffs .+= δu
        # exact area renormalization (constant shift of u)
        u = coeffs_scalar(ucoeffs, grid)
        uvals = real.(grid_values(u))
        area = integrate_unit(make_scalar(exp.(2 .* uvals) .* sdq̄, grid))
        ucoeffs[i00] -= sqrt(4π) * log(area / 4π) / 2
        flow_iters = iter
    end

    # --- Newton polish on the Liouville equation ---
    newton_iters = 0
    # The uniformization is "converged" only when the Newton residual actually
    # reaches `newton_tol` (genuine convergence to round-off), not when the
    # iteration merely stalls at the aliasing/near-kernel floor of an
    # under-resolved surface.
    newton_converged = false
    if use_newton
        L = Δ̄mat === nothing ? operator_matrix(f -> laplacian(ops̄, f), grid) : Δ̄mat
        R̄half = scalar_coeffs(R̄) ./ 2
        Fnorm_prev = Inf
        for iter in 1:newton_maxiter
            u = coeffs_scalar(ucoeffs, grid)
            uvals = real.(grid_values(u))
            e2u = exp.(2 .* uvals)
            F = L * ucoeffs .- R̄half .+ scalar_coeffs(make_scalar(e2u, grid))
            Fnorm = norm(F)
            if Fnorm ≤ newton_tol * n
                newton_converged = true   # genuine convergence to round-off
                break
            elseif Fnorm > 0.5 * Fnorm_prev
                break                     # stalled at the aliasing/near-kernel floor
            end
            Fnorm_prev = Fnorm
            J = L + 2 .* multiplication_matrix(make_scalar(e2u, grid))
            # SVD-regularized least-squares step: the Jacobian has a
            # three-dimensional near-kernel (Möbius gauge) at the solution.
            S = svd(J)
            cutoff = 1.0e-8 * S.S[1]
            δ = S.V * ((S.U' * (-F)) .* map(s -> s > cutoff ? 1 / s : 0.0, S.S))
            ucoeffs .+= δ
            newton_iters = iter
        end
    end

    # Re-normalize the area of q̊ to exactly 4π
    u = coeffs_scalar(ucoeffs, grid)
    uvals = real.(grid_values(u))
    areå = integrate_unit(make_scalar(exp.(2 .* uvals) .* real.(grid_values(ops̄.sqrtdetq)), grid))
    uvals .-= log(areå / 4π) / 2
    u = make_scalar(uvals, grid)

    Δu = laplacian(ops̄, u)
    Rvals = real.(exp.(-2 .* uvals) .* (R̄vals .- 2 .* grid_values(Δu)))
    R = make_scalar(Rvals, grid)
    resid = residual(Rvals)
    # Without the Newton polish, fall back to the flow tolerance.
    converged = use_newton ? newton_converged : (isfinite(resid) && resid ≤ flow_tol)
    return Uniformization(u, R, resid, flow_iters, newton_iters, converged)
end
