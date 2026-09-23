using AbstractSphericalHarmonics
using ApparentHorizonFinder
using KorzynskiSpin
using LinearAlgebra
using StaticArrays
using Test

include("kerr_schild.jl")

# `::SVector{3}` annotations mark these as per-point providers (auto-wrapped
# for the batched Cauchy-data interface).
const flat3 = (x::SVector{3}) -> SMatrix{3,3,Float64}(I)
const zero3 = (x::SVector{3}) -> zero(SMatrix{3,3,Float64})

"Best-fit rotation Λ with χ_i ≈ Λ_ij n_j, n the unit direction from `center`"
function fit_chi_rotation(res::SpinResult, center::SVector{3,Float64})
    M = zeros(3, 3)
    for ij in CartesianIndices(size(res.geometry.x))
        n = res.geometry.x[ij] - center
        n /= norm(n)
        χv = [real(grid_values(res.eigenfunctions.χ[i])[ij]) for i in 1:3]
        M .+= χv * n'
    end
    U, _, V = svd(M)
    return U * V'
end

@testset "KorzynskiSpin" begin
    @testset "Round sphere in flat space" begin
        lmax = 12
        res = horizon_spin(shape_embedding((θ, ϕ) -> 1.0), flat3, zero3; lmax=lmax)
        @test res.area ≈ 4π atol = 1.0e-12
        @test res.J ≈ 0 atol = 1.0e-12
        @test norm(res.Jvec) ≈ 0 atol = 1.0e-12
        @test norm(res.Kvec) ≈ 0 atol = 1.0e-12
        @test res.diagnostics[:gauss_bonnet] ≈ 0 atol = 1.0e-10
        @test res.diagnostics[:round_residual] < 1.0e-10
        @test res.diagnostics[:eigenvalue_offset] < 1.0e-9
        @test res.diagnostics[:takahashi_residual] < 1.0e-9
        @test maximum(abs.(real.(grid_values(res.uniformization.u)))) < 1.0e-10
    end

    @testset "Off-center sphere (round metric, distorted chart)" begin
        lmax = 16
        c = SVector(0.1, 0.15, -0.2)
        function h(θ, ϕ)
            r̂ = SVector(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))
            rc = dot(r̂, c)
            return rc + sqrt(1 - dot(c, c) + rc^2)
        end
        res = horizon_spin(shape_embedding(h), flat3, zero3; lmax=lmax)
        @test res.area ≈ 4π atol = 1.0e-10
        @test res.J ≈ 0 atol = 1.0e-10
        # u ≈ 0: the metric is already round, only the chart is distorted
        @test maximum(abs.(real.(grid_values(res.uniformization.u)))) < 1.0e-8
        @test res.diagnostics[:eigenvalue_offset] < 1.0e-8
        @test res.diagnostics[:takahashi_residual] < 1.0e-8
        # χ must reproduce the directions seen from the sphere's own center,
        # up to a proper rotation
        Λ = fit_chi_rotation(res, c)
        @test det(Λ) ≈ 1 atol = 1.0e-8
        err = maximum(CartesianIndices(size(res.geometry.x))) do ij
            n = res.geometry.x[ij] - c
            χv = SVector{3}(real(grid_values(res.eigenfunctions.χ[i])[ij]) for i in 1:3)
            norm(χv - Λ * n / norm(n))
        end
        @test err < 1.0e-7
    end

    @testset "Ellipsoid in flat space" begin
        abc = SVector(1.0, 1.2, 0.8)
        emb = (θ, ϕ) -> SVector(abc[1] * sin(θ) * cos(ϕ), abc[2] * sin(θ) * sin(ϕ), abc[3] * cos(θ))
        curvature_error = Float64[]
        for lmax in (16, 24)
            grid = EquiangularGrid(lmax)
            geom = surface_geometry(emb, flat3, zero3, grid)
            ops = MetricOps(geom.q)
            R = real_part(scalar_curvature(ops))
            @test integrate(geom, R) ≈ 8π atol = 1.0e-6
            # analytic Gauss curvature: K = 1/((abc)² (x²/a⁴+y²/b⁴+z²/c⁴)²)
            err = maximum(CartesianIndices(size(geom.x))) do ij
                x = geom.x[ij]
                w = x[1]^2 / abc[1]^4 + x[2]^2 / abc[2]^4 + x[3]^2 / abc[3]^4
                abs(real(grid_values(R)[ij]) - 2 / (prod(abc)^2 * w^2))
            end
            push!(curvature_error, err)
        end
        # spectral convergence
        @test curvature_error[2] < 5.0e-2 * curvature_error[1]
        @test curvature_error[2] < 1.0e-4

        res = horizon_spin(emb, flat3, zero3; lmax=24)
        @test res.J ≈ 0 atol = 1.0e-10
        @test res.diagnostics[:round_residual] < 1.0e-7
        @test res.diagnostics[:takahashi_residual] < 1.0e-7
    end

    @testset "Synthetic rotation form on the round sphere" begin
        lmax = 16
        grid = EquiangularGrid(lmax)
        emb = shape_embedding((θ, ϕ) -> 1.0)
        geom = surface_geometry(emb, flat3, zero3, grid)
        ops = MetricOps(geom.q)
        Δmat = operator_matrix(f -> laplacian(ops, f), grid)

        α, β, γc = 0.3, -0.2, 0.5
        f = scalar_field((θ, ϕ) -> α * cos(θ) + β * sin(θ) * cos(ϕ), grid)
        g = scalar_field((θ, ϕ) -> γc * sin(θ) * sin(ϕ) + 0.1 * (3cos(θ)^2 - 1), grid)
        df = differential(f)
        dg = differential(g)
        # ω = ⋆df + dg in dyad components: (⋆α)_1 = α_2, (⋆α)_2 = −α_1
        ω = map_fields((a, b) -> SVector{2}(a[2] + b[1], -a[1] + b[2]), df, dg)

        ωinv, gsol = hodge_fix(ops, ω; Δmat=Δmat)
        gdev = real.(grid_values(gsol)) .- real.(grid_values(g))
        @test maximum(gdev) - minimum(gdev) < 1.0e-12   # recovered up to a constant

        R̄ = make_scalar(fill(2.0 + 0im, size(grid_values(f))), grid)
        unif = uniformize(ops, R̄)
        eig = sphere_eigenfunctions(ops, unif.u; Δ̄mat=Δmat)
        gen = mobius_generators(ops, unif.u, eig)
        Jvec = SVector{3}(momentum_integral(geom, ωinv, gen.φ[i]) for i in 1:3)
        Kvec = SVector{3}(momentum_integral(geom, ωinv, gen.ξ[i]) for i in 1:3)

        # Analytic: J_i = −(1/3) Λ_ij c_j with c the l=1 coefficients of f
        # in the (x, y, z) basis, Λ the χ-frame rotation; K⃗ = 0.
        M = zeros(3, 3)
        for ij in CartesianIndices(size(geom.x))
            n = geom.x[ij]
            χv = [real(grid_values(eig.χ[i])[ij]) for i in 1:3]
            M .+= χv * n'
        end
        U, _, V = svd(M)
        Λr = U * V'
        c = SVector(β, 0.0, α)
        @test norm(Jvec - (-Λr * c / 3)) < 1.0e-12
        @test norm(Kvec) < 1.0e-12

        # Gauge invariance: J must not change when a gradient is added to ω
        Jvec2 = SVector{3}(momentum_integral(geom, hodge_fix(ops, map_fields((w, d) -> w + 2 .* d, ω, dg); Δmat=Δmat)[1], gen.φ[i]) for i in 1:3)
        @test norm(Jvec2 - Jvec) < 1.0e-12
    end

    @testset "Kerr horizon (Kerr–Schild slice)" begin
        M, a = 1.0, 0.6
        lmax = 20
        ks = KerrSchild(M, a)
        res = horizon_spin(ks_horizon_embedding(M, a), slice_metric(ks), slice_excurv(ks); lmax=lmax)
        @test res.area ≈ ks_horizon_area(M, a) atol = 1.0e-10
        # the points-matrix entry agrees with the callable entry
        grid = EquiangularGrid(lmax)
        emb = ks_horizon_embedding(M, a)
        pts = [SVector{3,Float64}(emb(θϕ...)) for θϕ in grid_coords(grid)]
        res_pts = horizon_spin(pts, slice_metric(ks), slice_excurv(ks))
        @test res_pts.J == res.J
        @test res.J ≈ M * a atol = 1.0e-12
        @test norm(res.Kvec) < 1.0e-10              # axisymmetric: K⃗ = 0 after gauge fixing
        @test res.axis_embedding ≈ SVector(0.0, 0.0, 1.0) atol = 1.0e-10
        @test res.diagnostics[:boost_consistency] < 1.0e-12
        @test res.diagnostics[:gauss_bonnet] ≈ 0 atol = 1.0e-8
        @test res.diagnostics[:round_residual] < 1.0e-9
    end

    @testset "Batched Cauchy-data provider interface" begin
        M, a = 1.0, 0.6
        lmax = 20
        ks = KerrSchild(M, a)
        emb = ks_horizon_embedding(M, a)

        # Reference: auto-detected per-point providers (argument annotated).
        ref = horizon_spin(emb, slice_metric(ks), slice_excurv(ks); lmax=lmax)

        # Native batched providers: called once with the whole grid of points,
        # returning an array of the same shape.  Also count the invocations.
        mcalls = Ref(0)
        kcalls = Ref(0)
        metric_batched = function (Xs)
            mcalls[] += 1
            return map(slice_metric(ks), Xs)
        end
        excurv_batched = function (Xs)
            kcalls[] += 1
            return map(slice_excurv(ks), Xs)
        end
        res_batched = horizon_spin(emb, metric_batched, excurv_batched; lmax=lmax)
        @test res_batched.J == ref.J
        @test res_batched.area == ref.area
        # A single batched call per provider, not one per collocation point.
        @test mcalls[] == 1
        @test kcalls[] == 1

        # Bare (untyped) per-point closures wrapped explicitly with `pointwise`.
        bare_metric = x -> slice_metric(ks)(x)
        bare_excurv = x -> slice_excurv(ks)(x)
        res_pointwise = horizon_spin(emb, pointwise(bare_metric), pointwise(bare_excurv); lmax=lmax)
        @test res_pointwise.J == ref.J
        @test res_pointwise.area == ref.area
    end

    @testset "Schwarzschild horizon (Kerr–Schild slice)" begin
        M = 1.0
        ks = KerrSchild(M, 0.0)
        res = horizon_spin(ks_horizon_embedding(M, 0.0), slice_metric(ks), slice_excurv(ks); lmax=12)
        @test res.area ≈ 16π * M^2 atol = 1.0e-10
        @test res.J ≈ 0 atol = 1.0e-12
        @test norm(res.Jvec) < 1.0e-12
        @test norm(res.Kvec) < 1.0e-12
    end

    @testset "Rotated and translated Kerr" begin
        M, a = 1.0, 0.6
        lmax = 20
        # rotation taking ẑ to n̂ (ZYZ Euler angles), plus a spatial translation
        n̂ = SVector(1.0, 2.0, 2.0) / 3
        ψ, θr = atan(n̂[2], n̂[1]), acos(n̂[3])
        b = SVector(0.3, -0.2, 0.1)
        m = translate(rotate(KerrSchild(M, a), ψ, θr, 0.0), SVector(0.0, b...))
        # rotation matrix R ẑ = n̂, for transforming the embedding
        cψ, sψ, cθ, sθ = cos(ψ), sin(ψ), cos(θr), sin(θr)
        Λ = SMatrix{3,3}(cψ * cθ, sψ * cθ, -sθ, -sψ, cψ, 0, cψ * sθ, sψ * sθ, cθ)
        @assert Λ * SVector(0.0, 0.0, 1.0) ≈ n̂

        emb0 = ks_horizon_embedding(M, a)
        emb = (θ, ϕ) -> Λ * emb0(θ, ϕ) + b

        res = horizon_spin(emb, slice_metric(m), slice_excurv(m); lmax=lmax)
        @test res.J ≈ M * a atol = 1.0e-11
        @test res.axis_embedding ≈ n̂ atol = 1.0e-9
    end

    @testset "Spectral convergence of J (Kerr)" begin
        M, a = 1.0, 0.6
        ks = KerrSchild(M, a)
        errs = [abs(horizon_spin(ks_horizon_embedding(M, a), slice_metric(ks), slice_excurv(ks); lmax=lmax).J - M * a) for
                lmax in (8, 12, 16)]
        @test errs[2] < errs[1]
        @test errs[3] < 1.0e-2 * errs[1]
        @test errs[3] < 1.0e-8
    end

    @testset "Boosted, rotated, translated Kerr with ApparentHorizonFinder shape" begin
        # The acid test of the Hodge gauge fixing (Korzyński 2007): a boost
        # genuinely changes the slicing (∂_t γ ≠ 0 enters K_ij), yet J and the
        # area are slice-independent.  The shape is not analytic input but
        # found numerically by ApparentHorizonFinder on the shared SphereGrid.
        M, a = 1.0, 0.6
        r₊ = M + sqrt(M^2 - a^2)
        n̂ = SVector(1.0, 2.0, 2.0) / 3
        ψ, θr = atan(n̂[2], n̂[1]), acos(n̂[3])
        b = SVector(0.3, -0.2, 0.1)

        # boost along the (rotated) spin axis: J, area, and axis all survive
        m = translate(boost(rotate(KerrSchild(M, a), ψ, θr, 0.0), 0.3 * n̂), SVector(0.0, b...))
        hor = find_horizon(slice_admvars(m), b + SVector(0.05, 0.0, 0.0), EquiangularGrid(15), 2.5, 0.0, 300; verbosity=0)
        @test hor.success
        @test hor.origin ≈ b atol = 1.0e-10              # recentring finds the translation
        res = horizon_spin(hor, slice_metric(m), slice_excurv(m))
        # (observed at lmax=15: area err 4e-10, J err 6e-11, axis err 2e-11)
        @test res.area ≈ 8π * M * r₊ atol = 1.0e-8       # slice-independent area
        @test res.J ≈ M * a atol = 1.0e-9                # tilted-foliation invariance of J
        @test res.axis_embedding ≈ n̂ atol = 1.0e-9       # boost ∥ axis preserves the axis

        # spectral resampling to a finer grid (observed: J err 2e-13)
        res2 = horizon_spin(hor, slice_metric(m), slice_excurv(m); grid=EquiangularGrid(19))
        @test res2.J ≈ M * a atol = 1.0e-11

        # the points-matrix entry is identical to the NamedTuple entry
        res3 = horizon_spin(horizon_points(hor), slice_metric(m), slice_excurv(m))
        @test res3.J == res.J

        # transverse boost: assert only the invariants (the embedding-axis
        # reporting proxy is coordinate-dependent under aberration)
        v⊥ = 0.25 * normalize(cross(n̂, SVector(0.0, 0.0, 1.0)))
        m⊥ = boost(rotate(KerrSchild(M, a), ψ, θr, 0.0), v⊥)
        hor⊥ = find_horizon(slice_admvars(m⊥), SVector(0.05, 0.0, 0.0), EquiangularGrid(15), 2.5, 0.0, 300; verbosity=0)
        @test hor⊥.success
        res⊥ = horizon_spin(hor⊥, slice_metric(m⊥), slice_excurv(m⊥))
        @test res⊥.area ≈ 8π * M * r₊ atol = 1.0e-8
        @test res⊥.J ≈ M * a atol = 1.0e-9
    end

    @testset "Boost identities" begin
        # eq. (21) reduces to |J⃗| for K⃗ = 0
        Jv = SVector(0.3, -0.1, 0.7)
        J, A, B = spin_from_invariants(Jv, zero(SVector{3,Float64}))
        @test J ≈ norm(Jv)
        # boost to the parallel frame: J⃗′ ∥ K⃗′ and |J⃗′| equals the invariant
        Kv = SVector(0.2, 0.5, -0.1)
        J2, _, _ = spin_from_invariants(Jv, Kv)
        β⃗ = parallel_frame_boost(Jv, Kv)
        @test 0 < norm(β⃗) < 1
        Jv′, Kv′ = mobius_boost(Jv, Kv, β⃗)
        @test norm(cross(Jv′, Kv′)) < 1.0e-12
        @test norm(Jv′) ≈ J2 atol = 1.0e-12
        # invariants are preserved by arbitrary boosts
        β⃗2 = SVector(0.1, -0.3, 0.2)
        Jv″, Kv″ = mobius_boost(Jv, Kv, β⃗2)
        @test dot(Jv″, Jv″) - dot(Kv″, Kv″) ≈ dot(Jv, Jv) - dot(Kv, Kv) atol = 1.0e-12
        @test dot(Jv″, Kv″) ≈ dot(Jv, Kv) atol = 1.0e-12
    end

    @testset "Balanced frame: uniqueness and chart independence" begin
        # an ellipsoid in a Möbius-distorted chart: the flow lands on an
        # unbalanced round metric, the balancing does not
        abc = SVector(1.0, 1.2, 0.8)
        r̂(θ, ϕ) = SVector(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))
        Λb = KorzynskiSpin.boost_matrix(SVector(0.15, -0.1, 0.2))
        res1 = horizon_spin((θ, ϕ) -> abc .* r̂(θ, ϕ), flat3, zero3; lmax=32)
        res2 = horizon_spin((θ, ϕ) -> abc .* KorzynskiSpin.mobius_point(Λb, r̂(θ, ϕ))[1], flat3, zero3; lmax=32)
        mp1 = horizon_multipoles(res1; lmax=4)
        mp2 = horizon_multipoles(res2; lmax=4)
        @test mp2.diagnostics[:balance_iters] ≥ 1
        for mp in (mp1, mp2)
            @test mp.diagnostics[:dipole_residual] < 1.0e-13
            @test mp.diagnostics[:I00_offset] < 1.0e-12
            @test mp.diagnostics[:conjugation] < 1.0e-12
            @test mp.diagnostics[:invariant_A] < 1.0e-12
            @test maximum(abs.(values(mp.L))) < 1.0e-12        # no rotation
        end
        # (observed: 2e-11)
        @test maximum(abs(mp1.I[k] - mp2.I[k]) for k in keys(mp1.I)) < 1.0e-9
        @test real(mp1.I[(2, 0)]) > 0.1                        # genuinely deformed
        @test maximum(abs(mp1.I[k]) for k in keys(mp1.I) if isodd(k[2])) < 1.0e-12   # reflection symmetry

        # Uniqueness: balancing from an arbitrarily boosted start lands on the
        # same round metric
        Λ0 = KorzynskiSpin.boost_matrix(SVector(-0.1, 0.3, 0.25))
        χ0 = KorzynskiSpin.point_triple(res1.eigenfunctions.χ)
        imgs = map(c -> KorzynskiSpin.mobius_point(Λ0, c), χ0)
        χp = ntuple(i -> make_scalar(map(p -> p[1][i], imgs), res1.grid), 3)
        up = make_scalar(real.(grid_values(res1.uniformization.u)) .+ map(p -> log(p[2]), imgs), res1.grid)
        bal0 = balance_frame(res1.geometry, res1.uniformization.u, res1.eigenfunctions.χ)
        balp = balance_frame(res1.geometry, up, χp)
        @test balp.converged && balp.iters ≥ 2
        @test maximum(abs.(real.(grid_values(balp.u)) .- real.(grid_values(bal0.u)))) < 1.0e-12
    end

    @testset "Kerr horizon multipoles (balanced frame)" begin
        M, a = 1.0, 0.6
        lmax = 20
        ks = KerrSchild(M, a)
        res = horizon_spin(ks_horizon_embedding(M, a), slice_metric(ks), slice_excurv(ks); lmax=lmax)
        mp = horizon_multipoles(res; lmax=6)
        @test mp.diagnostics[:dipole_residual] < 1.0e-13
        @test mp.diagnostics[:I00_offset] < 1.0e-12
        @test mp.diagnostics[:L00] < 1.0e-12
        @test mp.diagnostics[:invariant_A] < 1.0e-12
        @test mp.diagnostics[:invariant_B] < 1.0e-12
        # the canonical round sphere is Gourgoulhon et al.'s z(cos θ)
        # (observed: 1e-12)
        r₊ = M + sqrt(M^2 - a^2)
        zerr = maximum(CartesianIndices(size(res.geometry.x))) do ij
            abs(real(grid_values(mp.χ[3])[ij]) - kerr_balanced_z(M, a, res.geometry.x[ij][3] / r₊))
        end
        @test zerr < 1.0e-10
        @test mp.L1vec[3] > 0                                  # current dipole ∥ spin
        # closed-form multipoles (observed at lmax=20: ≤ 5e-12)
        for l in 0:6
            ref = kerr_multipole(M, a, l)
            @test mp.I[(l, 0)] ≈ real(ref) atol = 1.0e-10
            @test mp.L[(l, 0)] ≈ imag(ref) atol = 1.0e-10
            for m in (-l):l
                m == 0 && continue
                @test abs(mp.I[(l, m)]) + abs(mp.L[(l, m)]) < 1.0e-12   # axisymmetry
            end
        end

        # Schwarzschild: only the universal monopole survives
        res0 = horizon_spin(
            ks_horizon_embedding(M, 0.0), slice_metric(KerrSchild(M, 0.0)), slice_excurv(KerrSchild(M, 0.0)); lmax=12
        )
        mp0 = horizon_multipoles(res0; lmax=4)
        @test maximum(abs(mp0.I[k]) + abs(mp0.L[k]) for k in keys(mp0.I) if k[1] ≥ 1) < 1.0e-12

        # Rotated and translated data: the orientation fixing (ẑ ∥ current
        # dipole) makes the multipoles identical
        n̂ = SVector(1.0, 2.0, 2.0) / 3
        ψ, θr = atan(n̂[2], n̂[1]), acos(n̂[3])
        b = SVector(0.3, -0.2, 0.1)
        m = translate(rotate(ks, ψ, θr, 0.0), SVector(0.0, b...))
        cψ, sψ, cθ, sθ = cos(ψ), sin(ψ), cos(θr), sin(θr)
        Λ = SMatrix{3,3}(cψ * cθ, sψ * cθ, -sθ, -sψ, cψ, 0, cψ * sθ, sψ * sθ, cθ)
        emb0 = ks_horizon_embedding(M, a)
        resr = horizon_spin((θ, ϕ) -> Λ * emb0(θ, ϕ) + b, slice_metric(m), slice_excurv(m); lmax=lmax)
        mpr = horizon_multipoles(resr; lmax=6)
        @test maximum(abs(mpr.I[k] - mp.I[k]) + abs(mpr.L[k] - mp.L[k]) for k in keys(mp.I)) < 1.0e-10
    end

    @testset "Kerr multipoles on boosted slices (ApparentHorizonFinder shape)" begin
        # All cross-sections of a Killing horizon are isometric and Ψ₂ is
        # invariant, so the balanced multipoles must not depend on the slicing
        M, a = 1.0, 0.6
        n̂ = SVector(1.0, 2.0, 2.0) / 3
        ψ, θr = atan(n̂[2], n̂[1]), acos(n̂[3])
        v⊥ = 0.25 * normalize(cross(n̂, SVector(0.0, 0.0, 1.0)))
        for v⃗ in (0.3 * n̂, v⊥)
            m = boost(rotate(KerrSchild(M, a), ψ, θr, 0.0), v⃗)
            hor = find_horizon(slice_admvars(m), SVector(0.05, 0.0, 0.0), EquiangularGrid(15), 2.5, 0.0, 300; verbosity=0)
            @test hor.success
            res = horizon_spin(hor, slice_metric(m), slice_excurv(m); grid=EquiangularGrid(23))
            mp = horizon_multipoles(res; lmax=6)
            @test mp.diagnostics[:dipole_residual] < 1.0e-13
            @test mp.diagnostics[:invariant_A] < 1.0e-12
            # (observed: ≤ 1e-11)
            for l in 0:6
                ref = kerr_multipole(M, a, l)
                @test mp.I[(l, 0)] ≈ real(ref) atol = 1.0e-9
                @test mp.L[(l, 0)] ≈ imag(ref) atol = 1.0e-9
            end
            @test maximum(abs(mp.I[k]) + abs(mp.L[k]) for k in keys(mp.I) if k[2] ≠ 0) < 1.0e-9
        end
    end
end
