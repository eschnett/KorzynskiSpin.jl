using AbstractSphericalHarmonics
using ApparentHorizonFinder
using KorzynskiSpin
using LinearAlgebra
using StaticArrays
using Test

include("kerr_schild.jl")

const flat3 = x -> SMatrix{3,3,Float64}(I)
const zero3 = x -> zero(SMatrix{3,3,Float64})

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
        unif = uniformize(ops, R̄; Δ̄mat=Δmat)
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
end
