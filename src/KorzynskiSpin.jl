# Quasi-local spin of a distorted horizon via Korzyński's conformal
# decomposition (Class. Quantum Grav. 24, 5935 (2007), arXiv:0707.2824).
# The algorithm is worked out in detail in docs/algorithm.tex.

module KorzynskiSpin

using AbstractSphericalHarmonics
import FastSphericalHarmonics
using LinearAlgebra
using StaticArrays

export scalar_field, grid_values, make_scalar, grid_coords, integrate_unit, grad, differential
export imag_norm, real_part, symmetrize, map_fields, scalar_coeffs, coeffs_scalar
export SurfaceGeometry, surface_geometry, shape_embedding, integrate, pointwise
export MetricOps, scalar_curvature, laplacian, divergence, operator_matrix, multiplication_matrix
export Uniformization, uniformize
export SphereEigenfunctions, sphere_eigenfunctions, MobiusGenerators, mobius_generators
export hodge_fix, momentum_integral, spin_from_invariants, mobius_boost, parallel_frame_boost
export boosted_rotation_generators, SpinResult, horizon_spin

include("fields.jl")
include("geometry.jl")
include("calculus.jl")
include("uniformize.jl")
include("eigenfunctions.jl")
include("invariants.jl")

end
