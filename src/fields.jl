# Helpers for scalar/tensor fields on the sphere, built on
# AbstractSphericalHarmonics' `Tensor`/`SpinTensor` representation.
#
# Conventions (see docs/algorithm.tex §4.1):
# - Fields live on the Driscoll--Healy collocation grid of size
#   `ash_grid_size(lmax) = (nphi, ntheta)`.
# - Tensor components are stored in the orthonormal coordinate dyad
#   (eθ, eϕ) of the *unit coordinate sphere* q̂; covariant and
#   contravariant components coincide in this frame with respect to q̂.
# - `grad` is the covariant derivative ∇̂ with respect to q̂; the
#   derivative index is the last index.

const Scalar0 = SArray{Tuple{},ComplexF64,0,1}

"Coordinates (θ, ϕ) of all grid points"
grid_coords(lmax::Int) = [SVector{2}(ash_point_coord(ij, lmax)) for ij in CartesianIndices(ash_grid_size(lmax))]

"Evaluate `f(θ, ϕ)::Number` on the grid as a `Tensor{0}`"
function scalar_field(f, lmax::Int)
    values = [Scalar0(ComplexF64(f(θϕ[1], θϕ[2]))) for θϕ in grid_coords(lmax)]
    return Tensor{0}(values, lmax)
end

"Unwrap a `Tensor{0}` into a complex matrix of grid values"
grid_values(t::Tensor{0}) = map(v -> v[], t.values)

"Wrap a complex matrix of grid values into a `Tensor{0}`"
make_scalar(values::AbstractMatrix{<:Number}, lmax::Int) = Tensor{0}(Scalar0.(ComplexF64.(values)), lmax)

"Largest |imaginary part| over all components and points"
imag_norm(t::Tensor) = maximum(v -> maximum(abs.(imag.(v))), t.values)

"Drop imaginary parts (validity should be checked via `imag_norm`)"
real_part(t::Tensor{D}) where {D} = Tensor{D}(map(v -> ComplexF64.(real.(v)), t.values), t.lmax)

"Symmetrize the two indices of a rank-2 tensor"
symmetrize(t::Tensor{2}) = Tensor{2}(map(v -> (v + transpose(v)) / 2, t.values), t.lmax)

"Pointwise application: `map_fields(op, ts...)` applies `op` to the component arrays at each grid point"
function map_fields(op, ts::Tensor...)
    lmax = ts[1].lmax
    values = map((vs...) -> op(vs...), (t.values for t in ts)...)
    return Tensor(values, lmax)
end

"Spectral coefficients of a scalar field"
scalar_coeffs(t::Tensor{0}) = copy(SpinTensor(t).coeffs[])

"Scalar field from spectral coefficients"
function coeffs_scalar(c::AbstractVector{<:Complex}, lmax::Int)
    return Tensor(SpinTensor{0}(SArray{Tuple{}}((Vector{ComplexF64}(c),)), lmax))
end

"∮ f ε̂ over the unit coordinate sphere (exact for band-limited f)"
function integrate_unit(f::Tensor{0})
    c = SpinTensor(f).coeffs[]
    return sqrt(4π) * real(c[ash_mode_index(0, 0, 0, f.lmax)])
end

"Covariant derivative ∇̂ w.r.t. the unit coordinate sphere; derivative index last"
grad(t::Tensor{D}) where {D} = Tensor(tensor_gradient(SpinTensor(t)))

"Like `grad` but with 2/3-rule mode filtering (dealiasing)"
grad_filtered(t::Tensor{D}) where {D} = Tensor(filter_modes(tensor_gradient(SpinTensor(t))))

"Filter a tensor field with the 2/3 rule"
dealias(t::Tensor{D}) where {D} = Tensor(filter_modes(SpinTensor(t)))
