# KorzyńskiSpin.jl

- Mikołaj Korzyński, "Quasi-local angular momentum of non-symmetric
  isolated and dynamical horizons from the conformal decomposition of
  the metric", [arXiv:0707.2824 [gr-qc]](http://arxiv.org/abs/0707.2824)

- R. Gómez, L. Lehner, P. Papadopoulos, J. Winicour, "The eth
  formalism in numerical relativity",
  [arXiv:gr-qc/9702002](https://arxiv.org/abs/gr-qc/9702002)

- Carsten Gundlach, "Pseudo-spectral apparent horizon finders: an
  efficient new algorithm",
  [arXiv:gr-qc/9707050](https://arxiv.org/abs/gr-qc/9707050)

- Olaf Dreyer, Badri Krishnan, Erik Schnetter, Deirdre Shoemaker,
  "Introduction to Isolated Horizons in Numerical Relativity",
  [arXiv:gr-qc/0206008](https://arxiv.org/abs/gr-qc/0206008)

- Gregory B. Cook, "Initial Data for Numerical Relativity",
  [arXiv:gr-qc/0007085](https://arxiv.org/abs/gr-qc/0007085)

<!--
- Lap-Ming Lin, Jérôme Novak, "A new spectral apparent horizon finder
  for 3D numerical relativity",
  [arXiv:gr-qc/0702038](https://arxiv.org/abs/gr-qc/0702038)

- Antonios A. Tsokaros, Kōji Uryū, "Numerical method for binary black
  hole/neutron star initial data: Code test",
  [arXiv:gr-qc/0703030](https://arxiv.org/abs/gr-qc/0703030)
-->

- [List of formulas in Riemannian geometry](https://en.m.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry)
- [Ricci curvature](https://en.wikipedia.org/wiki/Ricci_curvature)
- [Ricci flow](https://en.wikipedia.org/wiki/Ricci_flow)
- [Riemann curvature tensor](https://en.wikipedia.org/wiki/Riemann_curvature_tensor)
- [Spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics)
- [Spin-weighted spherical harmonics](https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics)

- [AbstractSphericalHarmonics.jl](https://github.com/eschnett/AbstractSphericalHarmonics.jl)
- [FastSphericalHarmonics.jl](https://github.com/eschnett/FastSphericalHarmonics.jl)
- [SSHT.jl](https://github.com/eschnett/SSHT.jl)

## Competing work

- László B. Szabados, "Quasi-Local Energy-Momentum and Angular
  Momentum in GR: A Review Article", Living Reviews in Relativity
  **12**, 4 (2009),
  [DOI:10.12942/lrr-2009-4](https://doi.org/10.12942/lrr-2009-4)

- Gregory B. Cook, Bernard F. Whiting, "Approximate Killing Vectors on
  S^2", [arXiv:0706.0199 [gr-qc]](https://arxiv.org/abs/0706.0199)

- José Luis Jaramillo, Eric Gourgoulhon, "Mass and Angular Momentum in
  General Relativity", [arXiv:1001.5429
  [gr-qc]](https://arxiv.org/abs/1001.5429)

- Robert Owen, Alex S. Fox, John A. Freiberg, Terrence Pierre Jacques,
  "Black Hole Spin Axis in Numerical Relativity", [arXiv:1708.07325
  [gr-qc]](https://arxiv.org/abs/1708.07325)

## Notes

### Rotation one-form

    \omega_A = - (D_A l^b) k_b = (D_A k^b) l_b
    with   l^b k_b = -1

    q_ab := g_ab + l_a k_b + k_a l_b

    \omega^a = - k^c q^ab D_b l_c
    \Theta   = q^ab D_b l_a

See Dreyer et al.:

    \omega_a = K_ab s^b



### Conformal decomposition and transformations

Six vector fields:

Rotations about the three orthogonal axes:

    \phi_1 = [- \sin\phi, - \cot\theta \cos\phi]   (\curl y)
    \phi_2 = [  \cos\phi, - \cot\theta \sin\phi]   (\curl x)
    \phi_3 = [  0       ,   1                  ]   (\curl z)

Proper conformal transformations along the three axes:

    \xi_1 = [- \cos\theta \cos\phi,   \sin\phi / \sin\theta]   (\grad y)
    \xi_2 = [- \cos\theta \sin\phi, - \cos\phi / \sin\theta]   (\grad x)
    \xi_3 = [  \sin\theta         ,   0                    ]   (\grad z)

Integrated action of proper conformal transformation:

    \xi := n_i \xi_i
    with   n_i n_j \delta^ij = 1
    
    \phi\tilde_i = n_i (n_k \phi_k) + \cosh\lambda ((\delta_ik - n_i n_k) \phi_k) + \sinh\lambda (\epsilon_ijk n_j      \xi_k)
     \xi\tilde_i = n_i (n_k  \xi_k) - \sinh\lambda (\epsilon_ijk n_j      \phi_k) + \cosh\lambda ((\delta_ik - n_i n_k) \xi_k)
