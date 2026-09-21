"""D3Q19 lattice constants.

Fixed reference data for the standard 3D, 19-velocity lattice: rest + 6 axial
+ 12 face-diagonal directions.

Conventions used throughout the 3D solver:
  - distribution array ``f`` has shape ``(19, nz, ny, nx)``: axis 0 is the
    lattice direction, then spatial axes ordered ``(z, y, x)``.
  - ``C[i] = (cx, cy, cz)`` is the velocity of direction ``i``; streaming
    direction ``i`` shifts ``f[i]`` by ``+cx`` along x (axis 3), ``+cy`` along
    y (axis 2) and ``+cz`` along z (axis 1).
  - macroscopic velocity ``u`` has shape ``(3, nz, ny, nx)`` ordered
    ``(ux, uy, uz)``.
"""

import jax.numpy as jnp

# Discrete velocities (cx, cy, cz), shape (19, 3). Ordered as opposite pairs
# (1,2), (3,4), ... so OPP below is just a pair swap.
C = jnp.array(
    [
        [0, 0, 0],  # 0: rest
        [1, 0, 0],  # 1: +x          2: -x
        [-1, 0, 0],
        [0, 1, 0],  # 3: +y          4: -y
        [0, -1, 0],
        [0, 0, 1],  # 5: +z          6: -z
        [0, 0, -1],
        [1, 1, 0],  # 7: +x+y        8: -x-y
        [-1, -1, 0],
        [1, -1, 0],  # 9: +x-y       10: -x+y
        [-1, 1, 0],
        [1, 0, 1],  # 11: +x+z      12: -x-z
        [-1, 0, -1],
        [1, 0, -1],  # 13: +x-z      14: -x+z
        [-1, 0, 1],
        [0, 1, 1],  # 15: +y+z      16: -y-z
        [0, -1, -1],
        [0, 1, -1],  # 17: +y-z      18: -y+z
        [0, -1, 1],
    ],
    dtype=jnp.int32,
)

# Lattice weights, shape (19,). Sum to 1.
W = jnp.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)

# Opposite direction index: C[OPP[i]] == -C[i].
OPP = jnp.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=jnp.int32)

# Lattice speed of sound squared.
CS2 = 1.0 / 3.0

from bob import mrt as _mrt


# MRT moment basis for D3Q19, evaluated on C above so it matches our velocity
# ordering.  c2 = cx^2 + cy^2 + cz^2.  Row order:
#   0  rho      (conserved)
#   1  jx       (conserved)
#   2  jy       (conserved)
#   3  jz       (conserved)
#   4  3x^2-c^2 (shear — deviatoric stress)
#   5  y^2-z^2  (shear — deviatoric stress)
#   6  xy       (shear)
#   7  yz       (shear)
#   8  xz       (shear)
#   9  c^2      (bulk — trace of second moment)
#  10  (3c^2-5)x  (ghost — energy-flux x)
#  11  (3c^2-5)y  (ghost — energy-flux y)
#  12  (3c^2-5)z  (ghost — energy-flux z)
#  13  x(y^2-z^2) (ghost)
#  14  y(z^2-x^2) (ghost)
#  15  z(x^2-y^2) (ghost)
#  16  c^4        (ghost)
#  17  (3c^2-5)(3x^2-c^2) (ghost)
#  18  (3c^2-5)(y^2-z^2)  (ghost)
def _d3q19_polys():
    def p(f):
        return lambda cx, cy, cz: f(cx, cy, cz, cx * cx + cy * cy + cz * cz)

    return [
        p(lambda x, y, z, c2: 1.0),                                            # 0  rho       conserved
        p(lambda x, y, z, c2: x),                                              # 1  jx        conserved
        p(lambda x, y, z, c2: y),                                              # 2  jy        conserved
        p(lambda x, y, z, c2: z),                                              # 3  jz        conserved
        p(lambda x, y, z, c2: 3 * x * x - c2),                                # 4  3x^2-c^2  shear
        p(lambda x, y, z, c2: y * y - z * z),                                  # 5  y^2-z^2   shear
        p(lambda x, y, z, c2: x * y),                                          # 6  xy        shear
        p(lambda x, y, z, c2: y * z),                                          # 7  yz        shear
        p(lambda x, y, z, c2: x * z),                                          # 8  xz        shear
        p(lambda x, y, z, c2: c2),                                             # 9  c^2       bulk (trace)
        p(lambda x, y, z, c2: (3 * c2 - 5) * x),                              # 10 q-flux x  ghost
        p(lambda x, y, z, c2: (3 * c2 - 5) * y),                              # 11 q-flux y  ghost
        p(lambda x, y, z, c2: (3 * c2 - 5) * z),                              # 12 q-flux z  ghost
        p(lambda x, y, z, c2: x * (y * y - z * z)),                           # 13 m_x       ghost
        p(lambda x, y, z, c2: y * (z * z - x * x)),                           # 14 m_y       ghost
        p(lambda x, y, z, c2: z * (x * x - y * y)),                           # 15 m_z       ghost
        p(lambda x, y, z, c2: c2 * c2),                                        # 16 c^4 (eps) ghost
        p(lambda x, y, z, c2: (3 * c2 - 5) * (3 * x * x - c2)),              # 17 pi_x      ghost
        p(lambda x, y, z, c2: (3 * c2 - 5) * (y * y - z * z)),               # 18 pi_yz     ghost
    ]


M = _mrt.moment_matrix(C, _d3q19_polys())
Minv = jnp.linalg.inv(M)

RHO_IDX, JX_IDX, JY_IDX, JZ_IDX = 0, 1, 2, 3
CONSERVED_IDX = (0, 1, 2, 3)
SHEAR_IDX = (4, 5, 6, 7, 8)        # deviatoric stress -> kinematic viscosity (omega)
BULK_IDX = (9,)                    # trace 2nd moment = energy m_e (MF-LBM s_e)
GHOST_IDX = (10, 11, 12, 13, 14, 15, 16, 17, 18)

DEFAULT_S_BULK = 1.19
DEFAULT_S_GHOST = 1.2

# Per-moment fixed relaxation rates matching MF-LBM's default (mode 2) D3Q19 spectrum
# (Kernel_multiphase.F90 mrt==2). The lumped two-group model above relaxes all of
# {energy-flux q, t-moments, epsilon, 3rd-order pi} at one GHOST rate; the d'Humières
# spectrum gives them distinct rates. Used by mrt.build_S when s_bulk/s_ghost are
# left at their defaults. Shear (4-8) -> omega; conserved (0-3) -> 0.
#   idx 9        energy m_e            -> 1.19   (s_e)
#   idx 10,11,12 energy flux q         -> 1.20   (s_q)
#   idx 13,14,15 t-moments            -> 1.98   (s_t)
#   idx 16       epsilon m_e2          -> 1.40   (s_e2)
#   idx 17,18    3rd-order pi          -> 1.40   (s_pi)
GHOST_RATES = {9: 1.19, 10: 1.2, 11: 1.2, 12: 1.2, 13: 1.98, 14: 1.98, 15: 1.98, 16: 1.4, 17: 1.4, 18: 1.4}
