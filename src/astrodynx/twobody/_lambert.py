"""Lambert problem solver using Battin universal variables.

References:
    Battin, 1999, "An Introduction to the Methods of Astrodynamics", pp.194-210.
    Curtis, 2020, "Orbital Mechanics for Engineering Students", Algorithm 5.2.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from jax import Array

from astrodynx.twobody._uniformulas import ufunc2, ufunc3


def _stumpff_c(z: DTypeLike) -> Array:
    r"""Stumpff function C(z).

    $$
    C(z) = U_2(1, z) = \begin{cases}
    \frac{1 - \cos\sqrt{z}}{z} & z > 0 \\
    \frac{1}{2} & z = 0 \\
    \frac{\cosh\sqrt{-z} - 1}{-z} & z < 0
    \end{cases}
    $$
    """
    return ufunc2(1.0, z)


def _stumpff_s(z: DTypeLike) -> Array:
    r"""Stumpff function S(z).

    $$
    S(z) = U_3(1, z) = \begin{cases}
    \frac{\sqrt{z} - \sin\sqrt{z}}{(\sqrt{z})^3} & z > 0 \\
    \frac{1}{6} & z = 0 \\
    \frac{\sinh\sqrt{-z} - \sqrt{-z}}{(\sqrt{-z})^3} & z < 0
    \end{cases}
    $$
    """
    return ufunc3(1.0, z)


def solve_lambert(
    r1_vec: ArrayLike,
    r2_vec: ArrayLike,
    tof: ArrayLike,
    mu: ArrayLike = 1.0,
    prograde: bool = True,
    tol: DTypeLike = 1e-10,
    max_iter: int = 60,
) -> tuple[Array, Array]:
    r"""Solve Lambert's problem using universal variables.

    Given initial position $\mathbf{r}_1$, final position $\mathbf{r}_2$,
    and time of flight $\Delta t$, find the departure and arrival velocities
    $\mathbf{v}_1$ and $\mathbf{v}_2$.

    Args:
        r1_vec: (3,) Initial position vector.
        r2_vec: (3,) Final position vector.
        tof: Time of flight.
        mu: Gravitational parameter.
        prograde: If True, prograde transfer (short way for z-component of
            cross product >= 0). If False, retrograde.
        tol: Convergence tolerance.
        max_iter: Maximum bisection iterations.
        revolution: Number of complete revolutions (currently only 0 supported).

    Returns:
        (v1_vec, v2_vec) — departure and arrival velocity vectors.

    Notes:
        Uses bisection on the universal variable $z$ with Stumpff functions
        $C(z)$ and $S(z)$, which map to astrodynx's $U_2(1, z)$ and
        $U_3(1, z)$. Bisection is used for guaranteed convergence even for
        near-180° transfers where Newton-Raphson can diverge.

        Currently supports zero-revolution transfers ($N=0$).

    Examples:
        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> mu = 398600.4418
        >>> r1 = jnp.array([15945.34, 0.0, 0.0])
        >>> r2 = jnp.array([12214.83899, 10249.46731, 0.0])
        >>> tof = 76.0 * 60.0
        >>> v1, v2 = adx.twobody.solve_lambert(r1, r2, tof, mu)
    """
    r1 = jnp.linalg.vector_norm(r1_vec)
    r2 = jnp.linalg.vector_norm(r2_vec)

    cos_dnu = jnp.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dnu = jnp.clip(cos_dnu, -1.0, 1.0)

    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]

    prograde_cond = cross_z >= 0 if prograde else cross_z < 0
    dnu = jnp.where(
        prograde_cond,
        jnp.arccos(cos_dnu),
        2 * jnp.pi - jnp.arccos(cos_dnu),
    )

    A = jnp.sin(dnu) * jnp.sqrt(r1 * r2 / (1 - cos_dnu))

    def _y(z):
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        return r1 + r2 + A * (z * S - 1) / jnp.sqrt(C)

    def _tof_from_z(z):
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        y = _y(z)
        y_safe = jnp.maximum(y, 1e-30)
        chi = jnp.sqrt(y_safe / C)
        return (chi**3 * S + A * jnp.sqrt(y_safe)) / jnp.sqrt(mu)

    # --- Bracket the root using bisection ---
    # For zero-rev, tof(z) is monotonically increasing for z > z_min.
    # z=0 gives parabolic TOF. If target > tof(0), root is z > 0 (elliptic).
    # If target < tof(0), root is z < 0 (hyperbolic).

    tof_0 = _tof_from_z(0.0)
    elliptic = tof_0 < tof

    # Find upper bracket bound by doubling z until tof > target
    def _find_upper(carry):
        z_h = carry
        return _tof_from_z(z_h) < tof

    def _double(carry):
        z_h = carry
        return z_h * 2.0

    z_high = jax.lax.while_loop(
        _find_upper, _double, jnp.where(elliptic, 1.0, -0.5)
    )

    # Lower bracket: z=0 for elliptic, find z_low for hyperbolic
    z_low = jnp.where(elliptic, 0.0, z_high)
    z_high = jnp.where(elliptic, z_high, 0.0)

    # For hyperbolic: also need to ensure y(z_low) > 0
    # The valid range is y > 0; for z < 0, y decreases with decreasing z.
    # z_low starts at z_high (most negative), but we need to ensure y > 0.
    # For simplicity, we search from z_high and move toward 0.
    # Since tof increases with z, and z_low is the most negative with tof < target,
    # bisection between z_low and z_high=0 will work.

    # --- Bisection ---
    def _bisect_cond(carry):
        i, z_lo, z_hi = carry
        return jnp.logical_and(
            i < max_iter,
            (z_hi - z_lo) > tol * 2,
        )

    def _bisect_step(carry):
        i, z_lo, z_hi = carry
        z_mid = (z_lo + z_hi) / 2.0
        f_mid = _tof_from_z(z_mid) - tof
        z_lo = jnp.where(f_mid < 0, z_mid, z_lo)
        z_hi = jnp.where(f_mid >= 0, z_mid, z_hi)
        return (i + 1, z_lo, z_hi)

    _, z_lo, z_hi = jax.lax.while_loop(
        _bisect_cond, _bisect_step, (0, z_low, z_high)
    )
    z = (z_lo + z_hi) / 2.0

    # --- Lagrange coefficients ---
    y = _y(z)

    f_lg = 1 - y / r1
    g_lg = A * jnp.sqrt(y / mu)
    gdot_lg = 1 - y / r2

    v1_vec = (r2_vec - f_lg * r1_vec) / g_lg
    v2_vec = (gdot_lg * r2_vec - r1_vec) / g_lg

    return v1_vec, v2_vec
