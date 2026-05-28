"""Tests for Lambert solver.

Test cases:
1. Curtis Example 5.2 (ISS rendezvous)
2. Circular orbit transfers with known solutions
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import astrodynx as adx


class TestLambertCurtis52:
    """Curtis Example 5.2: ISS rendezvous.

    r1 = [15945.34, 0, 0] km
    r2 = [12214.83899, 10249.46731, 0] km
    dt = 76 min = 4560 s
    mu = 398600.4418 km^3/s^2
    """

    mu = 398600.4418
    r1 = jnp.array([15945.34, 0.0, 0.0])
    r2 = jnp.array([12214.83899, 10249.46731, 0.0])
    tof = 76.0 * 60.0

    def test_arrival_position(self):
        """Verify r2 = F*r1 + G*v1 via Kepler propagation."""
        v1, v2 = adx.twobody.solve_lambert(
            self.r1, self.r2, self.tof, self.mu, prograde=True
        )
        from astrodynx.twobody._uniformulas import sigma_fn
        from astrodynx.twobody._orb_integrals import semimajor_axis
        from astrodynx.twobody._kep_equ import solve_kepler_uni
        from astrodynx.twobody._uniformulas import ufunc2, ufunc1
        from astrodynx.twobody._lagrange import lagrange_F, lagrange_G

        mu = self.mu
        r1_mag = jnp.linalg.vector_norm(self.r1)
        v1_mag = jnp.linalg.vector_norm(v1)
        alpha = 1.0 / semimajor_axis(r1_mag, v1_mag, mu)
        sigma0 = sigma_fn(self.r1, v1, mu).item()

        chi = solve_kepler_uni(self.tof, alpha, r1_mag, sigma0, mu)
        U2 = ufunc2(chi, alpha)
        U1 = ufunc1(chi, alpha)

        f = lagrange_F(U2, r1_mag)
        g = lagrange_G(U1, U2, sigma0, r1_mag, mu)

        r2_computed = f * self.r1 + g * v1
        np.testing.assert_allclose(r2_computed, self.r2, atol=0.1)

    def test_finite_velocities(self):
        v1, v2 = adx.twobody.solve_lambert(
            self.r1, self.r2, self.tof, self.mu, prograde=True
        )
        assert jnp.all(jnp.isfinite(v1))
        assert jnp.all(jnp.isfinite(v2))
        assert jnp.linalg.vector_norm(v1) > 0
        assert jnp.linalg.vector_norm(v2) > 0

    def test_v1_reasonable_magnitude(self):
        """Orbital velocity at r=15945 km should be around 5 km/s."""
        v1, _ = adx.twobody.solve_lambert(
            self.r1, self.r2, self.tof, self.mu, prograde=True
        )
        v_circ = jnp.sqrt(self.mu / jnp.linalg.vector_norm(self.r1))
        v1_mag = jnp.linalg.vector_norm(v1)
        assert 0.5 * v_circ < v1_mag < 2.0 * v_circ


class TestLambertCircular:
    """Test with known circular orbit transfers."""

    mu = 1.0

    def test_quarter_orbit_transfer(self):
        """90° transfer on a circular orbit of radius 1."""
        r1 = jnp.array([1.0, 0.0, 0.0])
        r2 = jnp.array([0.0, 1.0, 0.0])
        tof = jnp.pi / 2

        v1, v2 = adx.twobody.solve_lambert(r1, r2, tof, self.mu, prograde=True)

        np.testing.assert_allclose(v1, jnp.array([0.0, 1.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(v2, jnp.array([-1.0, 0.0, 0.0]), atol=1e-8)

    def test_near_half_orbit(self):
        """Near-180° transfer on circular orbit (avoid degenerate 180°)."""
        r1 = jnp.array([1.0, 0.0, 0.0])
        r2 = jnp.array([-0.999, 0.0, 0.0447])
        tof = jnp.pi

        v1, v2 = adx.twobody.solve_lambert(r1, r2, tof, self.mu, prograde=True)

        assert jnp.all(jnp.isfinite(v1))
        assert jnp.all(jnp.isfinite(v2))
        assert jnp.linalg.vector_norm(v1) > 0

    def test_arrival_via_propagation(self):
        """Verify arrival at r2 for a non-trivial transfer."""
        r1 = jnp.array([1.0, 0.0, 0.0])
        r2 = jnp.array([0.5, 0.866, 0.0])
        tof = 2.0

        v1, v2 = adx.twobody.solve_lambert(r1, r2, tof, self.mu, prograde=True)

        from astrodynx.twobody._uniformulas import sigma_fn
        from astrodynx.twobody._orb_integrals import semimajor_axis
        from astrodynx.twobody._kep_equ import solve_kepler_uni
        from astrodynx.twobody._uniformulas import ufunc2, ufunc1
        from astrodynx.twobody._lagrange import lagrange_F, lagrange_G

        r1_mag = jnp.linalg.vector_norm(r1)
        v1_mag = jnp.linalg.vector_norm(v1)
        alpha = 1.0 / semimajor_axis(r1_mag, v1_mag, self.mu)
        sigma0 = sigma_fn(r1, v1, self.mu).item()
        chi = solve_kepler_uni(tof, alpha, r1_mag, sigma0, self.mu)

        f = lagrange_F(ufunc2(chi, alpha), r1_mag)
        g = lagrange_G(ufunc1(chi, alpha), ufunc2(chi, alpha), sigma0, r1_mag, self.mu)

        r2_computed = f * r1 + g * v1
        np.testing.assert_allclose(r2_computed, r2, atol=1e-8)


class TestLambertJIT:
    def test_jit_compilable(self):
        r1 = jnp.array([15945.34, 0.0, 0.0])
        r2 = jnp.array([12214.83899, 10249.46731, 0.0])
        tof = 4560.0
        mu = 398600.4418

        jitted = jax.jit(adx.twobody.solve_lambert, static_argnames=["prograde"])
        v1, v2 = jitted(r1, r2, tof, mu, prograde=True)

        assert jnp.all(jnp.isfinite(v1))
        assert jnp.all(jnp.isfinite(v2))
