"""Validation of the diffusion reduced-order models against exact results.

These tests do not compare one implementation against another; every assertion
is anchored to a closed-form property of the underlying PDE.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from cellkernel.rom import (
    FiniteVolumeDiffusion,
    PadeDiffusion,
    PolynomialDiffusion,
    SpectralDiffusion,
    exact_surface_transfer_function,
    low_frequency_series,
    make_rom,
    normalised_series_coefficients,
    pade_coefficients,
    zero_flux_eigenvalues,
)

R = 5.86e-6  # particle radius, m (Chen 2020 negative electrode)
D = 3.3e-14  # solid diffusivity, m2 s-1

ALL_KINDS = ["pade", "spectral", "fv", "poly"]


def _all_roms(order: int = 6) -> list:
    return [make_rom(kind, R, D, order=order) for kind in ALL_KINDS]


# --------------------------------------------------------------------------
# Exact series and transfer function
# --------------------------------------------------------------------------


def test_series_coefficients_match_known_rationals():
    """The normalised expansion is 1, 1/15, -1/525, 2/23625, -37/9095625."""
    got = normalised_series_coefficients(4)
    assert got[:5] == (
        Fraction(1),
        Fraction(1, 15),
        Fraction(-1, 525),
        Fraction(2, 23625),
        Fraction(-37, 9095625),
    )


def test_series_reproduces_mass_balance_and_offset():
    """Leading terms of G(s) must be 3/(Rs) + R/5D - R^3 s/175 D^2."""
    h = normalised_series_coefficients(2)
    assert float(3 * h[0]) == pytest.approx(3.0)
    assert float(3 * h[1]) == pytest.approx(1.0 / 5.0)
    assert float(3 * h[2]) == pytest.approx(-1.0 / 175.0)


def test_transfer_function_branches_agree_at_crossover():
    """Closed form and series must agree either side of the |xi| = 0.1 switch."""
    theta = R**2 / D
    # |xi| = 0.1 corresponds to s = 0.01 / theta.
    for scale in (0.7, 0.9, 1.1, 1.5):
        s = 0.01 * scale**2 / theta
        closed = (
            (R / D)
            * np.sinh(np.sqrt(s * theta))
            / (np.sqrt(s * theta) * np.cosh(np.sqrt(s * theta)) - np.sinh(np.sqrt(s * theta)))
        )
        series = low_frequency_series(s, R, D, order=8)
        assert abs(series - closed) / abs(closed) < 1e-10


def test_high_frequency_rolloff_is_inverse_sqrt():
    """G(s) -> (R/D)/(xi - 1) for large xi, so magnitude falls as s^-1/2.

    Also guards the overflow reformulation: the naive sinh/cosh expression
    returns NaN beyond xi ~ 710, whereas these must stay exact.
    """
    theta = R**2 / D
    for xi in (10.0, 1e2, 1e3, 1e4, 1e5):
        G = exact_surface_transfer_function(xi**2 / theta, R, D)
        assert np.isfinite(abs(G))
        assert abs(G) * (xi - 1.0) * D / R == pytest.approx(1.0, rel=1e-8)


# --------------------------------------------------------------------------
# Structural properties shared by every model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rom", _all_roms(), ids=ALL_KINDS)
def test_mass_balance_is_structurally_exact(rom):
    """d(c_bar)/dt must equal 3N/R exactly, for every model and order.

    The averaging row of ``C`` must annihilate ``A`` (no internal redistribution
    can change the mean) and must map ``B`` onto exactly ``3/R``.
    """
    A, B, C, Df = rom.continuous()
    row = C[1] @ A
    assert np.allclose(row, 0.0, atol=1e-9 * max(1.0, np.abs(A).max()))
    assert float((C[1] @ B).reshape(-1)[0]) == pytest.approx(3.0 / R, rel=1e-12)
    # The mean must have no direct feedthrough from flux.
    assert float(Df[1, 0]) == 0.0


@pytest.mark.parametrize("rom", _all_roms(), ids=ALL_KINDS)
def test_uniform_state_is_an_equilibrium(rom):
    """A uniformly loaded particle at zero flux must report c_surf = c_bar = c.

    Holds to machine precision for every model, finite volume included, because
    :meth:`~cellkernel.rom.base.DiffusionROM.discretise` imposes ``A u = u``
    explicitly instead of trusting the matrix exponential to preserve it. Without
    that projection this bound is a property of the linear-algebra backend rather
    than of the model: SciPy 1.15 leaves a residual of 1.004e-12 here where 1.18
    leaves under 1e-13.
    """
    c0 = 24000.0
    ss = rom.discretise(1.0)
    x = ss.initial_state(c0)
    for _ in range(50):
        c_surf, c_bar = ss.outputs(x, 0.0)
        assert c_surf == pytest.approx(c0, rel=1e-10)
        assert c_bar == pytest.approx(c0, rel=1e-12)
        x = ss.step(x, 0.0)


@pytest.mark.parametrize("rom", _all_roms(), ids=ALL_KINDS)
def test_discrete_mass_balance_matches_coulomb_count(rom):
    """Discretised c_bar must integrate flux exactly: c_bar = c0 + 3*dt*sum(N)/R."""
    dt = 0.5
    ss = rom.discretise(dt)
    rng = np.random.default_rng(0)
    flux = rng.normal(0.0, 1e-6, size=200)
    _, c_bar = ss.simulate(0.0, flux)
    expected = 3.0 * dt * np.concatenate([[0.0], np.cumsum(flux)[:-1]]) / R
    assert np.allclose(c_bar, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("rom", _all_roms(), ids=ALL_KINDS)
def test_zoh_discretisation_is_stable_at_huge_timestep(rom):
    """Exact exponential discretisation must not blow up even at dt >> R^2/D.

    Stability is asserted structurally rather than by taking the spectral radius
    of the whole discrete matrix, and the distinction is not pedantic.

    At ``dt = 100 R^2/D`` every shape mode has decayed to nothing while the
    conserved coordinate is untouched, so the entries of ``A`` span about fifteen
    orders of magnitude for a Pade model and eighteen for a high-order one.
    Eigenvalue extraction on a matrix that badly scaled is itself ill-conditioned,
    and the answer becomes a property of the LAPACK build: the same matrix that
    gives a spectral radius of exactly 1 under OpenBLAS returns 1.000065 under
    Accelerate. That is not an unstable discretisation, it is a numerically
    meaningless question asked of a well-behaved matrix.

    What is actually true, and what is checked here, is stronger and perfectly
    conditioned. The compact models put the conserved concentration in the first
    coordinate with no inflow, so ``A`` is block lower triangular: its first row
    is exactly ``[1, 0, ..., 0]`` and the remaining block carries only decaying
    modes. Finite volume has no such coordinate -- its conserved quantity is a
    weighted sum of states -- but its matrix is well scaled, so the direct
    spectral radius is meaningful there.
    """
    ss = rom.discretise(100.0 * rom.time_constant)
    A = ss.A

    if rom.name == "fv":
        assert np.max(np.abs(np.linalg.eigvals(A))) <= 1.0 + 1e-9
        return

    assert A[0, 0] == pytest.approx(1.0, abs=1e-12)
    if A.shape[0] > 1:
        assert np.allclose(A[0, 1:], 0.0, atol=1e-12)
        assert np.max(np.abs(np.linalg.eigvals(A[1:, 1:]))) < 1.0


@pytest.mark.parametrize("rom", _all_roms(), ids=ALL_KINDS)
@pytest.mark.parametrize("ratio", [1e-4, 1e-2, 1.0, 100.0, 1000.0])
def test_conservation_invariants_survive_any_timestep(rom, ratio):
    """The three conservation identities must hold exactly at every step size.

    ``A u = u``
        a uniformly loaded, unloaded particle is a fixed point;
    ``w A = w``
        the volume average is unchanged by the update;
    ``w B = 3 dt / R``
        and it advances by exactly the coulomb count.

    These are guaranteed by construction in continuous time, and
    :meth:`~cellkernel.rom.base.DiffusionROM.discretise` projects them back after
    sampling rather than assuming the matrix exponential preserved them. It does
    not always: at ``dt = 100 R^2/D`` the generator spans fifteen orders of
    magnitude, and SciPy 1.15 returned ``A[0, 0]`` as 0.99988 on one platform and
    1.000065 on another. The tolerance here is deliberately near machine epsilon,
    because these identities are exact statements and not approximations.
    """
    dt = ratio * rom.time_constant
    ss = rom.discretise(dt)
    u = ss.x0_from_uniform.reshape(-1)
    w = ss.C[1]

    assert np.max(np.abs(ss.A @ u - u)) <= 1e-12 * max(np.max(np.abs(u)), 1.0)
    assert np.max(np.abs(w @ ss.A - w)) <= 1e-12 * np.max(np.abs(w))
    target = 3.0 * dt / rom.radius
    assert float(w @ ss.B.reshape(-1)) == pytest.approx(target, rel=1e-12)


@pytest.mark.parametrize("rom", _all_roms(), ids=ALL_KINDS)
def test_rested_particle_stays_put_at_huge_timestep(rom):
    """The physical counterpart of the structural stability check.

    A uniformly loaded particle carrying no flux is a fixed point of the
    dynamics. Iterating it 500 times at an absurd step must not move it at all.
    This measures growth along an actually reachable trajectory, needs no
    eigendecomposition, and so is immune to the backend sensitivity described
    above -- genuine instability would still show up here, amplified 500-fold.
    """
    ss = rom.discretise(100.0 * rom.time_constant)
    x = ss.initial_state(24000.0)
    reference = np.linalg.norm(x)
    peak = reference
    for _ in range(500):
        x = ss.step(x, 0.0)
        peak = max(peak, float(np.linalg.norm(x)))
    assert peak / reference <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# Steady-state surface offset: the R / 5D identity
# --------------------------------------------------------------------------


def _steady_offset(rom, flux: float) -> float:
    """Surface-minus-average concentration after a long constant flux."""
    dt = 0.02 * rom.time_constant
    ss = rom.discretise(dt)
    x = ss.initial_state(0.0)
    for _ in range(4000):
        x = ss.step(x, flux)
    c_surf, c_bar = ss.outputs(x, flux)
    return c_surf - c_bar


def test_polynomial_offset_is_exact():
    """The two-state polynomial model hits R/5D identically, not approximately."""
    flux = 1e-6
    rom = PolynomialDiffusion(R, D)
    assert _steady_offset(rom, flux) == pytest.approx(R * flux / (5.0 * D), rel=1e-9)


def test_spectral_eigenvalue_sum_converges_to_one_tenth():
    """sum(1/lambda_k^2) over roots of tan(x) = x equals 1/10.

    Convergence is slow because lambda_k ~ (k + 1/2) pi, making the tail
    O(1/(pi^2 k)). The test asserts both the limit and that the residual shrinks
    at the predicted rate, which would catch a mis-bracketed root far more
    reliably than a loose tolerance on the sum alone.
    """
    residuals = []
    for count in (500, 2000, 8000):
        lam = zero_flux_eigenvalues(count)
        partial = float(np.sum(1.0 / lam**2))
        assert partial < 0.1
        residuals.append((count, 0.1 - partial))
    for count, residual in residuals:
        assert residual == pytest.approx(1.0 / (np.pi**2 * count), rel=0.05)
    assert residuals[-1][1] < 1.5e-5


def test_plain_modal_truncation_follows_partial_eigenvalue_sum():
    """Without residualisation the offset is 2 R flux / D times the partial sum."""
    flux = 1e-6
    rom = SpectralDiffusion(R, D, n_modes=8, residualise=False)
    predicted = 2.0 * R * flux / D * rom.steady_state_offset_factor()
    assert _steady_offset(rom, flux) == pytest.approx(predicted, rel=1e-6)


@pytest.mark.parametrize("n_modes", [1, 2, 3, 8])
def test_plain_modal_truncation_is_biased_low(n_modes):
    """Quantify the documented deficit: 49.5%, 66.3%, 74.7%, 88.7% of R/5D."""
    flux = 1e-6
    exact = R * flux / (5.0 * D)
    rom = SpectralDiffusion(R, D, n_modes=n_modes, residualise=False)
    recovered = _steady_offset(rom, flux) / exact
    expected = 10.0 * rom.steady_state_offset_factor()
    assert recovered == pytest.approx(expected, rel=1e-6)
    assert recovered < 0.9  # even eight modes fall well short


@pytest.mark.parametrize("n_modes", [0, 1, 3, 8])
def test_residualisation_makes_offset_exact_at_every_order(n_modes):
    """Static condensation restores the full R/5D offset for any mode count."""
    flux = 1e-6
    rom = SpectralDiffusion(R, D, n_modes=n_modes, residualise=True)
    assert _steady_offset(rom, flux) == pytest.approx(R * flux / (5.0 * D), rel=1e-9)


def test_residual_gain_is_the_discarded_tail():
    rom = SpectralDiffusion(R, D, n_modes=4, residualise=True)
    assert rom.residual_gain == pytest.approx(
        2.0 * R * (0.1 - rom.steady_state_offset_factor()) / D, rel=1e-12
    )
    assert SpectralDiffusion(R, D, n_modes=4, residualise=False).residual_gain == 0.0


@pytest.mark.parametrize("order", [3, 5, 7])
def test_pade_offset_approaches_exact(order):
    flux = 1e-6
    rom = PadeDiffusion(R, D, order=order)
    assert _steady_offset(rom, flux) == pytest.approx(R * flux / (5.0 * D), rel=1e-6)


def test_finite_volume_offset_converges_with_shells():
    """Flux-consistent surface extrapolation must converge to R/5D as N grows."""
    flux = 1e-6
    exact = R * flux / (5.0 * D)
    errors = []
    for n in (5, 10, 20, 40):
        rom = FiniteVolumeDiffusion(R, D, n_shells=n)
        errors.append(abs(_steady_offset(rom, flux) - exact) / exact)
    assert errors[-1] < 0.01
    # Monotone reduction, at close to second order.
    for a, b in zip(errors, errors[1:], strict=False):
        assert b < a


# --------------------------------------------------------------------------
# Frequency-domain accuracy
# --------------------------------------------------------------------------


def _worst_relative_error(rom, lo_decade: float, hi_decade: float, points: int = 40) -> float:
    """Worst relative transfer-function error over a dimensionless frequency band.

    Frequencies are specified as decades of ``omega * R^2 / D``. Using the
    dimensionless group rather than rad/s is what makes the tolerances below
    meaningful: for the Chen 2020 negative electrode ``R^2/D`` is about 1040 s,
    so ``omega * R^2/D = 1`` is already a 17-minute timescale and a tolerance
    quoted at "1 rad/s" would silently be testing a regime no reduced model of
    modest order can reach.
    """
    theta = rom.time_constant
    worst = 0.0
    for decade in np.logspace(lo_decade, hi_decade, points):
        s = 1j * decade / theta
        exact = rom.exact_transfer_function(s)
        worst = max(worst, abs(rom.transfer_function(s) - exact) / abs(exact))
    return worst


@pytest.mark.parametrize(
    ("kind", "order", "tol"),
    [
        # Quasi-steady to hour-scale: omega R^2/D in [1e-2, 1e1].
        ("poly", 2, 3e-2),
        ("pade", 3, 5e-4),
        ("pade", 5, 1e-8),
        ("pade", 8, 1e-12),
        ("spectral", 4, 1e-2),
        ("spectral", 9, 1e-3),
        ("fv", 20, 1e-2),
        ("fv", 40, 3e-3),
    ],
)
def test_accuracy_over_core_band(kind, order, tol):
    """Accuracy from quasi-steady load up to ten times the diffusion pole."""
    rom = make_rom(kind, R, D, order=order)
    assert _worst_relative_error(rom, -2.0, 1.0) < tol


@pytest.mark.parametrize("n_modes", [1, 2, 3, 5, 8])
def test_residualisation_improves_the_core_band(n_modes):
    """Condensing the tail must help, by at least tenfold, below the pole."""
    plain = _worst_relative_error(
        SpectralDiffusion(R, D, n_modes=n_modes, residualise=False), -2.0, 1.0
    )
    condensed = _worst_relative_error(
        SpectralDiffusion(R, D, n_modes=n_modes, residualise=True), -2.0, 1.0
    )
    assert condensed < plain / 10.0


def test_residualisation_costs_accuracy_far_above_the_pole():
    """The documented trade-off: feedthrough breaks the s^-1/2 roll-off.

    Guards against someone "improving" the default without realising that a
    non-zero D makes the model asymptote to a constant instead of decaying.
    """
    plain = _worst_relative_error(SpectralDiffusion(R, D, n_modes=1, residualise=False), 2.5, 3.5)
    condensed = _worst_relative_error(
        SpectralDiffusion(R, D, n_modes=1, residualise=True), 2.5, 3.5
    )
    assert condensed > plain


def test_two_residualised_modes_beat_five_plain_ones():
    band = (-2.0, 1.0)
    two = _worst_relative_error(SpectralDiffusion(R, D, n_modes=2, residualise=True), *band)
    five = _worst_relative_error(SpectralDiffusion(R, D, n_modes=5, residualise=False), *band)
    assert two < five / 10.0


@pytest.mark.parametrize(
    ("kind", "order", "tol"),
    [
        # Pulse and regen content: omega R^2/D up to 1e2.
        ("pade", 8, 1e-7),
        ("pade", 12, 1e-11),
    ],
)
def test_high_order_pade_holds_into_pulse_band(kind, order, tol):
    """Only high-order Pade tracks the exact response two decades past the pole."""
    rom = make_rom(kind, R, D, order=order)
    assert _worst_relative_error(rom, -2.0, 2.0) < tol


def test_pade_beats_alternatives_at_equal_state_count():
    """At five states Pade must be orders of magnitude better than the rest.

    This is the central design claim of the package: approximating the transfer
    function directly buys far more accuracy per state than discretising space
    (finite volume) or truncating a slowly converging modal series (spectral).
    """
    band = (-2.0, 1.0)
    pade = _worst_relative_error(PadeDiffusion(R, D, order=5), *band)
    spectral = _worst_relative_error(SpectralDiffusion(R, D, n_modes=4), *band)
    fv = _worst_relative_error(FiniteVolumeDiffusion(R, D, n_shells=5), *band)
    assert pade < spectral / 1e4
    assert pade < fv / 1e4


def test_polynomial_beats_two_state_pade():
    """The moment-closure model is the better two-state choice.

    A [1/1] Pade of the remainder spends its single dynamic state matching the
    series at the origin, whereas the Subramanian closure places its pole at
    30 D/R^2, which is closer to the true dominant relaxation.
    """
    band = (-2.0, 1.0)
    assert _worst_relative_error(PolynomialDiffusion(R, D), *band) < _worst_relative_error(
        PadeDiffusion(R, D, order=2), *band
    )


def test_pade_converges_monotonically_in_order():
    """Increasing Pade order must reduce worst-case error in the band."""
    omega = np.logspace(-3, 0, 30)

    def worst(order: int) -> float:
        rom = PadeDiffusion(R, D, order=order)
        return max(
            abs(rom.transfer_function(1j * w) - rom.exact_transfer_function(1j * w))
            / abs(rom.exact_transfer_function(1j * w))
            for w in omega
        )

    errs = [worst(k) for k in (2, 3, 4, 5, 6)]
    for a, b in zip(errs, errs[1:], strict=False):
        assert b < a


# --------------------------------------------------------------------------
# Pade coefficient machinery
# --------------------------------------------------------------------------


def test_pade_reproduces_series_to_expected_order():
    """H*M - N must vanish through order a^(m+n)."""
    m, n = 3, 3
    numer, denom = pade_coefficients(m, n)
    h = normalised_series_coefficients(m + n)
    for k in range(m + n + 1):
        lhs = sum(denom[j] * h[k - j] for j in range(0, min(k, n) + 1))
        rhs = numer[k] if k < len(numer) else Fraction(0)
        assert lhs == rhs, f"mismatch at order {k}"


def test_pade_coefficients_are_exact_rationals():
    numer, denom = pade_coefficients(2, 2)
    assert denom[0] == Fraction(1)
    assert all(isinstance(c, Fraction) for c in numer + denom)


def test_pade_high_order_is_numerically_usable():
    """Exact rational solve must stay well conditioned where a float solve fails."""
    rom = PadeDiffusion(R, D, order=10)
    poles = np.linalg.eigvals(rom.continuous()[0])
    # One pole at the origin (the mass balance), the rest strictly stable.
    at_origin = np.sum(np.abs(poles) < 1e-8)
    assert at_origin == 1
    assert np.all(np.real(poles[np.abs(poles) >= 1e-8]) < 0.0)


def test_first_order_pade_is_pure_coulomb_counting():
    rom = PadeDiffusion(R, D, order=1)
    assert rom.n_states == 1
    ss = rom.discretise(1.0)
    c_surf, c_bar = ss.outputs(ss.initial_state(100.0), 1e-6)
    assert c_surf == pytest.approx(c_bar)


# --------------------------------------------------------------------------
# Misc API
# --------------------------------------------------------------------------


def test_state_counts():
    assert PadeDiffusion(R, D, order=4).n_states == 4
    assert SpectralDiffusion(R, D, n_modes=4).n_states == 5
    assert FiniteVolumeDiffusion(R, D, n_shells=7).n_states == 7
    assert PolynomialDiffusion(R, D).n_states == 2


def test_make_rom_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown reduced-order model"):
        make_rom("nonsense", R, D)


@pytest.mark.parametrize("bad", [{"radius": -1.0}, {"diffusivity": 0.0}])
def test_invalid_geometry_rejected(bad):
    kwargs = {"radius": R, "diffusivity": D, **bad}
    with pytest.raises(ValueError):
        PolynomialDiffusion(**kwargs)
