"""Coverage of the potential functions and parameter helpers.

These are the least glamorous parts of the package and among the most
consequential: an open-circuit potential with the wrong slope makes every filter
in the package mis-scale its correction, and a parameter set that is not charge
balanced pushes the residual into whatever transport parameter is being fitted.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cellkernel.ocp import (
    TabulatedOCP,
    derivative_of,
    graphite_chen2020,
    graphite_chen2020_derivative,
    lfp_prada2013,
    nmc811_chen2020,
    nmc811_chen2020_derivative,
    numerical_derivative,
    tabulate,
)
from cellkernel.params import (
    balanced_stoichiometry_window,
    chen2020_nmc811_graphite,
    lfp_graphite,
)

# ------------------------------------------------------------------- potentials


@pytest.mark.parametrize(
    "fit,low,high",
    [
        # Bounds set from the fits' actual range over [0.01, 0.99], loosened
        # enough to be a sanity check rather than a snapshot. All three reach
        # well past their working window at the extremes, which is expected:
        # they are extrapolations there and the electrode never visits it.
        (graphite_chen2020, 0.02, 2.0),
        (nmc811_chen2020, 3.0, 4.8),
        (lfp_prada2013, 2.4, 4.3),
    ],
)
def test_potentials_stay_in_a_plausible_window(fit, low, high):
    values = fit(np.linspace(0.01, 0.99, 400))
    assert np.all(np.isfinite(values))
    assert values.min() > low
    assert values.max() < high


def test_the_phosphate_fit_survives_its_own_cancellation():
    """``lfp_prada2013`` contains two exponentials of order 1e8 that nearly cancel.

    They differ in the fifth significant figure of an exponent, so the result is
    a small difference of two very large numbers -- exactly the arrangement that
    produces garbage in floating point. It happens to be well conditioned over
    the range that matters, and this pins that rather than assuming it.
    """
    values = lfp_prada2013(np.linspace(0.001, 0.999, 2000))
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) < 0.0)
    assert values.max() < 5.0


@pytest.mark.parametrize("fit", [graphite_chen2020, nmc811_chen2020, lfp_prada2013])
def test_potentials_decrease_with_lithiation(fit):
    """More lithium in the host means a lower potential against lithium metal.

    True of every intercalation material, and a monotone potential is also what
    makes the state-of-charge inverse problem well posed.
    """
    values = fit(np.linspace(0.05, 0.95, 200))
    assert np.all(np.diff(values) < 0.0)


@pytest.mark.parametrize(
    "fit,exact",
    [
        (graphite_chen2020, graphite_chen2020_derivative),
        (nmc811_chen2020, nmc811_chen2020_derivative),
    ],
)
def test_closed_form_derivatives_match_differences(fit, exact):
    x = np.linspace(0.05, 0.95, 200)
    numeric = numerical_derivative(fit, x)
    analytic = exact(x)
    scale = np.maximum(np.abs(numeric), 1e-2)
    assert np.max(np.abs(analytic - numeric) / scale) < 1e-4


def test_derivative_of_prefers_the_closed_form():
    assert derivative_of(graphite_chen2020) is graphite_chen2020_derivative
    assert derivative_of(nmc811_chen2020) is nmc811_chen2020_derivative


def test_derivative_of_uses_an_interpolants_own_derivative():
    table = TabulatedOCP(np.linspace(0.0, 1.0, 11), np.linspace(1.5, 0.1, 11))
    assert derivative_of(table) == table.derivative


def test_derivative_of_falls_back_to_differencing():
    def custom(x):
        return 3.0 - 0.5 * np.asarray(x, dtype=float)

    derivative = derivative_of(custom)
    assert np.allclose(derivative(np.linspace(0.2, 0.8, 5)), -0.5, atol=1e-6)


def test_numerical_derivative_does_not_step_outside_the_domain():
    """The fits diverge violently outside ``[0, 1]``, so the stencil is clipped."""
    for x in (0.0, 1.0):
        value = numerical_derivative(graphite_chen2020, np.array([x]))
        assert np.all(np.isfinite(value))


# ---------------------------------------------------------------- tabulated OCP


def test_tabulated_ocp_requires_increasing_samples():
    with pytest.raises(ValueError, match="increasing"):
        TabulatedOCP(np.array([0.0, 0.5, 0.4]), np.array([1.0, 0.5, 0.4]))


def test_tabulated_ocp_requires_matching_lengths():
    with pytest.raises(ValueError, match="equal length"):
        TabulatedOCP(np.array([0.0, 1.0]), np.array([1.0]))


def test_tabulated_ocp_requires_two_samples():
    with pytest.raises(ValueError, match="two samples"):
        TabulatedOCP(np.array([0.0]), np.array([1.0]))


def test_tabulated_ocp_clamps_outside_its_range():
    table = TabulatedOCP(np.linspace(0.2, 0.8, 7), np.linspace(1.0, 0.2, 7))
    assert table(np.array([0.0])) == pytest.approx(table(np.array([0.2])))
    assert table(np.array([1.0])) == pytest.approx(table(np.array([0.8])))


def test_tabulated_ocp_preserves_monotonicity():
    """PCHIP rather than a natural spline, and the reason is not aesthetic.

    A natural spline through a flat plateau overshoots, and a locally positive
    dU/dx flips the sign of the Kalman gain and drives the filter away from the
    truth.
    """
    sto = np.array([0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0])
    potential = np.array([3.55, 3.44, 3.43, 3.42, 3.41, 3.35, 3.0])
    table = TabulatedOCP(sto, potential)
    dense = table(np.linspace(0.0, 1.0, 500))
    assert np.all(np.diff(dense) <= 1e-12)


def test_resample_lands_on_a_uniform_grid():
    table = TabulatedOCP(np.linspace(0.1, 0.9, 9), np.linspace(1.0, 0.1, 9))
    grid, values = table.resample(33)
    assert grid.size == values.size == 33
    assert np.allclose(np.diff(grid), grid[1] - grid[0])


def test_resample_rejects_a_degenerate_count():
    table = TabulatedOCP(np.linspace(0.1, 0.9, 9), np.linspace(1.0, 0.1, 9))
    with pytest.raises(ValueError, match="count"):
        table.resample(1)


def test_tabulate_error_falls_as_the_grid_refines():
    coarse = tabulate(graphite_chen2020, 33)
    fine = tabulate(graphite_chen2020, 129)
    assert fine.max_abs_error < coarse.max_abs_error / 4.0


def test_tabulate_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        tabulate(graphite_chen2020, 2)


# ------------------------------------------------------------------- parameters


@pytest.mark.parametrize("factory", [chen2020_nmc811_graphite, lfp_graphite])
def test_built_in_sets_are_charge_balanced(factory):
    cell = factory()
    assert cell.balance_error() < 1e-6
    assert cell.usable_capacity() == pytest.approx(cell.nominal_capacity, rel=1e-6)


@pytest.mark.parametrize("factory", [chen2020_nmc811_graphite, lfp_graphite])
def test_open_circuit_voltage_spans_the_stated_limits(factory):
    cell = factory()
    low, high = cell.voltage_limits
    assert float(cell.open_circuit_voltage(0.0)) == pytest.approx(low, abs=1e-3)
    assert float(cell.open_circuit_voltage(1.0)) == pytest.approx(high, abs=1e-3)


@pytest.mark.parametrize("factory", [chen2020_nmc811_graphite, lfp_graphite])
def test_open_circuit_voltage_is_monotone(factory):
    cell = factory()
    values = np.asarray(cell.open_circuit_voltage(np.linspace(0.0, 1.0, 400)))
    assert np.all(np.diff(values) > 0.0)


def test_soc_from_ocv_inverts_the_curve():
    cell = chen2020_nmc811_graphite()
    for soc in (0.05, 0.25, 0.5, 0.75, 0.95):
        voltage = float(cell.open_circuit_voltage(soc))
        assert cell.soc_from_ocv(voltage) == pytest.approx(soc, abs=1e-6)


def test_soc_from_ocv_clamps_beyond_the_curve():
    """A cold-boot reading below the 0% open-circuit voltage is routine."""
    cell = chen2020_nmc811_graphite()
    assert cell.soc_from_ocv(1.0) == 0.0
    assert cell.soc_from_ocv(9.0) == 1.0


def test_ocv_derivative_is_positive_and_matches_differences():
    cell = chen2020_nmc811_graphite()
    soc = np.linspace(0.05, 0.95, 60)
    analytic = np.asarray(cell.ocv_derivative(soc))
    step = 1e-5
    numeric = (
        np.asarray(cell.open_circuit_voltage(soc + step))
        - np.asarray(cell.open_circuit_voltage(soc - step))
    ) / (2.0 * step)
    assert np.all(analytic > 0.0)
    assert np.max(np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-3)) < 1e-3


def test_electrode_lookup_accepts_the_usual_aliases():
    cell = chen2020_nmc811_graphite()
    for alias in ("negative", "neg", "n", "anode"):
        assert cell._electrode(alias) is cell.negative
    for alias in ("positive", "pos", "p", "cathode"):
        assert cell._electrode(alias) is cell.positive
    with pytest.raises(ValueError, match="unknown electrode"):
        cell._electrode("middle")


def test_capacity_fade_narrows_both_windows():
    cell = chen2020_nmc811_graphite()
    aged = cell.with_capacity_fade(0.8)
    assert aged.nominal_capacity == pytest.approx(0.8 * cell.nominal_capacity)
    assert aged.usable_capacity() < cell.usable_capacity()
    for side in ("negative", "positive"):
        before = getattr(cell, side)
        after = getattr(aged, side)
        span_before = abs(before.stoich_at_100_soc - before.stoich_at_0_soc)
        span_after = abs(after.stoich_at_100_soc - after.stoich_at_0_soc)
        assert span_after == pytest.approx(0.8 * span_before, rel=1e-12)


def test_capacity_fade_rejects_a_nonsense_retention():
    cell = chen2020_nmc811_graphite()
    with pytest.raises(ValueError, match="retention"):
        cell.with_capacity_fade(0.0)
    with pytest.raises(ValueError, match="retention"):
        cell.with_capacity_fade(1.5)


def test_activation_energies_reject_negative_values():
    cell = chen2020_nmc811_graphite()
    with pytest.raises(ValueError, match="diffusion_negative"):
        cell.with_activation_energies(diffusion_negative=-1.0)


def test_activation_energies_are_applied_to_both_electrodes():
    cell = chen2020_nmc811_graphite().with_activation_energies(
        diffusion_negative=1000.0,
        diffusion_positive=2000.0,
        reaction_negative=3000.0,
        reaction_positive=4000.0,
    )
    assert cell.negative.diffusion_activation_energy == 1000.0
    assert cell.positive.diffusion_activation_energy == 2000.0
    assert cell.negative.reaction_activation_energy == 3000.0
    assert cell.positive.reaction_activation_energy == 4000.0
    # And they must actually change the transport, not merely be stored.
    warm = cell.negative.diffusivity_at(330.0, cell.reference_temperature)
    assert warm > cell.negative.diffusivity


def test_flux_and_current_scales_are_consistent():
    """``flux_scale`` is ``interfacial_current_scale`` divided by Faraday."""
    from cellkernel.params import FARADAY

    cell = chen2020_nmc811_graphite()
    for side in ("negative", "positive"):
        assert cell.flux_scale(side) == pytest.approx(
            cell.interfacial_current_scale(side) / FARADAY, rel=1e-12
        )


def test_balancing_rejects_an_unreachable_window():
    """Asking for more capacity than the coatings hold must fail loudly.

    And the message must name the achievable figures, since the usual cause is a
    capacity or voltage limit that the electrode pair simply cannot reach.
    """
    cell = chen2020_nmc811_graphite()
    with pytest.raises(ValueError, match="exceeds electrode throughput"):
        balanced_stoichiometry_window(
            cell.negative,
            cell.positive,
            electrode_area=cell.electrode_area,
            capacity=500.0,
            voltage_limits=cell.voltage_limits,
        )


def test_balancing_rejects_inverted_voltage_limits():
    cell = chen2020_nmc811_graphite()
    with pytest.raises(ValueError, match="increasing"):
        balanced_stoichiometry_window(
            cell.negative,
            cell.positive,
            electrode_area=cell.electrode_area,
            capacity=5.0,
            voltage_limits=(4.2, 2.5),
        )


def test_throughput_capacity_exceeds_usable_capacity():
    cell = chen2020_nmc811_graphite()
    for side in ("negative", "positive"):
        electrode = getattr(cell, side)
        assert electrode.throughput_capacity(cell.electrode_area) > electrode.usable_capacity(
            cell.electrode_area
        )


def test_lfp_plateau_is_genuinely_flat():
    """The property that makes phosphate chemistry hard to estimate.

    Across the plateau the open-circuit voltage barely moves, so voltage carries
    almost no state-of-charge information and a filter relying on it becomes
    weakly observable.
    """
    cell = lfp_graphite()
    slope = np.asarray(cell.ocv_derivative(np.linspace(0.3, 0.7, 40)))
    nmc = np.asarray(chen2020_nmc811_graphite().ocv_derivative(np.linspace(0.3, 0.7, 40)))
    assert slope.mean() < 0.5 * nmc.mean()


def test_thermal_time_constant():
    cell = chen2020_nmc811_graphite()
    thermal = cell.thermal
    expected = thermal.heat_capacity / (thermal.heat_transfer_coefficient * thermal.surface_area)
    assert thermal.time_constant == pytest.approx(expected, rel=1e-12)


def test_replacing_a_field_keeps_the_set_usable():
    cell = chen2020_nmc811_graphite()
    quieter = replace(cell, contact_resistance=0.002)
    assert quieter.contact_resistance == 0.002
    assert quieter.balance_error() == pytest.approx(cell.balance_error())
