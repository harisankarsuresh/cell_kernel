"""Validation of the charging protocols."""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.degradation import DegradationModel
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.protocols import (
    ChargeLimits,
    constant_current_constant_voltage,
    plating_limited_charge,
    plating_limited_current,
)

ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)


@pytest.fixture(scope="module")
def ageing(cell):
    return DegradationModel(cell)


@pytest.fixture(scope="module")
def limits():
    return ChargeLimits(max_c_rate=3.0, max_voltage=4.2, plating_margin=0.01)


def at_temperature(cell, temperature: float) -> SPM:
    return SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)


# ------------------------------------------------------------------ validation


def test_limits_reject_nonsense():
    with pytest.raises(ValueError, match="max_c_rate"):
        ChargeLimits(max_c_rate=0.0)
    with pytest.raises(ValueError, match="min_c_rate"):
        ChargeLimits(min_c_rate=-0.1)
    with pytest.raises(ValueError, match="min_c_rate"):
        ChargeLimits(max_c_rate=1.0, min_c_rate=2.0)


def test_charge_rejects_bad_update_interval(cell, ageing, limits):
    model = at_temperature(cell, 298.15)
    with pytest.raises(ValueError, match="update_every"):
        plating_limited_charge(model, ageing, limits, update_every=0)


# --------------------------------------------------------- the current limit


def test_limit_is_the_ceiling_when_nothing_binds(cell, ageing):
    """Warm and nearly empty, the model should not be the constraint."""
    model = at_temperature(cell, 308.15)
    generous = ChargeLimits(max_c_rate=0.2, plating_margin=0.0)
    x = model.initial_state(0.1)
    assert plating_limited_current(model, ageing, x, generous) == pytest.approx(
        0.2 * cell.nominal_capacity
    )


def test_limit_is_zero_when_nothing_is_safe(cell, ageing):
    """Very cold and nearly full: no charging current keeps the margin."""
    model = at_temperature(cell, 248.15)
    strict = ChargeLimits(max_c_rate=3.0, plating_margin=0.15)
    x = model.initial_state(0.99)
    assert plating_limited_current(model, ageing, x, strict) == 0.0


def test_the_returned_current_actually_meets_the_margin(cell, ageing, limits):
    """The point of the bisection: the answer must satisfy the constraint."""
    for temperature in (263.15, 283.15, 303.15):
        model = at_temperature(cell, temperature)
        for soc in (0.2, 0.5, 0.8, 0.95):
            x = model.initial_state(soc)
            current = plating_limited_current(model, ageing, x, limits)
            if current <= 0.0:
                continue
            potential = ageing.negative_potential(model, x, -current)
            assert potential >= limits.plating_margin - 1e-3, (
                f"{temperature - 273.15:.0f} C, soc {soc}: "
                f"phi {potential:.5f} below margin {limits.plating_margin}"
            )


def test_the_limit_is_tight_not_merely_safe(cell, ageing, limits):
    """A protocol that always returned zero would pass the test above.

    Just above the returned current the margin must be violated, or the bisection
    has stopped short and the charge is needlessly slow.
    """
    model = at_temperature(cell, 283.15)
    x = model.initial_state(0.7)
    current = plating_limited_current(model, ageing, x, limits)
    ceiling = limits.max_c_rate * cell.nominal_capacity
    assert 0.0 < current < ceiling, "this operating point should be model-limited"
    over = ageing.negative_potential(model, x, -(current + 0.02 * ceiling))
    assert over < limits.plating_margin


def test_the_limit_falls_with_state_of_charge(cell, ageing, limits):
    model = at_temperature(cell, 283.15)
    currents = [
        plating_limited_current(model, ageing, model.initial_state(soc), limits)
        for soc in (0.2, 0.5, 0.8, 0.95)
    ]
    assert currents == sorted(currents, reverse=True)


def test_the_limit_falls_with_temperature(cell, ageing, limits):
    currents = []
    for temperature in (263.15, 273.15, 288.15, 303.15):
        model = at_temperature(cell, temperature)
        currents.append(plating_limited_current(model, ageing, model.initial_state(0.7), limits))
    assert currents == sorted(currents)


def test_a_wider_margin_is_more_cautious(cell, ageing):
    model = at_temperature(cell, 283.15)
    x = model.initial_state(0.75)
    relaxed = plating_limited_current(
        model, ageing, x, ChargeLimits(max_c_rate=3.0, plating_margin=0.0)
    )
    strict = plating_limited_current(
        model, ageing, x, ChargeLimits(max_c_rate=3.0, plating_margin=0.05)
    )
    assert strict < relaxed


# ------------------------------------------------------------------ protocols


def test_conventional_charge_respects_its_voltage_limit(cell, limits):
    model = at_temperature(cell, 298.15)
    result = constant_current_constant_voltage(model, limits, 2.0, 0.2, 3600.0)
    assert result["voltage"].max() <= limits.max_voltage + 5e-3
    assert result["soc"][-1] > result["soc"][0]


def test_conventional_charge_current_only_falls(cell, limits):
    """Constant current, then taper. It must never ramp back up."""
    model = at_temperature(cell, 298.15)
    result = constant_current_constant_voltage(model, limits, 2.0, 0.2, 3600.0)
    magnitude = -result["current"]
    assert np.all(np.diff(magnitude) <= 1e-9)


def test_plating_limited_charge_never_plates(cell, ageing, limits):
    """The guarantee the protocol exists to provide, across temperature."""
    for temperature in (263.15, 283.15, 298.15):
        model = at_temperature(cell, temperature)
        result = plating_limited_charge(model, ageing, limits, 0.1, 5400.0, temperature)
        assert result["plating_potential"].min() >= -1e-3, (
            f"{temperature - 273.15:.0f} C: reached {result['plating_potential'].min():.5f} V"
        )
        assert result["soc"][-1] > 0.6, "and it must still charge the cell"


def test_conventional_charge_does_plate_in_the_cold(cell, ageing, limits):
    """The comparison that motivates the whole thing.

    At -5 C a conventional charge deposits metal at every rate offered, including
    1C, because terminal voltage gives the controller no sight of the electrode
    potential. The plating-limited protocol reaches a similar state of charge in a
    similar time without ever crossing the onset.
    """
    temperature = 268.15
    model = at_temperature(cell, temperature)
    conventional = constant_current_constant_voltage(
        model, limits, 1.0, 0.1, 5400.0, temperature, ageing=ageing
    )
    guarded = plating_limited_charge(model, ageing, limits, 0.1, 5400.0, temperature)

    assert conventional["plating_potential"].min() < 0.0, "expected the baseline to plate"
    assert guarded["plating_potential"].min() >= -1e-3
    # And the guarded protocol should not be dramatically slower for it.
    assert guarded["soc"][-1] > 0.75


def test_plating_potential_is_recorded_along_the_trajectory(cell, ageing, limits):
    """Not reconstructed from rested states afterwards.

    Reconstruction ignores the concentration gradient the charge itself built,
    which flatters the conventional protocol enough to reverse the conclusion.
    """
    model = at_temperature(cell, 273.15)
    result = constant_current_constant_voltage(
        model, limits, 2.0, 0.2, 1800.0, 273.15, ageing=ageing
    )
    assert "plating_potential" in result
    rested = np.array(
        [
            ageing.negative_potential(model, model.initial_state(s, 273.15), i)
            for s, i in zip(result["soc"][::300], result["current"][::300], strict=True)
        ]
    )
    along = result["plating_potential"][::300]
    assert np.any(along < rested - 1e-4), "rested reconstruction should be optimistic"


def test_omitting_ageing_omits_the_diagnostic(cell, limits):
    model = at_temperature(cell, 298.15)
    result = constant_current_constant_voltage(model, limits, 1.0, 0.3, 600.0)
    assert "plating_potential" not in result


def test_update_interval_does_not_change_the_answer_much(cell, ageing, limits):
    """Recomputing the setpoint every sample is wasteful, and unnecessary.

    The limit moves on the timescale of state of charge, which is minutes, while
    the loop runs at hertz.
    """
    model = at_temperature(cell, 283.15)
    fine = plating_limited_charge(model, ageing, limits, 0.2, 1800.0, 283.15, update_every=1)
    coarse = plating_limited_charge(model, ageing, limits, 0.2, 1800.0, 283.15, update_every=20)
    assert coarse["soc"][-1] == pytest.approx(fine["soc"][-1], abs=5e-3)
    assert coarse["plating_potential"].min() >= -1e-3


def test_charge_respects_the_rate_ceiling(cell, ageing):
    """Whatever the model permits, the hardware limit still binds."""
    model = at_temperature(cell, 313.15)
    capped = ChargeLimits(max_c_rate=0.5, max_voltage=4.2, plating_margin=0.0)
    result = plating_limited_charge(model, ageing, capped, 0.1, 1800.0, 313.15)
    assert np.max(-result["current"]) <= 0.5 * cell.nominal_capacity + 1e-9
