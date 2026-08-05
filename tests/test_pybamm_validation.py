"""Validation against PyBaMM.

Everywhere else this package checks itself against closed-form results and
against its own mirror. Those catch a great deal, but they cannot catch a
misunderstanding shared between a model and its test. This module checks against
a second, independent implementation of the same physics.

PyBaMM is an optional dependency and these tests skip without it. They are slow
-- the DFN comparisons dominate -- so they run in their own CI job rather than on
every developer's machine.

The comparison is set up to measure the physics rather than the bookkeeping.
Both packages are started from *identical* stoichiometries rather than from a
nominal state of charge, because each has its own way of mapping charge onto
electrode composition and comparing those would tell us nothing about the models.
"""

from __future__ import annotations

import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm", reason="PyBaMM is an optional dependency")

from cellkernel.models import SPM, SPMe  # noqa: E402
from cellkernel.params import from_pybamm  # noqa: E402

SOC0 = 0.9


@pytest.fixture(scope="module")
def cell():
    return from_pybamm(pybamm.ParameterValues("Chen2020"), name="chen2020-from-pybamm")


@pytest.fixture(scope="module")
def solve(cell):
    cache: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]] = {}

    def run(kind: str, c_rate: float, seconds: float):
        key = (kind, c_rate, seconds)
        if key in cache:
            return cache[key]
        values = pybamm.ParameterValues("Chen2020")
        values["Current function [A]"] = c_rate * 5.0
        values["Initial concentration in negative electrode [mol.m-3]"] = float(
            cell.negative.concentration(SOC0)
        )
        values["Initial concentration in positive electrode [mol.m-3]"] = float(
            cell.positive.concentration(SOC0)
        )
        model = {
            "SPM": pybamm.lithium_ion.SPM(),
            "SPMe": pybamm.lithium_ion.SPMe(),
            "DFN": pybamm.lithium_ion.DFN(),
        }[kind]
        solution = pybamm.Simulation(model, parameter_values=values).solve(
            np.arange(0.0, seconds + 1.0, 1.0),
            solver=pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-11),
        )
        cache[key] = (solution["Time [s]"].entries, solution["Voltage [V]"].entries)
        return cache[key]

    return run


def discrepancy(ours, reference) -> tuple[float, float]:
    """``(rmse, max)`` in volts, over the overlap of the two time bases."""
    times, values = reference
    mask = ours["time"] <= times[-1]
    error = ours["voltage"][mask] - np.interp(ours["time"][mask], times, values)
    return float(np.sqrt(np.mean(error**2))), float(np.max(np.abs(error)))


# ------------------------------------------------------------ the bridge itself


def test_the_imported_cell_is_charge_balanced(cell):
    """PyBaMM publishes loadings and stoichiometry limits independently.

    Taken verbatim they leave a percent-level imbalance. ``from_pybamm`` solves
    the window instead, so the imported cell must come out balanced and at the
    published capacity.
    """
    assert cell.balance_error() < 1e-6
    assert cell.nominal_capacity == pytest.approx(5.0, rel=1e-9)
    assert cell.usable_capacity() == pytest.approx(5.0, rel=1e-6)


def test_the_kinetics_are_read_from_the_source_not_guessed(cell):
    """The bug that hid behind everything else here.

    An earlier version hardcoded a reaction rate of 1e-6 because PyBaMM returns
    its exchange-current density as an expression node rather than a number, and
    the ``float()`` that failed was caught by a bare fallback. The published
    values are 6.48e-7 for this graphite and 3.42e-6 for the oxide, so the model
    was carrying kinetics wrong by 1.5x on one electrode and 5.3x on the other.
    It cost 23 mV against PyBaMM at 2C and was written off as a model difference.
    """
    assert cell.negative.reaction_rate == pytest.approx(6.48e-7, rel=1e-9)
    assert cell.positive.reaction_rate == pytest.approx(3.42e-6, rel=1e-9)


def test_the_electrolyte_transport_is_read_from_the_source(cell):
    """Likewise, and this one had been overridden by intuition.

    The default salt diffusivity in this package was nearly three times PyBaMM's,
    chosen because the depletion it produced looked more plausible. It was not:
    with the published value the electrolyte model agrees with a full
    Doyle-Fuller-Newman solution roughly seven times better.
    """
    values = pybamm.ParameterValues("Chen2020")
    assert cell.electrolyte_diffusivity == pytest.approx(1.7694e-10, rel=1e-3)
    assert cell.ionic_conductivity == pytest.approx(0.9487, rel=1e-3)
    assert cell.transference_number == values["Cation transference number"]
    assert cell.negative.porosity == values["Negative electrode porosity"]
    assert cell.positive.porosity == values["Positive electrode porosity"]
    assert cell.separator_porosity == values["Separator porosity"]


def test_reading_a_pybamm_expression_never_falls_back_silently():
    """PyBaMM parameter functions return expression nodes, not floats.

    Every one of them. A bridge that treats that as "value unavailable" and
    substitutes a default will build a plausible-looking cell that is not the one
    it was handed, which is the worst failure mode available to it.
    """
    from cellkernel.params import _as_float

    values = pybamm.ParameterValues("Chen2020")
    function = values["Negative electrode exchange-current density [A.m-2]"]
    node = function(1000.0, 15000.0, 33133.0, 298.15)
    assert not isinstance(node, float), "if PyBaMM starts returning floats, simplify this"
    assert _as_float(node) == pytest.approx(
        6.48e-7 * np.sqrt(1000.0) * np.sqrt(15000.0) * np.sqrt(33133.0 - 15000.0), rel=1e-9
    )


def test_the_imported_geometry_matches_the_source(cell):
    values = pybamm.ParameterValues("Chen2020")
    assert cell.negative.thickness == values["Negative electrode thickness [m]"]
    assert cell.positive.particle_radius == values["Positive particle radius [m]"]
    assert cell.negative.max_concentration == pytest.approx(
        values["Maximum concentration in negative electrode [mol.m-3]"]
    )
    assert cell.electrode_area == pytest.approx(
        values["Electrode height [m]"]
        * values["Electrode width [m]"]
        * values["Number of electrodes connected in parallel to make a cell"]
    )


# --------------------------------------------------------------- the same model


@pytest.mark.parametrize(
    "c_rate,seconds,tolerance",
    [(0.5, 1800.0, 1e-3), (1.0, 1200.0, 2e-3), (2.0, 700.0, 5e-3), (3.0, 400.0, 1e-2)],
)
def test_single_particle_models_agree(cell, solve, c_rate, seconds, tolerance):
    """The strongest form of the check: same physics, two implementations.

    Sub-millivolt at 0.5C and a few millivolt at 3C. The residual that remains is
    the reduced-order approximation of the particle, and it grows with rate
    because the concentration profile sharpens.
    """
    model = SPM(cell, dt=1.0, rom="pade", order=5)
    ours = model.simulate(np.full(int(seconds) + 1, c_rate * 5.0), soc0=SOC0)
    rmse, _ = discrepancy(ours, solve("SPM", c_rate, seconds))
    assert rmse < tolerance, f"{c_rate}C: rmse {1e3 * rmse:.2f} mV"


def test_the_residual_now_does_shrink_with_more_states(cell, solve):
    """It did not, until the kinetics were being read correctly.

    While the reaction rate was wrong, the gap against PyBaMM was a constant
    offset that no amount of refinement touched -- five families from six to
    forty-eight states gave the same answer, which correctly said the cause was
    not discretisation but was taken as evidence of an unexplained model
    difference. With the kinetics right, what is left does respond to
    refinement -- so part of it is now discretisation, as it should be.

    It does not go to zero, though. At 2C it falls from about 3.4 mV at order
    three to about 2.5 mV at order seven and then flattens, so a small model
    difference of a couple of millivolt remains. That is an order of magnitude
    below what it was and below anything a measurement front end would resolve,
    but it is a residual and this test records it as one rather than implying the
    two implementations now agree exactly.
    """
    reference = solve("SPM", 2.0, 700.0)
    errors = []
    for order in (3, 5, 7):
        model = SPM(cell, dt=1.0, rom="pade", order=order)
        ours = model.simulate(np.full(701, 10.0), soc0=SOC0)
        errors.append(discrepancy(ours, reference)[0])
    assert errors == sorted(errors, reverse=True), f"not monotone: {errors}"
    assert errors[-1] < 0.8 * errors[0], "refinement should help"
    assert errors[-1] > 1e-4, "but it does not converge to agreement; see the docstring"
    assert errors[-1] < 4e-3, "and what remains is small in absolute terms"


# ------------------------------------------------------- the electrolyte claim


@pytest.mark.parametrize("c_rate,seconds", [(0.5, 1800.0), (1.0, 1200.0), (2.0, 700.0)])
def test_resolving_the_electrolyte_moves_us_towards_the_full_model(cell, solve, c_rate, seconds):
    """The claim `SPMe` exists to make, checked against an outside reference.

    Adding salt transport should close some of the distance to a full
    Doyle-Fuller-Newman solution. It does, at every rate tried, roughly halving
    the discrepancy -- and this is the version of that statement that does not
    depend on any of this package's own assumptions.
    """
    steps = int(seconds) + 1
    drive = np.full(steps, c_rate * 5.0)
    reference = solve("DFN", c_rate, seconds)

    simple = SPM(cell, dt=1.0, rom="pade", order=5).simulate(drive, soc0=SOC0)
    resolved = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)).simulate(
        drive, soc0=SOC0
    )

    without, _ = discrepancy(simple, reference)
    with_electrolyte, _ = discrepancy(resolved, reference)
    assert with_electrolyte < without, (
        f"{c_rate}C: {1e3 * with_electrolyte:.1f} mV is no better than {1e3 * without:.1f} mV"
    )
    # Not a marginal improvement: close to an order of magnitude at every rate.
    assert with_electrolyte < 0.2 * without, (
        f"{c_rate}C: {1e3 * with_electrolyte:.1f} mV against {1e3 * without:.1f} mV"
    )


@pytest.mark.parametrize("c_rate,seconds,tolerance", [(0.5, 1800.0, 6e-3), (1.0, 1200.0, 1.2e-2)])
def test_our_electrolyte_model_tracks_pybamms(cell, solve, c_rate, seconds, tolerance):
    """Against PyBaMM's SPMe specifically, which is the like-for-like comparison."""
    ours = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)).simulate(
        np.full(int(seconds) + 1, c_rate * 5.0), soc0=SOC0
    )
    rmse, _ = discrepancy(ours, solve("SPMe", c_rate, seconds))
    assert rmse < tolerance, f"{c_rate}C: rmse {1e3 * rmse:.2f} mV"


@pytest.mark.parametrize("c_rate,seconds", [(0.5, 1800.0), (1.0, 1200.0), (2.0, 700.0)])
def test_the_electrolyte_model_agrees_with_the_full_solution(cell, solve, c_rate, seconds):
    """The headline, stated absolutely rather than as a ratio.

    Against a full Doyle-Fuller-Newman solution, with every parameter read from
    PyBaMM's own set: a few millivolt at half a C and under twenty at two C, from
    a seventeen-state linear model standing in for a discretised system of
    coupled partial differential equations.
    """
    ours = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)).simulate(
        np.full(int(seconds) + 1, c_rate * 5.0), soc0=SOC0
    )
    rmse, _ = discrepancy(ours, solve("DFN", c_rate, seconds))
    assert rmse < 0.02, f"{c_rate}C: rmse {1e3 * rmse:.2f} mV"


def test_the_linear_electrolyte_gives_up_at_high_rate(cell, solve):
    """Not a failure, the documented limit, and better asserted than described.

    Transport coefficients are held at bulk values, so the model is progressively
    optimistic as the electrolyte depletes. By 3C the discrepancy against a full
    solution is large, and `validity()` should already be saying so rather than
    leaving a caller to discover it from the voltage.
    """
    model = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6))
    drive = np.full(401, 15.0)
    ours = model.simulate(drive, soc0=SOC0)
    rmse, _ = discrepancy(ours, solve("DFN", 3.0, 400.0))
    assert rmse > 0.05, "if this ever gets small, the limitation note is stale"

    state = model.initial_state(SOC0)
    for _ in range(400):
        state = model.step(state, 15.0)
    assert model.validity(state) != "good"
