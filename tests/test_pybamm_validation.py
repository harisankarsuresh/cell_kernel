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


def test_single_particle_models_agree_at_low_rate(cell, solve):
    """The strongest form of the check: same physics, two implementations.

    At 0.5C the concentration profile is shallow, the kinetics are near-linear,
    and there is nothing left for the two to disagree about. Sub-millivolt is
    what agreement should look like here, and anything larger would point at a
    parameter being carried across wrongly.
    """
    model = SPM(cell, dt=1.0, rom="pade", order=5)
    ours = model.simulate(np.full(1801, 2.5), soc0=SOC0)
    rmse, worst = discrepancy(ours, solve("SPM", 0.5, 1800.0))
    assert rmse < 1e-3, f"rmse {1e3 * rmse:.2f} mV"
    assert worst < 2e-3, f"max {1e3 * worst:.2f} mV"


@pytest.mark.parametrize("c_rate,seconds", [(1.0, 1200.0), (2.0, 700.0)])
def test_single_particle_models_stay_close_at_higher_rate(cell, solve, c_rate, seconds):
    model = SPM(cell, dt=1.0, rom="pade", order=5)
    ours = model.simulate(np.full(int(seconds) + 1, c_rate * 5.0), soc0=SOC0)
    rmse, _ = discrepancy(ours, solve("SPM", c_rate, seconds))
    assert rmse < 0.03, f"{c_rate}C: rmse {1e3 * rmse:.2f} mV"


def test_the_high_rate_residual_is_not_a_discretisation_error(cell, solve):
    """Worth separating, because the two have opposite remedies.

    The gap against PyBaMM's own single particle model grows from 0.2 mV at 0.5C
    to about 23 mV at 2C. If that were the reduced-order approximation it would
    fall as states are added. It does not: five families spanning six to
    forty-eight states give the same answer to a fraction of a millivolt. So it
    is a difference in the models, not in how finely they are resolved, and
    adding states is not the fix.
    """
    reference = solve("SPM", 2.0, 700.0)
    results = {}
    for kind, order in (("pade", 3), ("pade", 7), ("spectral", 8), ("fv", 24)):
        model = SPM(cell, dt=1.0, rom=kind, order=order)
        ours = model.simulate(np.full(701, 10.0), soc0=SOC0)
        results[(kind, model.n_states)] = discrepancy(ours, reference)[0]

    values = list(results.values())
    assert max(values) - min(values) < 1e-3, (
        f"spread {1e3 * (max(values) - min(values)):.2f} mV across {list(results)}"
    )
    assert min(values) > 0.01, "and the residual itself does not vanish"


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
    assert with_electrolyte < 0.6 * without, "and the improvement should be substantial"


def test_our_electrolyte_model_tracks_pybamms(cell, solve):
    """Against PyBaMM's SPMe specifically, which is the like-for-like comparison."""
    ours = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)).simulate(
        np.full(1801, 2.5), soc0=SOC0
    )
    rmse, _ = discrepancy(ours, solve("SPMe", 0.5, 1800.0))
    assert rmse < 0.02, f"rmse {1e3 * rmse:.2f} mV"


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
