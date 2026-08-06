"""Validation against a real LG M50, not against another model.

Every other suite here answers "is this implemented correctly". Closed-form
identities, a NumPy mirror, PyBaMM -- all of them compare code against code or
against mathematics. None of them can say whether the result describes a cell.

This one can, and the answer is more sobering than the others. Against PyBaMM the
single particle model agrees to 0.26 mV. Against a real cell, with the same
literature parameters, it is out by tens of millivolts. Both numbers are correct
and they measure different things: the first says the implementation is faithful,
the second says the parameters belong to a different unit. Reporting only the
first would be misleading, so these tests exist to keep the second visible.

The dataset is not vendored -- it belongs to the PyBOP project. Run
``python -m cellkernel.data.reference`` or call ``download()`` to fetch it. These
tests skip without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.data import reference
from cellkernel.models import SPM, SPMe
from cellkernel.params import chen2020_nmc811_graphite, fit_stoichiometry_window

pytest.importorskip("scipy.io")


def _have_data() -> bool:
    try:
        reference.load_ocv()
    except (FileNotFoundError, OSError):
        return False
    return True


needs_data = pytest.mark.skipif(
    not _have_data(), reason="measured dataset not downloaded; see cellkernel.data.reference"
)


@pytest.fixture(scope="module")
def cell():
    """The literature parameter set, imported verbatim from PyBaMM.

    Explicitly *not* falling back to the built-in set, which is the same physics
    but carries a 10 mOhm lumped contact resistance standing in for losses
    PyBaMM's Chen2020 does not include. Handing that to `SPMe`, which computes
    the electrolyte from geometry, counts the electrolyte twice and makes it
    perform worse than `SPM`.

    An earlier version of this fixture did fall back, and the resulting failure
    -- only in the one CI job without PyBaMM installed -- is what surfaced the
    trap. `SPMe` warns about it now.
    """
    pybamm = pytest.importorskip("pybamm", reason="needed for a verbatim parameter set")
    from cellkernel.params import from_pybamm

    imported = from_pybamm(pybamm.ParameterValues("Chen2020"))
    assert imported.contact_resistance == 0.0, "expected no lumped resistance"
    return imported


@pytest.fixture(scope="module")
def measured_capacity():
    segment = reference.load_discharge("T25", "cRate_0p1C")
    return float(np.trapezoid(segment.current, segment.time) / 3600.0)


@pytest.fixture(scope="module")
def fitted(cell, measured_capacity):
    soc, volts = reference.load_ocv()
    return fit_stoichiometry_window(cell, soc, volts, capacity=measured_capacity)


def rmse(model_voltage, measured, floor: float = 2.7) -> float:
    """Millivolts, over the range above the cut-off where the model applies."""
    mask = measured > floor
    return 1e3 * float(np.sqrt(np.mean((model_voltage[mask] - measured[mask]) ** 2)))


# --------------------------------------------------------------------- loading


@needs_data
def test_the_dataset_covers_four_rates_and_four_temperatures():
    conditions = reference.available_conditions()
    assert len(conditions) >= 12
    assert {a for a, _ in conditions} == {"T0", "T10", "T25", "T45"}


@needs_data
def test_current_is_flipped_to_this_packages_convention():
    """The source file is charge-positive; getting this backwards is invisible."""
    segment = reference.load_discharge("T25", "cRate_1C")
    assert np.all(segment.current > 0.0), "discharge must be positive here"
    assert segment.current.mean() == pytest.approx(5.0, rel=0.02)


@needs_data
def test_segments_are_uniformly_sampled():
    """Source spacing varies from one second to a minute depending on the rate."""
    segment = reference.load_discharge("T25", "cRate_0p1C", dt=1.0)
    assert np.allclose(np.diff(segment.time), 1.0)


@needs_data
def test_the_measured_capacity_is_close_to_nameplate(measured_capacity):
    assert 4.5 < measured_capacity < 5.2


@needs_data
def test_download_is_a_no_op_when_the_files_are_present(tmp_path, monkeypatch):
    """It must not reach the network on every run, only when something is missing."""

    def refuse(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("download() tried to fetch an already-present file")

    for name in ("LGM50_5Ah_RateTest.mat", "LGM50_5Ah_OCV.mat"):
        (tmp_path / name).write_bytes(b"placeholder")
    monkeypatch.setattr(reference.urllib.request, "urlopen", refuse)
    assert reference.download(tmp_path) == tmp_path


@needs_data
def test_a_missing_dataset_says_how_to_get_it(tmp_path):
    """Rather than a bare FileNotFoundError from somewhere inside scipy."""
    with pytest.raises(FileNotFoundError, match="download"):
        reference.load_ocv(cache=tmp_path)


@needs_data
def test_unknown_conditions_are_rejected():
    with pytest.raises(ValueError, match="ambient"):
        reference.load_discharge("T99", "cRate_1C")
    with pytest.raises(ValueError, match="rate"):
        reference.load_discharge("T25", "cRate_9C")


# ------------------------------------------------------- the sobering compariso


@needs_data
def test_literature_parameters_do_not_reproduce_this_cell(cell):
    """The headline result, asserted so nobody has to take it on trust.

    Tens of millivolts on the open-circuit curve alone, before any dynamics are
    involved. This is not an implementation error -- the same code matches
    PyBaMM to a quarter of a millivolt -- it is what a parameter set fitted to
    one cell does on another.
    """
    soc, volts = reference.load_ocv()
    error = np.asarray(cell.open_circuit_voltage(soc)) - volts
    interior = (soc > 0.05) & (soc < 0.95)
    settled = 1e3 * float(np.sqrt(np.mean(error[interior] ** 2)))
    assert settled > 20.0, (
        f"only {settled:.1f} mV off; if the parameter set has improved this much, "
        "the README's claim about literature parameters needs revisiting"
    )
    assert settled < 100.0, f"{settled:.1f} mV is worse than expected"


@needs_data
def test_fitting_the_window_closes_most_of_the_open_circuit_gap(cell, fitted):
    soc, volts = reference.load_ocv()
    before = np.asarray(cell.open_circuit_voltage(soc)) - volts
    after = np.asarray(fitted.open_circuit_voltage(soc)) - volts
    assert np.sqrt(np.mean(after**2)) < 0.35 * np.sqrt(np.mean(before**2))


@needs_data
def test_the_fit_holds_capacity(fitted, measured_capacity):
    """Which is the whole reason capacity is a residual and not left free."""
    assert fitted.usable_capacity() == pytest.approx(measured_capacity, rel=5e-3)


@needs_data
def test_an_unconstrained_fit_is_degenerate(cell, measured_capacity):
    """Documented as a trap, so pinned here as one.

    With capacity free the solver buys open-circuit accuracy by stretching the
    state-of-charge axis. It produces a better-looking curve and a worse cell.
    """
    soc, volts = reference.load_ocv()
    loose = fit_stoichiometry_window(cell, soc, volts, capacity_weight=0.0)
    drift = abs(loose.usable_capacity() - measured_capacity) / measured_capacity
    assert drift > 0.03, (
        "the unconstrained fit is supposed to wander; if it no longer does, the "
        "warning in fit_stoichiometry_window can be softened"
    )


# ------------------------------------------------------------ under load


@needs_data
@pytest.mark.parametrize("rate", ["cRate_0p1C", "cRate_0p5C", "cRate_1C", "cRate_2C"])
def test_the_models_track_a_real_discharge_to_tens_of_millivolts(cell, rate):
    """Not millivolts. Tens of them, and the distinction matters."""
    segment = reference.load_discharge("T25", rate)
    model = SPM(cell, dt=1.0, rom="pade", order=5)
    run = model.simulate(segment.current, soc0=1.0)
    error = rmse(run["voltage"], segment.voltage)
    assert 10.0 < error < 250.0, f"{segment.c_rate}C: {error:.1f} mV"


@needs_data
@pytest.mark.parametrize("rate", ["cRate_1C", "cRate_2C"])
def test_resolving_the_electrolyte_helps_on_a_real_cell(cell, rate):
    """Independent confirmation of the electrolyte work, against measurement.

    The claim was previously supported only against a Doyle-Fuller-Newman
    solution, which is still model-against-model. On this cell at 2C the
    electrolyte model roughly halves the error against what was actually
    recorded.
    """
    segment = reference.load_discharge("T25", rate)
    simple = SPM(cell, dt=1.0, rom="pade", order=5).simulate(segment.current, soc0=1.0)
    resolved = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)).simulate(
        segment.current, soc0=1.0
    )
    without = rmse(simple["voltage"], segment.voltage)
    with_electrolyte = rmse(resolved["voltage"], segment.voltage)
    assert with_electrolyte < without, (
        f"{segment.c_rate}C: {with_electrolyte:.1f} mV against {without:.1f} mV"
    )


def test_a_lumped_contact_resistance_can_be_reconciled():
    """The trap this suite fell into, and the helper that removes it.

    The built-in set's 10 mOhm was fitted without an electrolyte model, so it
    already contains the loss `SPMe` computes from geometry. Counting it twice
    makes the more detailed model perform *worse* than the simpler one, and the
    symptom is a plausible voltage that is merely too low. It surfaced in the one
    CI job without PyBaMM installed, which fell back to this parameter set.
    """
    lumped = chen2020_nmc811_graphite()
    assert lumped.contact_resistance > 0.0
    probe = SPMe(lumped, dt=1.0, rom="pade", order=3)
    tidy = SPMe.reconcile(lumped)
    assert tidy.contact_resistance == pytest.approx(
        lumped.contact_resistance - probe.electrolyte_resistance
    )
    assert SPMe.reconcile(tidy).contact_resistance >= 0.0


def test_reconcile_floors_at_zero():
    """A set whose entire resistance is electrolyte must not go negative."""
    from dataclasses import replace

    cell = replace(chen2020_nmc811_graphite(), contact_resistance=1e-4)
    assert SPMe.reconcile(cell).contact_resistance == 0.0


@needs_data
def test_the_electrolyte_advantage_grows_with_rate(cell):
    """As it should: at low rate the electrolyte really is just a resistance."""
    ratios = {}
    for rate in ("cRate_0p1C", "cRate_2C"):
        segment = reference.load_discharge("T25", rate)
        simple = SPM(cell, dt=1.0, rom="pade", order=5).simulate(segment.current, soc0=1.0)
        resolved = SPMe(cell, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)).simulate(
            segment.current, soc0=1.0
        )
        ratios[rate] = rmse(resolved["voltage"], segment.voltage) / rmse(
            simple["voltage"], segment.voltage
        )
    assert ratios["cRate_2C"] < ratios["cRate_0p1C"]


# --------------------------------------------------------------- self-heating


@needs_data
def test_measured_self_heating_scales_the_way_the_model_says(cell):
    """A 2C discharge heats this cell by 33 K. Any isothermal model is wrong here.

    Not a comparison against `ThermalSPM` -- the lumped heat-transfer
    coefficient and surface area in the built-in parameter set are placeholders
    and would have to be fitted first. What is checked is the qualitative claim
    the thermal section rests on: dissipation goes as the square of current, so
    quadrupling the rate should raise the temperature rise far more than
    fourfold once the cell has less time to shed it.
    """
    rises = {}
    for rate in ("cRate_0p5C", "cRate_1C", "cRate_2C"):
        segment = reference.load_discharge("T25", rate)
        assert segment.temperature is not None
        rises[segment.c_rate] = segment.temperature_rise
    assert rises[0.5] < rises[1.0] < rises[2.0]
    assert rises[2.0] > 20.0, "a 2C discharge really does heat this cell substantially"
    # Superlinear in current, which is what makes an isothermal model fail fast.
    assert rises[2.0] / rises[1.0] > 2.0


@needs_data
def test_the_cold_cell_heats_more_not_less():
    """Sluggish transport means more dissipation, which is why cold is dangerous.

    The same 2C discharge raises the cell 42 K from 0 C ambient against 33 K from
    25 C. A thermal design validated at room temperature is not validated.
    """
    cold = reference.load_discharge("T0", "cRate_2C").temperature_rise
    warm = reference.load_discharge("T25", "cRate_2C").temperature_rise
    assert cold > warm
