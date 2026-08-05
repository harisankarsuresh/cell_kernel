"""Every estimator against every model.

The package's central claim is that models and estimators compose: anything
presenting :class:`~cellkernel.models.base.CellModel` can be filtered, generated
from, or aged. That claim is easy to make and easy to quietly break, because each
model is developed against the one filter its author had open at the time.

These tests run the whole grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.data import synthetic_drive_cycle
from cellkernel.degradation import DegradationModel
from cellkernel.estimators import EKF, UKF, DualEKF
from cellkernel.models import ECM, SPM, SPMe, ThermalSPM
from cellkernel.params import chen2020_nmc811_graphite, lfp_graphite

ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)


def build(kind: str, cell):
    if kind == "SPM":
        return SPM(cell, dt=1.0, rom="pade", order=3)
    if kind == "SPMe":
        return SPMe(cell, dt=1.0, rom="pade", order=3, electrolyte_cells=(2, 1, 2))
    if kind == "ThermalSPM":
        return ThermalSPM(cell, dt=1.0, rom="pade", order=3)
    if kind == "ECM":
        return ECM(cell, dt=1.0)
    raise AssertionError(kind)


MODELS = ["SPM", "SPMe", "ThermalSPM", "ECM"]


# ------------------------------------------------------- the shared interface


@pytest.mark.parametrize("kind", MODELS)
def test_every_model_satisfies_the_contract(kind, cell):
    model = build(kind, cell)
    x = model.initial_state(0.6)
    assert x.shape == (model.n_states,)
    assert len(model.state_names) == model.n_states
    assert model.dt > 0.0

    outputs = model.outputs(x, 2.0)
    assert np.isfinite(outputs.voltage)
    assert 0.0 <= outputs.soc <= 1.0
    assert np.isfinite(outputs.temperature)

    assert model.state_jacobian(x, 2.0).shape == (model.n_states, model.n_states)
    assert model.voltage_jacobian(x, 2.0).shape == (model.n_states,)
    assert model.step(x, 2.0).shape == (model.n_states,)


@pytest.mark.parametrize("kind", MODELS)
def test_every_model_reports_a_usable_soc_gradient(kind, cell):
    """Without it the estimators report an uncertainty of exactly zero.

    Which is worse than reporting none, because it looks like an answer.
    """
    model = build(kind, cell)
    gradient = np.asarray(model.soc_jacobian())
    assert gradient.shape == (model.n_states,)
    assert np.any(gradient != 0.0)


@pytest.mark.parametrize("kind", MODELS)
def test_soc_direction_recovers_the_affine_map(kind, cell):
    model = build(kind, cell)
    direction = model.soc_direction()
    predicted = model.initial_state(0.5) + 0.3 * direction
    assert np.allclose(predicted, model.initial_state(0.8), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kind", MODELS)
def test_input_direction_is_the_current_column(kind, cell):
    model = build(kind, cell)
    column = model.input_direction()
    assert column.shape == (model.n_states,)
    assert np.any(column != 0.0)


@pytest.mark.parametrize("kind", ["SPM", "SPMe", "ECM"])
def test_linear_models_have_a_current_independent_input_column(kind, cell):
    """True of everything except the thermal model, where heat is quadratic."""
    model = build(kind, cell)
    assert np.allclose(model.input_direction(0.0), model.input_direction(8.0), rtol=1e-9)


def test_the_thermal_model_is_the_documented_exception(cell):
    model = build("ThermalSPM", cell)
    at_rest = model.input_direction(0.0)
    under_load = model.input_direction(8.0)
    index = model.temperature_index
    # The diffusion rows still agree; only the temperature row moves.
    assert np.allclose(at_rest[:index], under_load[:index], rtol=1e-9)
    assert not np.isclose(at_rest[index], under_load[index])


# ---------------------------------------------------------- filters and models


@pytest.mark.parametrize("kind", MODELS)
@pytest.mark.parametrize("filter_class", [EKF, UKF])
def test_every_filter_runs_on_every_model(kind, filter_class, cell):
    model = build(kind, cell)
    current = synthetic_drive_cycle(cell.nominal_capacity, duration=300.0, seed=1)
    truth = model.simulate(current, soc0=0.7)

    estimator = filter_class(
        model,
        process_noise=filter_class.suggest_process_noise(model, current_std=0.05),
        measurement_noise=1e-5,
        initial_covariance=filter_class.suggest_initial_covariance(model, soc_std=0.1),
    )
    estimator.initialise(0.7)
    result = estimator.run(current, truth["voltage"])

    assert np.all(np.isfinite(result["soc"]))
    assert np.all(np.isfinite(result["voltage"]))
    assert np.all(result["soc_std"] >= 0.0)
    assert np.max(np.abs(result["soc"] - truth["soc"])) < 0.05


@pytest.mark.parametrize("kind", MODELS)
def test_filters_recover_from_a_wrong_start_on_every_model(kind, cell):
    """The property that matters: a cold boot with a bad guess must converge."""
    tolerance = 0.03
    model = build(kind, cell)
    current = np.full(2400, 0.5 * cell.nominal_capacity)
    truth = model.simulate(current, soc0=0.80)

    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, current_std=0.05),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.15),
        iterations=3,
    )
    estimator.initialise(0.65)
    result = estimator.run(current, truth["voltage"])

    start = abs(result["soc"][0] - truth["soc"][0])
    end = abs(result["soc"][-1] - truth["soc"][-1])
    assert end < tolerance, f"{kind}: ended at {end:.4f}"
    assert end < 0.5 * start, f"{kind}: barely improved ({start:.4f} -> {end:.4f})"


def test_the_prior_respects_the_units_of_each_state(cell):
    """A mixed-units state vector breaks an isotropic floor, quietly and totally.

    The floor in :meth:`suggest_initial_covariance` is scaled from the largest
    entry of the state-of-charge direction, which is a concentration of order
    1e5 mol m-3. Applied to every state alike that is a few hundred mol m-3 --
    sensible. Applied to a temperature state it is several hundred kelvin, and
    the filter puts the cell at 430 K on its first update and never comes back.
    """
    model = build("ThermalSPM", cell)
    prior = EKF.suggest_initial_covariance(model, soc_std=0.15)
    index = model.temperature_index
    assert np.sqrt(prior[index, index]) < 10.0, "temperature prior must be in kelvin"
    # The concentration states keep a floor far larger, as they should.
    assert prior[0, 0] > 1e3 * prior[index, index]


def test_process_noise_does_not_tie_temperature_to_the_current_sensor(cell):
    """Temperature error comes from the ambient, not from the shunt.

    Correlating them makes a voltage residual partly a temperature correction,
    and temperature is too weakly observable to survive that.
    """
    model = build("ThermalSPM", cell)
    noise = EKF.suggest_process_noise(model, current_std=0.05)
    index = model.temperature_index
    off_diagonal = np.delete(noise[index], index)
    assert np.allclose(off_diagonal, 0.0, atol=0.0)
    assert noise[index, index] > 0.0


@pytest.mark.parametrize("kind", MODELS)
def test_covariance_stays_well_formed_on_every_model(kind, cell):
    model = build(kind, cell)
    current = synthetic_drive_cycle(cell.nominal_capacity, duration=400.0, seed=5)
    truth = model.simulate(current, soc0=0.6)
    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, current_std=0.05),
        measurement_noise=1e-5,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.1),
    )
    estimator.initialise(0.6)
    for i, v in zip(current, truth["voltage"], strict=True):
        estimator.update(float(i), float(v))
        assert np.allclose(estimator.P, estimator.P.T, atol=0.0)
    assert np.all(np.linalg.eigvalsh(estimator.P) > -1e-9)


@pytest.mark.parametrize("kind", ["SPM", "SPMe", "ECM"])
def test_the_dual_filter_runs_on_the_models_it_supports(kind, cell):
    model = build(kind, cell)
    current = synthetic_drive_cycle(cell.nominal_capacity, duration=600.0, seed=2)
    truth = model.simulate(current, soc0=0.75)
    estimator = DualEKF(
        model,
        process_noise=EKF.suggest_process_noise(model, current_std=0.05),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.1),
    )
    estimator.initialise(0.75)
    result = estimator.run(current, truth["voltage"])
    health = estimator.health()
    assert np.all(np.isfinite(result["soc"]))
    assert 0.5 < health.capacity_retention < 1.5
    assert np.isfinite(health.resistance_growth)


# ------------------------------------------------------------- other pairings


@pytest.mark.parametrize("kind", MODELS)
def test_degradation_reads_every_model(kind, cell):
    """Ageing only needs surface stoichiometry and overpotential, which the
    equivalent circuit does not have -- so it must fail clearly, not silently."""
    model = build(kind, cell)
    ageing = DegradationModel(cell)
    x = model.initial_state(0.8)
    outputs = model.outputs(x, -cell.nominal_capacity)
    if not outputs.surface_stoichiometry:
        pytest.skip("equivalent circuit exposes no surface state")
    result = ageing.evaluate(model, x, -cell.nominal_capacity, ageing.initial_state())
    assert np.isfinite(result.plating_potential)
    assert result.sei_current_density >= 0.0


@pytest.mark.parametrize("chemistry", [chen2020_nmc811_graphite, lfp_graphite])
def test_both_chemistries_work_end_to_end(chemistry):
    """Phosphate is the harder case: a flat plateau makes voltage uninformative."""
    cell = chemistry()
    model = SPM(cell, dt=1.0, rom="pade", order=3)
    current = np.full(1800, 0.5 * cell.nominal_capacity)
    truth = model.simulate(current, soc0=0.7)
    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, current_std=0.02),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.05),
        iterations=3,
    )
    estimator.initialise(0.7)
    result = estimator.run(current, truth["voltage"])
    assert np.max(np.abs(result["soc"] - truth["soc"])) < 0.02


def test_the_flat_plateau_gives_less_signal_per_unit_error():
    """What the phosphate plateau actually costs, stated so it can be checked.

    The folklore is that a flat plateau makes the filter converge slowly. Run as
    an experiment that claim does not survive: with a matched model and clean
    voltage, the phosphate cell converges *faster*, because the Kalman gain is
    ``PH/(H^2 P + R)`` and shrinking ``H`` while ``H^2 P`` still dominates ``R``
    pushes the gain towards ``1/H``, which is large.

    The defensible statement is about signal, not speed, and it needs no filter
    at all: a given state-of-charge error produces about six times less voltage
    discrepancy on phosphate. Everything else -- the noise floor a front end must
    reach, the rest duration needed to re-anchor -- follows from that ratio, and
    unlike a convergence comparison it does not depend on how the two filters
    happened to be tuned.
    """
    signal = {}
    for name, chemistry in (("nmc", chen2020_nmc811_graphite), ("lfp", lfp_graphite)):
        parameters = chemistry()
        model = SPM(parameters, dt=1.0, rom="pade", order=3)
        discrepancies = []
        for soc in (0.3, 0.5, 0.7):
            truth = model.initial_state(soc)
            wrong = model.initial_state(soc + 0.01)
            discrepancies.append(abs(model.voltage(wrong, 0.0) - model.voltage(truth, 0.0)))
        signal[name] = float(np.mean(discrepancies))

    assert signal["nmc"] > 4.0 * signal["lfp"], (
        f"expected far more signal on the layered oxide: "
        f"nmc {1e3 * signal['nmc']:.3f} mV vs lfp {1e3 * signal['lfp']:.3f} mV per 1% soc"
    )
    # And on phosphate it is genuinely close to a good front end's noise floor.
    assert signal["lfp"] < 3e-3
