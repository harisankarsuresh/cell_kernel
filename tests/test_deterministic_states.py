"""States that are propagated but never corrected.

A state driven by the input alone, unaffected by the others, and starting from a
known condition is a function of the current history rather than something to be
estimated. Correcting it spends arithmetic on information the measurement does
not carry -- and because the covariance update is cubic in the number of
estimated states, the arithmetic is not a rounding error.

This matters most for `SPMe`, where the electrolyte block is over half the state
vector, and is the difference between that model being plausible on a
microcontroller and not.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.estimators import EKF, UKF
from cellkernel.models import ECM, SPM, SPMe, ThermalSPM
from cellkernel.params import chen2020_nmc811_graphite


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite()


@pytest.fixture(scope="module")
def model(cell):
    return SPMe(SPMe.reconcile(cell), dt=1.0, rom="pade", order=3, electrolyte_cells=(4, 3, 4))


# ------------------------------------------------------------------ the model


def test_only_the_electrolyte_model_declares_any(cell):
    """It is a structural property, so only the model that has one reports it."""
    assert SPM(cell, dt=1.0).deterministic_states == ()
    assert ECM(cell, dt=1.0).deterministic_states == ()
    assert ThermalSPM(cell, dt=1.0).deterministic_states == ()


def test_the_electrolyte_block_is_declared(model):
    declared = model.deterministic_states
    assert len(declared) == model._n_elec
    assert min(declared) == model._i_elec
    assert max(declared) == model.n_states - 1
    names = model.state_names
    assert all(names[index].startswith("elyte_") for index in declared)


def test_the_claim_behind_it_holds(model, cell):
    """The electrolyte really is a function of current alone.

    Two runs from the same current but wildly different solid states must reach
    identical electrolyte states. If that ever stops being true the block is no
    longer deterministic and this whole optimisation is unsound.
    """
    current = np.full(400, 2.0 * cell.nominal_capacity)
    low = model.initial_state(0.2)
    high = model.initial_state(0.9)
    for value in current:
        low = model.step(low, float(value))
        high = model.step(high, float(value))
    index = model._i_elec
    assert np.allclose(low[index:], high[index:], rtol=0, atol=1e-9)
    assert not np.allclose(low[:index], high[:index])


# --------------------------------------------------------------- the estimator


def test_the_mask_matches_what_the_model_declared(model, cell):
    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model),
    )
    mask = estimator.correction_mask
    assert mask.shape == (model.n_states,)
    assert np.all(mask[: model._i_elec] == 1.0)
    assert np.all(mask[model._i_elec :] == 0.0)


def test_declared_states_are_propagated_but_not_corrected(model, cell):
    """The distinction that makes this safe: they still move, just not from voltage."""
    current = np.full(600, 1.5 * cell.nominal_capacity)
    truth = model.simulate(current, soc0=0.8)
    rng = np.random.default_rng(0)
    measured = truth["voltage"] + rng.normal(0.0, 2e-3, truth["voltage"].size)

    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, current_std=0.05),
        measurement_noise=4e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.1),
    )
    estimator.initialise(0.8)
    open_loop = model.initial_state(0.8)
    index = model._i_elec
    for value, reading in zip(current, measured, strict=True):
        estimator.update(float(value), float(reading))
        open_loop = model.step(open_loop, float(value))

    # Propagated: they moved a long way from where they started.
    assert np.max(np.abs(estimator.x[index:] - model.initial_state(0.8)[index:])) > 50.0
    # Not corrected: they went exactly where open-loop simulation put them.
    assert np.allclose(estimator.x[index:], open_loop[index:], rtol=0, atol=1e-9)


def test_the_solid_states_are_still_corrected(model, cell):
    """Guards against masking too much."""
    current = np.full(600, 1.0 * cell.nominal_capacity)
    truth = model.simulate(current, soc0=0.8)
    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, current_std=0.05),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.15),
        iterations=3,
    )
    estimator.initialise(0.65)
    result = estimator.run(current, truth["voltage"])
    assert abs(result["soc"][-1] - truth["soc"][-1]) < 0.03


def test_accuracy_is_not_the_price(model, cell):
    """Skipping the correction must not cost the estimate.

    Run against a filter that is allowed to correct everything, on a drive cycle
    hard enough to move the electrolyte across most of its range. If this ever
    fails, the saving is not free and the model should stop declaring the block.
    """

    class Unrestricted(EKF):
        """Corrects every state, ignoring what the model advises."""

        @property
        def correction_mask(self) -> np.ndarray:
            return np.ones(self.model.n_states)

    rng = np.random.default_rng(3)
    current = np.repeat(rng.uniform(-1.0, 2.5, 30), 25) * cell.nominal_capacity
    truth = model.simulate(current, soc0=0.85)
    measured = truth["voltage"] + rng.normal(0.0, 2e-3, truth["voltage"].size)

    def run(cls) -> np.ndarray:
        estimator = cls(
            model,
            process_noise=EKF.suggest_process_noise(model, current_std=0.05),
            measurement_noise=4e-6,
            initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.1),
        )
        estimator.initialise(0.85)
        return np.array(
            [
                estimator.update(float(i), float(v)).soc - reference
                for i, v, reference in zip(current, measured, truth["soc"], strict=True)
            ]
        )

    masked = run(EKF)
    full = run(Unrestricted)
    assert not np.allclose(masked, full), "the two should not be bit-identical"

    settled = slice(300, None)
    masked_rmse = float(np.sqrt(np.mean(masked[settled] ** 2)))
    full_rmse = float(np.sqrt(np.mean(full[settled] ** 2)))
    assert masked_rmse < max(2.0 * full_rmse, 0.005), (
        f"masked {1e2 * masked_rmse:.4f}% against full {1e2 * full_rmse:.4f}%"
    )
    assert masked_rmse < 0.01


def test_the_unscented_filter_honours_it_too(model, cell):
    """Both filters read the same declaration, so neither can drift from the other."""
    current = np.full(300, 1.0 * cell.nominal_capacity)
    truth = model.simulate(current, soc0=0.8)
    estimator = UKF(
        model,
        process_noise=UKF.suggest_process_noise(model, current_std=0.05),
        measurement_noise=1e-5,
        initial_covariance=UKF.suggest_initial_covariance(model, soc_std=0.1),
    )
    assert np.all(estimator.correction_mask[model._i_elec :] == 0.0)
    estimator.initialise(0.8)
    result = estimator.run(current, truth["voltage"])
    assert np.all(np.isfinite(result["soc"]))


def test_the_saving_is_worth_having(model):
    """Cubic in the estimated count, so removing half the states is not half the work."""
    total = model.n_states
    estimated = total - len(model.deterministic_states)
    full = total**3
    split = estimated**3 + len(model.deterministic_states) ** 2
    assert split < 0.35 * full, f"{full} -> {split} is not much of a saving"
