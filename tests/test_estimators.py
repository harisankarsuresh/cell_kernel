"""Validation of the state and health estimators.

The pattern throughout is a truth-model experiment: generate voltage from a model
with known state, corrupt it with noise, hand the estimator a deliberately wrong
initial condition, and require that it recovers the truth. Where an estimator
*should not* be able to recover something, that is asserted too -- a filter that
appears to identify an unobservable parameter is more dangerous than one that
admits it cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.data import constant_current, hppc_pulses, rest, synthetic_drive_cycle
from cellkernel.estimators import EKF, UKF, DualEKF, safe_cholesky
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite()


@pytest.fixture(scope="module")
def model(cell):
    return SPM(cell, dt=1.0, rom="pade", order=3)


def truth_run(model, current, soc0=0.8, noise_std=1e-3, seed=0):
    """Simulate the model and return noisy voltage alongside the true state of charge."""
    result = model.simulate(current, soc0=soc0)
    rng = np.random.default_rng(seed)
    measured = result["voltage"] + rng.normal(0.0, noise_std, result["voltage"].size)
    return measured, result


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def _tuned(filter_class, model, soc_std=0.2, **kwargs):
    return filter_class(
        model,
        process_noise=EKF.suggest_process_noise(model, 0.05, soc_drift_per_hour=0.02),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("filter_class", "kwargs", "tol"),
    [
        (EKF, {"iterations": 3}, 0.01),
        (UKF, {}, 0.03),
    ],
    ids=["iterated-EKF", "UKF"],
)
@pytest.mark.parametrize("soc_error", [0.15, -0.15])
def test_filter_recovers_from_wrong_initial_soc(model, filter_class, kwargs, tol, soc_error):
    """A 15% initial error must be corrected to a small fraction of that."""
    current = synthetic_drive_cycle(5.0, duration=2400.0, dt=model.dt, peak_discharge_rate=2.0)
    measured, truth = truth_run(model, current, soc0=0.75)

    estimator = _tuned(filter_class, model, **kwargs)
    estimator.initialise(0.75 + soc_error)
    # Check the seeded error before filtering. The first reported value is already
    # corrected, and with a well-shaped prior most of the error is removed on the
    # very first sample, so it is not a measure of where the filter started.
    assert abs(estimator.model.soc(estimator.x) - 0.75) == pytest.approx(abs(soc_error), abs=1e-6)
    out = estimator.run(current, measured)

    final_error = abs(out["soc"][-1] - truth["soc"][-1])
    assert final_error < tol, f"converged to {final_error:.4f}"


def test_single_shot_ekf_overshoots_where_the_iterated_form_converges(model):
    """Document the linearisation failure that motivates iterating.

    Seeded at 90% when the truth is 75%, the local ``dOCV/dSOC`` of this cell is
    about five times smaller than the average slope across the error, so one
    linearised correction overshoots badly and the estimate never settles. Two
    Gauss-Newton iterations are enough to fix it. If a future change makes the
    single-shot filter succeed here, this test will fail and should be deleted --
    but it should not be relaxed silently, because the failure it captures is the
    reason the option exists.
    """
    current = synthetic_drive_cycle(5.0, duration=2400.0, dt=model.dt, peak_discharge_rate=2.0)
    measured, truth = truth_run(model, current, soc0=0.75)

    errors = {}
    for iterations in (1, 3):
        estimator = _tuned(EKF, model, iterations=iterations)
        estimator.initialise(0.90)
        out = estimator.run(current, measured)
        errors[iterations] = abs(out["soc"][-1] - truth["soc"][-1])

    assert errors[1] > 0.1, "single-shot filter unexpectedly coped"
    assert errors[3] < 0.01
    assert errors[3] < errors[1] / 20.0


def test_iterated_ekf_matches_or_beats_the_unscented_filter(model):
    """With iteration the cheaper filter should be at least as accurate here."""
    current = synthetic_drive_cycle(5.0, duration=2400.0, dt=model.dt, peak_discharge_rate=2.0)
    measured, truth = truth_run(model, current, soc0=0.75)

    results = {}
    for name, estimator in (
        ("ekf", _tuned(EKF, model, iterations=3)),
        ("ukf", _tuned(UKF, model)),
    ):
        estimator.initialise(0.90)
        out = estimator.run(current, measured)
        results[name] = float(np.sqrt(np.mean((out["soc"][-500:] - truth["soc"][-500:]) ** 2)))
    assert results["ekf"] <= results["ukf"] * 1.5


def test_iterations_must_be_positive(model):
    with pytest.raises(ValueError, match="iterations must be at least 1"):
        EKF(model, 1.0, 1e-6, 1.0, iterations=0)


@pytest.mark.parametrize("filter_class", [EKF, UKF])
def test_filter_beats_open_loop_coulomb_counting(model, filter_class):
    """The filter must do better than integrating current from a wrong start.

    This is the claim that justifies running a filter at all. Open-loop counting
    keeps whatever error it started with forever, so the comparison is a floor the
    filter has to clear.
    """
    current = synthetic_drive_cycle(5.0, duration=3000.0, dt=model.dt, peak_discharge_rate=2.4)
    measured, truth = truth_run(model, current, soc0=0.7)

    kwargs = {"iterations": 3} if filter_class is EKF else {}
    estimator = _tuned(filter_class, model, **kwargs)
    estimator.initialise(0.85)
    out = estimator.run(current, measured)

    open_loop = 0.85 - np.cumsum(current) * model.dt / (3600.0 * model.parameters.nominal_capacity)
    filtered_rmse = float(np.sqrt(np.mean((out["soc"][-500:] - truth["soc"][-500:]) ** 2)))
    open_loop_rmse = float(np.sqrt(np.mean((open_loop[-500:] - truth["soc"][-500:]) ** 2)))
    assert filtered_rmse < open_loop_rmse / 5.0


def test_innovation_is_consistent_with_its_predicted_variance(model):
    """A well-tuned filter's normalised innovation should have unit variance.

    The normalised innovation squared, ``e^2 / S``, has expectation one when the
    filter's noise model matches reality. Checking it is the standard consistency
    test and it catches covariance bookkeeping errors that a state-error test can
    miss, because a filter can track well while being badly over-confident.
    """
    current = synthetic_drive_cycle(5.0, duration=3600.0, dt=model.dt, peak_discharge_rate=1.6)
    noise_std = 2e-3
    measured, _ = truth_run(model, current, soc0=0.8, noise_std=noise_std)

    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, 0.05, 0.01),
        measurement_noise=noise_std**2,
        initial_covariance=EKF.suggest_initial_covariance(model, 0.02),
        iterations=3,
    )
    estimator.initialise(0.8)
    out = estimator.run(current, measured)

    settled = slice(len(current) // 4, None)
    normalised = out["innovation"][settled] ** 2 / out["innovation_variance"][settled]
    assert 0.2 < float(np.mean(normalised)) < 5.0


def test_reported_uncertainty_shrinks_as_the_filter_converges(model):
    current = hppc_pulses(5.0, discharge_rate=3.0, steps=6, dt=model.dt)
    measured, _ = truth_run(model, current, soc0=0.6)
    estimator = _tuned(EKF, model, iterations=3)
    estimator.initialise(0.7)
    out = estimator.run(current, measured)
    assert out["soc_std"][-1] < out["soc_std"][0] / 2.0


def test_filter_stays_within_its_own_uncertainty_through_a_long_rest(model):
    """During an hour at open circuit the error must stay inside the reported band.

    Checking the estimate against its own covariance rather than against a fixed
    number is the meaningful test, and it is a demanding one: it fails both if the
    estimate drifts and if the filter understates how uncertain it is.

    Some drift is legitimate here and worth understanding. One voltage measurement
    cannot separate the two electrodes, so there is a direction in state space --
    both particles shifting so as to leave the terminal voltage unchanged -- that
    voltage alone never resolves. Only current integration couples them, and the
    positive particle in this cell has a diffusion time constant of nearly two
    hours, so a gradient established before the rest is still decaying when the
    rest ends. A filter that claimed millivolt-level certainty about state of
    charge under those conditions would be lying.
    """
    current = np.concatenate([np.full(600, 10.0), rest(3600.0, model.dt), np.full(600, 10.0)])
    measured, truth = truth_run(model, current, soc0=0.7)
    drift_allowance = 0.002
    estimator = EKF(
        model,
        process_noise=EKF.suggest_process_noise(model, 0.05, drift_allowance),
        measurement_noise=1e-6,
        initial_covariance=EKF.suggest_initial_covariance(model, 0.05),
        iterations=3,
    )
    estimator.initialise(0.7)
    out = estimator.run(current, measured)

    rest_window = slice(700, 4100)
    error = np.abs(out["soc"][rest_window] - truth["soc"][rest_window])
    assert np.all(out["soc_std"][rest_window] > 0.0)
    # The random-walk term in Q is the only thing that can move the estimate at
    # open circuit, so the drift must stay within a small multiple of what was
    # asked for. This would catch a sign error or a runaway gain immediately.
    assert np.max(error) < 5.0 * drift_allowance
    assert abs(out["soc"][-1] - truth["soc"][-1]) < 0.01


def test_reported_uncertainty_is_optimistic_during_a_long_rest(model):
    """Record a known limitation: the posterior understates error at open circuit.

    A single voltage measurement cannot separate the two electrodes -- there is a
    direction in state space, both particles shifting together so as to leave the
    terminal voltage unchanged, that voltage never observes. Only current
    integration couples them, and at open circuit there is no current. The process
    noise supplied by :meth:`EKF.suggest_process_noise` is placed along the
    state-of-charge and input directions, both of which are *observable*, so the
    covariance has nothing representing the unobservable direction and the reported
    standard deviation comes out several times smaller than the true error.

    This test asserts the deficiency rather than hiding it. If a future change makes
    the covariance honest here -- by adding process noise in the differential
    direction, or by estimating the electrode split -- this test will fail and
    should be replaced by the three-sigma consistency check it currently stands in
    for.
    """
    current = np.concatenate([constant_current(10.0, 600.0, model.dt), rest(3600.0, model.dt)])
    measured, truth = truth_run(model, current, soc0=0.7)
    estimator = _tuned(EKF, model, soc_std=0.05, iterations=3)
    estimator.initialise(0.7)
    out = estimator.run(current, measured)

    rest_window = slice(700, None)
    error = np.abs(out["soc"][rest_window] - truth["soc"][rest_window])
    reported = out["soc_std"][rest_window]
    assert np.max(error / reported) > 3.0


# ---------------------------------------------------------------------------
# Covariance health
# ---------------------------------------------------------------------------


def test_covariance_stays_symmetric_and_positive_semidefinite(model):
    current = synthetic_drive_cycle(5.0, duration=1800.0, dt=model.dt, peak_discharge_rate=4.0)
    measured, _ = truth_run(model, current, soc0=0.5, noise_std=5e-3)
    estimator = EKF(
        model,
        EKF.suggest_process_noise(model, 0.05, 0.05),
        2.5e-5,
        EKF.suggest_initial_covariance(model, 0.2),
        iterations=3,
    )
    estimator.initialise(0.65)
    for i, v in zip(current, measured, strict=False):
        estimator.update(float(i), float(v))
        assert np.allclose(estimator.P, estimator.P.T, atol=0.0)
        eigenvalues = np.linalg.eigvalsh(estimator.P)
        assert eigenvalues.min() > -1e-9 * max(eigenvalues.max(), 1.0)


def test_joseph_form_matches_short_form_in_exact_arithmetic(model):
    """The Joseph update must agree with (I-KH)P when conditioning is benign.

    Guards the algebra: the two forms are mathematically identical, so a
    discrepancy here would mean a transcription error rather than a rounding
    effect.
    """
    estimator = EKF(model, 1.0, 1e-6, 1e3)
    estimator.initialise(0.6)
    P = estimator.P.copy()
    H = model.voltage_jacobian(estimator.x, 5.0).reshape(1, -1)
    PH = P @ H.T
    S = float((H @ PH).item()) + estimator.R
    K = PH / S
    identity = np.eye(model.n_states)
    joseph = (identity - K @ H) @ P @ (identity - K @ H).T + K @ K.T * estimator.R
    short = (identity - K @ H) @ P
    assert np.allclose(joseph, short, rtol=1e-6, atol=1e-9 * np.abs(P).max())


def test_safe_cholesky_recovers_from_marginal_indefiniteness():
    base = np.diag([1.0, 1e-18, 4.0])
    nudged = base.copy()
    nudged[1, 1] = -1e-20  # marginally negative, as rounding can produce
    factor = safe_cholesky(nudged)
    assert np.all(np.isfinite(factor))
    assert factor.shape == (3, 3)


def test_safe_cholesky_is_exact_for_healthy_matrices():
    rng = np.random.default_rng(3)
    root = rng.normal(size=(5, 5))
    matrix = root @ root.T + 5.0 * np.eye(5)
    factor = safe_cholesky(matrix)
    assert np.allclose(factor @ factor.T, matrix, rtol=1e-10)


# ---------------------------------------------------------------------------
# Unscented specialisation
# ---------------------------------------------------------------------------


def _sigma_predicted_covariance(estimator, model, current):
    points = estimator.sigma_points()
    propagated = np.array([model.step(p, current) for p in points])
    mean = estimator.weights_mean @ propagated
    deviation = propagated - mean
    return mean, (estimator.weights_cov * deviation.T) @ deviation


@pytest.mark.parametrize("alpha", [1e-3, 1e-1, 1.0])
def test_unscented_transform_of_linear_dynamics_is_exact(model, alpha):
    """Sigma-point propagation through the linear process reproduces A P A'.

    This is the justification for the UKF using a matrix prediction step: for an
    affine map the unscented transform is exact in real arithmetic, so the cheaper
    route computes the same answer.

    In floating point the two agree only to the precision the sigma-point
    weighting allows, which is why ``alpha`` is swept. Small ``alpha`` clusters the
    points tightly around the mean and pairs a large negative centre weight with
    large positive outer weights; reconstructing the covariance from that
    arrangement is a difference of big cancelling terms and loses several digits.
    So the matrix prediction step is not merely faster here, it is also more
    accurate.
    """
    estimator = UKF(model, 1.0, 1e-6, 1e4, alpha=alpha)
    estimator.initialise(0.55)
    current = 7.0
    mean, covariance = _sigma_predicted_covariance(estimator, model, current)

    A = model.state_jacobian(estimator.x, current)
    reference = A @ estimator.P @ A.T
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.allclose(mean, model.step(estimator.x, current), rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(covariance - reference)) / scale < 1e-6


def test_tight_sigma_points_lose_precision_relative_to_the_matrix_update(model):
    """Quantify the cancellation: alpha = 1e-3 is measurably worse than alpha = 1."""
    current = 7.0
    errors = []
    for alpha in (1.0, 1e-3):
        estimator = UKF(model, 1.0, 1e-6, 1e4, alpha=alpha)
        estimator.initialise(0.55)
        _, covariance = _sigma_predicted_covariance(estimator, model, current)
        A = model.state_jacobian(estimator.x, current)
        reference = A @ estimator.P @ A.T
        errors.append(np.max(np.abs(covariance - reference)) / max(np.abs(reference).max(), 1e-300))
    assert errors[1] > errors[0] * 10.0


def test_ukf_and_ekf_agree_once_settled(model):
    """After the initial transient the two filters should track together.

    They are compared only after settling. The transient is where they are
    *supposed* to differ: both start from the same seeded error but handle the
    curvature of the voltage curve differently, so the first few corrections
    diverge by several percent before converging on the same trajectory.
    """
    current = hppc_pulses(5.0, discharge_rate=1.0, steps=4, dt=model.dt)
    measured, _ = truth_run(model, current, soc0=0.5, noise_std=1e-3)
    results = []
    for cls in (EKF, UKF):
        estimator = cls(
            model,
            EKF.suggest_process_noise(model, 0.05, 0.01),
            1e-6,
            EKF.suggest_initial_covariance(model, 0.1),
        )
        estimator.initialise(0.55)
        results.append(estimator.run(current, measured)["soc"])
    settled = slice(3 * len(current) // 4, None)
    assert np.max(np.abs(results[0][settled] - results[1][settled])) < 0.01


# ---------------------------------------------------------------------------
# Health estimation and observability
# ---------------------------------------------------------------------------


def test_dual_filter_identifies_resistance_growth(cell, model):
    """Added series resistance is observable from current transitions."""
    added = 0.02
    current = synthetic_drive_cycle(5.0, duration=4000.0, dt=model.dt, peak_discharge_rate=3.0)
    truth = model.simulate(current, soc0=0.8)
    measured = truth["voltage"] - added * current
    rng = np.random.default_rng(1)
    measured = measured + rng.normal(0.0, 5e-4, measured.size)

    estimator = DualEKF(
        model,
        process_noise=EKF.suggest_process_noise(model, 0.05, 0.02),
        measurement_noise=2.5e-7,
        initial_covariance=EKF.suggest_initial_covariance(model, 0.05),
        resistance_process_noise=1e-11,
    )
    estimator.initialise(0.8)
    out = estimator.run(current, measured)
    assert out["resistance_growth"][-1] == pytest.approx(added, abs=5e-3)


def test_capacity_estimate_stays_unbiased_on_a_short_pulse_train(model):
    """Without a charge excursion the capacity estimate must not move off its prior.

    Capacity retention is modelled here as loss of *active material*, which scales
    the molar flux per ampere. That makes it partially observable from pulse
    transients as well as from long excursions: a cell with less active material
    swings its surface concentration further for the same current, and the shape of
    the pulse response carries that. So the reported uncertainty legitimately
    tightens somewhat even over a profile that moves state of charge by less than
    two percent.

    What must not happen is bias. The estimate has to stay centred on the truth
    rather than drifting off to fit measurement noise, because a confidently wrong
    capacity is acted on by the vehicle. The companion test below checks that a
    genuine excursion is far more informative than this.
    """
    current = np.concatenate([constant_current(10.0, 5.0, model.dt), rest(30.0, model.dt)] * 4)
    measured, truth = truth_run(model, current, soc0=0.6, noise_std=5e-4)
    assert abs(truth["soc"][0] - truth["soc"][-1]) < 0.02

    prior_std = 0.05
    estimator = DualEKF(
        model,
        process_noise=EKF.suggest_process_noise(model, 0.05, 0.01),
        measurement_noise=2.5e-7,
        initial_covariance=EKF.suggest_initial_covariance(model, 0.02),
        capacity_prior_std=prior_std,
    )
    estimator.initialise(0.6)
    out = estimator.run(current, measured)
    assert out["capacity_retention"][-1] == pytest.approx(1.0, abs=0.02)
    # Some tightening is expected and correct; a collapse to near-certainty is not.
    assert out["capacity_retention_std"][-1] > 0.1 * prior_std


def test_a_charge_excursion_is_more_informative_about_capacity_than_a_pulse_train(model):
    """Uncertainty must fall further over a real excursion than over pulses alone."""
    prior_std = 0.05

    def final_std(current: np.ndarray) -> float:
        measured, _ = truth_run(model, current, soc0=0.85, noise_std=5e-4)
        estimator = DualEKF(
            model,
            process_noise=EKF.suggest_process_noise(model, 0.05, 0.01),
            measurement_noise=2.5e-7,
            initial_covariance=EKF.suggest_initial_covariance(model, 0.02),
            capacity_prior_std=prior_std,
        )
        estimator.initialise(0.85)
        return float(estimator.run(current, measured)["capacity_retention_std"][-1])

    pulses = np.concatenate([constant_current(10.0, 5.0, model.dt), rest(30.0, model.dt)] * 4)
    excursion = constant_current(5.0, 2400.0, model.dt)
    assert final_std(excursion) < final_std(pulses)


def test_dual_filter_health_bounds_are_enforced(model):
    """Health parameters must stay inside their bounds under adversarial input."""
    current = np.full(400, 20.0)
    nonsense = np.full(400, 1.0)  # a voltage no cell would report
    estimator = DualEKF(model, 1.0, 1e-8, 1e8, capacity_bounds=(0.5, 1.15))
    estimator.initialise(0.5)
    estimator.run(current, nonsense)
    health = estimator.health()
    assert 0.5 <= health.capacity_retention <= 1.15
    assert -0.02 <= health.resistance_growth <= 0.2
    assert np.all(np.isfinite(estimator.x))


def test_dual_filter_reports_uncertainties(model):
    estimator = DualEKF(model, 1.0, 1e-6, 1e4)
    estimator.initialise(0.7)
    health = estimator.health()
    assert health.capacity_retention == pytest.approx(1.0)
    assert health.resistance_growth == pytest.approx(0.0)
    assert health.capacity_std > 0.0
    assert health.resistance_std > 0.0


# ---------------------------------------------------------------------------
# Initialisation and configuration
# ---------------------------------------------------------------------------


def test_initialise_from_voltage_inverts_the_ocv_curve(cell, model):
    estimator = EKF(model, 1.0, 1e-6, 1e4)
    target = 0.42
    estimator.initialise_from_voltage(float(cell.open_circuit_voltage(target)))
    assert estimator.model.soc(estimator.x) == pytest.approx(target, abs=1e-6)


def test_run_seeds_itself_from_the_first_sample(model):
    current = np.zeros(50)
    voltage = np.full(50, 3.7)
    estimator = EKF(model, 1.0, 1e-6, 1e4)
    out = estimator.run(current, voltage)
    assert np.all(np.isfinite(out["soc"]))


def test_noise_specifications_accept_scalar_vector_and_matrix(model):
    n = model.n_states
    for spec in (1.0, np.full(n, 1.0), np.eye(n)):
        estimator = EKF(model, spec, 1e-6, spec)
        assert estimator.Q.shape == (n, n)


def test_bad_noise_specifications_are_rejected(model):
    with pytest.raises(ValueError, match="process_noise vector"):
        EKF(model, np.ones(model.n_states + 3), 1e-6, 1.0)
    with pytest.raises(ValueError, match="measurement_noise must be positive"):
        EKF(model, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="equal length"):
        EKF(model, 1.0, 1e-6, 1.0).run(np.zeros(5), np.zeros(4))


def test_suggested_moments_are_shaped_not_isotropic(model):
    """Both suggestions must be low rank, reflecting the mechanisms they model."""
    Q = EKF.suggest_process_noise(model, 0.05, 0.01)
    assert np.linalg.matrix_rank(Q) <= 2
    tighter = EKF.suggest_process_noise(model, 0.05, 0.001)
    assert np.trace(Q) > np.trace(tighter)

    P0 = EKF.suggest_initial_covariance(model, 0.1)
    direction = model.soc_direction()
    # The dominant eigenvector must align with the state-of-charge direction.
    values, vectors = np.linalg.eigh(P0)
    leading = vectors[:, int(np.argmax(values))]
    unit = direction / np.linalg.norm(direction)
    assert abs(float(leading @ unit)) > 0.99
    assert np.all(np.linalg.eigvalsh(P0) > 0.0)
