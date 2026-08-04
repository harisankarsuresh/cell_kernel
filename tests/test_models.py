"""Validation of the cell models.

Assertions are anchored to physics that must hold regardless of implementation:
rest voltage equals open-circuit voltage, discharged charge equals capacity,
analytic Jacobians match finite differences, and the single particle model and
equivalent circuit converge on each other in the quasi-static limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.models import ECM, SPM
from cellkernel.params import chen2020_nmc811_graphite, lfp_graphite

ROM_KINDS = ["pade", "spectral", "fv", "poly"]


def assert_jacobian_close(analytic: np.ndarray, numeric: np.ndarray, tol: float = 1e-4) -> None:
    """Compare Jacobians with a tolerance scaled per input direction.

    Central differencing perturbs one state at a time, so the roundoff error in
    a given column is set by that column's step size and is roughly *uniform in
    absolute terms* across the column. Entries far smaller than the column's
    largest entry are therefore resolved only in relative terms much worse than
    the column as a whole -- a finite-volume state matrix legitimately spans
    ``1e-10`` to ``1`` within one column. Comparing entry by entry with a
    relative tolerance would demand precision the reference simply does not
    have, so each column is compared against its own scale.
    """
    analytic = np.atleast_2d(analytic)
    numeric = np.atleast_2d(numeric)
    assert analytic.shape == numeric.shape
    scale = np.maximum(np.abs(numeric).max(axis=0), 1e-300)
    worst = np.max(np.abs(analytic - numeric) / scale)
    assert worst < tol, f"worst column-scaled Jacobian error {worst:.3e} exceeds {tol:.1e}"


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite()


@pytest.fixture(scope="module")
def spm(cell):
    return SPM(cell, dt=1.0, rom="pade", order=4)


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ROM_KINDS)
@pytest.mark.parametrize("soc", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_rest_voltage_equals_open_circuit_voltage(cell, kind, soc):
    """At zero current the overpotentials vanish and V must equal OCV exactly."""
    model = SPM(cell, dt=1.0, rom=kind, order=4)
    x = model.initial_state(soc)
    assert model.voltage(x, 0.0) == pytest.approx(float(cell.open_circuit_voltage(soc)), abs=1e-9)


@pytest.mark.parametrize("kind", ROM_KINDS)
def test_rested_state_is_stationary(cell, kind):
    """A rested cell at open circuit must not drift.

    Five thousand seconds of open circuit, so any per-step round-off in the
    discrete update accumulates 500 times over. The finite-volume model gets a
    looser voltage bound for the reason given in
    ``test_uniform_state_is_an_equilibrium``: its uniform state is preserved only
    to the accuracy of the exponential of a stiff generator, which depends on the
    linear-algebra backend. Even so, 1 microvolt over eighty minutes is four
    orders of magnitude below the noise floor of any measurement front end.
    """
    model = SPM(cell, dt=10.0, rom=kind, order=4)
    x = model.initial_state(0.5)
    v0 = model.voltage(x, 0.0)
    for _ in range(500):
        x = model.step(x, 0.0)
    tol = 1e-6 if kind == "fv" else 1e-9
    assert model.voltage(x, 0.0) == pytest.approx(v0, abs=tol)
    assert model.soc(x) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("soc", [0.1, 0.5, 0.9])
def test_reported_soc_matches_requested(spm, soc):
    assert spm.soc(spm.initial_state(soc)) == pytest.approx(soc, abs=1e-9)


# ---------------------------------------------------------------------------
# Capacity and directionality
# ---------------------------------------------------------------------------


def test_discharge_lowers_voltage_and_charge_raises_it(spm):
    x = spm.initial_state(0.5)
    v_rest = spm.voltage(x, 0.0)
    assert spm.voltage(x, 5.0) < v_rest
    assert spm.voltage(x, -5.0) > v_rest


def test_overpotential_signs_oppose_current(spm):
    """On discharge the negative overpotential is positive and vice versa."""
    x = spm.initial_state(0.5)
    eta_n, eta_p = spm.outputs(x, 5.0).overpotential
    assert eta_n > 0.0 and eta_p < 0.0
    eta_n, eta_p = spm.outputs(x, -5.0).overpotential
    assert eta_n < 0.0 and eta_p > 0.0


def test_soc_tracks_coulomb_count(cell, spm):
    """Reported state of charge must follow integrated current exactly.

    The bulk concentration is driven by the structurally exact mass balance, so
    this is a strong statement: it holds to numerical precision, not to within a
    modelling tolerance.
    """
    current = 2.5
    steps = 1800
    result = spm.simulate(np.full(steps, current), soc0=0.9)
    expected = 0.9 - current * np.arange(steps) * spm.dt / (3600.0 * cell.nominal_capacity)
    assert np.allclose(result["soc"], expected, atol=1e-9)


@pytest.mark.parametrize("c_rate", [0.2, 0.5, 1.0])
def test_full_discharge_delivers_about_nominal_capacity(cell, c_rate):
    """Discharging from 100% to the lower cut-off must deliver close to nominal.

    Delivered charge falls short of nominal as the rate rises, because
    overpotential and surface depletion bring the terminal voltage to the cut-off
    before the bulk is exhausted. That gap is the physics the model exists to
    capture, so the test asserts both that the capacity is sensible and that it
    decreases monotonically with rate.
    """
    model = SPM(cell, dt=1.0, rom="pade", order=4)
    current = c_rate * cell.nominal_capacity
    result = model.simulate(np.full(20000, current), soc0=1.0)
    below = result["voltage"] <= cell.voltage_limits[0]
    assert below.any(), "cell never reached the lower cut-off"
    steps = int(np.argmax(below))
    delivered = current * steps * model.dt / 3600.0
    assert 0.80 * cell.nominal_capacity < delivered <= 1.001 * cell.nominal_capacity


def test_delivered_capacity_decreases_with_rate(cell):
    model = SPM(cell, dt=1.0, rom="pade", order=4)
    delivered = []
    for c_rate in (0.2, 0.5, 1.0, 2.0):
        current = c_rate * cell.nominal_capacity
        result = model.simulate(np.full(20000, current), soc0=1.0)
        below = result["voltage"] <= cell.voltage_limits[0]
        steps = int(np.argmax(below))
        delivered.append(current * steps * model.dt / 3600.0)
    for a, b in zip(delivered, delivered[1:], strict=False):
        assert b < a


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ROM_KINDS)
@pytest.mark.parametrize("current", [0.0, 1.0, -1.0, 10.0, -10.0])
@pytest.mark.parametrize("soc", [0.15, 0.5, 0.85])
def test_spm_voltage_jacobian_matches_finite_difference(cell, kind, current, soc):
    model = SPM(cell, dt=1.0, rom=kind, order=4)
    x = model.initial_state(soc)
    analytic = model.voltage_jacobian(x, current)
    numeric = model.numerical_voltage_jacobian(x, current)
    scale = max(np.max(np.abs(numeric)), 1e-300)
    assert np.max(np.abs(analytic - numeric)) / scale < 1e-4


@pytest.mark.parametrize("kind", ROM_KINDS)
def test_spm_state_jacobian_is_exact(cell, kind):
    """Linear dynamics mean the state Jacobian is the state matrix itself."""
    model = SPM(cell, dt=1.0, rom=kind, order=4)
    x = model.initial_state(0.6)
    assert_jacobian_close(model.state_jacobian(x, 3.0), model.numerical_state_jacobian(x, 3.0))


def test_state_jacobian_does_not_depend_on_operating_point(spm):
    a = spm.state_jacobian(spm.initial_state(0.2), -8.0)
    b = spm.state_jacobian(spm.initial_state(0.9), 8.0)
    assert np.array_equal(a, b)


def test_soc_jacobian_matches_finite_difference(spm):
    x = spm.initial_state(0.5)
    analytic = spm.soc_jacobian()
    numeric = np.zeros(spm.n_states)
    for i in range(spm.n_states):
        h = 1e-4 * max(abs(x[i]), 1.0)
        hi, lo = x.copy(), x.copy()
        hi[i] += h
        lo[i] -= h
        numeric[i] = (spm.soc(hi) - spm.soc(lo)) / (2.0 * h)
    assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-12)


def test_ecm_jacobians(cell):
    model = ECM(cell, dt=1.0)
    x = model.initial_state(0.55)
    x = model.step(x, 4.0)
    assert_jacobian_close(model.voltage_jacobian(x, 4.0), model.numerical_voltage_jacobian(x, 4.0))
    assert_jacobian_close(model.state_jacobian(x, 4.0), model.numerical_state_jacobian(x, 4.0))


# ---------------------------------------------------------------------------
# Equivalent circuit behaviour
# ---------------------------------------------------------------------------


def test_ecm_rc_branch_is_exactly_discretised(cell):
    """A single RC branch must relax as exp(-t/tau), not as an Euler approximation."""
    resistance, tau = 0.02, 50.0
    model = ECM(cell, dt=5.0, series_resistance=0.0, rc_pairs=((resistance, tau),))
    x = model.initial_state(0.5)
    current = 3.0
    for _ in range(4000):  # reach the steady branch voltage
        x = model.step(x, current)
    assert x[1] == pytest.approx(resistance * current, rel=1e-9)
    # Then relax at open circuit and check the decay constant.
    v0 = x[1]
    for _ in range(10):
        x = model.step(x, 0.0)
    assert x[1] == pytest.approx(v0 * np.exp(-10 * model.dt / tau), rel=1e-9)


def test_ecm_coulomb_counts(cell):
    model = ECM(cell, dt=1.0)
    result = model.simulate(np.full(600, 5.0), soc0=1.0)
    expected = 1.0 - 5.0 * np.arange(600) / (3600.0 * cell.nominal_capacity)
    assert np.allclose(result["soc"], expected, atol=1e-12)


def test_ecm_rejects_bad_time_constants(cell):
    with pytest.raises(ValueError, match="time constants"):
        ECM(cell, dt=1.0, rc_pairs=((0.01, 0.0),))


def test_spm_and_ecm_agree_at_low_rate(cell):
    """At C/20 both models should be within a few tens of millivolts.

    Not an identity: the equivalent circuit has no concentration gradient, so it
    cannot reproduce surface depletion. But at a low enough rate that effect is
    small and the two must not disagree wildly, which catches sign errors and
    unit slips in either one.
    """
    current = cell.nominal_capacity / 20.0
    steps = 4000
    spm = SPM(cell, dt=10.0, rom="pade", order=5)
    ecm = ECM(cell, dt=10.0, series_resistance=0.02, rc_pairs=((0.005, 60.0),))
    v_spm = spm.simulate(np.full(steps, current), soc0=0.95)["voltage"]
    v_ecm = ecm.simulate(np.full(steps, current), soc0=0.95)["voltage"]
    assert np.max(np.abs(v_spm - v_ecm)) < 0.05


# ---------------------------------------------------------------------------
# Surface depletion: the effect an equivalent circuit cannot represent
# ---------------------------------------------------------------------------


def test_surface_depletes_faster_than_bulk_under_load(spm):
    """Under discharge the negative surface stoichiometry must fall below bulk."""
    x = spm.initial_state(0.7)
    for _ in range(120):
        x = spm.step(x, 15.0)
    out = spm.outputs(x, 15.0)
    neg = spm.parameters.negative
    x_bulk = neg.stoichiometry(out.soc)
    assert out.surface_stoichiometry[0] < x_bulk


def test_voltage_recovers_after_load_removal(spm):
    """Relaxation must be a real dynamic effect, not an instantaneous drop."""
    x = spm.initial_state(0.6)
    for _ in range(300):
        x = spm.step(x, 20.0)
    v_loaded = spm.voltage(x, 20.0)
    v_immediate = spm.voltage(x, 0.0)
    for _ in range(1800):
        x = spm.step(x, 0.0)
    v_relaxed = spm.voltage(x, 0.0)
    assert v_loaded < v_immediate < v_relaxed


# ---------------------------------------------------------------------------
# Temperature and robustness
# ---------------------------------------------------------------------------


def test_at_temperature_rebuilds_consistently(cell, spm):
    cold = spm.at_temperature(263.15)
    assert cold.temperature == pytest.approx(263.15)
    assert cold.n_states == spm.n_states
    # With zero activation energies only the 2RT/F kinetic prefactor changes, so
    # a rested cell is unaffected while a loaded one sees proportionally smaller
    # overpotentials at lower temperature.
    x = cold.initial_state(0.5)
    warm_x = spm.initial_state(0.5)
    assert cold.voltage(x, 0.0) == pytest.approx(spm.voltage(warm_x, 0.0), abs=1e-9)

    cold_eta = cold.outputs(x, 10.0).overpotential
    warm_eta = spm.outputs(warm_x, 10.0).overpotential
    ratio = 263.15 / 298.15
    for c, w in zip(cold_eta, warm_eta, strict=False):
        assert abs(w) > 1e-3, "overpotential is implausibly small; check i0 units"
        assert c == pytest.approx(w * ratio, rel=1e-9)


def test_exchange_current_density_is_physically_plausible(spm):
    """Guard the i0 units directly.

    Reported exchange current densities for graphite and layered-oxide electrodes
    sit in the range of roughly 0.1 to 30 A m-2. An extra Faraday constant in the
    expression would put it near 1e4, which produces a model with essentially no
    charge-transfer resistance.
    """
    x = spm.initial_state(0.5)
    cs_n, _, cs_p, _ = spm._concentrations(x, 0.0)
    for c, electrode in ((cs_n, "negative"), (cs_p, "positive")):
        i0 = spm._exchange_current(c, electrode)
        assert 0.01 < i0 < 100.0, f"{electrode} i0 = {i0:.4g} A/m2 is out of range"


def test_overpotential_grows_with_rate(spm):
    """Kinetic overpotential must be a meaningful fraction of the voltage drop."""
    x = spm.initial_state(0.5)
    magnitudes = []
    for c_rate in (0.2, 1.0, 3.0):
        eta_n, eta_p = spm.outputs(x, c_rate * 5.0).overpotential
        magnitudes.append(abs(eta_n) + abs(eta_p))
    for a, b in zip(magnitudes, magnitudes[1:], strict=False):
        assert b > a
    assert magnitudes[-1] > 0.05  # tens of millivolts at 3C, not microvolts


def test_extreme_state_does_not_produce_nan(cell):
    """A filter can transiently push concentration out of range; that must not poison it."""
    model = SPM(cell, dt=1.0, rom="poly")
    for soc in (-0.5, 0.0, 1.0, 1.5):
        x = model.initial_state(soc)
        for current in (0.0, 200.0, -200.0):
            assert np.isfinite(model.voltage(x, current))
            assert np.all(np.isfinite(model.voltage_jacobian(x, current)))


def test_lfp_cell_runs_and_is_less_observable(cell):
    """The LFP set must simulate, and its OCV slope must be far smaller than NMC."""
    lfp = lfp_graphite()
    model = SPM(lfp, dt=1.0, rom="pade", order=4)
    result = model.simulate(np.full(3600, lfp.nominal_capacity / 2.0), soc0=0.9)
    assert np.all(np.isfinite(result["voltage"]))
    mid = np.linspace(0.3, 0.9, 25)
    assert np.median(np.abs(lfp.ocv_derivative(mid))) < 0.2 * np.median(
        np.abs(cell.ocv_derivative(mid))
    )


def test_rejects_non_positive_dt(cell):
    with pytest.raises(ValueError, match="dt must be positive"):
        SPM(cell, dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        ECM(cell, dt=-1.0)
