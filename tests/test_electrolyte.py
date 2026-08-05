"""Validation of salt transport and the single particle model with electrolyte."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cellkernel.models import SPM, SPMe
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.rom import ElectrolyteDiffusion

# The shipped contact resistance is a lumped fit that already contains the
# electrolyte, so it is reduced here by the modelled electrolyte resistance.
# Leaving it alone would double-count and is the most likely mistake anyone
# makes moving a parameter set from SPM to SPMe.
ELECTROLYTE_RESISTANCE = 4.043e-3


@pytest.fixture(scope="module")
def base():
    return chen2020_nmc811_graphite()


@pytest.fixture(scope="module")
def cell(base):
    return replace(base, contact_resistance=base.contact_resistance - ELECTROLYTE_RESISTANCE)


@pytest.fixture(scope="module")
def model(cell):
    return SPMe(cell, dt=1.0, rom="pade", order=3)


def build_electrolyte(cell, cells=(4, 3, 4)) -> ElectrolyteDiffusion:
    return ElectrolyteDiffusion(
        thickness_negative=cell.negative.thickness,
        thickness_separator=cell.separator_thickness,
        thickness_positive=cell.positive.thickness,
        porosity_negative=cell.negative.porosity,
        porosity_separator=cell.separator_porosity,
        porosity_positive=cell.positive.porosity,
        diffusivity=cell.electrolyte_diffusivity,
        transference_number=cell.transference_number,
        electrode_area=cell.electrode_area,
        bruggeman=cell.bruggeman,
        cells_negative=cells[0],
        cells_separator=cells[1],
        cells_positive=cells[2],
    )


# ------------------------------------------------------------------ validation


def test_rejects_unphysical_parameters(cell):
    with pytest.raises(ValueError, match="porosity"):
        build_electrolyte(replace(cell, separator_porosity=0.0))
    with pytest.raises(ValueError, match="transference"):
        build_electrolyte(replace(cell, transference_number=1.0))
    with pytest.raises(ValueError, match="cells_negative"):
        build_electrolyte(cell, cells=(0, 3, 4))


# ---------------------------------------------------------------- conservation


def test_total_salt_is_conserved_exactly(cell):
    """No current, no chemistry and no boundary can create salt.

    The source integrates to zero over the sandwich by construction, so total
    salt is a conserved quantity of the continuous system, and the discretisation
    imposes it rather than inheriting it from the accuracy of a matrix
    exponential.
    """
    electrolyte = build_electrolyte(cell)
    space = electrolyte.discretise(1.0)
    widths = electrolyte.widths
    porosity = np.array([electrolyte.porosity[k] for k in electrolyte.region_index])
    weights = porosity * widths
    weights = weights / weights.sum()

    rng = np.random.default_rng(0)
    x = space.initial_state(1000.0)
    for _ in range(2000):
        x = space.step(x, float(rng.normal(0.0, 15.0)))
        assert float(weights @ x) == pytest.approx(1000.0, abs=1e-7)


def test_uniform_field_is_a_fixed_point(cell):
    electrolyte = build_electrolyte(cell)
    space = electrolyte.discretise(1.0)
    x = space.initial_state(1000.0)
    for _ in range(500):
        x = space.step(x, 0.0)
        assert np.allclose(x, 1000.0, rtol=1e-12)


@pytest.mark.parametrize("dt_ratio", [1e-3, 1.0, 10.0])
def test_conservation_holds_at_any_step(cell, dt_ratio):
    electrolyte = build_electrolyte(cell)
    space = electrolyte.discretise(dt_ratio * electrolyte.time_constant)
    widths = electrolyte.widths
    porosity = np.array([electrolyte.porosity[k] for k in electrolyte.region_index])
    weights = porosity * widths
    weights = weights / weights.sum()
    u = np.ones(space.n_states)
    b = space.B.reshape(-1)
    assert np.allclose(space.A @ u, u, atol=1e-12)
    assert np.allclose(weights @ space.A, weights, rtol=1e-11)
    # Relative to the size of B, not absolute: the source entries are of order
    # 1e-2 to 1e0 depending on step, so an absolute bound would be testing the
    # magnitude of B rather than the cancellation.
    assert abs(float(weights @ b)) <= 1e-12 * max(float(np.max(np.abs(b))), 1.0)


# ------------------------------------------------------------------ steady state


@pytest.mark.parametrize("c_rate", [0.5, 1.0, 2.0, 3.0])
def test_simulated_steady_state_matches_the_analytic_solve(cell, c_rate):
    """Two independent routes to the same answer.

    The analytic route solves the singular continuous system with the mean
    pinned; the simulated route iterates the discretisation to convergence.
    Nothing is shared between them beyond the state matrices, so agreement is
    evidence that both the discretisation and the steady-state solve are right.
    """
    electrolyte = build_electrolyte(cell)
    space = electrolyte.discretise(1.0)
    current = cell.nominal_capacity * c_rate

    x = space.initial_state(1000.0)
    for _ in range(20000):
        x = space.step(x, current)
    simulated_n, simulated_p = space.averages(x)
    analytic_n, analytic_p = electrolyte.steady_state_split(current)

    assert simulated_n - 1000.0 == pytest.approx(analytic_n, abs=1e-8)
    assert simulated_p - 1000.0 == pytest.approx(analytic_p, abs=1e-8)


def test_depletion_is_linear_in_current(cell):
    """The transport model is linear, so doubling the current doubles the split."""
    electrolyte = build_electrolyte(cell)
    single = electrolyte.steady_state_split(cell.nominal_capacity)
    double = electrolyte.steady_state_split(2.0 * cell.nominal_capacity)
    assert double[0] == pytest.approx(2.0 * single[0], rel=1e-12)
    assert double[1] == pytest.approx(2.0 * single[1], rel=1e-12)


def test_discharge_enriches_the_negative_and_depletes_the_positive(cell):
    """Sign check, and it is the one most easily got backwards.

    On discharge lithium leaves the negative particles into the electrolyte and
    is consumed at the positive, so salt piles up on the negative side.
    """
    electrolyte = build_electrolyte(cell)
    negative, positive = electrolyte.steady_state_split(cell.nominal_capacity)
    assert negative > 0.0
    assert positive < 0.0
    charge_n, charge_p = electrolyte.steady_state_split(-cell.nominal_capacity)
    assert charge_n == pytest.approx(-negative, rel=1e-12)
    assert charge_p == pytest.approx(-positive, rel=1e-12)


def test_steady_state_split_converges_with_grid_refinement(cell):
    values = [
        build_electrolyte(cell, cells=c).steady_state_split(2.0 * cell.nominal_capacity)[0]
        for c in ((2, 1, 2), (4, 3, 4), (8, 6, 8), (16, 12, 16))
    ]
    errors = [abs(v - values[-1]) for v in values[:-1]]
    assert errors[0] > errors[1] > errors[2]


def test_thinner_separator_gives_a_smaller_split(cell):
    """A physical monotonicity: less distance to diffuse, less gradient."""
    thick = build_electrolyte(replace(cell, separator_thickness=40e-6))
    thin = build_electrolyte(replace(cell, separator_thickness=8e-6))
    current = 2.0 * cell.nominal_capacity
    assert thin.steady_state_split(current)[0] < thick.steady_state_split(current)[0]


def test_faster_salt_diffusion_gives_a_smaller_split(cell):
    slow = build_electrolyte(replace(cell, electrolyte_diffusivity=2.0e-10))
    fast = build_electrolyte(replace(cell, electrolyte_diffusivity=2.0e-9))
    current = 2.0 * cell.nominal_capacity
    assert fast.steady_state_split(current)[0] < slow.steady_state_split(current)[0]


# ---------------------------------------------------------------------- ohmic


def test_ohmic_resistance_matches_the_closed_form(cell):
    electrolyte = build_electrolyte(cell)
    kappa = cell.ionic_conductivity
    expected = (
        cell.negative.thickness / (3.0 * kappa * cell.negative.porosity**cell.bruggeman)
        + cell.separator_thickness / (kappa * cell.separator_porosity**cell.bruggeman)
        + cell.positive.thickness / (3.0 * kappa * cell.positive.porosity**cell.bruggeman)
    ) / cell.electrode_area
    assert electrolyte.ohmic_resistance(kappa) == pytest.approx(expected, rel=1e-12)


def test_separator_carries_no_factor_of_three(cell):
    """The thirds are physics, not symmetry.

    Ionic current in a coating falls linearly from the separator face to the
    current collector as it is handed to the solid phase, which is where the
    factor of three comes from. The separator carries the full current
    everywhere, so tripling its thickness must cost three times as much
    resistance as tripling it would if it were treated like a coating.
    """
    thin = build_electrolyte(replace(cell, separator_thickness=10e-6))
    thick = build_electrolyte(replace(cell, separator_thickness=40e-6))
    kappa = cell.ionic_conductivity
    delta = thick.ohmic_resistance(kappa) - thin.ohmic_resistance(kappa)
    expected = 30e-6 / (kappa * cell.separator_porosity**cell.bruggeman) / cell.electrode_area
    assert delta == pytest.approx(expected, rel=1e-12)


def test_ohmic_resistance_rejects_nonpositive_conductivity(cell):
    with pytest.raises(ValueError, match="conductivity"):
        build_electrolyte(cell).ohmic_resistance(0.0)


# ----------------------------------------------------------------------- SPMe


def test_rest_voltage_equals_open_circuit(model, cell):
    for soc in (0.15, 0.5, 0.85):
        x = model.initial_state(soc)
        assert model.voltage(x, 0.0) == pytest.approx(
            float(cell.open_circuit_voltage(soc)), abs=1e-9
        )


def test_no_concentration_overpotential_at_rest(model):
    x = model.initial_state(0.6)
    assert model.decompose(x, 0.0)["concentration_overpotential"] == pytest.approx(0.0, abs=1e-15)


def test_concentration_overpotential_opposes_discharge(model, cell):
    """Salt piles up on the negative side, so the log ratio is negative."""
    current = 2.0 * cell.nominal_capacity
    x = model.initial_state(0.6)
    for _ in range(200):
        x = model.step(x, current)
    assert model.decompose(x, current)["concentration_overpotential"] < 0.0


def test_concentration_overpotential_is_nearly_antisymmetric_in_current(model, cell):
    """Nearly, not exactly, and the gap is physical rather than numerical.

    Salt transport is linear, so the concentration *deviations* under equal and
    opposite currents are exact mirrors. The overpotential is not, because it
    goes as the logarithm of a ratio and the two coatings hold different amounts
    of electrolyte -- 85 um at 25% porosity against 76 um at 33.5%. The negative
    side therefore swings by a different amount than the positive, and
    ``ln((c-a)/(c+b))`` is only antisymmetric when ``a == b``. The residual is a
    few percent, and a test asserting exact antisymmetry would be asserting
    something false.
    """
    forward = model.initial_state(0.5)
    backward = model.initial_state(0.5)
    current = cell.nominal_capacity
    for _ in range(120):
        forward = model.step(forward, current)
        backward = model.step(backward, -current)
    a = model.decompose(forward, current)["concentration_overpotential"]
    b = model.decompose(backward, -current)["concentration_overpotential"]
    assert a < 0.0 < b
    assert a == pytest.approx(-b, rel=0.06)

    # The underlying concentration deviations, however, are exact mirrors.
    fn, fp = model.electrolyte_concentrations(forward)
    bn, bp = model.electrolyte_concentrations(backward)
    nominal = cell.electrolyte_concentration
    assert fn - nominal == pytest.approx(-(bn - nominal), rel=1e-10)
    assert fp - nominal == pytest.approx(-(bp - nominal), rel=1e-10)


def test_electrolyte_term_takes_time_to_build(model, cell):
    """The reason a fitted resistance cannot stand in for it.

    An ohmic term appears instantly; this one grows over tens of seconds, set by
    the sandwich thickness rather than the particle. A constant resistance fitted
    to the settled value is wrong for the first minute of every transient.
    """
    current = 2.0 * cell.nominal_capacity
    x = model.initial_state(0.7)
    magnitudes = []
    for step in range(241):
        if step in (1, 10, 40, 120, 240):
            magnitudes.append(abs(model.decompose(x, current)["concentration_overpotential"]))
        x = model.step(x, current)
    assert magnitudes == sorted(magnitudes), "should grow monotonically"
    assert magnitudes[0] < 0.2 * magnitudes[-1], "should start far below its settled value"
    assert magnitudes[-1] == pytest.approx(magnitudes[-2], rel=0.05), "should settle"


def test_state_jacobian_is_exact_and_constant(model, cell):
    """Every transport process here is linear, so the prediction step is exact."""
    for c_rate in (0.0, 1.0, -2.0):
        current = cell.nominal_capacity * c_rate
        x = model.initial_state(0.5)
        for _ in range(100):
            x = model.step(x, current)
        analytic = model.state_jacobian(x, current)
        numeric = model.numerical_state_jacobian(x, current)
        assert np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric) < 1e-7
    # And constant: independent of both state and current.
    a = model.state_jacobian(model.initial_state(0.2), 0.0)
    b = model.state_jacobian(model.initial_state(0.9), 12.0)
    assert np.allclose(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("soc", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("c_rate", [0.0, 1.0, -1.0, 3.0])
def test_voltage_jacobian_matches_central_differences(model, cell, soc, c_rate):
    current = cell.nominal_capacity * c_rate
    x = model.initial_state(soc)
    for _ in range(200):
        x = model.step(x, current)
    analytic = model.voltage_jacobian(x, current)
    numeric = model.numerical_voltage_jacobian(x, current)
    assert np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric) < 1e-6


def test_electrolyte_states_affect_voltage(model, cell):
    """Guards against the electrolyte block being computed and then ignored."""
    current = 2.0 * cell.nominal_capacity
    x = model.initial_state(0.6)
    for _ in range(150):
        x = model.step(x, current)
    gradient = model.voltage_jacobian(x, current)
    assert np.any(np.abs(gradient[model._i_elec :]) > 0.0)


def test_matches_spm_at_low_rate_once_resistance_is_reconciled(base, cell, model):
    """At 0.2C the electrolyte really is just a resistance, and the two agree."""
    spm = SPM(base, dt=1.0, rom="pade", order=3)
    current = 0.2 * cell.nominal_capacity
    xe, xs = model.initial_state(0.7), spm.initial_state(0.7)
    worst = 0.0
    for _ in range(600):
        worst = max(worst, abs(model.voltage(xe, current) - spm.voltage(xs, current)))
        xe, xs = model.step(xe, current), spm.step(xs, current)
    assert worst < 5e-3, f"low-rate disagreement {1e3 * worst:.2f} mV"


def test_diverges_from_spm_at_high_rate(base, cell, model):
    """And at 3C it does not, which is the whole point of the model."""
    spm = SPM(base, dt=1.0, rom="pade", order=3)
    current = 3.0 * cell.nominal_capacity
    xe, xs = model.initial_state(0.7), spm.initial_state(0.7)
    for _ in range(300):
        xe, xs = model.step(xe, current), spm.step(xs, current)
    assert abs(model.voltage(xe, current) - spm.voltage(xs, current)) > 0.02


def test_validity_degrades_as_the_electrolyte_empties(model, cell):
    """The model must say when it has left the range it can represent."""
    x = model.initial_state(0.8)
    assert model.validity(x) == "good"
    assert model.depletion(x) == pytest.approx(1.0, rel=1e-9)

    for _ in range(600):
        x = model.step(x, 6.0 * cell.nominal_capacity)
    assert model.depletion(x) < 0.6
    assert model.validity(x) in {"degraded", "extrapolating"}


def test_voltage_stays_finite_under_severe_depletion(model, cell):
    """Past the validity limit the answer is wrong, but it must not be NaN.

    A linear transport model will happily predict negative salt concentration.
    The square roots and the logarithm are floored so that an estimator meeting
    this condition degrades instead of poisoning its covariance with NaN.
    """
    x = model.initial_state(0.9)
    for _ in range(2000):
        x = model.step(x, 12.0 * cell.nominal_capacity)
        assert np.all(np.isfinite(x))
    assert np.isfinite(model.voltage(x, 12.0 * cell.nominal_capacity))
    assert model.validity(x) == "extrapolating"


def test_state_layout_and_naming(model):
    names = model.state_names
    assert len(names) == model.n_states
    assert sum(1 for n in names if n.startswith("elyte_")) == model._n_elec
    assert names[model._i_elec].startswith("elyte_")


def test_series_resistance_adds_the_electrolyte(model, cell):
    assert model.series_resistance == pytest.approx(
        model.electrolyte_resistance + cell.contact_resistance, rel=1e-12
    )
    assert model.electrolyte_resistance > 0.0
