"""Validation of the interphase-growth and lithium-plating degradation model."""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.degradation import (
    DegradationModel,
    DegradationState,
    PlatingParameters,
    SEIParameters,
)
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite

ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)
DAY = 24.0 * 3600.0


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)


@pytest.fixture(scope="module")
def model(cell):
    return SPM(cell, dt=1.0, rom="pade", order=3)


@pytest.fixture(scope="module")
def ageing(cell):
    return DegradationModel(cell)


def at_temperature(cell, temperature: float) -> SPM:
    return SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)


# ------------------------------------------------------------------ validation


def test_parameters_reject_nonsense():
    with pytest.raises(ValueError, match="reaction_rate"):
        SEIParameters(reaction_rate=0.0)
    with pytest.raises(ValueError, match="initial_thickness"):
        SEIParameters(initial_thickness=0.0)
    with pytest.raises(ValueError, match="transfer_coefficient"):
        PlatingParameters(transfer_coefficient=1.0)
    with pytest.raises(ValueError, match="stripping_efficiency"):
        PlatingParameters(stripping_efficiency=1.5)
    with pytest.raises(ValueError, match="safety_margin"):
        PlatingParameters(safety_margin=-0.01)


def test_step_rejects_nonpositive_dt(model, ageing):
    with pytest.raises(ValueError, match="dt"):
        ageing.step(model, model.initial_state(0.5), 0.0, ageing.initial_state(), 0.0)


def test_age_over_cycle_validates_its_window(model, ageing):
    with pytest.raises(ValueError, match="soc_low"):
        ageing.age_over_cycle(model, ageing.initial_state(), soc_low=0.9, soc_high=0.1)
    with pytest.raises(ValueError, match="samples"):
        ageing.age_over_cycle(model, ageing.initial_state(), samples=1)


# --------------------------------------------------------------------- plating


@pytest.mark.parametrize("temperature", [263.15, 298.15, 318.15])
def test_plating_current_vanishes_exactly_at_the_onset(ageing, temperature):
    """The thermodynamic condition, and the reason Butler-Volmer is used.

    A bare Tafel term has no reverse branch and predicts deposition at every
    potential, including at rest. Integrated over a month of storage that plates
    out an entire cell. The two branches must cancel exactly at zero.
    """
    assert ageing.plating_current_density(0.0, temperature) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize("temperature", [263.15, 298.15])
def test_plating_deposits_below_and_strips_above(ageing, temperature):
    assert ageing.plating_current_density(-0.05, temperature) > 0.0
    assert ageing.plating_current_density(+0.05, temperature) < 0.0


def test_plating_current_is_monotone_in_potential(ageing):
    values = [ageing.plating_current_density(phi, 298.15) for phi in (0.1, 0.05, 0.0, -0.05, -0.1)]
    assert values == sorted(values), "more negative potential must plate harder"


def test_plating_potential_falls_with_charge_rate(model, ageing, cell):
    """Faster charging pushes the electrode closer to lithium metal."""
    potentials = []
    for c_rate in (0.5, 1.0, 2.0, 3.0):
        current = -c_rate * cell.nominal_capacity
        x = model.initial_state(0.8)
        for _ in range(120):
            x = model.step(x, current)
        potentials.append(ageing.negative_potential(model, x, current))
    assert potentials == sorted(potentials, reverse=True)


def test_plating_potential_falls_with_state_of_charge(model, ageing, cell):
    """A fuller electrode has less room, so its potential sits lower."""
    current = -1.5 * cell.nominal_capacity
    potentials = []
    for soc in (0.3, 0.6, 0.8, 0.95):
        x = model.initial_state(soc)
        for _ in range(120):
            x = model.step(x, current)
        potentials.append(ageing.negative_potential(model, x, current))
    assert potentials == sorted(potentials, reverse=True)


def test_plating_is_worse_in_the_cold(cell, ageing):
    """The defining characteristic, and the reason fast charge is temperature gated.

    Requires temperature-dependent transport to come out right: with zero
    activation energies only the 2RT/F prefactor responds, and the trend inverts.
    """
    potentials = []
    for temperature in (263.15, 273.15, 283.15, 298.15, 313.15):
        m = at_temperature(cell, temperature)
        current = -2.0 * cell.nominal_capacity
        x = m.initial_state(0.8)
        for _ in range(120):
            x = m.step(x, current)
        potentials.append(ageing.negative_potential(m, x, current))
    assert potentials == sorted(potentials), "colder must mean lower potential"


def test_no_plating_on_discharge(model, ageing, cell):
    for c_rate in (0.5, 1.0, 2.0):
        current = c_rate * cell.nominal_capacity
        x = model.initial_state(0.5)
        for _ in range(120):
            x = model.step(x, current)
        assert ageing.negative_potential(model, x, current) > 0.0


def test_plating_margin_is_stricter_than_the_onset(ageing, model, cell):
    state = ageing.initial_state()
    current = -1.0 * cell.nominal_capacity
    x = model.initial_state(0.85)
    for _ in range(120):
        x = model.step(x, current)
    outputs = ageing.evaluate(model, x, current, state)
    assert outputs.plating_margin < outputs.plating_potential
    assert outputs.plating_margin == pytest.approx(
        outputs.plating_potential - ageing.plating.safety_margin
    )


# ------------------------------------------------------------ plating inventory


def test_stripping_cannot_recover_more_than_was_deposited(cell, ageing, model):
    """The bookkeeping bug this test exists to prevent.

    With a single running total, stripping decrements it and the *same* lithium
    becomes available to strip again on the next sample. Over a discharge leg
    sampled two dozen times that recovers an amount which should have been
    permanent, several times over. Plated and dead inventories are separate so
    metal that has lost contact leaves the strippable pool for good.
    """
    state = DegradationState(film_thickness=5e-9, lithium_plated=0.010)
    x = model.initial_state(0.5)
    # A long discharge: plenty of opportunity to strip.
    for _ in range(50):
        ageing.step(model, x, 2.0 * cell.nominal_capacity, state, 600.0)
    assert state.lithium_plated >= 0.0
    assert state.lithium_dead <= 0.010 + 1e-12
    recovered = 0.010 - state.lithium_dead
    assert recovered <= 0.010 * ageing.plating.stripping_efficiency + 1e-12


def test_dead_lithium_never_decreases(cell, ageing, model):
    state = ageing.initial_state()
    previous = 0.0
    for temperature in (263.15, 263.15, 298.15, 263.15):
        m = at_temperature(cell, temperature)
        ageing.age_over_cycle(m, state, c_rate=2.0, temperature=temperature)
        assert state.lithium_dead >= previous
        previous = state.lithium_dead


def test_fully_reversible_plating_leaves_nothing_dead(cell, model):
    ageing = DegradationModel(cell, plating=PlatingParameters(stripping_efficiency=1.0))
    state = DegradationState(film_thickness=5e-9, lithium_plated=0.005)
    x = model.initial_state(0.5)
    for _ in range(40):
        ageing.step(model, x, 2.0 * cell.nominal_capacity, state, 600.0)
    assert state.lithium_dead == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------------- interphase growth


def test_film_only_ever_grows(model, ageing):
    state = ageing.initial_state()
    x = model.initial_state(0.7)
    previous = state.film_thickness
    for _ in range(100):
        ageing.step(model, x, 0.0, state, DAY)
        assert state.film_thickness > previous
        previous = state.film_thickness


def test_growth_becomes_diffusion_limited(model, ageing):
    """Early on the reaction is kinetically limited; the film it builds takes over.

    That crossover is why capacity loss bends from linear to a square root, and
    reporting which branch is active distinguishes a cell being driven hard from
    one that is simply old.
    """
    state = ageing.initial_state()
    x = model.initial_state(0.9)
    first = ageing.evaluate(model, x, 0.0, state)
    for _ in range(2000):
        ageing.step(model, x, 0.0, state, DAY)
    later = ageing.evaluate(model, x, 0.0, state)
    assert later.sei_limited_by == "diffusion"
    assert later.sei_current_density < first.sei_current_density


def test_diffusion_limited_growth_follows_a_square_root(model, ageing):
    """The signature of a self-limiting film: thickness goes as sqrt(t).

    Measured between two late times, both well inside the diffusion-limited
    regime, so the early kinetic transient does not contaminate the exponent.
    """
    state = ageing.initial_state()
    x = model.initial_state(0.9)
    thickness = {}
    for day in range(1, 4001):
        ageing.step(model, x, 0.0, state, DAY)
        if day in (500, 4000):
            thickness[day] = state.film_thickness
    ratio = thickness[4000] / thickness[500]
    assert ratio == pytest.approx(np.sqrt(8.0), rel=0.1)


def test_calendar_ageing_is_worse_at_high_state_of_charge(model, ageing):
    """A lithiated electrode sits at a lower potential, which drives the reaction.

    This is why storage recommendations exist, and it only appears if the kinetic
    branch is not utterly swamped by diffusion -- a purely diffusion-limited
    model has no potential dependence at all.
    """
    losses = []
    for soc in (0.1, 0.5, 0.9):
        state = ageing.initial_state()
        x = model.initial_state(soc)
        for _ in range(365):
            ageing.step(model, x, 0.0, state, DAY)
        losses.append(state.lithium_lost)
    assert losses[0] < losses[1] < losses[2]
    assert losses[2] > 2.0 * losses[0]


def test_interphase_growth_accelerates_with_temperature(cell, ageing):
    losses = []
    for temperature in (283.15, 298.15, 318.15):
        m = at_temperature(cell, temperature)
        state = ageing.initial_state()
        x = m.initial_state(0.7, temperature)
        for _ in range(200):
            ageing.step(m, x, 0.0, state, DAY, temperature=temperature)
        losses.append(state.lithium_lost)
    assert losses[0] < losses[1] < losses[2]


def test_film_resistance_grows_with_the_film(ageing):
    thin = DegradationState(film_thickness=5e-9)
    thick = DegradationState(film_thickness=50e-9)
    assert ageing.film_resistance(thick) == pytest.approx(
        10.0 * ageing.film_resistance(thin), rel=1e-12
    )


def test_areal_and_total_film_resistance_differ_by_the_area(ageing):
    """Confusing the two costs a factor of the electrode area.

    The areal value multiplies an interfacial current *density*; the total value
    multiplies cell current. Using the total where the areal belongs makes the
    film's effect on the side reactions vanish, which is how the error hides.
    """
    state = DegradationState(film_thickness=20e-9)
    assert ageing.film_areal_resistance(state) == pytest.approx(
        ageing.film_resistance(state) * ageing.negative_area, rel=1e-12
    )
    assert ageing.negative_area > 1.0


# ------------------------------------------------------------------ accounting


def test_capacity_retention_accounts_for_every_inventory(ageing, cell):
    state = DegradationState(
        film_thickness=5e-9, lithium_lost=0.05, lithium_plated=0.02, lithium_dead=0.03
    )
    expected = 1.0 - (0.05 + 0.02 + 0.03) / cell.nominal_capacity
    assert ageing.capacity_retention(state) == pytest.approx(expected, rel=1e-12)


def test_capacity_retention_is_clamped_at_zero(ageing, cell):
    state = DegradationState(film_thickness=5e-9, lithium_lost=10.0 * cell.nominal_capacity)
    assert ageing.capacity_retention(state) == 0.0


def test_fresh_cell_is_undamaged(ageing, cell):
    state = ageing.initial_state()
    assert ageing.capacity_retention(state) == pytest.approx(1.0)
    assert state.lithium_lost == 0.0
    assert state.lithium_plated == 0.0
    assert state.lithium_dead == 0.0
    assert state.film_thickness == pytest.approx(ageing.sei.initial_thickness)


def test_throughput_counts_both_directions(model, ageing, cell):
    state = ageing.initial_state()
    x = model.initial_state(0.5)
    ageing.step(model, x, +cell.nominal_capacity, state, 3600.0)
    ageing.step(model, x, -cell.nominal_capacity, state, 3600.0)
    assert state.throughput == pytest.approx(2.0 * cell.nominal_capacity, rel=1e-12)


def test_state_copy_is_independent(ageing):
    state = ageing.initial_state()
    clone = state.copy()
    clone.lithium_lost = 1.0
    assert state.lithium_lost == 0.0


# ---------------------------------------------------------------- the U shape


def test_ageing_is_u_shaped_in_temperature(cell, ageing):
    """The result worth having: neither hot nor cold is safe, for different reasons.

    Interphase growth is Arrhenius and gets worse with heat. Plating is driven by
    sluggish transport and gets worse with cold. Their sum has a minimum
    somewhere in the middle, and where that minimum sits depends on how hard the
    cell is being charged -- which is exactly why a thermal management system has
    a target band rather than a ceiling.
    """
    retentions = {}
    for temperature in (263.15, 283.15, 298.15, 313.15, 328.15):
        m = at_temperature(cell, temperature)
        state = ageing.initial_state()
        for _ in range(200):
            ageing.age_over_cycle(m, state, c_rate=1.0, temperature=temperature)
        retentions[temperature] = ageing.capacity_retention(state)

    best = max(retentions, key=retentions.get)
    assert 273.0 < best < 323.0, f"optimum should be temperate, got {best - 273.15:.0f} C"
    assert retentions[263.15] < retentions[best]
    assert retentions[328.15] < retentions[best]


def test_the_cold_side_is_plating_and_the_hot_side_is_interphase(cell, ageing):
    """Not just that the curve is U-shaped, but that each arm has the right cause."""
    cold = ageing.initial_state()
    hot = ageing.initial_state()
    for temperature, state in ((263.15, cold), (328.15, hot)):
        m = at_temperature(cell, temperature)
        for _ in range(200):
            ageing.age_over_cycle(m, state, c_rate=1.0, temperature=temperature)
    assert cold.lithium_dead > cold.lithium_lost, "cold loss should be plating"
    assert hot.lithium_lost > hot.lithium_dead, "hot loss should be interphase growth"


def test_the_safe_temperature_rises_with_charge_rate(cell, ageing):
    """Faster charging pushes the plating arm further up the temperature axis."""

    def best_temperature(c_rate: float) -> float:
        scores = {}
        for temperature in (283.15, 298.15, 313.15, 328.15):
            m = at_temperature(cell, temperature)
            state = ageing.initial_state()
            for _ in range(150):
                ageing.age_over_cycle(m, state, c_rate=c_rate, temperature=temperature)
            scores[temperature] = ageing.capacity_retention(state)
        return max(scores, key=scores.get)

    assert best_temperature(2.0) > best_temperature(0.5)


def test_faster_cycling_ages_faster(cell, ageing):
    retentions = []
    for c_rate in (0.5, 1.0, 2.0):
        m = at_temperature(cell, 288.15)
        state = ageing.initial_state()
        for _ in range(200):
            ageing.age_over_cycle(m, state, c_rate=c_rate, temperature=288.15)
        retentions.append(ageing.capacity_retention(state))
    assert retentions[0] > retentions[1] > retentions[2]


def test_evaluate_does_not_mutate_the_state(model, ageing, cell):
    state = ageing.initial_state()
    before = state.copy()
    ageing.evaluate(model, model.initial_state(0.8), -cell.nominal_capacity, state)
    assert state.film_thickness == before.film_thickness
    assert state.lithium_lost == before.lithium_lost
    assert state.lithium_plated == before.lithium_plated
