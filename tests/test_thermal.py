"""Validation of the coupled electro-thermal model and its temperature schedule."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cellkernel.data import synthetic_drive_cycle
from cellkernel.models import SPM, ThermalSPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.rom import ScheduledStateSpace, make_rom, schedule_over_temperature

# Activation energies are not shipped with the parameter sets on purpose, so the
# thermal tests supply representative ones. Without them only the 2RT/F kinetic
# prefactor responds to temperature and the schedule has nothing to interpolate.
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
def model(cell):
    return ThermalSPM(cell, dt=1.0, rom="pade", order=3)


def conductance(cell) -> float:
    return cell.thermal.heat_transfer_coefficient * cell.thermal.surface_area


# ---------------------------------------------------------------- construction


def test_requires_thermal_parameters(cell):
    without = replace(cell, thermal=None)
    with pytest.raises(ValueError, match="thermal"):
        ThermalSPM(without, dt=1.0)


def test_state_layout(model):
    assert model.n_states == model.temperature_index + 1
    assert model.state_names[-1] == "temperature"
    assert model.state_names[model.temperature_index] == "temperature"


def test_initial_state_carries_requested_temperature(model):
    for temperature in (253.15, 280.0, 298.15, 330.0):
        z = model.initial_state(0.5, temperature)
        assert model.temperature(z) == pytest.approx(temperature)
        assert model.outputs(z, 0.0).temperature == pytest.approx(temperature)


def test_rested_voltage_matches_open_circuit(model, cell):
    """At zero current the overpotentials vanish at any temperature."""
    for temperature in (263.15, 298.15, 323.15):
        for soc in (0.2, 0.5, 0.8):
            z = model.initial_state(soc, temperature)
            expected = float(cell.open_circuit_voltage(soc))
            assert model.voltage(z, 0.0) == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------- thermal physics


def test_temperature_relaxes_to_ambient_at_rest(model):
    """No current means no heat, so the cell must decay to ambient.

    Twenty thousand seconds is sixteen thermal time constants, which leaves
    ``25 * exp(-16)`` -- about 3 microkelvin -- of the initial offset. The bound
    is set from that rather than from machine precision, because the residual
    here is the physics of an exponential, not round-off.
    """
    z = model.initial_state(0.5, model.ambient + 25.0)
    for _ in range(20000):
        z = model.step(z, 0.0)
    assert model.temperature(z) == pytest.approx(model.ambient, abs=1e-4)
    assert model.temperature(z) > model.ambient, "decay must approach from above"


def test_thermal_relaxation_follows_the_exact_exponential(model):
    """The unforced node must decay as exp(-t/tau), not approximately."""
    tau = model.parameters.thermal.time_constant
    start = model.ambient + 20.0
    z = model.initial_state(0.5, start)
    for step in range(1, 400):
        z = model.step(z, 0.0)
        expected = model.ambient + 20.0 * np.exp(-step * model.dt / tau)
        assert model.temperature(z) == pytest.approx(expected, rel=1e-10)


def test_exact_integration_beats_forward_euler_at_a_long_step(cell):
    """The thermal update is exact for any step; forward Euler would not be.

    At dt = 3 tau an explicit update overshoots into oscillation. This asserts the
    exact solution is used by checking a single long step against the closed form.
    """
    tau = cell.thermal.time_constant
    long_step = 3.0 * tau
    model = ThermalSPM(cell, dt=long_step, rom="pade", order=3)
    z = model.initial_state(0.5, model.ambient + 30.0)
    z = model.step(z, 0.0)
    expected = model.ambient + 30.0 * np.exp(-1.0 * 3.0)
    assert model.temperature(z) == pytest.approx(expected, rel=1e-12)
    # Forward Euler at this step would give ambient + 30*(1 - 3) = ambient - 60.
    assert model.temperature(z) > model.ambient


@pytest.mark.parametrize("c_rate", [0.3, 0.5, 1.0])
def test_steady_state_satisfies_the_energy_balance(cell, c_rate):
    """At thermal equilibrium, generation must equal the loss to ambient."""
    model = ThermalSPM(cell, dt=5.0, rom="pade", order=3)
    current = cell.nominal_capacity * c_rate
    z = model.initial_state(0.6, model.ambient)
    for _ in range(40):
        z = model.step(z, current)

    # Hold the electrical state and drive the thermal node to convergence, so the
    # balance is tested without state of charge drifting underneath it.
    heat = model.heat_generation(z, current)["total"]
    steady = model.ambient + heat / conductance(cell)
    decay = np.exp(-model.dt / cell.thermal.time_constant)
    temperature = model.temperature(z)
    # 20000 steps at dt = 5 s is eighty time constants, so the transient is gone
    # to well below double precision and what remains is the fixed point itself.
    for _ in range(20000):
        rise = heat / conductance(cell)
        temperature = model.ambient + rise + (temperature - model.ambient - rise) * decay
    assert temperature == pytest.approx(steady, rel=1e-12)
    assert heat == pytest.approx(conductance(cell) * (steady - model.ambient), rel=1e-12)


def test_irreversible_heat_is_never_negative(model, cell):
    """Dissipation cannot cool the cell, for either sign of current."""
    for c_rate in (-3.0, -1.0, -0.2, 0.2, 1.0, 3.0):
        current = cell.nominal_capacity * c_rate
        z = model.initial_state(0.5, 298.15)
        for _ in range(60):
            z = model.step(z, current)
        assert model.heat_generation(z, current)["irreversible"] >= 0.0


def test_reversible_heat_changes_sign_with_current(model, cell):
    """Entropic heat is signed; it is the only term that can cool a cell."""
    z = model.initial_state(0.5, 298.15)
    discharge = model.heat_generation(z, cell.nominal_capacity)["reversible"]
    charge = model.heat_generation(z, -cell.nominal_capacity)["reversible"]
    assert discharge == pytest.approx(-charge, rel=1e-12)
    assert discharge != 0.0


def test_zero_current_generates_no_heat(model):
    z = model.initial_state(0.5, 310.0)
    heat = model.heat_generation(z, 0.0)
    assert heat["irreversible"] == pytest.approx(0.0, abs=1e-15)
    assert heat["reversible"] == pytest.approx(0.0, abs=1e-15)


def test_irreversible_heat_grows_with_rate(model, cell):
    """Roughly quadratic: doubling the current more than doubles dissipation."""
    values = []
    for c_rate in (0.5, 1.0, 2.0):
        current = cell.nominal_capacity * c_rate
        z = model.initial_state(0.6, 298.15)
        for _ in range(60):
            z = model.step(z, current)
        values.append(model.heat_generation(z, current)["irreversible"])
    assert values[1] > 2.0 * values[0]
    assert values[2] > 2.0 * values[1]


def test_self_heating_raises_temperature_under_load(model, cell):
    result = model.simulate(np.full(1200, cell.nominal_capacity), soc0=0.9)
    assert result["temperature"][-1] > result["temperature"][0] + 3.0
    assert np.all(np.diff(result["temperature"]) > -1e-9), "temperature dipped under load"


# ------------------------------------------------------------------- schedule


def test_blended_systems_are_cached_within_a_sample(cell):
    """Evaluating one sample must not re-blend the schedule six times.

    Concentrations, voltage, heat generation and the step itself all need the
    same matrices at the same temperature. Caching one deep is worth about half
    the model's runtime, and this pins it so a later refactor cannot quietly
    undo it.
    """
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3)
    z = model.initial_state(0.6, 295.0)
    model.outputs(z, 5.0)
    first = model._systems(295.0)
    model.heat_generation(z, 5.0)
    model.step(z, 5.0)
    second = model._systems(295.0)
    # Identity, not equality: a fresh blend would allocate new objects.
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert model._cache_temperature == 295.0


def test_cache_refreshes_when_temperature_moves(cell):
    """And it must not go stale, which is the failure mode a cache introduces."""
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3)
    cold = model.initial_state(0.6, 265.0)
    warm = model.initial_state(0.6, 320.0)
    first = model.voltage(cold, 8.0)
    second = model.voltage(warm, 8.0)
    again = model.voltage(cold, 8.0)
    assert first != second
    assert again == pytest.approx(first, rel=1e-15)


def test_schedule_reproduces_grid_points_exactly(cell):
    """On a grid point the interpolant must return that system unchanged."""
    grid = np.linspace(263.15, 323.15, 7)
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3, temperature_grid=grid)
    for index, temperature in enumerate(grid):
        got = model.schedule_negative.at(float(temperature))
        want = model.schedule_negative.systems[index]
        assert np.allclose(got.A, want.A, rtol=0, atol=1e-14)
        assert np.allclose(got.B, want.B, rtol=0, atol=1e-14)


def test_schedule_clamps_outside_its_range(cell):
    grid = np.linspace(273.15, 313.15, 5)
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3, temperature_grid=grid)
    below = model.schedule_negative.at(200.0)
    at_low = model.schedule_negative.systems[0]
    assert np.allclose(below.A, at_low.A)
    above = model.schedule_negative.at(400.0)
    at_high = model.schedule_negative.systems[-1]
    assert np.allclose(above.A, at_high.A)


def test_schedule_slope_is_zero_outside_the_grid(cell):
    grid = np.linspace(273.15, 313.15, 5)
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3, temperature_grid=grid)
    for temperature in (250.0, 350.0):
        dA, dB = model.schedule_negative.slope(temperature)
        assert np.all(dA == 0.0)
        assert np.all(dB == 0.0)


def test_schedule_slope_matches_a_central_difference(cell):
    grid = np.linspace(263.15, 323.15, 7)
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3, temperature_grid=grid)
    schedule = model.schedule_negative
    for temperature in (271.0, 295.0, 311.0):  # interior, away from grid points
        h = 1e-4
        numeric = (schedule.at(temperature + h).A - schedule.at(temperature - h).A) / (2 * h)
        analytic, _ = schedule.slope(temperature)
        scale = max(float(np.max(np.abs(numeric))), 1e-30)
        assert np.max(np.abs(analytic - numeric)) / scale < 1e-6


def test_factor_blending_beats_linear_in_temperature(cell):
    """The whole reason the schedule blends on the Arrhenius factor.

    Linear interpolation in temperature fits a straight line through an
    exponential and is worst where the exponential is steepest, which is the cold
    end -- precisely where a physics-based estimator is most needed, because that
    is where plating limits bite. Blending on the factor removes almost all of it.
    """
    grid = np.linspace(253.15, 333.15, 9)
    electrode = cell.negative
    reference = cell.reference_temperature

    def build(temperature: float):
        return make_rom(
            "pade",
            electrode.particle_radius,
            electrode.diffusivity_at(temperature, reference),
            order=3,
        )

    smart = schedule_over_temperature(
        build,
        grid,
        1.0,
        activation_energy=electrode.diffusion_activation_energy,
        reference_temperature=reference,
    )
    naive = ScheduledStateSpace(temperatures=grid, systems=smart.systems)

    probe = 255.15  # near the cold end, deliberately between grid points
    exact = build(probe).discretise(1.0)
    scale = float(np.max(np.abs(exact.A)))
    smart_error = float(np.max(np.abs(smart.at(probe).A - exact.A))) / scale
    naive_error = float(np.max(np.abs(naive.at(probe).A - exact.A))) / scale
    assert smart_error < naive_error / 10.0, (
        f"factor blending {smart_error:.2e} should beat linear {naive_error:.2e}"
    )


def test_schedule_error_falls_as_the_grid_is_refined(cell):
    errors = []
    for count in (5, 9, 17):
        model = ThermalSPM(
            cell,
            dt=1.0,
            rom="pade",
            order=3,
            temperature_grid=np.linspace(253.15, 333.15, count),
        )
        errors.append(model.scheduling_error()["negative_A"])
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_scheduled_voltage_tracks_an_exactly_rebuilt_model(cell):
    """The schedule must not cost more than a couple of millivolts anywhere.

    Compared against an isothermal model rebuilt exactly at each temperature,
    with the thermal node frozen by an enormous heat capacity so the only
    difference is the interpolation.
    """
    current = np.full(600, cell.nominal_capacity * 2.0)
    for temperature_c in (-18.0, -3.0, 22.0, 47.0):
        temperature = temperature_c + 273.15
        reference = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)
        frozen = replace(
            cell,
            thermal=replace(cell.thermal, heat_capacity=1e12, ambient_temperature=temperature),
        )
        scheduled = ThermalSPM(frozen, dt=1.0, rom="pade", order=3, ambient=temperature)
        want = reference.simulate(current, soc0=0.9)["voltage"]
        got = scheduled.simulate(current, soc0=0.9, temperature=temperature)["voltage"]
        worst = float(np.max(np.abs(got - want)))
        assert worst < 5e-3, f"{temperature_c} C: schedule cost {1e3 * worst:.2f} mV"


# ------------------------------------------------------------------ Jacobians


@pytest.mark.parametrize("soc", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("c_rate", [0.0, 1.0, -1.0, 3.0])
def test_state_jacobian_matches_central_differences(model, cell, soc, c_rate):
    current = cell.nominal_capacity * c_rate
    z = model.initial_state(soc, 298.15)
    for _ in range(150):
        z = model.step(z, current)
    analytic = model.state_jacobian(z, current)
    numeric = model.numerical_state_jacobian(z, current)
    scale = max(float(np.linalg.norm(numeric)), 1e-30)
    assert np.linalg.norm(analytic - numeric) / scale < 1e-6


@pytest.mark.parametrize("soc", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("c_rate", [0.0, 1.0, -1.0, 3.0])
def test_voltage_jacobian_matches_central_differences(model, cell, soc, c_rate):
    current = cell.nominal_capacity * c_rate
    z = model.initial_state(soc, 298.15)
    for _ in range(150):
        z = model.step(z, current)
    analytic = model.voltage_jacobian(z, current)
    numeric = model.numerical_voltage_jacobian(z, current)
    scale = max(float(np.linalg.norm(numeric)), 1e-30)
    assert np.linalg.norm(analytic - numeric) / scale < 1e-6


def test_temperature_column_of_the_jacobian_is_not_zero(model, cell):
    """Guards against the temperature coupling being silently dropped.

    Every term in the temperature column comes from a different mechanism, so a
    plausible-looking implementation that forgets one still passes a smoke test.
    This asserts the column is populated where it must be.
    """
    current = cell.nominal_capacity * 2.0
    z = model.initial_state(0.6, 293.0)
    for _ in range(200):
        z = model.step(z, current)
    jac = model.state_jacobian(z, current)
    column = jac[:, model.temperature_index]
    assert np.any(np.abs(column[: model.temperature_index]) > 0.0), (
        "diffusion states must respond to temperature through the schedule"
    )
    assert abs(model.voltage_jacobian(z, current)[model.temperature_index]) > 0.0


# -------------------------------------------------------- against the isothermal


def test_matches_isothermal_when_temperature_is_pinned(cell):
    """With heat capacity enormous, the thermal model must reduce to the isothermal one."""
    temperature = 298.15
    frozen = replace(
        cell,
        thermal=replace(cell.thermal, heat_capacity=1e14, ambient_temperature=temperature),
    )
    thermal = ThermalSPM(frozen, dt=1.0, rom="pade", order=3, ambient=temperature)
    isothermal = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)
    current = synthetic_drive_cycle(cell.nominal_capacity, duration=400.0, seed=2)
    a = thermal.simulate(current, soc0=0.8, temperature=temperature)
    b = isothermal.simulate(current, soc0=0.8)
    assert np.max(np.abs(a["voltage"] - b["voltage"])) < 5e-3
    assert np.max(np.abs(a["soc"] - b["soc"])) < 1e-9


def test_thermal_and_isothermal_diverge_in_the_cold(cell):
    """The case for carrying a temperature state at all.

    A cell self-heating from -15 C behaves nothing like one held there, because
    diffusivity climbs by a factor of several as it warms. If this difference were
    small the thermal model would not be worth its cost.
    """
    ambient = 258.15
    thermal = ThermalSPM(cell, dt=1.0, rom="pade", order=3, ambient=ambient)
    isothermal = SPM(cell, dt=1.0, rom="pade", order=3, temperature=ambient)
    current = np.full(900, cell.nominal_capacity * 2.0)
    a = thermal.simulate(current, soc0=0.9, temperature=ambient)
    b = isothermal.simulate(current, soc0=0.9)
    assert a["temperature"][-1] - ambient > 5.0
    assert np.max(np.abs(a["voltage"] - b["voltage"])) > 0.1


def test_warmer_cell_polarises_less(cell):
    """Higher temperature must mean less overpotential at the same current."""
    losses = []
    for ambient in (263.15, 283.15, 303.15):
        frozen = replace(
            cell,
            thermal=replace(cell.thermal, heat_capacity=1e14, ambient_temperature=ambient),
        )
        model = ThermalSPM(frozen, dt=1.0, rom="pade", order=3, ambient=ambient)
        current = cell.nominal_capacity * 2.0
        z = model.initial_state(0.6, ambient)
        for _ in range(300):
            z = model.step(z, current)
        terms = model.outputs(z, current)
        losses.append(abs(terms.overpotential[0]) + abs(terms.overpotential[1]))
    assert losses[0] > losses[1] > losses[2]
