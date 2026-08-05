"""Validation of the temperature-scheduled code generator."""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.codegen import generate, generate_scheduled, spec_from_thermal_spm
from cellkernel.codegen.thermal_spec import (
    ThermalReferenceEstimator,
    scheduled_table_domain,
)
from cellkernel.data import synthetic_drive_cycle
from cellkernel.models import SPM, ThermalSPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.verify import find_compiler, verify_scheduled

ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)
HAS_CC = find_compiler() is not None
needs_cc = pytest.mark.skipif(not HAS_CC, reason="no C compiler on PATH")


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)


@pytest.fixture(scope="module")
def model(cell):
    return ThermalSPM(cell, dt=1.0, rom="pade", order=3)


@pytest.fixture(scope="module")
def profile(cell):
    return synthetic_drive_cycle(cell.nominal_capacity, duration=400.0, seed=4)


# ------------------------------------------------------------ spec extraction


def test_spec_shapes(model):
    spec = spec_from_thermal_spm(model)
    grid = spec.n_temperatures
    assert spec.negative.A_grid.shape == (grid, spec.n_negative, spec.n_negative)
    assert spec.positive.B_grid.shape == (grid, spec.n_positive)
    assert spec.negative.c_surface_grid.shape == (grid, spec.n_negative)
    assert spec.negative.d_surface_grid.shape == (grid,)
    assert spec.n_states == spec.n_negative + spec.n_positive


def test_spec_blend_reproduces_the_python_schedule(model):
    """The spec's own blending must agree with the model it was extracted from."""
    spec = spec_from_thermal_spm(model)
    for temperature in (255.0, 271.3, 298.15, 317.7):
        a_spec, b_spec, c_spec, d_spec = spec.matrices(spec.negative, temperature)
        want = model.schedule_negative.at(temperature)
        assert np.allclose(a_spec, want.A, rtol=0, atol=1e-14)
        assert np.allclose(c_spec, want.C[0], rtol=0, atol=1e-14)
        assert d_spec == pytest.approx(float(want.D[0, 0]), abs=1e-14)
        # The spec folds the flux conversion into B before blending while the
        # model blends first and converts after. Those are the same number in
        # exact arithmetic and differ by rounding in floating point, so this is a
        # relative comparison rather than an absolute one.
        assert np.allclose(b_spec, want.B.reshape(-1) * model._flux_neg, rtol=1e-12, atol=0.0)


def test_spec_initial_state_matches_the_model(model):
    spec = spec_from_thermal_spm(model)
    for soc in (0.0, 0.35, 1.0):
        want = model.initial_state(soc, model.ambient)[: spec.n_states]
        assert np.allclose(spec.initial_state(soc, model.ambient), want, rtol=1e-12)


def test_bulk_row_must_be_temperature_independent(model):
    """Guards the assumption that lets the bulk row be emitted once.

    The conserved functional cannot depend on diffusivity -- how much lithium is
    in a particle is not a function of how fast it moves -- so a temperature
    dependence here would mean something upstream is wrong. The extractor checks
    rather than assumes, and this exercises that check.
    """
    spec = spec_from_thermal_spm(model)
    for side, schedule in (
        (spec.negative, model.schedule_negative),
        (spec.positive, model.schedule_positive),
    ):
        for system in schedule.systems:
            assert np.allclose(system.C[1], side.c_bulk, rtol=0, atol=1e-13)


def test_table_domain_uses_the_coldest_grid_point(model, cell):
    """Surface excursion is largest where diffusivity is smallest.

    A table sized at 25 C saturates at -20 C, where diffusivity is roughly a
    twelfth of its reference value, so the domain must be computed cold.
    """
    for side in ("negative", "positive"):
        cold_lo, cold_hi = scheduled_table_domain(model, side, max_c_rate=3.0)
        electrode = cell._electrode(side)
        window_lo, window_hi = sorted((electrode.stoich_at_0_soc, electrode.stoich_at_100_soc))
        assert cold_lo <= window_lo and cold_hi >= window_hi
        assert 0.0 <= cold_lo < cold_hi <= 1.0

    # And it must be wider than the same calculation at the reference temperature.
    from cellkernel.codegen.spec import table_domain

    isothermal = SPM(cell, dt=1.0, rom="pade", order=3)
    warm_lo, warm_hi = table_domain(isothermal, "positive", max_c_rate=3.0)
    cold_lo, cold_hi = scheduled_table_domain(model, "positive", max_c_rate=3.0)
    assert (cold_hi - cold_lo) >= (warm_hi - warm_lo)


# --------------------------------------------------------------- NumPy mirror


@pytest.mark.parametrize("temperature", [258.15, 275.0, 298.15, 320.0])
def test_mirror_voltage_matches_an_isothermal_python_model(cell, temperature):
    """At a fixed temperature the scheduled mirror must agree with a rebuilt SPM.

    Tolerance is the schedule's own interpolation cost, which
    ``test_thermal.py`` measures independently; this checks that the mirror does
    not add anything on top of it.
    """
    model = ThermalSPM(cell, dt=1.0, rom="pade", order=3, ambient=temperature)
    spec = spec_from_thermal_spm(model)
    mirror = ThermalReferenceEstimator(spec)
    mirror.init(0.8, temperature)
    reference = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)
    x = reference.initial_state(0.8)

    current = 2.0 * cell.nominal_capacity
    worst = 0.0
    for _ in range(200):
        worst = max(
            worst, abs(mirror.voltage(current, temperature) - reference.voltage(x, current))
        )
        mirror.predict(current, temperature)
        x = reference.step(x, current)
    assert worst < 6e-3, f"{temperature} K: mirror differs by {1e3 * worst:.2f} mV"


def test_mirror_covariance_stays_symmetric(model):
    spec = spec_from_thermal_spm(model)
    mirror = ThermalReferenceEstimator(spec)
    mirror.init(0.5, 290.0)
    for _ in range(200):
        mirror.predict(5.0, 290.0)
        mirror.update(5.0, 290.0, 3.7)
        assert np.allclose(mirror.P, mirror.P.T, atol=0.0)
        assert np.all(np.linalg.eigvalsh(mirror.P) > -1e-12)


def test_mirror_clamps_outside_the_grid(model):
    """Beyond the grid the coefficients must hold, not extrapolate."""
    spec = spec_from_thermal_spm(model)
    grid = spec.temperature_grid
    below = spec.matrices(spec.negative, float(grid[0]) - 40.0)[0]
    at_low = spec.matrices(spec.negative, float(grid[0]))[0]
    assert np.allclose(below, at_low)
    above = spec.matrices(spec.negative, float(grid[-1]) + 40.0)[0]
    at_high = spec.matrices(spec.negative, float(grid[-1]))[0]
    assert np.allclose(above, at_high)


# ------------------------------------------------------------------ emission


def test_generate_scheduled_writes_expected_files(model, tmp_path):
    project = generate_scheduled(model, tmp_path / "gen", precision="float")
    for name in (
        "cellkernel_scheduled.h",
        "cellkernel_scheduled.c",
        "ck_harness.c",
        "Makefile",
        "CMakeLists.txt",
        "BUDGET.txt",
    ):
        assert (project.directory / name).is_file(), name
    source = (project.directory / "cellkernel_scheduled.c").read_text()
    assert "malloc" not in source
    header = (project.directory / "cellkernel_scheduled.h").read_text()
    assert "TEMPERATURE IS AN INPUT, NOT A STATE" in header
    assert project.scheduled is True


def test_scheduled_costs_more_flash_than_isothermal(cell, model, tmp_path):
    """The schedule is not free, and the budget should say so."""
    isothermal = generate(
        SPM(cell, dt=1.0, rom="pade", order=3), tmp_path / "iso", precision="float"
    )
    scheduled = generate_scheduled(model, tmp_path / "sch", precision="float")
    assert scheduled.budget.flash_bytes > isothermal.budget.flash_bytes
    assert scheduled.budget.ram_bytes > isothermal.budget.ram_bytes
    # But not unboundedly so: the potential tables dominate and are shared.
    assert scheduled.budget.flash_bytes < 3 * isothermal.budget.flash_bytes


# ------------------------------------------------------- compile and compare


@needs_cc
def test_generated_scheduled_c_compiles_without_warnings(model, tmp_path):
    from cellkernel.verify import compile_project

    project = generate_scheduled(model, tmp_path / "gen", precision="float")
    executable, warnings = compile_project(project)
    assert executable.is_file()
    assert warnings == "", warnings


@needs_cc
@pytest.mark.parametrize("temperature_c", [-15.0, 5.0, 25.0, 50.0])
def test_generated_c_matches_the_mirror_at_fixed_temperature(
    model, profile, tmp_path, temperature_c
):
    project = generate_scheduled(model, tmp_path / f"t{temperature_c}", precision="double")
    report = verify_scheduled(project, model, profile, temperature_c + 273.15, initial_soc=0.85)
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 1e-12
    assert report.max_soc_error_vs_mirror < 1e-12


@needs_cc
def test_generated_c_matches_the_mirror_while_temperature_ramps(model, profile, tmp_path):
    """The interesting case: temperature crossing grid boundaries mid-run."""
    project = generate_scheduled(model, tmp_path / "ramp", precision="double")
    ramp = np.linspace(256.0, 322.0, profile.size)
    report = verify_scheduled(project, model, profile, ramp, initial_soc=0.85)
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 1e-12


@needs_cc
def test_generated_c_matches_in_single_precision(model, profile, tmp_path):
    project = generate_scheduled(model, tmp_path / "single", precision="float")
    ramp = np.linspace(260.0, 315.0, profile.size)
    report = verify_scheduled(project, model, profile, ramp, initial_soc=0.85)
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 5e-4


@needs_cc
def test_filter_path_matches_the_mirror(model, profile, tmp_path):
    project = generate_scheduled(model, tmp_path / "filt", precision="double")
    report = verify_scheduled(project, model, profile, 268.15, initial_soc=0.6, mode="filter")
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 1e-11
    assert report.max_soc_error_vs_mirror < 1e-11


@needs_cc
def test_generated_c_clamps_beyond_the_grid(model, profile, tmp_path):
    """Feeding an out-of-range temperature must not diverge or produce NaN."""
    project = generate_scheduled(model, tmp_path / "clamp", precision="double")
    report = verify_scheduled(project, model, profile, 210.0, initial_soc=0.8)
    assert report.passed, report.summary()
    assert np.isfinite(report.max_voltage_error_vs_mirror)
