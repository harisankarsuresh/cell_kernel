"""Tests for code generation and the generated-versus-Python cross-check."""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.codegen import estimate_budget, generate
from cellkernel.codegen.spec import (
    ReferenceEstimator,
    spec_from_spm,
    table_backed_model,
    table_domain,
)
from cellkernel.data import synthetic_drive_cycle
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.verify import find_compiler, verify

CELL = chen2020_nmc811_graphite()
HAS_CC = find_compiler() is not None
needs_cc = pytest.mark.skipif(not HAS_CC, reason="no C compiler on PATH")


@pytest.fixture(scope="module")
def profile() -> np.ndarray:
    return synthetic_drive_cycle(CELL.nominal_capacity, duration=600.0, seed=3)


def build(rom: str = "pade", order: int = 3) -> SPM:
    return SPM(CELL, dt=1.0, rom=rom, order=order)


# ----------------------------------------------------------------- spec extraction


def test_spec_shapes_match_model():
    model = build()
    spec = spec_from_spm(model)
    assert spec.n_states == model.n_states
    assert spec.A.shape == (spec.n_states, spec.n_states)
    assert spec.B.shape == (spec.n_states,)
    assert spec.n_negative + spec.n_positive == spec.n_states


def test_spec_transition_reproduces_model_step():
    """The extracted (A, B) must reproduce the model exactly, not approximately.

    The single particle model has linear dynamics, so this is an identity rather
    than a linearisation, and any discrepancy is an extraction bug.
    """
    model = build()
    spec = spec_from_spm(model)
    rng = np.random.default_rng(0)
    x = model.initial_state(0.6)
    for current in (0.0, 5.0, -3.0, 15.0):
        for _ in range(5):
            expected = model.step(x, current)
            actual = spec.A @ x + spec.B * current
            assert np.allclose(actual, expected, rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(x))))
            x = expected
        x = model.initial_state(rng.uniform(0.2, 0.9))


def test_spec_initial_state_matches_model():
    model = build()
    spec = spec_from_spm(model)
    for soc in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(spec.initial_state(soc), model.initial_state(soc), rtol=1e-12)


def test_probed_kinetics_reject_a_changed_model():
    """Extraction validates the assumed functional form instead of trusting it."""
    model = build()

    def wrong_exchange_current(c_surf, electrode):
        return 1.0 + 0.0 * c_surf  # constant: not k*sqrt(c)*sqrt(cmax-c)

    model._exchange_current = wrong_exchange_current
    with pytest.raises(ValueError, match="not of the form"):
        spec_from_spm(model)


# --------------------------------------------------------------------- OCP tables


def test_table_domain_covers_surface_excursion():
    """The table must extend past the bulk window, or it saturates under load."""
    model = build()
    for side in ("negative", "positive"):
        el = CELL._electrode(side)
        lo, hi = table_domain(model, side, max_c_rate=3.0)
        window_lo, window_hi = sorted((el.stoich_at_0_soc, el.stoich_at_100_soc))
        assert lo <= window_lo and hi >= window_hi
        assert 0.0 <= lo < hi <= 1.0


def test_table_domain_widens_with_rate():
    model = build()
    narrow = table_domain(model, "positive", max_c_rate=0.5)
    wide = table_domain(model, "positive", max_c_rate=4.0)
    assert wide[0] <= narrow[0] and wide[1] >= narrow[1]


def test_table_error_falls_as_points_squared():
    """Linear interpolation is second order, so doubling points quarters the error."""
    model = build()
    coarse = spec_from_spm(model, table_points=65).negative.ocp_table.max_abs_error
    fine = spec_from_spm(model, table_points=257).negative.ocp_table.max_abs_error
    ratio = coarse / fine
    assert 8.0 < ratio < 24.0, f"expected ~16x improvement, got {ratio:.1f}x"


# ----------------------------------------------------------------- NumPy mirror


@pytest.mark.parametrize("rom,order", [("poly", 2), ("spectral", 3), ("pade", 3)])
def test_mirror_matches_table_backed_model(rom, order, profile):
    """The mirror must reproduce the model it claims to mirror, to round-off."""
    model = build(rom, order)
    spec = spec_from_spm(model)
    reference = table_backed_model(model, spec)
    expected = reference.simulate(profile, soc0=0.8)

    est = ReferenceEstimator(spec)
    est.init(0.8)
    for k, current in enumerate(profile):
        assert abs(est.voltage(float(current)) - expected["voltage"][k]) < 1e-11
        assert abs(est.soc() - expected["soc"][k]) < 1e-12
        est.predict(float(current))


def test_mirror_voltage_jacobian_matches_central_difference():
    model = build()
    spec = spec_from_spm(model)
    est = ReferenceEstimator(spec)
    est.init(0.7)
    for _ in range(60):
        est.predict(8.0)
    analytic = est.voltage_jacobian(8.0)
    numeric = np.empty_like(analytic)
    base = est.x.copy()
    for i in range(spec.n_states):
        h = 1e-6 * max(abs(base[i]), 1.0)
        est.x = base.copy()
        est.x[i] += h
        hi = est.voltage(8.0)
        est.x = base.copy()
        est.x[i] -= h
        lo = est.voltage(8.0)
        numeric[i] = (hi - lo) / (2 * h)
    est.x = base
    scale = max(np.linalg.norm(numeric), 1e-30)
    assert np.linalg.norm(analytic - numeric) / scale < 1e-6


def test_mirror_covariance_stays_symmetric_and_positive():
    model = build()
    spec = spec_from_spm(model)
    est = ReferenceEstimator(spec)
    est.init(0.5)
    for _ in range(300):
        est.predict(5.0)
        est.update(5.0, 3.7)
        assert np.allclose(est.P, est.P.T, atol=0.0), "covariance lost symmetry"
        assert np.all(np.linalg.eigvalsh(est.P) > -1e-12), "covariance lost positivity"


# ------------------------------------------------------------------- emission


def test_generate_writes_expected_files(tmp_path):
    project = generate(build(), tmp_path / "gen", precision="float")
    for name in (
        "cellkernel_estimator.h",
        "cellkernel_estimator.c",
        "ck_harness.c",
        "Makefile",
        "CMakeLists.txt",
        "BUDGET.txt",
    ):
        assert (project.directory / name).is_file(), name
    source = (project.directory / "cellkernel_estimator.c").read_text()
    assert "malloc" not in source, "generated code must not allocate"
    assert "static" in source


def test_generated_constants_round_trip_exactly(tmp_path):
    """Emitted literals must read back bit-identically in double precision."""
    project = generate(build(), tmp_path / "gen", precision="double")
    source = (project.directory / "cellkernel_estimator.c").read_text()
    for value in project.spec.A.reshape(-1):
        if value != 0.0:
            assert repr(float(value)) in source


def test_float_and_double_differ_only_in_typedef(tmp_path):
    single = generate(build(), tmp_path / "f", precision="float")
    double = generate(build(), tmp_path / "d", precision="double")
    assert "typedef float ck_real_t;" in (single.directory / "cellkernel_estimator.h").read_text()
    assert "typedef double ck_real_t;" in (double.directory / "cellkernel_estimator.h").read_text()
    assert single.budget.ram_bytes * 2 == double.budget.ram_bytes


# --------------------------------------------------------------------- budget


def test_budget_scales_quadratically_with_states():
    small = estimate_budget(spec_from_spm(build("pade", 2)), "float")
    large = estimate_budget(spec_from_spm(build("pade", 5)), "float")
    assert large.n_states > small.n_states
    # RAM is dominated by the covariance, which is n^2.
    assert large.ram_bytes > small.ram_bytes
    assert large.estimated_cycles > small.estimated_cycles


def test_budget_reports_nonzero_resources():
    budget = estimate_budget(spec_from_spm(build()), "float")
    assert budget.flash_bytes > 0
    assert budget.ram_bytes > 0
    assert budget.estimated_cycles > 0
    assert "flash" in budget.summary()


# ------------------------------------------------- compile and cross-verify


@needs_cc
def test_generated_c_compiles_without_warnings(tmp_path):
    from cellkernel.verify import compile_project

    project = generate(build(), tmp_path / "gen", precision="float")
    executable, warnings = compile_project(project)
    assert executable.is_file()
    assert warnings == "", f"generated C produced warnings:\n{warnings}"


@needs_cc
@pytest.mark.parametrize("rom,order", [("poly", 2), ("spectral", 3), ("pade", 3), ("pade", 4)])
def test_generated_c_matches_python_double(tmp_path, profile, rom, order):
    model = build(rom, order)
    project = generate(model, tmp_path / f"{rom}{order}", precision="double")
    report = verify(project, model, profile, initial_soc=0.85)
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 1e-12
    # End-to-end error against the full model is the lookup table, nothing else.
    assert report.max_voltage_error_total < 5e-4, report.summary()


@needs_cc
def test_generated_c_matches_python_single_precision(tmp_path, profile):
    model = build()
    project = generate(model, tmp_path / "single", precision="float")
    report = verify(project, model, profile, initial_soc=0.85)
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 5e-4


@needs_cc
def test_filter_path_matches_python(tmp_path, profile):
    """The Kalman path, not just the forward model, is verified."""
    model = build()
    project = generate(model, tmp_path / "filt", precision="double")
    report = verify(project, model, profile, initial_soc=0.55, mode="filter")
    assert report.passed, report.summary()
    assert report.max_voltage_error_vs_mirror < 1e-11
    assert report.max_soc_error_vs_mirror < 1e-11


def test_filter_converges_from_a_wrong_initial_soc():
    """The filter must recover from a bad cold-boot estimate.

    Convergence is fast -- a single voltage measurement removes most of a 30%
    error -- because the initial covariance is rank one along ``d(state)/d(soc)``.
    That tells the filter the error lies in exactly one direction, so the scalar
    voltage residual is enough to determine it. A diagonal prior spreads the same
    uncertainty over every coordinate, and then one scalar measurement cannot
    resolve it; that version of this test diverged.
    """
    model = build()
    spec = spec_from_spm(model, measurement_noise=1e-6)
    current = np.full(2400, CELL.nominal_capacity * 0.5)
    truth = model.simulate(current, soc0=0.80)

    est = ReferenceEstimator(spec)
    est.init(0.50)
    initial_error = abs(est.soc() - truth["soc"][0])
    assert initial_error > 0.25, "test should start badly wrong"

    errors = []
    for k, i_k in enumerate(current):
        est.predict(float(i_k))
        est.update(float(i_k), float(truth["voltage"][k]))
        errors.append(abs(est.soc() - truth["soc"][k]))

    assert errors[0] < 0.05, f"first correction should be decisive, got {errors[0]:.4f}"
    assert errors[-1] < 0.01, f"failed to converge: final error {errors[-1]:.4f}"
    assert errors[-1] < initial_error / 20.0


def test_filter_rejects_voltage_noise():
    """Estimated state of charge must be far quieter than the measurement."""
    model = build()
    spec = spec_from_spm(model, measurement_noise=1e-4)
    current = np.full(1800, CELL.nominal_capacity * 0.5)
    truth = model.simulate(current, soc0=0.75)
    rng = np.random.default_rng(11)
    noisy = truth["voltage"] + rng.normal(0.0, 0.01, truth["voltage"].size)

    est = ReferenceEstimator(spec)
    est.init(0.75)
    estimated = np.empty(current.size)
    for k, i_k in enumerate(current):
        est.predict(float(i_k))
        est.update(float(i_k), float(noisy[k]))
        estimated[k] = est.soc()

    error = estimated - truth["soc"]
    assert np.max(np.abs(error)) < 0.02, f"tracking error {np.max(np.abs(error)):.4f}"
    # The estimate should not simply follow the noise: successive differences of
    # the estimate must be far smaller than the noise would imply if it did.
    assert np.std(np.diff(error)) < 1e-3


def test_diagonal_prior_is_worse_than_structured():
    """Documents why the structured covariance exists, rather than asserting it.

    Kept as a regression guard: if someone replaces the rank-one prior with a
    diagonal one, this test records that the estimate degrades badly, which is the
    failure that motivated the current formulation.
    """
    model = build()
    structured = spec_from_spm(model, measurement_noise=1e-6)
    n = structured.n_states
    diagonal = replace_covariance(structured, np.diag(np.diag(structured.initial_covariance)))

    current = np.full(1200, CELL.nominal_capacity * 0.5)
    truth = model.simulate(current, soc0=0.80)

    def run(spec) -> float:
        est = ReferenceEstimator(spec)
        est.init(0.50)
        for k, i_k in enumerate(current):
            est.predict(float(i_k))
            est.update(float(i_k), float(truth["voltage"][k]))
        return abs(est.soc() - truth["soc"][-1])

    assert n > 2
    assert run(structured) < run(diagonal)


def replace_covariance(spec, initial_covariance):
    """Return a copy of ``spec`` with a different initial covariance."""
    from dataclasses import replace as dc_replace

    return dc_replace(spec, initial_covariance=initial_covariance)
