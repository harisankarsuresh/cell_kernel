"""The plating limiter, in the firmware.

The degradation model computes a plating margin and the protocol module inverts
it into a safe charging current, but both live in Python. This is that same
calculation on the microcontroller, which is where a charge setpoint actually has
to be produced.

It costs almost nothing to carry: the emitted ``ck_voltage`` already forms the
negative electrode potential as a sub-expression and discards it.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from cellkernel.codegen import generate, spec_from_spm
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.verify import find_compiler

HAS_CC = find_compiler() is not None
needs_cc = pytest.mark.skipif(not HAS_CC, reason="no C compiler on PATH")
MARGIN = 0.01


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite()


@pytest.fixture(scope="module")
def model(cell):
    return SPM(cell, dt=1.0, rom="pade", order=3)


@pytest.fixture(scope="module")
def spec(model):
    return spec_from_spm(model)


@pytest.fixture(scope="module")
def mirror(spec):
    from cellkernel.codegen.spec import ReferenceEstimator

    return ReferenceEstimator(spec)


# ------------------------------------------------------------------ the mirror


def test_ceiling_is_the_rate_the_tables_were_sized_for(spec, cell):
    """Searching above it would describe the lookup table, not the cell."""
    assert spec.current_ceiling == pytest.approx(3.0 * cell.nominal_capacity)


def test_plating_potential_tracks_the_full_model(model, spec, mirror, cell):
    """The table-backed mirror must agree with the analytic model it stands for.

    Not to machine precision. The potential is read from a lookup table, and
    graphite tabulates badly -- its several sharp stage transitions cost about
    5 mV at the default 257 points, against 0.13 mV for the layered oxide
    opposite it. The bound here is the table's own declared error, which is the
    honest one: anything larger would mean the emitted table is worse than it
    claims.
    """
    from cellkernel.degradation import DegradationModel

    ageing = DegradationModel(cell)
    declared = spec.negative.ocp_table.max_abs_error
    worst = 0.0
    for soc in (0.2, 0.5, 0.8):
        for c_rate in (0.0, -0.5, -1.0, -2.0):
            current = c_rate * cell.nominal_capacity
            x = model.initial_state(soc)
            mirror.init(soc)
            for _ in range(60):
                x = model.step(x, current)
                mirror.predict(current)
            exact = ageing.negative_potential(model, x, current)
            worst = max(worst, abs(mirror.plating_potential(current) - exact))
    assert worst <= 1.5 * declared, (
        f"{1e3 * worst:.2f} mV against a declared table error of {1e3 * declared:.2f} mV"
    )


def test_table_error_is_emitted_so_the_margin_can_be_sized(tmp_path, model, spec):
    """A plating margin below the table error measures the table, not the cell.

    That figure was previously visible only in a build log, which is the wrong
    place for a number the firmware's safety argument rests on.
    """
    out = tmp_path / "budget"
    generate(model, out, precision="double")
    header = (out / "cellkernel_estimator.h").read_text()
    assert "CK_OCP_ERROR_NEG" in header
    assert "CK_OCP_ERROR_POS" in header
    # Graphite really is the harder one, by about an order of magnitude.
    assert spec.negative.ocp_table.max_abs_error > 10.0 * spec.positive.ocp_table.max_abs_error


def test_more_table_points_shrink_the_error(model):
    """So the reader of that define knows what lever to pull."""
    from cellkernel.codegen.spec import ocp_table_for

    errors = [
        ocp_table_for(model, "negative", table_points=n).max_abs_error
        for n in (129, 257, 513, 1025)
    ]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 0.1 * errors[0]


def test_limiter_returns_the_ceiling_when_nothing_binds(mirror, spec):
    mirror.init(0.05)
    assert mirror.max_charge_current(MARGIN, spec.current_ceiling) == spec.current_ceiling


def test_limiter_returns_zero_when_nothing_is_safe(mirror, spec):
    mirror.init(0.999)
    assert mirror.max_charge_current(0.15, spec.current_ceiling) == 0.0


def test_limiter_rejects_a_nonpositive_ceiling(mirror):
    mirror.init(0.5)
    assert mirror.max_charge_current(MARGIN, 0.0) == 0.0


def test_the_returned_current_meets_the_margin(mirror, spec):
    for soc in (0.2, 0.4, 0.6, 0.8, 0.95):
        mirror.init(soc)
        allowed = mirror.max_charge_current(MARGIN, spec.current_ceiling)
        if allowed <= 0.0:
            continue
        assert mirror.plating_potential(-allowed) >= MARGIN - 1e-9


def test_the_limit_is_tight_not_merely_safe(mirror, spec):
    """A limiter that always returned zero would pass the test above."""
    mirror.init(0.7)
    allowed = mirror.max_charge_current(MARGIN, spec.current_ceiling)
    assert 0.0 < allowed < spec.current_ceiling
    over = allowed + 0.01 * spec.current_ceiling
    assert mirror.plating_potential(-over) < MARGIN


def test_the_limit_falls_as_the_cell_fills(mirror, spec):
    limits = []
    for soc in (0.2, 0.4, 0.6, 0.8, 0.95):
        mirror.init(soc)
        limits.append(mirror.max_charge_current(MARGIN, spec.current_ceiling))
    assert limits == sorted(limits, reverse=True)


def test_a_wider_margin_is_more_cautious(mirror, spec):
    mirror.init(0.7)
    relaxed = mirror.max_charge_current(0.0, spec.current_ceiling)
    strict = mirror.max_charge_current(0.05, spec.current_ceiling)
    assert strict < relaxed


# ------------------------------------------------------------- generated code


@needs_cc
def test_generated_c_matches_its_mirror(tmp_path, model, spec, cell):
    """Including the bisection, which should be bit-identical.

    A fixed iteration count and no convergence test means the two implementations
    take exactly the same path, so anything other than an exact match on the
    charge limit would mean the arithmetic differs somewhere upstream.
    """
    out = tmp_path / "plating"
    generate(model, out, precision="double")
    binary = out / "harness"
    subprocess.run(
        [
            find_compiler(),
            "-std=c99",
            "-O2",
            "-o",
            str(binary),
            str(out / "ck_harness.c"),
            str(out / "cellkernel_estimator.c"),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )

    current = np.full(400, -1.0 * cell.nominal_capacity)
    rows = "\n".join(f"{float(i):.17g},{3.9:.17g}" for i in current)
    done = subprocess.run(
        [str(binary), "openloop", "0.30"],
        input=rows,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = done.stdout.strip().splitlines()
    columns = {name: k for k, name in enumerate(lines[0].split(","))}
    assert "plating" in columns and "charge_limit" in columns
    emitted = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])

    from cellkernel.codegen.spec import ReferenceEstimator

    reference = ReferenceEstimator(spec)
    reference.init(0.30)
    expected = []
    for i in current:
        expected.append(
            (
                reference.plating_potential(float(i)),
                reference.max_charge_current(MARGIN, spec.current_ceiling),
            )
        )
        reference.predict(float(i))
    expected = np.array(expected)

    potential_error = np.max(np.abs(emitted[:, columns["plating"]] - expected[:, 0]))
    limit_error = np.max(np.abs(emitted[:, columns["charge_limit"]] - expected[:, 1]))
    assert potential_error < 1e-12, f"plating potential differs by {potential_error:.2e} V"
    assert limit_error == 0.0, f"bisection differs by {limit_error:.2e} A"


# ----------------------------------------------------- the scheduled variant


@pytest.fixture(scope="module")
def thermal_cell():
    return chen2020_nmc811_graphite().with_activation_energies(
        diffusion_negative=35_000.0,
        diffusion_positive=30_000.0,
        reaction_negative=35_000.0,
        reaction_positive=17_800.0,
    )


@pytest.fixture(scope="module")
def thermal_mirror(thermal_cell):
    from cellkernel.codegen import spec_from_thermal_spm
    from cellkernel.codegen.thermal_spec import ThermalReferenceEstimator
    from cellkernel.models import ThermalSPM

    model = ThermalSPM(thermal_cell, dt=1.0, rom="pade", order=3)
    spec = spec_from_thermal_spm(model)
    return ThermalReferenceEstimator(spec), spec


def test_the_scheduled_limiter_collapses_in_the_cold(thermal_mirror, thermal_cell):
    """The version that earns its keep.

    The isothermal generator can only answer for the one temperature it was
    built at, and plating is a cold-weather failure. On this cell the safe rate
    at 70% state of charge falls from 1.3C at 25 C to 0.26C at -10 C -- a factor
    of five that a fixed-rate charger has to either give up or ignore.
    """
    mirror, spec = thermal_mirror
    limits = {}
    for temperature in (263.15, 273.15, 283.15, 298.15, 313.15):
        mirror.init(0.7, temperature)
        limits[temperature] = mirror.max_charge_current(temperature, MARGIN, spec.current_ceiling)
    ordered = [limits[t] for t in sorted(limits)]
    assert ordered == sorted(ordered), "colder must allow less"
    assert limits[298.15] > 4.0 * limits[263.15]


def test_the_scheduled_limiter_meets_its_margin(thermal_mirror):
    mirror, spec = thermal_mirror
    for temperature in (263.15, 288.15, 313.15):
        for soc in (0.2, 0.5, 0.8, 0.95):
            mirror.init(soc, temperature)
            allowed = mirror.max_charge_current(temperature, MARGIN, spec.current_ceiling)
            if allowed <= 0.0:
                continue
            assert mirror.plating_potential(-allowed, temperature) >= MARGIN - 1e-9


@needs_cc
def test_scheduled_c_matches_its_mirror(tmp_path, thermal_cell, thermal_mirror):
    """Same bit-identical expectation as the isothermal case, across temperature."""
    from cellkernel.codegen import generate_scheduled
    from cellkernel.models import ThermalSPM

    mirror, spec = thermal_mirror
    model = ThermalSPM(thermal_cell, dt=1.0, rom="pade", order=3)
    out = tmp_path / "sched"
    generate_scheduled(model, out, precision="double")
    binary = out / "harness"
    subprocess.run(
        [
            find_compiler(),
            "-std=c99",
            "-O2",
            "-o",
            str(binary),
            str(out / "ck_harness.c"),
            str(out / "cellkernel_scheduled.c"),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )
    # A temperature ramp, so the blend crosses grid boundaries mid-run.
    steps = 300
    current = np.full(steps, -1.0 * thermal_cell.nominal_capacity)
    temperature = np.linspace(265.0, 305.0, steps)
    # The scheduled harness reads current, temperature, voltage.
    rows = "\n".join(
        f"{float(i):.17g},{float(t):.17g},{3.9:.17g}"
        for i, t in zip(current, temperature, strict=True)
    )
    done = subprocess.run(
        [str(binary), "openloop", "0.30"],
        input=rows,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = done.stdout.strip().splitlines()
    columns = {name: k for k, name in enumerate(lines[0].split(","))}
    assert "plating" in columns and "charge_limit" in columns
    emitted = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])

    mirror.init(0.30, float(temperature[0]))
    expected = []
    for i, t in zip(current, temperature, strict=True):
        expected.append(
            (
                mirror.plating_potential(float(i), float(t)),
                mirror.max_charge_current(float(t), MARGIN, spec.current_ceiling),
            )
        )
        mirror.predict(float(i), float(t))
    expected = np.array(expected)

    potential_error = np.max(np.abs(emitted[:, columns["plating"]] - expected[:, 0]))
    limit_error = np.max(np.abs(emitted[:, columns["charge_limit"]] - expected[:, 1]))
    assert potential_error < 1e-12, f"plating potential differs by {potential_error:.2e} V"
    assert limit_error == 0.0, f"bisection differs by {limit_error:.2e} A"


@needs_cc
def test_generated_limiter_actually_tapers(tmp_path, model, cell):
    """End to end: the firmware answer must fall as the cell fills."""
    out = tmp_path / "taper"
    generate(model, out, precision="double")
    binary = out / "harness"
    subprocess.run(
        [
            find_compiler(),
            "-std=c99",
            "-O2",
            "-o",
            str(binary),
            str(out / "ck_harness.c"),
            str(out / "cellkernel_estimator.c"),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )
    current = np.full(1500, -1.0 * cell.nominal_capacity)
    rows = "\n".join(f"{float(i):.17g},{3.9:.17g}" for i in current)
    done = subprocess.run(
        [str(binary), "openloop", "0.20"],
        input=rows,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = done.stdout.strip().splitlines()
    columns = {name: k for k, name in enumerate(lines[0].split(","))}
    data = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])

    limit = data[:, columns["charge_limit"]]
    potential = data[:, columns["plating"]]
    assert limit[0] > limit[-1], "the limit must taper as the cell fills"
    assert np.all(limit >= 0.0)
    assert np.all(np.isfinite(potential))
    # Charging at 1C throughout, the electrode should stay above the onset.
    assert potential.min() > 0.0
