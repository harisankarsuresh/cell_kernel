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


def test_plating_potential_tracks_the_full_model(model, mirror, cell):
    """The table-backed mirror must agree with the analytic model it stands for.

    Not to machine precision -- the potential is read from a 257-point table and
    that is the dominant error -- but to well inside the margin a controller
    holds, or the limiter would be acting on table resolution.
    """
    from cellkernel.degradation import DegradationModel

    ageing = DegradationModel(cell)
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
    assert worst < 0.2 * MARGIN, f"table error {1e3 * worst:.2f} mV eats the margin"


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
