"""Measuring the generated estimator on a real ARM toolchain.

The resource budget in :mod:`cellkernel.codegen.budget` counts data structures
and models arithmetic. That is a useful thing to have before a toolchain is set
up, and it was also quietly wrong in a way only a measurement could reveal: it
reported flash as tables only, on the stated grounds that code was "small next to
the tables", when in fact the code is about as large again.

These tests need ``arm-none-eabi-gcc``, and the instruction counts additionally
need ``qemu-system-arm``. Both skip when absent.
"""

from __future__ import annotations

import pytest

from cellkernel.codegen import (
    CORTEX_M0,
    CORTEX_M4F,
    estimate_budget,
    find_arm_toolchain,
    find_qemu,
    generate,
    measure_arm_footprint,
    measure_arm_instructions,
    spec_from_spm,
)
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite

needs_arm = pytest.mark.skipif(
    find_arm_toolchain() is None, reason="no arm-none-eabi-gcc on this machine"
)
needs_qemu = pytest.mark.skipif(find_qemu() is None, reason="no qemu-system-arm")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    cell = chen2020_nmc811_graphite()
    model = SPM(cell, dt=1.0, rom="pade", order=3)
    out = tmp_path_factory.mktemp("arm")
    generate(model, out, precision="float")
    return out, model


# ------------------------------------------------------------------- footprint


@needs_arm
def test_it_cross_compiles_for_a_cortex_m4f(project):
    out, _ = project
    measurement = measure_arm_footprint(out, target=CORTEX_M4F, optimisation="-Os")
    assert measurement.text_bytes > 0
    assert measurement.rodata_bytes > 0
    assert measurement.flash_bytes == (
        measurement.text_bytes + measurement.rodata_bytes + measurement.data_bytes
    )


@needs_arm
def test_the_estimator_needs_no_static_ram(project):
    """No globals, so an instance costs exactly what the caller allocates.

    Asserted here rather than taken from the source, because a global introduced
    by accident -- a cached table, a lazily initialised constant -- would break
    reentrancy across cells in a pack and would not be visible from Python.
    """
    out, _ = project
    measurement = measure_arm_footprint(out, target=CORTEX_M4F)
    assert measurement.bss_bytes == 0
    assert measurement.data_bytes == 0


@needs_arm
def test_the_table_model_is_accurate_but_the_flash_model_is_not(project):
    """Precisely which part of the budget can be trusted.

    The table count is exact arithmetic and lands within a few percent of what
    the linker reports. The flash figure is that same number, and it omits the
    code -- which is not a rounding error, it is another 85% on top.
    """
    out, model = project
    measurement = measure_arm_footprint(out, target=CORTEX_M4F, optimisation="-Os")
    budget = estimate_budget(spec_from_spm(model), precision="float")

    tables = abs(budget.flash_bytes - measurement.rodata_bytes) / measurement.rodata_bytes
    assert tables < 0.05, f"table model off by {100 * tables:.1f}%"
    assert measurement.text_bytes > 0.5 * measurement.rodata_bytes, (
        "code is a large fraction of flash, so a tables-only figure understates it"
    )


@needs_arm
def test_a_core_without_an_fpu_costs_more_code(project):
    """Every floating-point operation becomes a library call on a Cortex-M0+."""
    out, _ = project
    with_fpu = measure_arm_footprint(out, target=CORTEX_M4F, optimisation="-Os")
    without = measure_arm_footprint(out, target=CORTEX_M0, optimisation="-Os")
    assert without.text_bytes > with_fpu.text_bytes
    assert without.rodata_bytes == with_fpu.rodata_bytes, "tables do not care"


@needs_arm
def test_optimising_for_speed_costs_flash(project):
    out, _ = project
    small = measure_arm_footprint(out, optimisation="-Os")
    fast = measure_arm_footprint(out, optimisation="-O2")
    assert fast.text_bytes > small.text_bytes


def test_it_refuses_rather_than_guesses_without_a_toolchain(tmp_path, monkeypatch):
    """A missing toolchain must be an error, not a silently substituted estimate."""
    monkeypatch.setattr("cellkernel.codegen.measure.shutil.which", lambda _: None)
    monkeypatch.setattr("cellkernel.codegen.measure.find_arm_toolchain", lambda: None)
    with pytest.raises(RuntimeError, match="arm-none-eabi-gcc"):
        measure_arm_footprint(tmp_path)


# ---------------------------------------------------------------- instructions


@needs_arm
@needs_qemu
def test_instructions_per_step_are_measurable(project):
    """And considerably more than the model predicts.

    The modelled cycle count is built from an operation tally and a table of
    assumed costs. Measured against an emulated core it is optimistic by roughly
    a factor of two and a half -- and QEMU itself counts instructions, not
    cycles, so real silicon is further away still. Anyone sizing a task period
    from the modelled number would be badly wrong.
    """
    out, model = project
    count = measure_arm_instructions(out, optimisation="-Os")
    budget = estimate_budget(spec_from_spm(model), precision="float")

    assert 1_000 < count < 100_000, f"implausible instruction count {count}"
    assert count > 2.0 * budget.estimated_cycles, (
        f"measured {count:.0f} against modelled {budget.estimated_cycles}; "
        "if the model has improved, update this test and the README"
    )


@needs_arm
@needs_qemu
def test_the_measurement_is_reproducible(project):
    """Emulated instruction counts are deterministic; if this flakes, it is a bug."""
    out, _ = project
    first = measure_arm_instructions(out, optimisation="-Os")
    second = measure_arm_instructions(out, optimisation="-Os")
    assert first == pytest.approx(second, rel=1e-12)


@needs_arm
@needs_qemu
def test_a_larger_model_costs_more_instructions(tmp_path):
    """Covariance propagation is quadratic in state count, so this should bite."""
    cell = chen2020_nmc811_graphite()
    counts = {}
    for order in (2, 4):
        out = tmp_path / f"order{order}"
        generate(SPM(cell, dt=1.0, rom="pade", order=order), out, precision="float")
        counts[order] = measure_arm_instructions(out, optimisation="-Os")
    assert counts[4] > 2.0 * counts[2], f"{counts}"
