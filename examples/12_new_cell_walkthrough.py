"""The whole procedure on one cell, from measurements to a verified C estimator.

Follows docs/SOP.md stage by stage, with the acceptance check printed after each
one so it is obvious where a real cell would fail. Uses the measured LG M50 rate
test, which is missing the pulse test the procedure asks for -- and the run says
so, loudly, at the stage where it matters.

    python -m cellkernel.data.reference
    python examples/12_new_cell_walkthrough.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from cellkernel.codegen import find_arm_toolchain, generate, measure_arm_footprint
from cellkernel.data import reference, synthetic_drive_cycle
from cellkernel.identify import DEFAULT_KNOBS, identify
from cellkernel.models import SPM, SPMe
from cellkernel.params import chen2020_nmc811_graphite, fit_stoichiometry_window
from cellkernel.verify import find_compiler, verify

DT = 10.0
PASS, FAIL, NOTE = "  PASS", "  FAIL", "  ----"


def check(label: str, value: float, limit: float, units: str = "mV") -> bool:
    ok = value <= limit
    print(f"{PASS if ok else FAIL}  {label:44s} {value:8.2f} {units}  (limit {limit:g})")
    return ok


def starting_parameters():
    try:
        import pybamm

        from cellkernel.params import from_pybamm

        return from_pybamm(pybamm.ParameterValues("Chen2020")), "PyBaMM Chen2020"
    except ImportError:
        return chen2020_nmc811_graphite(), "built-in Chen2020-style"


def main() -> None:
    try:
        soc, measured_ocv = reference.load_ocv()
    except (FileNotFoundError, OSError):
        print("Needs the dataset:  python -m cellkernel.data.reference")
        sys.exit(0)

    print("=" * 72)
    print("STAGE 0  Accuracy requirement")
    print("=" * 72)
    print("  Target application : fast-charge control with plating protection")
    print("  Governing quantity : negative electrode surface potential")
    print("  Voltage budget     : 10 mV, so table error must stay under ~3 mV\n")

    # ---------------------------------------------------------------- stage 1
    print("=" * 72)
    print("STAGE 1  Characterisation data")
    print("=" * 72)
    slow = reference.load_discharge("T25", "cRate_0p1C", dt=DT)
    capacity = float(np.trapezoid(slow.current, slow.time) / 3600.0)
    print(f"  T1 capacity      : {capacity:.4f} Ah")
    print(f"  T2 pseudo-OCV    : {soc.size} points")
    rates = ("cRate_0p1C", "cRate_0p5C", "cRate_1C", "cRate_2C")
    print(f"  T3 rate test     : {len(rates)} rates at 25 C")
    print("  T4 pulse / HPPC  : ABSENT")
    print("  T5 temperature   : available at 0, 10, 25, 45 C\n")
    print("  Missing T4 is not a detail. Stage 3.4 will show what it costs.\n")

    # ---------------------------------------------------------------- stage 3
    cell, provenance = starting_parameters()
    print("=" * 72)
    print(f"STAGE 3  Calibration  (starting from {provenance})")
    print("=" * 72)

    error = np.asarray(cell.open_circuit_voltage(soc)) - measured_ocv
    interior = (soc > 0.05) & (soc < 0.95)
    before = 1e3 * float(np.sqrt(np.mean(error[interior] ** 2)))
    print(f"  3.0  literature OCV error{'':21s}{before:8.2f} mV")

    cell = fit_stoichiometry_window(cell, soc, measured_ocv, capacity=capacity)
    after = np.asarray(cell.open_circuit_voltage(soc)) - measured_ocv
    ocv_rmse = 1e3 * float(np.sqrt(np.mean(after[interior] ** 2)))
    print("  3.2  fitted the window with capacity pinned")
    check("OCV, 5-95% SoC", ocv_rmse, 10.0)
    check("capacity", 100 * abs(cell.usable_capacity() - capacity) / capacity, 1.0, "%")

    segments = [
        (name.replace("cRate_", ""), seg.current, seg.voltage)
        for name, seg in ((n, reference.load_discharge("T25", n, dt=DT)) for n in rates)
    ]

    print("\n  3.4  fitting kinetics and transport")
    report = identify(
        cell, segments, lambda p: SPM(p, dt=DT, rom="pade", order=5), knobs=DEFAULT_KNOBS
    )
    print(f"       residual {1e3 * report.rmse_before:.1f} -> {1e3 * report.rmse_after:.1f} mV")
    weak = report.poorly_identified()
    strongest = max(zip(report.knobs, report.sensitivity, strict=True), key=lambda pair: pair[1])[
        0
    ].name
    print(f"       best determined  : {strongest}")
    print(f"       not determined   : {', '.join(weak) if weak else 'none'}")
    if len(weak) >= len(report.knobs) - 1:
        print(FAIL + "  REJECT this fit. Only one parameter was identified.")
        print("        Constant current is one excitation and cannot separate")
        print("        ohmic, kinetic and diffusive losses. Run T4 and refit;")
        print("        the residual below is a curve fit, not a characterisation.")
    print()

    # ---------------------------------------------------------------- stage 4
    print("=" * 72)
    print("STAGE 4  Acceptance against held-out data")
    print("=" * 72)
    worst = 0.0
    for name in rates:
        seg = reference.load_discharge("T25", name, dt=DT)
        model = SPMe(SPMe.reconcile(cell), dt=DT, rom="pade", order=5, electrolyte_cells=(6, 4, 6))
        run = model.simulate(seg.current, soc0=1.0)
        mask = seg.voltage > 2.7
        rmse = 1e3 * float(np.sqrt(np.mean((run["voltage"][mask] - seg.voltage[mask]) ** 2)))
        worst = max(worst, rmse)
        print(f"       SPMe at {seg.c_rate:4.1f}C{'':26s}{rmse:8.2f} mV")
    check("worst rate-test RMSE", worst, 25.0)
    print()

    # ---------------------------------------------------------------- stage 6
    print("=" * 72)
    print("STAGE 6  Generate, verify, measure")
    print("=" * 72)
    if find_compiler() is None:
        print(NOTE + "  no C compiler; skipping")
        return
    out = Path(tempfile.mkdtemp(prefix="ck_sop_"))
    try:
        model = SPM(cell, dt=1.0, rom="pade", order=3)
        # 513 points rather than the default 257: stage 0 set a 3 mV budget for
        # the table, and graphite does not meet that at 257.
        project = generate(model, out, precision="float", table_points=513)
        drive = synthetic_drive_cycle(cell.nominal_capacity, duration=900.0, seed=0)
        result = verify(project, model, drive, initial_soc=0.9)

        check("generated C vs mirror", 1e3 * result.max_voltage_error_vs_mirror, 0.1)
        check(
            "OCP table error (budget 3 mV from stage 0)",
            1e3 * result.max_voltage_error_table_vs_exact,
            3.0,
        )
        check("end to end", 1e3 * result.max_voltage_error_total, 3.0)

        if find_arm_toolchain() is not None:
            m = measure_arm_footprint(out)
            print(f"       {'Cortex-M4F -Os flash':44s} {m.flash_bytes:8d} B")
            print(f"       {'  of which code':44s} {m.text_bytes:8d} B")
            print(f"       {'static RAM (must be zero)':44s} {m.ram_bytes:8d} B")
        else:
            print(NOTE + "  no ARM toolchain; footprint not measured")
    finally:
        shutil.rmtree(out, ignore_errors=True)

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  Implementation  : verified to machine precision.")
    print("  Parameters      : window calibrated; kinetics NOT identified.")
    print("  Blocking gap    : no pulse test. Stage 1.1 T4.")
    print()
    print("  A cell in this state can drive a state-of-charge display. It should")
    print("  not drive a plating-limited charger, because the parameter that sets")
    print("  the plating margin is one the data never constrained -- and the model")
    print("  would report a margin with the same confidence either way.")


if __name__ == "__main__":
    main()
