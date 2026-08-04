"""Generate an embedded C estimator, compile it, and measure the divergence.

This is the end-to-end claim of the package. It emits a C99 estimator for a
physics-based cell model, builds it with warnings as errors, replays the same
current profile through the compiled binary and through Python, and reports how
far apart they are -- separating code-generation error from the approximations the
generated code makes deliberately.

Requires a C compiler on PATH (gcc or clang), or ``CC`` pointing at one.
"""

from __future__ import annotations

import sys

from cellkernel.codegen import generate
from cellkernel.data import synthetic_drive_cycle
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.verify import find_compiler, verify

OUTPUT = "build/example_estimator"


def main() -> int:
    compiler = find_compiler()
    if compiler is None:
        print("No C compiler found. Install gcc or clang, or set CC.", file=sys.stderr)
        return 2
    print(f"compiler: {compiler}")
    print()

    cell = chen2020_nmc811_graphite()
    model = SPM(cell, dt=1.0, rom="pade", order=3)
    current = synthetic_drive_cycle(
        cell.nominal_capacity, duration=1200.0, dt=model.dt, peak_discharge_rate=2.5
    )

    for precision in ("double", "float"):
        project = generate(model, f"{OUTPUT}_{precision}", precision=precision)
        print("=" * 72)
        print(project)
        print()
        print(project.budget.summary())
        print()
        for mode in ("openloop", "filter"):
            report = verify(project, model, current, initial_soc=0.9, mode=mode)
            print(report.summary())
            print()

    print("=" * 72)
    print("What the three legs mean")
    print()
    print("  The first leg -- generated C against a NumPy mirror with identical loop")
    print("  order -- is the only one that measures the code generator. In double")
    print("  precision it lands at machine epsilon, which is bit-level agreement.")
    print("  In single precision it is a few microvolts, three orders of magnitude")
    print("  below the noise floor of any real measurement front end.")
    print()
    print("  The lookup-table leg is a deliberate approximation, not a defect. It is")
    print("  reported so the trade against flash is explicit: halving table_points")
    print("  halves the flash and roughly quadruples this error.")
    print()
    print("  Keeping them apart is what makes a discrepancy actionable. A few")
    print("  millivolts of table resolution is unremarkable; a few millivolts of")
    print("  arithmetic error means something is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
