"""Fitting a model to a real cell, and finding out what the data did not tell you.

The previous example showed a literature parameter set missing a real LG M50 by
tens of millivolts, and that re-solving the electrode balance fixes the static
part but not the behaviour under load. This one fits the rest -- and then asks
the question that matters more than the fit: which of those parameters did the
measurement actually determine?

    python -m cellkernel.data.reference
    python examples/11_identify_from_data.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from cellkernel.data import reference
from cellkernel.identify import DEFAULT_KNOBS, identify
from cellkernel.models import SPM, SPMe
from cellkernel.params import chen2020_nmc811_graphite, fit_stoichiometry_window

# Constant-current segments contain no information above about 0.1 Hz, and the
# zero-order-hold discretisation is exact for piecewise-constant current, so
# fitting at a ten-second step costs nothing and takes twenty times less time
# than fitting at one second. The fitted values agree to within a few percent.
DT = 10.0
RATES = ("cRate_0p1C", "cRate_0p5C", "cRate_1C", "cRate_2C")


def literature_cell():
    try:
        import pybamm

        from cellkernel.params import from_pybamm

        return from_pybamm(pybamm.ParameterValues("Chen2020"))
    except ImportError:
        return chen2020_nmc811_graphite()


def main() -> None:
    try:
        soc, ocv = reference.load_ocv()
    except (FileNotFoundError, OSError):
        print("Dataset not present. Fetch it with:\n  python -m cellkernel.data.reference")
        sys.exit(0)

    slow = reference.load_discharge("T25", "cRate_0p1C")
    capacity = float(np.trapezoid(slow.current, slow.time) / 3600.0)
    cell = fit_stoichiometry_window(literature_cell(), soc, ocv, capacity=capacity)
    print("starting from the literature set with its electrode balance re-solved")
    print(f"  capacity {cell.usable_capacity():.4f} Ah\n")

    segments = [
        (name.replace("cRate_", ""), seg.current, seg.voltage)
        for name, seg in ((name, reference.load_discharge("T25", name, dt=DT)) for name in RATES)
    ]

    for label, builder in (
        ("SPM", lambda p: SPM(p, dt=DT, rom="pade", order=5)),
        ("SPMe", lambda p: SPMe(p, dt=DT, rom="pade", order=5, electrolyte_cells=(6, 4, 6))),
    ):
        started = time.perf_counter()
        report = identify(cell, segments, builder, knobs=DEFAULT_KNOBS)
        print(f"===== {label}  ({time.perf_counter() - started:.0f} s) =====")
        print(report.summary())
        print()

    print(
        "The number to take from this is not the residual. It is the sensitivity\n"
        "column. Against constant-current discharges, the series resistance is the\n"
        "only thing the data determines -- every physical parameter comes back with\n"
        "a sensitivity a few percent of it, and the correlation report catches the\n"
        "positive reaction rate trading against resistance at rho above 0.95.\n"
        "\n"
        "So the fit is real and the parameters are not. A constant current applies\n"
        "one steady excitation, and one excitation cannot separate an ohmic drop\n"
        "from a charge-transfer overpotential from a diffusion limitation: over a\n"
        "whole discharge they all look like a voltage that is lower than it should\n"
        "be. Breaking that apart needs excitation with structure in it -- current\n"
        "pulses, an interrupted rest, or impedance spectroscopy -- because those\n"
        "separate the three by their timescales rather than their magnitudes.\n"
        "\n"
        "A fitting routine that reported only the residual would have presented\n"
        "six confident numbers here, five of which mean nothing."
    )


if __name__ == "__main__":
    main()
