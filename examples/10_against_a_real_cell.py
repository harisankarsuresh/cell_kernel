"""What a literature parameter set is actually worth on a real cell.

Every other example here compares code against mathematics or against another
model. This one compares against an LG M50 on a cycler, and the result is the
most important number in the project -- because it is the one that separates
"the implementation is right" from "the model describes your cell".

Needs the dataset, which is not distributed with this package:

    python -m cellkernel.data.reference
    python examples/10_against_a_real_cell.py
"""

from __future__ import annotations

import sys

import numpy as np

from cellkernel.data import reference
from cellkernel.models import SPM, SPMe
from cellkernel.params import chen2020_nmc811_graphite, fit_stoichiometry_window


def literature_cell():
    try:
        import pybamm

        from cellkernel.params import from_pybamm

        return from_pybamm(pybamm.ParameterValues("Chen2020"))
    except ImportError:
        return chen2020_nmc811_graphite()


def rmse(model_voltage, measured, floor: float = 2.7) -> float:
    mask = measured > floor
    return 1e3 * float(np.sqrt(np.mean((model_voltage[mask] - measured[mask]) ** 2)))


def main() -> None:
    try:
        soc, measured_ocv = reference.load_ocv()
    except (FileNotFoundError, OSError):
        print("Dataset not present. Fetch it with:\n  python -m cellkernel.data.reference")
        sys.exit(0)

    cell = literature_cell()
    slow = reference.load_discharge("T25", "cRate_0p1C")
    capacity = float(np.trapezoid(slow.current, slow.time) / 3600.0)

    print("cell            : LG M50, 5 Ah nameplate")
    print(f"measured capacity: {capacity:.4f} Ah at 0.1C to 2.5 V")
    print(f"parameter set   : {cell.name}\n")

    print("Open-circuit voltage, literature parameters against measurement")
    error = np.asarray(cell.open_circuit_voltage(soc)) - measured_ocv
    interior = (soc > 0.05) & (soc < 0.95)
    print(f"  5%..95% of charge   rmse {1e3 * np.sqrt(np.mean(error[interior] ** 2)):6.1f} mV")
    print(f"  worst point         {1e3 * np.max(np.abs(error)):6.1f} mV\n")
    print(
        "  For scale: the same code agrees with PyBaMM to 0.26 mV. That number\n"
        "  says the implementation is faithful. This one says the parameters\n"
        "  describe a different unit -- a different sample of the same design,\n"
        "  differently formed and differently aged. They are separate claims and\n"
        "  only the second one limits what you can predict about your cell.\n"
    )

    fitted = fit_stoichiometry_window(cell, soc, measured_ocv, capacity=capacity)
    after = np.asarray(fitted.open_circuit_voltage(soc)) - measured_ocv
    print("After fitting the stoichiometry window -- four numbers, capacity pinned")
    print(f"  5%..95% of charge   rmse {1e3 * np.sqrt(np.mean(after[interior] ** 2)):6.1f} mV")
    print(
        f"  negative window     {cell.negative.stoich_at_0_soc:.4f}"
        f" -> {cell.negative.stoich_at_100_soc:.4f}"
        f"   becomes {fitted.negative.stoich_at_0_soc:.4f}"
        f" -> {fitted.negative.stoich_at_100_soc:.4f}"
    )
    print(f"  capacity            {fitted.usable_capacity():.4f} Ah\n")

    loose = fit_stoichiometry_window(cell, soc, measured_ocv, capacity_weight=0.0)
    loose_error = np.asarray(loose.open_circuit_voltage(soc)) - measured_ocv
    print("  The same fit with capacity left free is a trap:")
    print(
        f"    open-circuit rmse {1e3 * np.sqrt(np.mean(loose_error[interior] ** 2)):5.1f} mV"
        f"   but capacity {loose.usable_capacity():.4f} Ah"
    )
    print(
        "    It buys a better curve by stretching the charge axis, and every\n"
        "    discharge then runs on a mis-scaled clock. Pin the capacity.\n"
    )

    print("Under load at 25 C, root-mean-square voltage error in mV")
    print(f"  {'rate':>6s} {'SPM':>8s} {'SPMe':>8s} {'SPM fit':>8s} {'SPMe fit':>9s}")
    print("  " + "-" * 46)
    for name in ("cRate_0p1C", "cRate_0p5C", "cRate_1C", "cRate_2C"):
        segment = reference.load_discharge("T25", name)
        row = f"  {segment.c_rate:5.1f}C"
        for params in (cell, fitted):
            for builder in (
                lambda p: SPM(p, dt=1.0, rom="pade", order=5),
                lambda p: SPMe(p, dt=1.0, rom="pade", order=5, electrolyte_cells=(6, 4, 6)),
            ):
                run = builder(params).simulate(segment.current, soc0=1.0)
                row += f" {rmse(run['voltage'], segment.voltage):8.1f}"
        print(row)
    print(
        "\n  Two things to take from that table. Resolving the electrolyte helps\n"
        "  on a real cell and helps most at high rate, which is the same\n"
        "  conclusion the Doyle-Fuller-Newman comparison reached and is worth\n"
        "  more here because nothing about it is circular. And fitting the window\n"
        "  helps, but not nearly as much as it helped the open-circuit curve --\n"
        "  what remains is kinetics and transport, which no amount of window\n"
        "  adjustment reaches.\n"
    )

    print("Measured self-heating, which is why the thermal model exists")
    print(f"  {'ambient':>8s} {'rate':>6s} {'rise':>7s}")
    for ambient in ("T0", "T25"):
        for name in ("cRate_0p5C", "cRate_1C", "cRate_2C"):
            segment = reference.load_discharge(ambient, name)
            if segment.temperature is None:
                continue
            print(
                f"  {segment.ambient - 273.15:7.0f}C {segment.c_rate:5.1f}C "
                f"{segment.temperature_rise:6.1f} K"
            )
    print(
        "\n  A 2C discharge raises this cell 33 K from room temperature and 42 K\n"
        "  from freezing -- colder is worse, because sluggish transport dissipates\n"
        "  more. Any isothermal model calibrated at 25 C is describing a cell that\n"
        "  no longer exists a few minutes into the discharge."
    )


if __name__ == "__main__":
    main()
