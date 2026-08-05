"""Charging as fast as the physics allows, instead of as fast as you dared guess.

A constant-current, constant-voltage charge picks a rate low enough to be safe at
the worst case it expects, then holds it. What actually limits charging is the
negative electrode potential against lithium metal, which is not observable from
the terminals -- so a controller that can only see voltage has to be conservative,
and in the cold even a conservative choice is not safe.

    python examples/08_fast_charge.py
"""

from __future__ import annotations

import numpy as np

from cellkernel.degradation import DegradationModel
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.protocols import (
    ChargeLimits,
    constant_current_constant_voltage,
    plating_limited_charge,
    plating_limited_current,
)

ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)


def main() -> None:
    cell = chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)
    ageing = DegradationModel(cell)
    limits = ChargeLimits(max_c_rate=3.0, max_voltage=4.2, plating_margin=0.01)

    print(f"cell   : {cell.name}")
    print(
        f"limits : {limits.max_c_rate:.0f}C ceiling, {limits.max_voltage} V, "
        f"{1e3 * limits.plating_margin:.0f} mV plating margin\n"
    )

    print("The current the model says is safe, in C-rate")
    print(f"  {'soc':>6s}" + "".join(f"{t:>9.0f}C" for t in (-10, 0, 10, 25, 40)))
    print("  " + "-" * 56)
    for soc in (0.1, 0.3, 0.5, 0.7, 0.9):
        cells = []
        for temperature_c in (-10, 0, 10, 25, 40):
            model = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature_c + 273.15)
            allowed = plating_limited_current(model, ageing, model.initial_state(soc), limits)
            cells.append(f"{allowed / cell.nominal_capacity:9.2f}")
        print(f"  {soc:6.2f}" + "".join(cells))
    print(
        "\n  This is the table a fast-charge controller wants and cannot measure."
        "\n  It falls with state of charge because the electrode fills, and with"
        "\n  cold because transport slows. A single fixed rate has to sit under"
        "\n  the worst cell in this grid.\n"
    )

    print("Charging from 10%, two hours available")
    print(
        f"  {'T':>7s} {'protocol':>18s} {'to 80%':>9s}"
        f" {'end soc':>8s} {'min phi':>9s} {'plating':>9s}"
    )
    print("  " + "-" * 66)
    for temperature_c in (-5.0, 10.0, 25.0):
        temperature = temperature_c + 273.15
        model = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)
        runs = [
            (
                f"CCCV {rate:.0f}C",
                constant_current_constant_voltage(
                    model, limits, rate, 0.1, 7200.0, temperature, ageing=ageing
                ),
            )
            for rate in (1.0, 2.0, 3.0)
        ]
        runs.append(
            (
                "plating-limited",
                plating_limited_charge(model, ageing, limits, 0.1, 7200.0, temperature),
            )
        )
        for label, result in runs:
            potential = result["plating_potential"]
            reached = np.where(result["soc"] >= 0.8)[0]
            to_80 = f"{result['time'][reached[0]] / 60:8.1f}m" if reached.size else "     n/a"
            minutes_plating = float(np.sum(potential < 0.0)) / 60.0
            print(
                f"  {temperature_c:+6.1f}C {label:>18s} {to_80} "
                f"{result['soc'][-1]:8.3f} {potential.min():+9.4f} "
                f"{minutes_plating:8.1f}m"
            )
        print()

    print(
        "  Read the cold block first. At -5 C the conventional charge deposits\n"
        "  metal at every rate on offer, including 1C, for something like ten\n"
        "  minutes of every charge. Nothing in its terminal measurements tells it\n"
        "  so. The guarded protocol reaches a comparable state of charge in a\n"
        "  comparable time and never crosses the onset.\n"
    )
    print(
        "  At 25 C the picture reverses: nothing plates, and the guarded protocol\n"
        "  is slower than simply charging at 3C. That is the honest trade. It buys\n"
        "  a guarantee, and a guarantee costs something exactly when it was not\n"
        "  needed. Whether that is worth it depends on how much of the year the\n"
        "  pack spends cold."
    )


if __name__ == "__main__":
    main()
