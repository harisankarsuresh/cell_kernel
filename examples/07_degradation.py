"""Predicting ageing, rather than tracking it.

The dual filter watches capacity drift and reports where it is now. This predicts
where it is going, from the two mechanisms that dominate a graphite cell's life:
interphase growth, which is slow and worsens with heat, and lithium plating,
which is fast and worsens with cold.

    python examples/07_degradation.py
"""

from __future__ import annotations

from cellkernel.degradation import DegradationModel
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite

ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)
DAY = 24.0 * 3600.0


def main() -> None:
    cell = chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)
    model = SPM(cell, dt=1.0, rom="pade", order=3)
    ageing = DegradationModel(cell)

    print(f"cell                : {cell.name}")
    print(f"negative interfacial: {ageing.negative_area:.2f} m2")
    print(f"formation film      : {1e9 * ageing.sei.initial_thickness:.1f} nm\n")

    print("Where plating starts: negative electrode potential against lithium")
    print("  (charging; below zero the cell is depositing metal)")
    print(f"  {'soc':>5s}" + "".join(f"{c:>9.1f}C" for c in (0.5, 1.0, 2.0, 3.0)))
    print("  " + "-" * 45)
    for soc in (0.3, 0.6, 0.8, 0.95):
        cells = []
        for c_rate in (0.5, 1.0, 2.0, 3.0):
            current = -c_rate * cell.nominal_capacity
            state = model.initial_state(soc)
            for _ in range(120):
                state = model.step(state, current)
            cells.append(f"{ageing.negative_potential(model, state, current):9.4f}")
        print(f"  {soc:5.2f}" + "".join(cells))
    print(
        "\n  This table is the reason fast charge tapers. It is not the cell voltage"
        "\n  that limits it, and not the bulk state of charge either: it is the"
        "\n  potential at the particle surface, which an equivalent circuit model"
        "\n  does not have and therefore cannot protect against.\n"
    )

    print("Calendar ageing depends on how full you store it (one year, 25 C)")
    for soc in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        state = ageing.initial_state()
        x = model.initial_state(soc)
        for _ in range(365):
            ageing.step(model, x, 0.0, state, DAY)
        print(
            f"  stored at {soc:.0%}   retention {ageing.capacity_retention(state):.4f}"
            f"   film {1e9 * state.film_thickness:6.2f} nm"
        )
    print(
        "\n  A lithiated electrode sits at a lower potential, which drives the"
        "\n  interphase reaction harder. Hence the storage recommendation.\n"
    )

    print("The film limits its own growth, so loss bends to a square root")
    state = ageing.initial_state()
    x = model.initial_state(0.9)
    for day in range(1, 3651):
        outputs = ageing.step(model, x, 0.0, state, DAY)
        if day in (1, 3, 10, 30, 100, 365, 1000, 3650):
            print(
                f"  day {day:5d}   film {1e9 * state.film_thickness:7.2f} nm"
                f"   retention {ageing.capacity_retention(state):.4f}"
                f"   limited by {outputs.sei_limited_by}"
            )
    print(
        "\n  Early growth is kinetically limited and roughly linear; the crossover"
        "\n  here falls in the first week. After it the film is thick enough to"
        "\n  throttle solvent transport, it limits its own growth, and the loss"
        "\n  goes as the square root of time -- 1.8% in the first year, another"
        "\n  1.5% over the next nine.\n"
    )

    print("Cycle ageing, 300 cycles at 1C, by temperature")
    print(
        f"  {'T':>7s} {'interphase':>12s} {'dead metal':>12s} {'retention':>11s} {'dominant':>10s}"
    )
    print("  " + "-" * 56)
    scores = {}
    for temperature_c in (-10.0, 0.0, 10.0, 25.0, 40.0, 55.0):
        temperature = temperature_c + 273.15
        cold_model = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)
        state = ageing.initial_state()
        for _ in range(300):
            ageing.age_over_cycle(cold_model, state, c_rate=1.0, temperature=temperature)
        retention = ageing.capacity_retention(state)
        scores[temperature_c] = retention
        dominant = "plating" if state.lithium_dead > state.lithium_lost else "interphase"
        print(
            f"  {temperature_c:+6.1f}C {1e3 * state.lithium_lost:11.2f}m "
            f"{1e3 * state.lithium_dead:11.2f}m {retention:11.4f} {dominant:>10s}"
        )
    best = max(scores, key=scores.get)
    print(
        f"\n  Best at {best:+.0f} C. Neither extreme is safe, and for opposite"
        "\n  reasons: heat drives the interphase, cold drives plating. That is why"
        "\n  a thermal management system targets a band rather than a ceiling, and"
        "\n  the band moves up as charging gets faster -- at 2C on this cell the"
        "\n  optimum is warmer still, because the plating arm reaches further."
    )


if __name__ == "__main__":
    main()
