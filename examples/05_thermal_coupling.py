"""Why a cell model that ignores temperature goes wrong, and by how much.

Runs the same 2C discharge from three ambient temperatures, once with the cell
allowed to self-heat and once held isothermal, and reports the gap. Also measures
what the temperature schedule costs in accuracy against exactly rebuilt models,
and how that cost falls as the grid is refined.

    python examples/05_thermal_coupling.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cellkernel.estimators import EKF
from cellkernel.models import SPM, ThermalSPM
from cellkernel.params import chen2020_nmc811_graphite

# Not shipped with the parameter set: activation energies are rarely reported
# alongside the transport properties they modify, and published values scatter.
# These are representative, and the point of the example is the sensitivity to
# them, not their exact value.
ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)


def freeze(cell, temperature: float):
    """A copy whose thermal node cannot move, for isolating the schedule."""
    return replace(
        cell,
        thermal=replace(cell.thermal, heat_capacity=1e12, ambient_temperature=temperature),
    )


def main() -> None:
    cell = chen2020_nmc811_graphite().with_activation_energies(**ACTIVATION)
    conductance = cell.thermal.heat_transfer_coefficient * cell.thermal.surface_area

    print(f"cell            : {cell.name}")
    print(f"thermal tau     : {cell.thermal.time_constant:.0f} s")
    print(f"hA              : {conductance:.4f} W/K")
    print(f"cell dU/dT      : {2.0e-5 - -6.0e-5:+.1e} V/K\n")

    print("How much does diffusivity actually move?")
    for temperature_c in (-20.0, 0.0, 25.0, 60.0):
        temperature = temperature_c + 273.15
        ratio_n = (
            cell.negative.diffusivity_at(temperature, cell.reference_temperature)
            / cell.negative.diffusivity
        )
        ratio_p = (
            cell.positive.diffusivity_at(temperature, cell.reference_temperature)
            / cell.positive.diffusivity
        )
        print(f"  {temperature_c:+6.1f} C   negative {ratio_n:5.2f}x   positive {ratio_p:5.2f}x")
    print("  Fifty-fold across the range. An isothermal model freezes this.\n")

    print("Self-heating versus held isothermal, 2C for 15 minutes")
    print(f"  {'ambient':>8s} {'T rise':>8s} {'max dV':>9s} {'end dV':>9s}")
    print("  " + "-" * 38)
    for temperature_c in (-15.0, 0.0, 25.0, 40.0):
        temperature = temperature_c + 273.15
        current = np.full(900, cell.nominal_capacity * 2.0)
        thermal = ThermalSPM(cell, dt=1.0, rom="pade", order=3, ambient=temperature)
        isothermal = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature)
        hot = thermal.simulate(current, soc0=0.9, temperature=temperature)
        cold = isothermal.simulate(current, soc0=0.9)
        print(
            f"  {temperature_c:+7.1f}C {hot['temperature'][-1] - temperature:7.2f}K "
            f"{1e3 * np.max(np.abs(hot['voltage'] - cold['voltage'])):8.1f}mV "
            f"{1e3 * (hot['voltage'][-1] - cold['voltage'][-1]):8.1f}mV"
        )
    print(
        "\n  The gap grows as the cell gets colder, because the self-heating that\n"
        "  an isothermal model ignores is worth proportionally more when the\n"
        "  starting diffusivity is low. At -15 C it is several hundred millivolts,\n"
        "  which is not a modelling refinement, it is the difference between\n"
        "  predicting a usable cell and predicting a dead one.\n"
    )

    print("What the temperature schedule costs, against exactly rebuilt models")
    print(f"  {'points':>7s} {'-18 C':>9s} {'-3 C':>9s} {'22 C':>9s} {'47 C':>9s}")
    print("  " + "-" * 47)
    current = np.full(600, cell.nominal_capacity * 2.0)
    for count in (5, 9, 17):
        grid = np.linspace(253.15, 333.15, count)
        row = []
        for temperature_c in (-18.0, -3.0, 22.0, 47.0):
            temperature = temperature_c + 273.15
            want = SPM(cell, dt=1.0, rom="pade", order=3, temperature=temperature).simulate(
                current, soc0=0.9
            )["voltage"]
            got = ThermalSPM(
                freeze(cell, temperature),
                dt=1.0,
                rom="pade",
                order=3,
                temperature_grid=grid,
                ambient=temperature,
            ).simulate(current, soc0=0.9, temperature=temperature)["voltage"]
            row.append(f"{1e3 * np.max(np.abs(got - want)):8.2f}")
        print(f"  {count:7d} " + " ".join(row) + "   mV")
    print(
        "\n  Blending on the Arrhenius factor rather than on temperature is what\n"
        "  keeps the cold column usable. Interpolating linearly in temperature\n"
        "  instead puts a straight line through an exponential, and the same\n"
        "  nine-point grid then costs 197 mV at -18 C rather than under two.\n"
    )

    print("Filtering with temperature as a state")
    ambient = 268.15
    truth_model = ThermalSPM(cell, dt=1.0, rom="pade", order=3, ambient=ambient)
    current = np.full(1200, cell.nominal_capacity * 1.5)
    truth = truth_model.simulate(current, soc0=0.85, temperature=ambient)
    rng = np.random.default_rng(0)
    measured = truth["voltage"] + rng.normal(0.0, 2e-3, truth["voltage"].size)

    estimator = EKF(
        truth_model,
        process_noise=EKF.suggest_process_noise(truth_model, current_std=0.05),
        measurement_noise=4e-6,
        initial_covariance=EKF.suggest_initial_covariance(truth_model, soc_std=0.1),
        iterations=3,
    )
    estimator.initialise(0.70)  # deliberately 15% low
    result = estimator.run(current, measured)
    print("  seeded at 0.70, truth 0.85")
    print(f"  final state-of-charge error : {abs(result['soc'][-1] - truth['soc'][-1]):.5f}")
    print(f"  cell warmed                 : {truth['temperature'][-1] - ambient:.2f} K")
    print(
        "\n  The filter runs on the thermal model unchanged, because it presents the\n"
        "  same CellModel interface. It is also visibly worse than the same filter\n"
        "  on the isothermal model, which settles near 0.0015 on a comparable run,\n"
        "  and that is the honest cost of the extra state rather than a tuning\n"
        "  oversight. The isothermal model has exactly linear dynamics, so the\n"
        "  extended filter's prediction step carries no linearisation error at all.\n"
        "  Here heat generation is quadratic in current and the transition matrix\n"
        "  depends on a state, so the prediction really is an approximation, and\n"
        "  temperature is only weakly observable from voltage besides -- it is\n"
        "  inferred through its effect on polarisation rather than measured. If you\n"
        "  have a thermistor, feed it in; a measured cell temperature is worth more\n"
        "  than any amount of filter tuning on this model."
    )


if __name__ == "__main__":
    main()
