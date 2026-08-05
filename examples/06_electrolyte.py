"""When the electrolyte stops behaving like a resistor.

A single particle model absorbs the electrolyte into one fitted series
resistance. That works, exactly, right up until it does not. This script finds
the boundary and measures what is on the other side of it.

    python examples/06_electrolyte.py
"""

from __future__ import annotations

from dataclasses import replace

from cellkernel.models import SPM, SPMe
from cellkernel.params import chen2020_nmc811_graphite


def main() -> None:
    base = chen2020_nmc811_graphite()
    spm = SPM(base, dt=1.0, rom="pade", order=3)

    probe = SPMe(base, dt=1.0, rom="pade", order=3)
    electrolyte_resistance = probe.electrolyte_resistance
    # The shipped contact resistance is a lumped fit that already contains the
    # electrolyte. Modelling the electrolyte explicitly means taking it back out,
    # or the same loss is counted twice.
    cell = replace(base, contact_resistance=base.contact_resistance - electrolyte_resistance)
    model = SPMe(cell, dt=1.0, rom="pade", order=3)

    print(f"cell                 : {base.name}")
    print(f"states               : SPM {spm.n_states}, SPMe {model.n_states}")
    print(f"salt diffusion time  : {model.electrolyte.time_constant:.0f} s")
    print(f"electrolyte ohmic    : {1e3 * electrolyte_resistance:.3f} mOhm")
    print(f"lumped fit it replaces: {1e3 * base.contact_resistance:.3f} mOhm\n")

    print("Steady-state salt split across the sandwich, from 1000 mol/m3")
    print(f"  {'rate':>6s} {'negative':>10s} {'positive':>10s} {'lowest':>9s} {'verdict':>15s}")
    print("  " + "-" * 54)
    for c_rate in (0.5, 1.0, 2.0, 3.0, 5.0):
        negative, positive = model.electrolyte.steady_state_split(cell.nominal_capacity * c_rate)
        lowest = (1000.0 + min(negative, positive)) / 1000.0
        verdict = "good" if lowest >= 0.6 else ("degraded" if lowest >= 0.3 else "extrapolating")
        print(f"  {c_rate:5.1f}C {negative:+10.1f} {positive:+10.1f} {lowest:9.3f} {verdict:>15s}")
    print(
        "\n  Transport here is linear, so the split scales exactly with current."
        "\n  Real electrolytes do not: conductivity and diffusivity both fall as"
        "\n  the salt runs out, so past roughly 3C this model is optimistic and"
        "\n  says so rather than quietly extrapolating.\n"
    )

    print("The part a resistance cannot do: it takes time to build")
    print("  2C step from rest, concentration overpotential and its ohmic equivalent")
    current = 2.0 * cell.nominal_capacity
    state = model.initial_state(0.7)
    for step in range(241):
        if step in (0, 2, 5, 10, 20, 40, 80, 160, 240):
            terms = model.decompose(state, current)
            equivalent = -1e3 * terms["concentration_overpotential"] / current
            print(
                f"    t = {step:4d} s   eta_c = "
                f"{1e3 * terms['concentration_overpotential']:8.3f} mV"
                f"   equivalent {equivalent:6.3f} mOhm"
            )
        state = model.step(state, current)
    print(
        "\n  A fitted series resistance has to choose one row of that table. Fit it"
        "\n  to a ten-second pulse and it under-predicts sustained discharge; fit it"
        "\n  to a settled value and it over-predicts every transient. This is the"
        "\n  error that does not go away by fitting harder.\n"
    )

    print("SPMe against SPM after 300 s")
    print(f"  {'rate':>6s} {'V(SPMe)':>9s} {'V(SPM)':>9s} {'gap':>9s} {'eta_c':>9s}")
    print("  " + "-" * 48)
    for c_rate in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0):
        current = cell.nominal_capacity * c_rate
        with_electrolyte = model.initial_state(0.7)
        without = spm.initial_state(0.7)
        for _ in range(300):
            with_electrolyte = model.step(with_electrolyte, current)
            without = spm.step(without, current)
        terms = model.decompose(with_electrolyte, current)
        gap = terms["voltage"] - spm.voltage(without, current)
        print(
            f"  {c_rate:5.1f}C {terms['voltage']:9.4f} "
            f"{spm.voltage(without, current):9.4f} {1e3 * gap:8.1f}mV "
            f"{1e3 * terms['concentration_overpotential']:8.2f}mV"
        )
    print(
        "\n  Below about 1C the two agree to a few millivolts and the simpler model"
        "\n  is the better choice: fewer states, exactly linear, nothing to explain."
        "\n  Use this one when the duty cycle spends real time above that.\n"
    )

    print("Cost")
    print(f"  SPM  : {spm.n_states:2d} states, covariance {spm.n_states**2:3d} entries")
    print(f"  SPMe : {model.n_states:2d} states, covariance {model.n_states**2:3d} entries")
    coarse = SPMe(cell, dt=1.0, rom="pade", order=3, electrolyte_cells=(2, 1, 2))
    fine_split = model.electrolyte.steady_state_split(2.0 * cell.nominal_capacity)[0]
    coarse_split = coarse.electrolyte.steady_state_split(2.0 * cell.nominal_capacity)[0]
    print(
        f"  coarse electrolyte grid (2,1,2): {coarse.n_states} states, "
        f"{100 * abs(coarse_split - fine_split) / abs(fine_split):.1f}% error on the split"
    )
    print(
        "\n  The electrolyte block dominates the state count, and a Kalman filter"
        "\n  pays for it quadratically. Coarsening the grid is the first lever if"
        "\n  that matters; the salt profile is smooth and does not need many cells."
    )


if __name__ == "__main__":
    main()
