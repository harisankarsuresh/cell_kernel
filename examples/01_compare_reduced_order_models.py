"""Accuracy and cost of the four solid-diffusion reduced-order models.

Compares each against the exact transfer function of spherical diffusion across
the frequency range that matters for a vehicle duty cycle, and prints the
steady-state surface offset each one predicts against the analytic ``R N / 5D``.

Run with matplotlib installed to also write ``rom_comparison.png``.
"""

from __future__ import annotations

import numpy as np

from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.rom import make_rom

BANDS = {"quasi-steady (<=1)": (-2.0, 0.0), "hour-scale (<=10)": (-2.0, 1.0),
         "pulse (<=100)": (-2.0, 2.0)}


def worst_error(rom, lo: float, hi: float, points: int = 60) -> float:
    theta = rom.time_constant
    worst = 0.0
    for decade in np.logspace(lo, hi, points):
        s = 1j * decade / theta
        exact = rom.exact_transfer_function(s)
        worst = max(worst, abs(rom.transfer_function(s) - exact) / abs(exact))
    return worst


def steady_offset(rom, flux: float) -> float:
    dt = 0.02 * rom.time_constant
    ss = rom.discretise(dt)
    x = ss.initial_state(0.0)
    for _ in range(4000):
        x = ss.step(x, flux)
    surface, average = ss.outputs(x, flux)
    return surface - average


def main() -> None:
    cell = chen2020_nmc811_graphite()
    radius = cell.negative.particle_radius
    diffusivity = cell.negative.diffusivity
    flux = 1e-6

    print(f"Graphite particle: R = {radius:.3e} m, D = {diffusivity:.3e} m2/s")
    print(f"Diffusion time constant R^2/D = {radius**2 / diffusivity:,.0f} s")
    print()

    header = f"{'model':<12}{'states':>7}" + "".join(f"{name:>22}" for name in BANDS)
    print(header)
    print("-" * len(header))

    configurations = [("pade", k) for k in (2, 3, 5, 8)]
    configurations += [("spectral", k) for k in (2, 3, 5, 8)]
    configurations += [("fv", k) for k in (5, 10, 20, 40)]
    configurations += [("poly", 2)]

    for kind, order in configurations:
        rom = make_rom(kind, radius, diffusivity, order=order)
        cells = "".join(f"{worst_error(rom, *band):>22.3e}" for band in BANDS.values())
        print(f"{kind:<12}{rom.n_states:>7}{cells}")

    print()
    exact_offset = radius * flux / (5.0 * diffusivity)
    print(f"Steady-state surface offset under constant flux (exact: {exact_offset:.6f} mol/m3)")
    print()
    print(f"{'model':<12}{'states':>7}{'offset':>16}{'relative error':>18}")
    print("-" * 53)
    for kind, order in [("poly", 2), ("spectral", 4), ("pade", 3), ("pade", 6), ("fv", 10),
                        ("fv", 40)]:
        rom = make_rom(kind, radius, diffusivity, order=order)
        offset = steady_offset(rom, flux)
        error = abs(offset - exact_offset) / exact_offset
        print(f"{kind:<12}{rom.n_states:>7}{offset:>16.6f}{error:>18.3e}")

    print()
    print("The polynomial model reproduces R N / 5D identically with two states,")
    print("because the moment closure is constructed to. The residualised spectral")
    print("model does too, by folding the truncated modes' static gain into a")
    print("feedthrough term. Finite volume converges to it at second order.")

    try:
        plot(radius, diffusivity)
    except ImportError:
        print()
        print("(install matplotlib to also write rom_comparison.png)")


def plot(radius: float, diffusivity: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = radius**2 / diffusivity
    dimensionless = np.logspace(-2, 2.5, 300)
    omega = dimensionless / theta

    fig, (ax_mag, ax_err) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    exact = np.array([make_rom("pade", radius, diffusivity, 3).exact_transfer_function(1j * w)
                      for w in omega])
    ax_mag.loglog(dimensionless, np.abs(exact), "k", lw=2.5, label="exact PDE")

    for kind, order, style in [("poly", 2, "--"), ("spectral", 5, "-."), ("fv", 10, ":"),
                               ("pade", 3, "-"), ("pade", 6, "-")]:
        rom = make_rom(kind, radius, diffusivity, order=order)
        response = np.array([rom.transfer_function(1j * w) for w in omega])
        label = f"{kind} ({rom.n_states} states)"
        ax_mag.loglog(dimensionless, np.abs(response), style, lw=1.4, label=label)
        ax_err.loglog(dimensionless, np.abs(response - exact) / np.abs(exact), style,
                      lw=1.4, label=label)

    ax_mag.set_ylabel(r"$|c_{surf}/N|$")
    ax_mag.set_title("Surface concentration response of a spherical particle")
    ax_mag.legend(fontsize=8)
    ax_mag.grid(alpha=0.3, which="both")

    ax_err.set_xlabel(r"dimensionless frequency  $\omega R^2 / D$")
    ax_err.set_ylabel("relative error")
    ax_err.axhline(1e-3, color="r", ls="--", lw=0.8, alpha=0.6)
    ax_err.text(1.2e-2, 1.3e-3, "0.1%", color="r", fontsize=8)
    ax_err.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("rom_comparison.png", dpi=140)
    print()
    print("wrote rom_comparison.png")


if __name__ == "__main__":
    main()
