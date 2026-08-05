"""Command-line interface.

Each subcommand maps onto one thing a person actually wants to do::

    cellkernel roms                       compare reduced-order models
    cellkernel generate build/est         emit a C estimator
    cellkernel verify build/est           compile it and check it against Python
    cellkernel simulate --out run.csv     produce a synthetic record
    cellkernel charge --temperature 263   find the safe charging rate
    cellkernel age --cycles 500           project capacity fade

Run ``cellkernel <command> --help`` for options.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .data import synthetic_drive_cycle
from .models import SPM
from .params import CellParameters, chen2020_nmc811_graphite, lfp_graphite
from .rom import make_rom

_CHEMISTRIES = {"nmc": chen2020_nmc811_graphite, "lfp": lfp_graphite}
_ROMS = ("pade", "spectral", "fv", "poly")


def _cell(name: str) -> CellParameters:
    try:
        return _CHEMISTRIES[name]()
    except KeyError:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"unknown chemistry {name!r}") from None


def _build_model(args: argparse.Namespace) -> SPM:
    return SPM(
        _cell(args.chemistry),
        dt=args.dt,
        rom=args.rom,
        order=args.order,
        temperature=args.temperature,
    )


def _add_model_options(parser: argparse.ArgumentParser, temperature: bool = True) -> None:
    parser.add_argument(
        "--chemistry", choices=sorted(_CHEMISTRIES), default="nmc", help="built-in parameter set"
    )
    parser.add_argument(
        "--rom", choices=_ROMS, default="pade", help="diffusion reduced-order model"
    )
    parser.add_argument("--order", type=int, default=3, help="states per electrode")
    parser.add_argument("--dt", type=float, default=1.0, help="sample period in seconds")
    if temperature:
        # Suppressed by the subcommands that sweep temperature and therefore need
        # to take a list rather than a single value.
        parser.add_argument(
            "--temperature", type=float, default=None, help="isothermal temperature in kelvin"
        )


def _cmd_roms(args: argparse.Namespace) -> int:
    """Print accuracy against the exact PDE and state count for each model."""
    cell = _cell(args.chemistry)
    radius = cell.negative.particle_radius
    diffusivity = cell.negative.diffusivity
    theta = radius**2 / diffusivity

    print(f"{args.chemistry} negative electrode: R = {radius:.3e} m, D = {diffusivity:.3e} m2/s")
    print(f"diffusion time constant R^2/D = {theta:,.0f} s")
    print()
    print("Worst relative error in the surface-concentration response, by")
    print("dimensionless frequency band (omega * R^2 / D):")
    print()
    bands = [(-2.0, 0.0), (-2.0, 1.0), (-2.0, 2.0)]
    header = f"  {'model':<14}{'states':>7}" + "".join(f"{f'<=1e{int(hi)}':>12}" for _, hi in bands)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kind in _ROMS:
        for order in (2, 3, 5, 8):
            if kind == "poly" and order != 2:
                continue
            rom = make_rom(kind, radius, diffusivity, order=order)
            cells = [f"{_worst_error(rom, lo, hi):>12.2e}" for lo, hi in bands]
            print(f"  {kind:<14}{rom.n_states:>7}" + "".join(cells))
    print()
    print("Pade approximates the transfer function directly and buys the most")
    print("accuracy per state. Spectral has a diagonal state matrix, so it is the")
    print("cheapest per state to evaluate. Finite volume is the only one that keeps")
    print("a resolved interior profile. Polynomial is the cheapest that still gets")
    print("the steady-state surface offset exactly right.")
    return 0


def _worst_error(rom, lo_decade: float, hi_decade: float, points: int = 30) -> float:
    theta = rom.time_constant
    worst = 0.0
    for decade in np.logspace(lo_decade, hi_decade, points):
        s = 1j * decade / theta
        exact = rom.exact_transfer_function(s)
        worst = max(worst, abs(rom.transfer_function(s) - exact) / abs(exact))
    return worst


def _cmd_generate(args: argparse.Namespace) -> int:
    from .codegen import generate, generate_scheduled

    if getattr(args, "scheduled", False):
        from .models import ThermalSPM

        cell = _cell(args.chemistry)
        # A schedule over a cell with no activation energies would interpolate
        # between identical matrices, so representative values are supplied when
        # the parameter set leaves them at zero. Fit your own before shipping.
        if cell.negative.diffusion_activation_energy == 0.0:
            cell = cell.with_activation_energies(
                diffusion_negative=35_000.0,
                diffusion_positive=30_000.0,
                reaction_negative=35_000.0,
                reaction_positive=17_800.0,
            )
            print(
                "note: parameter set carries no activation energies; using "
                "representative literature values for the schedule.\n",
                file=sys.stderr,
            )
        model = ThermalSPM(cell, dt=args.dt, rom=args.rom, order=args.order)
        project = generate_scheduled(
            model,
            args.output,
            precision=args.precision,
            table_points=args.table_points,
            max_c_rate=args.max_c_rate,
        )
    else:
        project = generate(
            _build_model(args),
            args.output,
            precision=args.precision,
            table_points=args.table_points,
            max_c_rate=args.max_c_rate,
        )
    print(project)
    print()
    print(project.budget.summary())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .codegen import generate
    from .verify import find_compiler, verify

    compiler = args.compiler or find_compiler()
    if compiler is None:
        print(
            "no C compiler found. Install gcc or clang, or set CC to point at one.",
            file=sys.stderr,
        )
        return 2

    model = _build_model(args)
    project = generate(
        model,
        args.output,
        precision=args.precision,
        table_points=args.table_points,
        max_c_rate=args.max_c_rate,
    )
    current = synthetic_drive_cycle(
        model.parameters.nominal_capacity,
        duration=args.duration,
        dt=args.dt,
        peak_discharge_rate=args.max_c_rate,
    )
    failures = 0
    for mode in ("openloop", "filter"):
        report = verify(project, model, current, initial_soc=args.soc, compiler=compiler, mode=mode)
        print(report.summary())
        print()
        failures += 0 if report.passed else 1
    print(project.budget.summary())
    return 1 if failures else 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    model = _build_model(args)
    current = synthetic_drive_cycle(
        model.parameters.nominal_capacity,
        duration=args.duration,
        dt=args.dt,
        peak_discharge_rate=args.max_c_rate,
    )
    result = model.simulate(current, soc0=args.soc)
    if args.noise > 0.0:
        rng = np.random.default_rng(args.seed)
        result["voltage"] = result["voltage"] + rng.normal(0.0, args.noise, result["voltage"].size)

    columns = ("time", "current", "voltage", "soc")
    destination = Path(args.out) if args.out else None
    handle = destination.open("w", newline="", encoding="utf-8") if destination else sys.stdout
    try:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in zip(*(result[key] for key in columns), strict=False):
            writer.writerow(f"{value:.9g}" for value in row)
    finally:
        if destination:
            handle.close()
    if destination:
        print(
            f"wrote {result['time'].size} samples to {destination} "
            f"(soc {result['soc'][0]:.4f} -> {result['soc'][-1]:.4f}, "
            f"voltage {result['voltage'].min():.4f} .. {result['voltage'].max():.4f} V)"
        )
    return 0


#: Representative activation energies, used by the subcommands that need
#: temperature-dependent transport. The built-in parameter sets deliberately ship
#: without them -- see ``CellParameters.with_activation_energies`` -- so the CLI
#: supplies a set and says so rather than silently modelling a cell whose
#: diffusivity does not respond to temperature at all.
_DEFAULT_ACTIVATION = dict(
    diffusion_negative=35_000.0,
    diffusion_positive=30_000.0,
    reaction_negative=35_000.0,
    reaction_positive=17_800.0,
)


def _thermal_cell(name: str) -> CellParameters:
    cell = _cell(name)
    if cell.negative.diffusion_activation_energy > 0.0:
        return cell
    print(
        "note: this parameter set ships no activation energies, so representative "
        "ones are being used. Supply your own with with_activation_energies() "
        "before trusting any number that depends on temperature.",
        file=sys.stderr,
    )
    return cell.with_activation_energies(**_DEFAULT_ACTIVATION)


def _cmd_charge(args: argparse.Namespace) -> int:
    from .degradation import DegradationModel
    from .protocols import ChargeLimits, plating_limited_current

    cell = _thermal_cell(args.chemistry)
    ageing = DegradationModel(cell)
    limits = ChargeLimits(
        max_c_rate=args.max_c_rate,
        max_voltage=cell.voltage_limits[1],
        plating_margin=args.margin,
    )
    temperatures = args.temperature or [263.15, 273.15, 283.15, 298.15, 313.15]

    print(f"{cell.name}: largest charge rate holding the electrode")
    print(f"{1e3 * args.margin:.0f} mV above the lithium plating onset, in C\n")
    print(f"  {'soc':>6s}" + "".join(f"{t - 273.15:>9.0f}C" for t in temperatures))
    print("  " + "-" * (6 + 10 * len(temperatures)))
    for soc in (0.1, 0.3, 0.5, 0.7, 0.9):
        cells = []
        for temperature in temperatures:
            model = SPM(cell, dt=args.dt, rom=args.rom, order=args.order, temperature=temperature)
            allowed = plating_limited_current(model, ageing, model.initial_state(soc), limits)
            cells.append(f"{allowed / cell.nominal_capacity:9.2f}")
        print(f"  {soc:6.2f}" + "".join(cells))
    print(
        "\nA fixed-rate charger has to sit under the worst entry in this grid.\n"
        "The limit is set by the potential at the particle surface, which is not\n"
        "observable from the terminals -- which is the whole reason for the model."
    )
    return 0


def _cmd_age(args: argparse.Namespace) -> int:
    from .degradation import DegradationModel

    cell = _thermal_cell(args.chemistry)
    ageing = DegradationModel(cell)
    temperatures = args.temperature or [263.15, 283.15, 298.15, 313.15, 328.15]

    print(f"{cell.name}: {args.cycles} cycles at {args.c_rate:g}C\n")
    print(
        f"  {'T':>8s} {'interphase':>12s} {'dead metal':>12s} {'retention':>11s} {'dominant':>11s}"
    )
    print("  " + "-" * 58)
    best = None
    for temperature in temperatures:
        model = SPM(cell, dt=args.dt, rom=args.rom, order=args.order, temperature=temperature)
        state = ageing.initial_state()
        for _ in range(args.cycles):
            ageing.age_over_cycle(model, state, c_rate=args.c_rate, temperature=temperature)
        retention = ageing.capacity_retention(state)
        dominant = "plating" if state.lithium_dead > state.lithium_lost else "interphase"
        print(
            f"  {temperature - 273.15:+7.1f}C {1e3 * state.lithium_lost:11.2f}m "
            f"{1e3 * state.lithium_dead:11.2f}m {retention:11.4f} {dominant:>11s}"
        )
        if best is None or retention > best[1]:
            best = (temperature, retention)
    assert best is not None
    print(
        f"\nBest at {best[0] - 273.15:+.0f} C. Neither extreme is safe, and for opposite\n"
        "reasons: heat drives interphase growth, cold drives plating. Parameters\n"
        "here are representative rather than fitted, so read the shape of the\n"
        "curve and not the absolute numbers."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellkernel",
        description=(
            "Physics-based lithium-ion state estimation, from reduced-order "
            "electrochemistry to verified embedded C."
        ),
    )
    parser.add_argument("--version", action="version", version=f"cellkernel {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    roms = subparsers.add_parser("roms", help="compare diffusion reduced-order models")
    roms.add_argument("--chemistry", choices=sorted(_CHEMISTRIES), default="nmc")
    roms.set_defaults(func=_cmd_roms)

    generate_cmd = subparsers.add_parser("generate", help="emit a C estimator")
    generate_cmd.add_argument("output", help="output directory")
    _add_model_options(generate_cmd)
    generate_cmd.add_argument("--precision", choices=("float", "double"), default="double")
    generate_cmd.add_argument("--table-points", type=int, default=257)
    generate_cmd.add_argument(
        "--max-c-rate", type=float, default=3.0, help="highest C-rate the tables must cover"
    )
    generate_cmd.add_argument(
        "--scheduled",
        action="store_true",
        help="emit a temperature-scheduled estimator that takes measured "
        "temperature as an input, valid across a range rather than at one point",
    )
    generate_cmd.set_defaults(func=_cmd_generate)

    verify_cmd = subparsers.add_parser(
        "verify", help="generate, compile and compare against Python"
    )
    verify_cmd.add_argument("output", help="output directory")
    _add_model_options(verify_cmd)
    verify_cmd.add_argument("--precision", choices=("float", "double"), default="double")
    verify_cmd.add_argument("--table-points", type=int, default=257)
    verify_cmd.add_argument("--max-c-rate", type=float, default=3.0)
    verify_cmd.add_argument("--duration", type=float, default=900.0, help="profile length, seconds")
    verify_cmd.add_argument("--soc", type=float, default=0.9, help="initial state of charge")
    verify_cmd.add_argument("--compiler", default=None, help="C compiler to use")
    verify_cmd.set_defaults(func=_cmd_verify)

    simulate = subparsers.add_parser("simulate", help="write a synthetic record as CSV")
    _add_model_options(simulate)
    simulate.add_argument("--out", default=None, help="output CSV (default: stdout)")
    simulate.add_argument("--duration", type=float, default=1800.0)
    simulate.add_argument("--max-c-rate", type=float, default=2.0)
    simulate.add_argument("--soc", type=float, default=0.95)
    simulate.add_argument("--noise", type=float, default=0.0, help="voltage noise std, volts")
    simulate.add_argument("--seed", type=int, default=0)
    simulate.set_defaults(func=_cmd_simulate)

    charge = subparsers.add_parser("charge", help="largest charge rate that avoids lithium plating")
    _add_model_options(charge, temperature=False)
    charge.add_argument(
        "--temperature",
        type=float,
        nargs="*",
        default=None,
        help="temperatures in kelvin (default: a spread from -10 to 40 C)",
    )
    charge.add_argument("--max-c-rate", type=float, default=3.0, help="hardware ceiling")
    charge.add_argument(
        "--margin", type=float, default=0.01, help="volts to hold above the plating onset"
    )
    charge.set_defaults(func=_cmd_charge)

    age = subparsers.add_parser("age", help="project capacity fade over cycles")
    _add_model_options(age, temperature=False)
    age.add_argument(
        "--temperature",
        type=float,
        nargs="*",
        default=None,
        help="temperatures in kelvin (default: a spread from -10 to 55 C)",
    )
    age.add_argument("--cycles", type=int, default=300)
    age.add_argument("--c-rate", type=float, default=1.0)
    age.set_defaults(func=_cmd_age)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
