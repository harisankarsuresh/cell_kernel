"""Charging protocols that use the model rather than a lookup table.

A constant-current, constant-voltage charge is a compromise made by a controller
that cannot see inside the cell. It picks a current low enough to be safe at the
worst state of charge and temperature it expects, holds it until the terminal
voltage reaches a limit, then tapers. The current is therefore too low for most of
the charge and, at low temperature, can still be too high at the end.

What actually limits charging is not terminal voltage. It is the potential of the
negative electrode against lithium metal: below zero the cell deposits metal
instead of intercalating it, which loses capacity irreversibly and, if the deposit
grows far enough, ends in a short. That potential depends on the *surface*
stoichiometry of the particle, on temperature and on rate, and it is not
observable from the terminals -- which is precisely why an equivalent-circuit
controller has to be conservative.

A physics-based model has it. :func:`plating_limited_current` inverts the
relationship: given a state and a temperature, it returns the largest charging
current that keeps the electrode a stated margin above the plating onset. Feeding
that back as the setpoint gives a charge that is aggressive where it can be and
cautious where it must be, which is most of what a fast-charge protocol is trying
to achieve.

This is the payoff of the degradation model, and the reason plating is reported
as a margin in volts rather than folded into a capacity-fade number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .degradation import DegradationModel

__all__ = [
    "ChargeLimits",
    "plating_limited_current",
    "constant_current_constant_voltage",
    "plating_limited_charge",
]


@dataclass(frozen=True)
class ChargeLimits:
    """Envelope a charging protocol must stay inside.

    Attributes
    ----------
    max_c_rate
        Ceiling on charge rate, whatever the model says is safe. Cell hardware,
        connectors and thermal management all have limits of their own.
    max_voltage
        Terminal voltage cut-off, V.
    plating_margin
        Volts to hold above the plating onset. The potential being compared is a
        model output with its own error, and the consequence of being wrong is
        not symmetric, so this should not be zero.
    min_c_rate
        Rate below which charging is considered finished.
    """

    max_c_rate: float = 3.0
    max_voltage: float = 4.2
    plating_margin: float = 0.02
    min_c_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.max_c_rate <= 0.0:
            raise ValueError("max_c_rate must be positive")
        if self.min_c_rate < 0.0:
            raise ValueError("min_c_rate must be non-negative")
        if self.min_c_rate >= self.max_c_rate:
            raise ValueError("min_c_rate must be below max_c_rate")


def plating_limited_current(
    model,
    ageing: DegradationModel,
    x: np.ndarray,
    limits: ChargeLimits,
    temperature: float | None = None,
    tolerance: float = 1e-4,
) -> float:
    """Largest charging current holding the electrode above the plating onset.

    Returns a magnitude in amperes; the caller applies the sign. Zero means no
    charging current is safe at this state, which happens at high state of charge
    in the cold.

    Found by bisection rather than in closed form. The relationship between
    current and electrode potential runs through the Butler-Volmer inverse and,
    in the thermal model, through a temperature-dependent schedule as well, so
    there is no analytic inverse -- but the potential is monotone decreasing in
    charging current, which is exactly the condition bisection needs. Twenty-odd
    evaluations settle it to well under a milliamp, and each is a voltage
    evaluation rather than a simulation.

    Parameters
    ----------
    model
        Any :class:`~cellkernel.models.base.CellModel`.
    ageing
        Supplies the plating criterion.
    x
        Present cell state.
    limits
        Envelope to respect.
    temperature
        Cell temperature in kelvin. Taken from the model if omitted.
    tolerance
        Bisection tolerance as a fraction of ``max_c_rate``.
    """
    capacity = model.parameters.nominal_capacity
    ceiling = limits.max_c_rate * capacity

    def margin(current_magnitude: float) -> float:
        potential = ageing.negative_potential(model, x, -current_magnitude)
        return potential - limits.plating_margin

    # If the ceiling is already safe there is nothing to solve.
    if margin(ceiling) >= 0.0:
        return ceiling
    # If even a vanishing current plates, no charging is safe.
    if margin(0.0) < 0.0:
        return 0.0

    low, high = 0.0, ceiling
    for _ in range(60):
        if (high - low) <= tolerance * ceiling:
            break
        middle = 0.5 * (low + high)
        if margin(middle) >= 0.0:
            low = middle
        else:
            high = middle
    return low


def constant_current_constant_voltage(
    model,
    limits: ChargeLimits,
    c_rate: float,
    soc0: float = 0.1,
    duration: float = 7200.0,
    temperature: float | None = None,
    ageing: DegradationModel | None = None,
) -> dict[str, np.ndarray]:
    """The conventional protocol, for comparison.

    Constant current until the terminal voltage reaches the limit, then a current
    that decays to hold it there. The voltage hold is a proportional correction
    rather than an exact solve, which is what a real charger does and is adequate
    because the loop is fast against the cell.

    Pass ``ageing`` to have the plating potential recorded along the trajectory.
    It is worth doing: the whole point of the comparison is what happens to the
    negative electrode, and reconstructing that afterwards from rested states
    ignores the concentration gradient and flatters this protocol substantially.
    """
    steps = int(round(duration / model.dt))
    x = model.initial_state(soc0, temperature)
    capacity = model.parameters.nominal_capacity
    current = c_rate * capacity

    keys = ["time", "current", "voltage", "soc"]
    if ageing is not None:
        keys.append("plating_potential")
    out = {key: np.empty(steps) for key in keys}
    for k in range(steps):
        voltage = model.voltage(x, -current)
        if voltage > limits.max_voltage:
            # Back off in proportion to the overshoot, floored at zero.
            current = max(0.0, current - 50.0 * capacity * (voltage - limits.max_voltage))
        out["time"][k] = k * model.dt
        out["current"][k] = -current
        out["voltage"][k] = model.voltage(x, -current)
        out["soc"][k] = model.soc(x)
        if ageing is not None:
            out["plating_potential"][k] = ageing.negative_potential(model, x, -current)
        x = model.step(x, -current)
    return out


def plating_limited_charge(
    model,
    ageing: DegradationModel,
    limits: ChargeLimits,
    soc0: float = 0.1,
    duration: float = 7200.0,
    temperature: float | None = None,
    update_every: int = 10,
) -> dict[str, np.ndarray]:
    """Charge at the largest current the plating criterion allows.

    At every ``update_every`` samples the setpoint is recomputed from the present
    state, then clipped by the voltage limit as well -- the plating criterion
    protects the negative electrode, and the voltage limit protects the positive
    one, so both are needed.

    Recomputing every sample would be wasteful: the limit moves on the timescale
    of the state of charge, which is minutes, while the loop runs at hertz.

    Returns the same keys as
    :func:`constant_current_constant_voltage`, plus ``plating_potential`` so the
    two can be compared on the quantity that actually matters.
    """
    if update_every < 1:
        raise ValueError("update_every must be at least 1")
    steps = int(round(duration / model.dt))
    x = model.initial_state(soc0, temperature)
    capacity = model.parameters.nominal_capacity

    keys = ("time", "current", "voltage", "soc", "plating_potential")
    out = {key: np.empty(steps) for key in keys}
    current = 0.0
    for k in range(steps):
        if k % update_every == 0:
            current = plating_limited_current(model, ageing, x, limits, temperature)
        voltage = model.voltage(x, -current)
        if voltage > limits.max_voltage:
            current = max(0.0, current - 50.0 * capacity * (voltage - limits.max_voltage))
        out["time"][k] = k * model.dt
        out["current"][k] = -current
        out["voltage"][k] = model.voltage(x, -current)
        out["soc"][k] = model.soc(x)
        out["plating_potential"][k] = ageing.negative_potential(model, x, -current)
        x = model.step(x, -current)
    return out
