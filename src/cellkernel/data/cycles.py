"""Synthetic current profiles, in amperes with discharge positive."""

from __future__ import annotations

import numpy as np

__all__ = [
    "cccv_charge",
    "constant_current",
    "hppc_pulses",
    "rest",
    "synthetic_drive_cycle",
]


def constant_current(amplitude: float, duration: float, dt: float = 1.0) -> np.ndarray:
    """A constant current of ``amplitude`` amperes for ``duration`` seconds."""
    return np.full(max(int(round(duration / dt)), 1), float(amplitude))


def rest(duration: float, dt: float = 1.0) -> np.ndarray:
    """Zero current for ``duration`` seconds."""
    return constant_current(0.0, duration, dt)


def hppc_pulses(
    capacity: float,
    discharge_rate: float = 1.0,
    charge_rate: float = 0.75,
    pulse: float = 10.0,
    relax: float = 40.0,
    between: float = 300.0,
    steps: int = 9,
    dt: float = 1.0,
) -> np.ndarray:
    """A hybrid pulse power characterisation profile.

    Each step is a discharge pulse, a rest, a charge pulse, a rest, then a longer
    constant-current discharge to move to the next state of charge.

    This is the workhorse test for identifying resistance and short-timescale
    dynamics, because it separates timescales by construction: the instantaneous
    step is ohmic, the pulse decay is charge transfer, and the long rest exposes
    diffusion. A single constant-current discharge cannot separate them, which is
    why fitting to one tends to produce parameter sets that reproduce that
    discharge and nothing else.

    Parameters
    ----------
    capacity
        Cell capacity in ampere hours, used to convert C-rates to amperes.
    discharge_rate, charge_rate
        Pulse magnitudes as C-rates.
    pulse
        Pulse duration in seconds.
    relax
        Rest after each pulse, in seconds.
    between
        Duration of the constant-current segment between pulse pairs.
    steps
        Number of state-of-charge points.
    """
    segments = []
    for _ in range(steps):
        segments.append(constant_current(discharge_rate * capacity, pulse, dt))
        segments.append(rest(relax, dt))
        segments.append(constant_current(-charge_rate * capacity, pulse, dt))
        segments.append(rest(relax, dt))
        segments.append(constant_current(0.5 * capacity, between, dt))
        segments.append(rest(relax, dt))
    return np.concatenate(segments)


def cccv_charge(
    capacity: float, rate: float = 1.0, cc_duration: float = 2400.0, taper: float = 1800.0,
    dt: float = 1.0
) -> np.ndarray:
    """A constant-current, tapering-current charge profile.

    The taper is an exponential decay standing in for the constant-voltage phase.
    A true constant-voltage phase requires closing a loop on the model, which
    belongs in a simulation driver rather than in a fixed profile; the decay
    reproduces the shape well enough to exercise the parts of a model that matter
    here, and it keeps the profile independent of whatever model consumes it.
    """
    cc = constant_current(-rate * capacity, cc_duration, dt)
    n_taper = max(int(round(taper / dt)), 1)
    tau = n_taper / 4.0
    cv = -rate * capacity * np.exp(-np.arange(n_taper) / tau)
    return np.concatenate([cc, cv])


def synthetic_drive_cycle(
    capacity: float,
    duration: float = 1800.0,
    dt: float = 1.0,
    peak_discharge_rate: float = 3.0,
    peak_regen_rate: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """A drive-cycle-like current profile with realistic spectral content.

    Not a reproduction of any standard cycle -- WLTP and the US federal schedules
    are defined as vehicle speed traces, and converting one to cell current needs a
    vehicle model, a gearbox, an inverter efficiency map and a pack configuration,
    none of which belong in a cell-level library.

    What matters for exercising a diffusion model is the *spectrum*, and that is
    what this reproduces: power drawn in bursts of a few seconds to a minute,
    superimposed on slower trends, with regenerative braking events and idle
    periods. The result is broadband over roughly 0.01 to 0.5 Hz, which straddles
    the surface-diffusion timescale of a typical electrode particle and is
    therefore where reduced-order models are actually distinguished from one
    another.

    Constructed by summing sinusoids at incommensurate periods, clipping to the
    requested rate limits and gating with idle intervals -- reproducible from
    ``seed``, and free of the discontinuities that a purely piecewise-constant
    random profile would introduce.
    """
    rng = np.random.default_rng(seed)
    n = max(int(round(duration / dt)), 1)
    t = np.arange(n) * dt

    signal = np.zeros(n)
    for period in (7.0, 13.0, 29.0, 61.0, 137.0, 313.0):
        signal += rng.uniform(0.4, 1.0) * np.sin(
            2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi)
        )
    signal /= np.max(np.abs(signal))

    # Bias towards discharge, as any real duty cycle is: regeneration recovers
    # only a fraction of the energy spent.
    signal = 0.35 + 0.65 * signal

    # Idle gating: a few stops of tens of seconds each.
    gate = np.ones(n)
    k = 0
    while k < n:
        run = int(rng.integers(int(120 / dt), int(420 / dt)))
        stop = int(rng.integers(int(10 / dt), int(60 / dt)))
        gate[k + run : k + run + stop] = 0.0
        k += run + stop
    signal *= gate

    current = capacity * np.where(
        signal >= 0.0, peak_discharge_rate * signal, peak_regen_rate * signal
    )
    return np.clip(
        current, -peak_regen_rate * capacity, peak_discharge_rate * capacity
    )
