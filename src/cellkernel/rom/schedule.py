"""Gain scheduling of a discretised diffusion model over temperature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DiffusionROM, DiscreteStateSpace

__all__ = ["ScheduledStateSpace", "schedule_over_temperature"]

#: Molar gas constant, J mol-1 K-1. Defined here rather than imported from
#: :mod:`cellkernel.params` so that :mod:`cellkernel.rom` stays free of any
#: dependency on the parameter layer.
GAS_CONSTANT = 8.31446261815324

#: Kept in step with :mod:`cellkernel.params`. Defined here rather than imported
#: because ``cellkernel.rom`` deliberately does not depend on the parameter layer:
#: a reduced-order diffusion model should be usable knowing only a radius and a
#: diffusivity.
_TEMPERATURE_FLOOR = 173.15
_TEMPERATURE_CEILING = 373.15


@dataclass(frozen=True)
class ScheduledStateSpace:
    """A discretised diffusion model evaluated at a grid of temperatures.

    Solid diffusivity follows an Arrhenius law, so the state matrices depend on
    temperature -- and they depend on it through a matrix exponential, which is
    not something a microcontroller can evaluate. Every practical
    temperature-aware estimator therefore schedules: the matrices are computed
    offline at a handful of temperatures and blended online.

    This class holds that schedule and does the blending. Interpolation is linear
    in temperature between bracketing grid points and clamped outside the grid.

    Blending on diffusivity, not on temperature
    -------------------------------------------
    The obvious scheme -- linear in temperature between bracketing points -- is
    measurably poor at low temperature, and the reason is worth stating because
    it decides the design.

    At any sample period short against the diffusion time constant, and
    ``dt / (R^2/D)`` is of order 1e-4 for a realistic cell, the discrete matrix
    is close to :math:`A \\approx I + A_c(D)\\,\\Delta t`, and the generator
    :math:`A_c` is *proportional to* ``D``. So the matrices are very nearly
    affine in diffusivity -- and diffusivity is exponential in :math:`1/T`.
    Interpolating linearly in temperature therefore fits a straight line through
    an exponential, and does so worst at the cold end where the exponential is
    steepest. Measured on a graphite electrode with a 35 kJ mol-1 activation
    energy, a 10 K grid gave 5 mV of voltage error at most temperatures and
    197 mV at -18 C.

    Blending on the Arrhenius factor instead makes the interpolation exact in the
    limit where that affine relationship holds, which removes the low-temperature
    penalty almost entirely and leaves an error that is genuinely second order in
    the grid spacing. The online cost is one exponential per electrode per step,
    against a saving of roughly a factor of four in grid points for the same
    accuracy -- a good trade in flash, and a better one in honesty.

    When ``activation_energy`` is zero the factor is constant, there is nothing to
    blend on, and the class falls back to linear interpolation in temperature.

    Attributes
    ----------
    temperatures
        Grid points in kelvin, strictly increasing.
    systems
        One :class:`~cellkernel.rom.base.DiscreteStateSpace` per grid point.
    activation_energy
        Arrhenius activation energy of the diffusivity these systems were built
        with, J mol-1. Zero disables factor blending.
    reference_temperature
        Temperature at which the diffusivity is quoted, kelvin.
    """

    temperatures: np.ndarray
    systems: tuple[DiscreteStateSpace, ...]
    activation_energy: float = 0.0
    reference_temperature: float = 298.15
    #: Arrhenius factor at each grid point, precomputed in ``__post_init__``.
    _grid_factors: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        temps = np.asarray(self.temperatures, dtype=float).reshape(-1)
        if temps.size != len(self.systems):
            raise ValueError("temperatures and systems must have equal length")
        if temps.size < 2:
            raise ValueError("need at least two grid points to schedule")
        if np.any(np.diff(temps) <= 0.0):
            raise ValueError("temperatures must be strictly increasing")
        object.__setattr__(self, "temperatures", temps)
        # Grid factors never change, and recomputing two exponentials per blend
        # was a measurable share of the runtime before they were cached here.
        if self.activation_energy == 0.0:
            grid_factors = np.ones_like(temps)
        else:
            grid_factors = np.exp(
                self.activation_energy
                / GAS_CONSTANT
                * (1.0 / self.reference_temperature - 1.0 / temps)
            )
        object.__setattr__(self, "_grid_factors", grid_factors)

    def factor(self, temperature: float) -> float:
        """Arrhenius scaling of diffusivity relative to the reference temperature.

        Clamped to a physical temperature range for the same reason as
        :func:`cellkernel.params._arrhenius`: an unscented filter carrying
        temperature as a state will place sigma points outside it, and ``1/T``
        then overflows the exponential and puts infinities into the covariance.
        Only the blend weight is affected, and the weight is clipped to ``[0, 1]``
        immediately afterwards regardless.
        """
        if self.activation_energy == 0.0:
            return 1.0
        bounded = min(max(float(temperature), _TEMPERATURE_FLOOR), _TEMPERATURE_CEILING)
        return float(
            np.exp(
                self.activation_energy
                / GAS_CONSTANT
                * (1.0 / self.reference_temperature - 1.0 / bounded)
            )
        )

    @property
    def n_states(self) -> int:
        return self.systems[0].n_states

    @property
    def dt(self) -> float:
        return self.systems[0].dt

    @property
    def bounds(self) -> tuple[float, float]:
        """Lowest and highest scheduled temperature, in kelvin."""
        return float(self.temperatures[0]), float(self.temperatures[-1])

    def weights(self, temperature: float) -> tuple[int, int, float]:
        """Return ``(lower, upper, blend)`` for a temperature.

        The bracket is found on temperature, which is monotone and cheap to
        search. The blend fraction is computed on the Arrhenius factor, for the
        reason given in the class docstring.

        ``blend`` is clamped to ``[0, 1]`` so that a temperature outside the grid
        holds the nearest endpoint rather than extrapolating. Clamping is the
        safe failure: an estimator that meets an out-of-range temperature should
        degrade to its nearest calibrated point, not extrapolate an exponential
        it was never fitted over.
        """
        temps = self.temperatures
        upper = int(np.searchsorted(temps, float(temperature), side="right"))
        upper = min(max(upper, 1), temps.size - 1)
        lower = upper - 1

        if self.activation_energy == 0.0:
            span = temps[upper] - temps[lower]
            blend = (float(temperature) - temps[lower]) / span
        else:
            factors = self._grid_factors
            low = factors[lower]
            span = factors[upper] - low
            if abs(span) < 1e-300:  # pragma: no cover - degenerate grid
                blend = 0.0
            else:
                blend = (self.factor(float(temperature)) - low) / span
        return lower, upper, float(min(max(blend, 0.0), 1.0))

    def at(self, temperature: float) -> DiscreteStateSpace:
        """Interpolated state-space system at ``temperature``."""
        lower, upper, blend = self.weights(temperature)
        a, b = self.systems[lower], self.systems[upper]
        return DiscreteStateSpace(
            A=(1.0 - blend) * a.A + blend * b.A,
            B=(1.0 - blend) * a.B + blend * b.B,
            C=(1.0 - blend) * a.C + blend * b.C,
            D=(1.0 - blend) * a.D + blend * b.D,
            dt=a.dt,
            x0_from_uniform=(1.0 - blend) * a.x0_from_uniform + blend * b.x0_from_uniform,
        )

    def blend_derivative(self, temperature: float) -> float:
        """``d(blend)/dT`` at ``temperature``, zero outside the grid.

        With factor blending the interpolation weight is a nonlinear function of
        temperature, so this is not simply one over the interval width. Using
        :math:`f'(T) = f(T) E_a / (R_g T^2)`,

        .. math::

            \\frac{d\\,\\mathrm{blend}}{dT}
                = \\frac{f(T)\\,E_a}{R_g T^{2}\\,(f_{hi} - f_{lo})}.
        """
        temps = self.temperatures
        temperature = float(temperature)
        if temperature <= temps[0] or temperature >= temps[-1]:
            return 0.0
        lower, upper, _ = self.weights(temperature)
        if self.activation_energy == 0.0:
            return 1.0 / (temps[upper] - temps[lower])
        factors = self._grid_factors
        low = factors[lower]
        span = factors[upper] - low
        if abs(span) < 1e-300:  # pragma: no cover - degenerate grid
            return 0.0
        derivative = (
            self.factor(temperature)
            * self.activation_energy
            / (GAS_CONSTANT * temperature * temperature)
        )
        return float(derivative / span)

    def slope(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dA/dT, dB/dT)`` of the interpolant at ``temperature``.

        Zero outside the grid, where :meth:`weights` clamps. This is the
        sensitivity an extended Kalman filter needs in order to know that a
        change in temperature moves the diffusion state, and it is available in
        closed form because the interpolation is linear in the blend weight even
        though the weight itself is not linear in temperature.
        """
        temps = self.temperatures
        if temperature <= temps[0] or temperature >= temps[-1]:
            n = self.n_states
            return np.zeros((n, n)), np.zeros((n, 1))
        lower, upper, _ = self.weights(temperature)
        rate = self.blend_derivative(temperature)
        a, b = self.systems[lower], self.systems[upper]
        return (b.A - a.A) * rate, (b.B - a.B) * rate

    def output_slope(self, temperature: float) -> tuple[np.ndarray, float]:
        """Return ``(dC_surface/dT, dD_surface/dT)`` for the surface row."""
        temps = self.temperatures
        if temperature <= temps[0] or temperature >= temps[-1]:
            return np.zeros(self.n_states), 0.0
        lower, upper, _ = self.weights(temperature)
        rate = self.blend_derivative(temperature)
        a, b = self.systems[lower], self.systems[upper]
        return (b.C[0] - a.C[0]) * rate, float((b.D[0, 0] - a.D[0, 0]) * rate)

    def interpolation_error(self, rom_at: callable, samples: int = 41) -> dict[str, float]:
        """Largest relative error of the interpolant against exact rebuilds.

        Rebuilds the model exactly at ``samples`` temperatures spread across the
        grid -- deliberately offset from the grid points, where the interpolant
        is exact by construction and would flatter itself -- and compares.

        Parameters
        ----------
        rom_at
            Callable mapping a temperature in kelvin to a
            :class:`~cellkernel.rom.base.DiffusionROM`.
        samples
            Number of test temperatures.
        """
        lo, hi = self.bounds
        probes = np.linspace(lo, hi, samples)
        worst_a = 0.0
        worst_b = 0.0
        for temperature in probes:
            exact = rom_at(float(temperature)).discretise(self.dt)
            approx = self.at(float(temperature))
            scale_a = max(float(np.max(np.abs(exact.A))), 1e-30)
            scale_b = max(float(np.max(np.abs(exact.B))), 1e-30)
            worst_a = max(worst_a, float(np.max(np.abs(approx.A - exact.A))) / scale_a)
            worst_b = max(worst_b, float(np.max(np.abs(approx.B - exact.B))) / scale_b)
        return {"max_relative_A": worst_a, "max_relative_B": worst_b}


def schedule_over_temperature(
    build: callable,
    temperatures,
    dt: float,
    activation_energy: float = 0.0,
    reference_temperature: float = 298.15,
) -> ScheduledStateSpace:
    """Discretise a diffusion model across a temperature grid.

    Parameters
    ----------
    build
        Callable mapping a temperature in kelvin to a
        :class:`~cellkernel.rom.base.DiffusionROM`.
    temperatures
        Grid points in kelvin.
    dt
        Sample period, shared by every point on the grid.
    activation_energy, reference_temperature
        Arrhenius parameters of the diffusivity, used for factor blending. Pass
        them whenever the diffusivity is temperature dependent; leaving the
        energy at zero reverts to linear interpolation in temperature, which is
        markedly worse at the cold end.
    """
    temps = np.asarray(temperatures, dtype=float).reshape(-1)
    systems = []
    for temperature in temps:
        model = build(float(temperature))
        if not isinstance(model, DiffusionROM):
            raise TypeError("build must return a DiffusionROM")
        systems.append(model.discretise(dt))
    return ScheduledStateSpace(
        temperatures=temps,
        systems=tuple(systems),
        activation_energy=float(activation_energy),
        reference_temperature=float(reference_temperature),
    )
