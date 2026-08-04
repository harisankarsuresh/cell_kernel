"""Equivalent-circuit cell model, as a baseline and a fallback."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..params import CellParameters
from .base import CellModel, ModelOutputs

__all__ = ["ECM"]


class ECM(CellModel):
    """Thevenin equivalent-circuit model with a series resistance and ``m`` RC pairs.

    .. math::

        V = \\mathrm{OCV}(z) - I R_0 - \\sum_{i=1}^{m} v_i,
        \\qquad
        \\frac{dz}{dt} = -\\frac{I}{3600 Q},
        \\qquad
        \\tau_i \\frac{dv_i}{dt} = R_i I - v_i .

    Each RC branch is discretised exactly for piecewise-constant current,

    .. math::

        v_{i,k+1} = e^{-\\Delta t / \\tau_i} v_{i,k}
                    + R_i \\left( 1 - e^{-\\Delta t/\\tau_i} \\right) I_k,

    rather than by forward Euler. With a 1 s sample period and a 5 s time
    constant, Euler would misplace the branch pole by about 10%, which a fit
    absorbs by distorting ``R_i`` and ``tau_i`` and which then fails to transfer
    to a different sample rate.

    This model is included for three reasons: it is the incumbent in almost every
    shipped battery-management system, so it is the benchmark any physics-based
    approach has to beat; it is a useful sanity check, because for slow duty
    cycles it should agree closely with the single particle model; and it is the
    right fallback when a cell has not been characterised well enough to
    parameterise electrochemistry.

    Its structural limitation is that state of charge is pure coulomb counting.
    Nothing in the model represents a concentration gradient, so after a hard
    pulse it cannot distinguish "the surface is depleted but the bulk is full"
    from "the cell is empty". That distinction is exactly what the single particle
    model provides and is where the two diverge under aggressive load.

    Parameters
    ----------
    parameters
        Cell parameter set, used for the open-circuit voltage curve and capacity.
    dt
        Sample period in seconds.
    series_resistance
        Ohmic resistance ``R0`` in ohms.
    rc_pairs
        Sequence of ``(resistance_ohm, time_constant_s)`` pairs.
    capacity
        Usable capacity in ampere hours. Defaults to the parameter set's nominal
        capacity.
    """

    def __init__(
        self,
        parameters: CellParameters,
        dt: float = 1.0,
        series_resistance: float = 0.02,
        rc_pairs: Sequence[tuple[float, float]] = ((0.01, 30.0), (0.005, 300.0)),
        capacity: float | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.parameters = parameters
        self.dt = float(dt)
        self.series_resistance = float(series_resistance)
        self.rc_pairs = tuple((float(r), float(t)) for r, t in rc_pairs)
        if any(t <= 0.0 for _, t in self.rc_pairs):
            raise ValueError("RC time constants must be positive")
        self.capacity = float(capacity if capacity is not None else parameters.nominal_capacity)
        if self.capacity <= 0.0:
            raise ValueError("capacity must be positive")

        self._decay = np.array([np.exp(-self.dt / t) for _, t in self.rc_pairs])
        self._gain = np.array([r * (1.0 - np.exp(-self.dt / t)) for r, t in self.rc_pairs])
        self._soc_per_amp_second = 1.0 / (3600.0 * self.capacity)

    @property
    def n_states(self) -> int:
        return 1 + len(self.rc_pairs)

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("soc",) + tuple(f"v_rc_{i}" for i in range(len(self.rc_pairs)))

    def initial_state(self, soc: float, temperature: float | None = None) -> np.ndarray:
        return np.concatenate([[float(soc)], np.zeros(len(self.rc_pairs))])

    def step(self, x: np.ndarray, current: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        current = float(current)
        out = np.empty_like(x)
        out[0] = x[0] - current * self.dt * self._soc_per_amp_second
        out[1:] = self._decay * x[1:] + self._gain * current
        return out

    def outputs(self, x: np.ndarray, current: float) -> ModelOutputs:
        x = np.asarray(x, dtype=float).reshape(-1)
        current = float(current)
        voltage = (
            float(self.parameters.open_circuit_voltage(x[0]))
            - current * self.series_resistance
            - float(np.sum(x[1:]))
        )
        return ModelOutputs(
            voltage=voltage,
            soc=float(x[0]),
            temperature=self.parameters.reference_temperature,
        )

    def state_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        jac = np.zeros((self.n_states, self.n_states))
        jac[0, 0] = 1.0
        for i, decay in enumerate(self._decay, start=1):
            jac[i, i] = decay
        return jac

    def voltage_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        grad = np.zeros(self.n_states)
        grad[0] = float(self.parameters.ocv_derivative(x[0]))
        grad[1:] = -1.0
        return grad

    def soc_jacobian(self) -> np.ndarray:
        grad = np.zeros(self.n_states)
        grad[0] = 1.0
        return grad
