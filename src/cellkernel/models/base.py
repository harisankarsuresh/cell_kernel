"""Common interface for discrete-time cell models."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

__all__ = ["CellModel", "ModelOutputs"]


@dataclass(frozen=True)
class ModelOutputs:
    """Everything a model reports for one sample.

    Attributes
    ----------
    voltage
        Terminal voltage in volts.
    soc
        State of charge in ``[0, 1]``, from the bulk concentration rather than
        from integrated current.
    temperature
        Cell temperature in kelvin. Equal to the ambient value for isothermal
        models.
    surface_stoichiometry
        ``(negative, positive)`` stoichiometry at the particle surface. Empty for
        equivalent-circuit models.
    overpotential
        ``(negative, positive)`` reaction overpotential in volts. Empty for
        equivalent-circuit models.
    """

    voltage: float
    soc: float
    temperature: float
    surface_stoichiometry: tuple[float, ...] = ()
    overpotential: tuple[float, ...] = ()


class CellModel(abc.ABC):
    """A fixed-step discrete-time model of a single cell.

    Sign convention throughout the package: **current is positive on
    discharge**. This follows the convention used by cell manufacturers and most
    test equipment, so recorded data usually needs no sign change.

    The sample period is fixed at construction rather than passed to
    :meth:`step`. That is a deliberate constraint. Every reduced-order model here
    is discretised by matrix exponential at build time, which is what makes the
    online update a plain matrix-vector product with no stability condition; a
    variable step would mean either re-exponentiating a matrix on the
    microcontroller or falling back to a conditionally stable integrator, and
    both defeat the purpose. A battery-management task runs on a fixed schedule
    anyway.

    Subclasses must keep :meth:`step` free of state mutation: it takes a state
    vector and returns a new one. The estimators rely on being able to call it
    repeatedly from perturbed states, which is impossible if the model holds
    internal state.
    """

    #: Sample period in seconds.
    dt: float

    @property
    @abc.abstractmethod
    def n_states(self) -> int:
        """Length of the state vector."""

    @property
    @abc.abstractmethod
    def state_names(self) -> tuple[str, ...]:
        """Human-readable name for each state, used in reports and generated C."""

    @abc.abstractmethod
    def initial_state(self, soc: float, temperature: float | None = None) -> np.ndarray:
        """State vector for a cell resting at ``soc`` and ``temperature``."""

    @abc.abstractmethod
    def step(self, x: np.ndarray, current: float) -> np.ndarray:
        """Advance one sample under constant ``current`` in amperes."""

    @abc.abstractmethod
    def outputs(self, x: np.ndarray, current: float) -> ModelOutputs:
        """Evaluate all reported quantities without advancing the state."""

    @abc.abstractmethod
    def state_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Jacobian ``d(step)/dx``, shape ``(n_states, n_states)``."""

    @abc.abstractmethod
    def voltage_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Gradient of terminal voltage with respect to the state, shape ``(n_states,)``."""

    # ------------------------------------------------------------------ helpers

    def soc_direction(self) -> np.ndarray:
        """State-space direction along which a rested cell's charge varies.

        Returns the vector ``d`` with ``initial_state(z + dz) = initial_state(z) +
        d dz``. Because :meth:`initial_state` is affine in state of charge for every
        model here, a central difference recovers it exactly.

        This direction is what makes a sensible filter prior possible. Saying "I do
        not know the state of charge" does *not* mean every state is independently
        uncertain: a cell that has rested has no concentration gradient no matter
        how full it is, so the uncertainty is confined almost entirely to this one
        direction. An isotropic prior instead spreads the same variance over the
        gradient coordinates, and since the Kalman gain is proportional to ``P
        H'``, a voltage correction is then absorbed mostly by the gradient states
        rather than by the bulk concentration. The filter fits the measurement
        while leaving its state-of-charge error largely untouched, and can wander
        several percent during a long rest -- exactly when the open-circuit voltage
        should be pinning it down.
        """
        lo = self.initial_state(0.4)
        hi = self.initial_state(0.6)
        return (hi - lo) / 0.2

    def input_direction(self, current: float = 0.0) -> np.ndarray:
        """State-space column through which current enters, per ampere per sample.

        This is the direction a current-measurement error perturbs the state
        along, and therefore the correct shape for a process-noise covariance.

        Evaluated as ``step(x, current + 1) - step(x, current)``. For the models
        whose process update is linear in current -- the single particle model,
        the electrolyte model and the equivalent circuit -- this is exact and
        independent of where it is taken.

        :class:`~cellkernel.models.thermal.ThermalSPM` is the exception, because
        heat generation is quadratic in current, so its temperature row genuinely
        depends on the operating point and ``current`` lets a caller who knows
        the duty cycle ask about a representative one.

        A one-sided difference, deliberately. A central difference would be the
        obvious choice and is wrong here: dissipation is even in current, so
        differencing symmetrically about zero cancels the resistive heating
        exactly and leaves only the reversible term. The temperature row then
        understates the thermal response by orders of magnitude, and a filter
        given that as its process-noise shape trusts its own temperature
        prediction far too much and diverges.
        """
        reference = self.initial_state(0.5)
        return self.step(reference, current + 1.0) - self.step(reference, current)

    def voltage(self, x: np.ndarray, current: float) -> float:
        """Terminal voltage in volts."""
        return self.outputs(x, current).voltage

    def soc(self, x: np.ndarray) -> float:
        """State of charge implied by the state vector."""
        return self.outputs(x, 0.0).soc

    def simulate(
        self,
        current: np.ndarray,
        soc0: float = 1.0,
        temperature: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Replay a current sequence from a rested initial condition.

        Returns arrays of equal length keyed ``time``, ``current``, ``voltage``,
        ``soc`` and ``temperature``. Outputs at index ``k`` are evaluated with
        the state at ``k`` and the current at ``k``, before stepping.
        """
        current = np.asarray(current, dtype=float).reshape(-1)
        x = self.initial_state(soc0, temperature)
        n = current.size
        voltage = np.empty(n)
        soc = np.empty(n)
        temp = np.empty(n)
        for k in range(n):
            out = self.outputs(x, current[k])
            voltage[k] = out.voltage
            soc[k] = out.soc
            temp[k] = out.temperature
            x = self.step(x, current[k])
        return {
            "time": np.arange(n, dtype=float) * self.dt,
            "current": current,
            "voltage": voltage,
            "soc": soc,
            "temperature": temp,
        }

    def numerical_voltage_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Central-difference voltage gradient, for checking the analytic one."""
        return _adaptive_gradient(
            lambda z: np.array([self.voltage(z, current)]), np.asarray(x, dtype=float)
        ).reshape(-1)

    def numerical_state_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Central-difference state Jacobian, for checking the analytic one."""
        return _adaptive_gradient(lambda z: self.step(z, current), np.asarray(x, dtype=float))


def _adaptive_gradient(
    fn, x: np.ndarray, target_signal: float = 1e-7, max_growth: int = 60
) -> np.ndarray:
    """Central-difference Jacobian with a per-state step chosen to clear roundoff.

    A fixed relative step is not usable on these models. State vectors legitimately
    span many orders of magnitude -- a Pade realisation of a particle can hold a
    bulk concentration near 1e4 alongside filter coordinates near 1e10 -- and
    several states are exactly zero at a rested initial condition, so there is no
    local magnitude to scale by at all. With a step that is too small the
    difference of two nearly equal outputs is dominated by cancellation, and the
    finite-difference "reference" ends up less accurate than the analytic
    derivative it is supposed to be checking. That failure mode is particularly
    misleading because it looks like a bug in the analytic expression.

    Here the step for each state starts small and grows geometrically until the
    change in the output rises clearly above the floating-point noise floor,
    estimated as ``eps`` times the output magnitude. The step therefore ends up as
    small as the arithmetic allows and no smaller, which keeps truncation error
    low while guaranteeing a meaningful signal.
    """
    x = np.asarray(x, dtype=float)
    base = np.asarray(fn(x), dtype=float).reshape(-1)
    noise = np.finfo(float).eps * np.maximum(np.abs(base).max(), 1e-30)
    jac = np.zeros((base.size, x.size))
    for i in range(x.size):
        h = 1e-8 * max(abs(x[i]), 1.0)
        delta = np.zeros_like(base)
        for _ in range(max_growth):
            hi, lo = x.copy(), x.copy()
            hi[i] += h
            lo[i] -= h
            delta = np.asarray(fn(hi), dtype=float).reshape(-1) - np.asarray(
                fn(lo), dtype=float
            ).reshape(-1)
            if np.abs(delta).max() > max(noise / target_signal, 1e-300):
                break
            h *= 4.0
        jac[:, i] = delta / (2.0 * h)
    return jac
