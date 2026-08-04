"""Joint estimation of cell state and slow health parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import CellModel
from .base import Estimator, EstimatorOutputs, symmetrise

__all__ = ["DualEKF", "HealthEstimate"]


@dataclass(frozen=True)
class HealthEstimate:
    """Health parameters and their uncertainty.

    Attributes
    ----------
    capacity_retention
        Usable capacity as a fraction of nominal. One means a fresh cell.
    resistance_growth
        Additional series resistance in ohms, above the nominal contact
        resistance already in the parameter set.
    capacity_std, resistance_std
        Standard deviations from the augmented covariance. These matter more than
        the point estimates: capacity is only weakly observable, and a
        confident-looking value with a wide standard deviation should not be
        reported to the vehicle.
    """

    capacity_retention: float
    resistance_growth: float
    capacity_std: float
    resistance_std: float


class DualEKF(Estimator):
    """Extended Kalman filter over the cell state augmented with two health parameters.

    The state vector is :math:`[x, \\theta_Q, \\theta_R]`, where :math:`\\theta_Q`
    is capacity retention and :math:`\\theta_R` is added series resistance. The
    health parameters follow a random walk with very small process noise, which is
    what makes them settle on slowly varying values instead of chasing noise:

    .. math::

        x_{k+1} = f(x_k, I_k / \\theta_Q),
        \\qquad
        \\theta_{k+1} = \\theta_k + w_k,
        \\qquad
        V_k = h(x_k, I_k) - \\theta_R I_k .

    Capacity retention divides the current in the *state* update: on a cell that
    has lost inventory, the same ampere-hour throughput moves the concentration
    further, so a given charge covers more of the state-of-charge range. Resistance
    growth enters only the measurement. Keeping the two effects in separate places
    is what makes them separately identifiable -- if both distorted the voltage in
    the same way there would be nothing to distinguish them.

    All augmented Jacobians are exact. The sensitivity of the state update to
    capacity retention is

    .. math::

        \\frac{\\partial x_{k+1}}{\\partial \\theta_Q}
            = -\\frac{I_k}{\\theta_Q^{2}}
              \\bigl( f(x_k, 1) - f(x_k, 0) \\bigr),

    where the bracket is the input column of the process model, recovered by two
    evaluations because the update is exactly linear in current. No finite
    differencing is involved.

    Notes
    -----
    **Observability.** Resistance growth is observable from any current step: it
    produces an instantaneous voltage offset proportional to current. Capacity
    retention is not. It only reveals itself through the *rate* at which
    open-circuit voltage moves for a given throughput, so identifying it requires
    a substantial state-of-charge excursion -- tens of percent, not a pulse. Run
    this filter on a short drive cycle and the capacity estimate will barely move
    from its prior, correctly, and its reported standard deviation will say so.
    The test suite checks both behaviours.

    This is the honest limitation of online capacity estimation, and it is why
    production systems fold in long-horizon evidence such as full-charge events
    rather than trusting a filter to converge from arbitrary data.
    """

    def __init__(
        self,
        model: CellModel,
        process_noise: float | np.ndarray = 1.0,
        measurement_noise: float = 1e-6,
        initial_covariance: float | np.ndarray = 1e4,
        capacity_process_noise: float = 1e-12,
        resistance_process_noise: float = 1e-14,
        capacity_prior_std: float = 0.05,
        resistance_prior_std: float = 5e-3,
        capacity_bounds: tuple[float, float] = (0.5, 1.15),
        resistance_bounds: tuple[float, float] = (-0.02, 0.2),
    ) -> None:
        super().__init__(model, process_noise, measurement_noise, initial_covariance)
        n = model.n_states
        self.n_model = n
        self.capacity_bounds = capacity_bounds
        self.resistance_bounds = resistance_bounds

        # Rebuild the augmented moments around the base-class ones.
        self.Q = _augment_diag(self.Q, capacity_process_noise, resistance_process_noise)
        self.P0 = _augment_diag(self.P0, capacity_prior_std**2, resistance_prior_std**2)
        self.x = np.zeros(n + 2)
        self.P = self.P0.copy()
        self._A_model = model.state_jacobian(model.initial_state(0.5), 0.0)

    # ------------------------------------------------------------------ set-up

    def initialise(self, soc: float, temperature: float | None = None) -> None:
        base = self.model.initial_state(soc, temperature)
        self.x = np.concatenate([base, [1.0, 0.0]])
        self.P = self.P0.copy()
        self._initialised = True

    @property
    def model_state(self) -> np.ndarray:
        return self.x[: self.n_model]

    @property
    def capacity_retention(self) -> float:
        return float(self.x[self.n_model])

    @property
    def resistance_growth(self) -> float:
        return float(self.x[self.n_model + 1])

    def health(self) -> HealthEstimate:
        """Current health estimate with uncertainties."""
        i = self.n_model
        return HealthEstimate(
            capacity_retention=float(self.x[i]),
            resistance_growth=float(self.x[i + 1]),
            capacity_std=float(np.sqrt(max(self.P[i, i], 0.0))),
            resistance_std=float(np.sqrt(max(self.P[i + 1, i + 1], 0.0))),
        )

    # ------------------------------------------------------------------ update

    def _voltage(self, x: np.ndarray, current: float) -> float:
        return self.model.voltage(x[: self.n_model], current) - x[self.n_model + 1] * current

    def update(self, current: float, voltage: float) -> EstimatorOutputs:
        if not self._initialised:
            self.initialise_from_voltage(voltage)

        model = self.model
        n = self.n_model
        current = float(current)

        predicted = self._voltage(self.x, current)
        H = np.zeros((1, n + 2))
        H[0, :n] = model.voltage_jacobian(self.model_state, current)
        H[0, n + 1] = -current

        innovation = float(voltage) - predicted
        PH = self.P @ H.T
        S = float((H @ PH).item()) + self.R
        K = PH / S

        self.x = self.x + (K * innovation).reshape(-1)
        self._clip_health()

        identity = np.eye(n + 2)
        KH = K @ H
        self.P = symmetrise((identity - KH) @ self.P @ (identity - KH).T + K @ K.T * self.R)

        corrected = model.outputs(self.model_state, current)
        soc_std = self.soc_std()

        # ---- prediction, with exact augmented Jacobian
        retention = max(self.capacity_retention, self.capacity_bounds[0])
        effective = current / retention
        input_column = model.step(self.model_state, 1.0) - model.step(self.model_state, 0.0)

        F = np.zeros((n + 2, n + 2))
        F[:n, :n] = self._A_model
        F[:n, n] = -input_column * current / retention**2
        F[n, n] = 1.0
        F[n + 1, n + 1] = 1.0

        self.x = np.concatenate(
            [model.step(self.model_state, effective), [retention, self.resistance_growth]]
        )
        self.P = symmetrise(F @ self.P @ F.T + self.Q)

        return EstimatorOutputs(
            soc=corrected.soc,
            voltage=corrected.voltage,
            innovation=innovation,
            innovation_variance=S,
            soc_std=soc_std,
        )

    def _clip_health(self) -> None:
        """Keep health parameters inside physically meaningful bounds.

        A capacity retention that wanders to zero or negative would divide the
        current by something near zero and destroy the state. Clipping is crude
        compared with a constrained filter, but it is bounded, cheap and
        predictable, which is what matters in a safety-adjacent loop.
        """
        i = self.n_model
        self.x[i] = float(np.clip(self.x[i], *self.capacity_bounds))
        self.x[i + 1] = float(np.clip(self.x[i + 1], *self.resistance_bounds))

    def _soc_gradient(self) -> np.ndarray:
        grad = np.zeros(self.n_model + 2)
        grad[: self.n_model] = np.asarray(self.model.soc_jacobian()).reshape(-1)
        return grad

    def run(
        self, current: np.ndarray, voltage: np.ndarray, soc0: float | None = None
    ) -> dict[str, np.ndarray]:
        """Filter a record, additionally returning health trajectories."""
        current = np.asarray(current, dtype=float).reshape(-1)
        voltage = np.asarray(voltage, dtype=float).reshape(-1)
        retention = np.empty(current.size)
        resistance = np.empty(current.size)
        retention_std = np.empty(current.size)
        resistance_std = np.empty(current.size)

        original_update = self.update
        index = {"k": 0}

        def recording_update(i: float, v: float) -> EstimatorOutputs:
            result = original_update(i, v)
            k = index["k"]
            health = self.health()
            retention[k] = health.capacity_retention
            resistance[k] = health.resistance_growth
            retention_std[k] = health.capacity_std
            resistance_std[k] = health.resistance_std
            index["k"] = k + 1
            return result

        self.update = recording_update  # type: ignore[method-assign]
        try:
            out = super().run(current, voltage, soc0)
        finally:
            del self.update
        out["capacity_retention"] = retention
        out["resistance_growth"] = resistance
        out["capacity_retention_std"] = retention_std
        out["resistance_growth_std"] = resistance_std
        return out


def _augment_diag(block: np.ndarray, *extra: float) -> np.ndarray:
    """Extend a covariance block with additional independent diagonal entries."""
    n = block.shape[0]
    out = np.zeros((n + len(extra), n + len(extra)))
    out[:n, :n] = block
    for i, value in enumerate(extra):
        out[n + i, n + i] = float(value)
    return out
