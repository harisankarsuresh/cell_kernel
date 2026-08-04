"""Shared machinery for the state estimators."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from ..models.base import CellModel

__all__ = ["Estimator", "EstimatorOutputs", "symmetrise"]


@dataclass(frozen=True)
class EstimatorOutputs:
    """Result of one estimator update.

    Attributes
    ----------
    soc
        Corrected state-of-charge estimate.
    voltage
        Model voltage after correction, for residual plots.
    innovation
        Measured minus predicted voltage before correction, in volts. The single
        most useful diagnostic a filter produces: a well-tuned filter leaves a
        zero-mean innovation with the variance predicted by ``innovation_variance``,
        and any structure in it points at a model error rather than noise.
    innovation_variance
        Predicted variance of the innovation, ``H P H' + R``.
    soc_std
        Standard deviation of the state-of-charge estimate, propagated from the
        state covariance.
    """

    soc: float
    voltage: float
    innovation: float
    innovation_variance: float
    soc_std: float


def symmetrise(matrix: np.ndarray) -> np.ndarray:
    """Force exact symmetry of a covariance matrix.

    Covariance updates are symmetric in exact arithmetic but not in floating
    point, and the asymmetry compounds. Once a covariance drifts far enough from
    symmetric its Cholesky factorisation fails or, worse, succeeds with a
    slightly indefinite matrix and the filter silently diverges. Averaging with
    the transpose costs almost nothing and removes the failure mode.
    """
    return 0.5 * (matrix + matrix.T)


class Estimator(abc.ABC):
    """Base class for recursive state estimators.

    Parameters
    ----------
    model
        The cell model to filter against.
    process_noise
        Either a scalar applied to every state, a vector of per-state variances,
        or a full covariance matrix. Units are squared state units per sample.
    measurement_noise
        Voltage measurement variance in V2. For a 12-bit converter over a 5 V
        span the quantisation alone contributes about ``(1.2e-3)^2 / 12``, but
        the dominant term in practice is model error rather than sensor noise, so
        this is usually tuned upward.
    initial_covariance
        Initial state covariance, in the same forms accepted by
        ``process_noise``.
    """

    def __init__(
        self,
        model: CellModel,
        process_noise: float | np.ndarray,
        measurement_noise: float,
        initial_covariance: float | np.ndarray,
    ) -> None:
        self.model = model
        n = model.n_states
        self.Q = _as_covariance(process_noise, n, "process_noise")
        self.R = float(measurement_noise)
        if self.R <= 0.0:
            raise ValueError("measurement_noise must be positive")
        self.P0 = _as_covariance(initial_covariance, n, "initial_covariance")
        self.x = np.zeros(n)
        self.P = self.P0.copy()
        self._initialised = False

    # ------------------------------------------------------------------ set-up

    def initialise(self, soc: float, temperature: float | None = None) -> None:
        """Seed the filter at a known state of charge."""
        self.x = self.model.initial_state(soc, temperature)
        self.P = self.P0.copy()
        self._initialised = True

    def initialise_from_voltage(self, voltage: float, temperature: float | None = None) -> None:
        """Seed the filter from a rest voltage measurement.

        This is how a battery-management unit starts after a long key-off: invert
        the open-circuit voltage curve. It is only trustworthy if the cell really
        has rested, and on a flat-plateau chemistry it is barely trustworthy even
        then, which is why the initial covariance matters.
        """
        params = getattr(self.model, "parameters", None)
        if params is None:  # pragma: no cover - defensive
            raise TypeError("model does not expose a parameter set")
        self.initialise(params.soc_from_ocv(voltage), temperature)

    # ------------------------------------------------------------------ update

    @abc.abstractmethod
    def update(self, current: float, voltage: float) -> EstimatorOutputs:
        """Correct with a voltage measurement, then predict one step ahead."""

    def run(
        self, current: np.ndarray, voltage: np.ndarray, soc0: float | None = None
    ) -> dict[str, np.ndarray]:
        """Filter a whole record and return per-sample diagnostics."""
        current = np.asarray(current, dtype=float).reshape(-1)
        voltage = np.asarray(voltage, dtype=float).reshape(-1)
        if current.size != voltage.size:
            raise ValueError("current and voltage must have equal length")
        if soc0 is not None:
            self.initialise(soc0)
        elif not self._initialised:
            self.initialise_from_voltage(float(voltage[0]))

        n = current.size
        out = {
            key: np.empty(n)
            for key in ("soc", "voltage", "innovation", "innovation_variance", "soc_std")
        }
        for k in range(n):
            result = self.update(float(current[k]), float(voltage[k]))
            out["soc"][k] = result.soc
            out["voltage"][k] = result.voltage
            out["innovation"][k] = result.innovation
            out["innovation_variance"][k] = result.innovation_variance
            out["soc_std"][k] = result.soc_std
        out["time"] = np.arange(n, dtype=float) * self.model.dt
        out["current"] = current
        out["measured_voltage"] = voltage
        return out

    # ----------------------------------------------------------------- helpers

    def soc_std(self) -> float:
        """Standard deviation of the reported state of charge."""
        grad = self._soc_gradient()
        return float(np.sqrt(max(grad @ self.P @ grad, 0.0)))

    def _soc_gradient(self) -> np.ndarray:
        jac = getattr(self.model, "soc_jacobian", None)
        if jac is not None:
            return np.asarray(jac()).reshape(-1)
        grad = np.zeros(self.model.n_states)  # pragma: no cover - defensive
        return grad


def _as_covariance(value: float | np.ndarray, n: int, label: str) -> np.ndarray:
    """Expand a scalar, vector or matrix specification into an ``(n, n)`` covariance."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(n) * float(arr)
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(f"{label} vector must have length {n}, got {arr.size}")
        return np.diag(arr)
    if arr.shape != (n, n):
        raise ValueError(f"{label} matrix must be {n}x{n}, got {arr.shape}")
    return symmetrise(arr.copy())
