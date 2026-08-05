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

    # These two live on the base class rather than on one filter because they
    # depend only on the model. Every estimator here needs the same shaped
    # moments, and having them reachable from only one of them was an accident
    # of which filter was written first.

    @staticmethod
    def suggest_initial_covariance(
        model: CellModel,
        soc_std: float = 0.1,
        gradient_std_fraction: float = 0.02,
        temperature_std: float = 2.0,
    ) -> np.ndarray:
        """Initial covariance for a stated state-of-charge uncertainty on a rested cell.

        Seeding a filter is where scaling mistakes do the most damage: too small a
        prior locks it onto a wrong start that no later data can undo, and a
        badly *shaped* prior sends corrections into the wrong states. Expressing
        the prior as "I know state of charge to within 10%" is something an
        engineer can judge; picking a variance in squared moles per cubic metre is
        not.

        The result is a rank-one covariance along
        :meth:`~cellkernel.models.base.CellModel.soc_direction` plus a small
        isotropic floor:

        .. math::

            P_0 = \\sigma_z^{2} d d^{\\!\\top}
                  + \\bigl( \\gamma \\sigma_z \\| d \\|_\\infty \\bigr)^{2} I .

        The rank-one term encodes the physically correct statement -- a rested cell
        of unknown charge is uncertain in its overall lithium content and in
        nothing else. The floor keeps the matrix positive definite and admits a
        little uncertainty about the internal gradient, which matters when the cell
        was not in fact fully rested at power-up.

        The floor is *isotropic*, which is only defensible while every state
        carries the same units. It is derived from the largest entry of ``d``,
        which for a diffusion model is a concentration of order 1e5 mol m-3, so
        the floor is a few hundred of those -- small, and sensible.

        Applied to a model that also carries temperature, the same number becomes
        a prior standard deviation of several hundred kelvin. The filter then puts
        the cell at 430 K on its first update, at −14 K on its fifth, and never
        recovers; the concentration states are dragged along because the
        temperature-dependent matrices go with them. The temperature state is
        therefore excluded from the floor and given a prior of its own, which a
        pack with thermistors knows to a couple of kelvin anyway.

        Parameters
        ----------
        model
            Model whose state space the covariance refers to.
        soc_std
            Prior standard deviation of state of charge.
        gradient_std_fraction
            Isotropic floor as a fraction of the dominant prior scale. Raise it if
            the cell may be seeded shortly after a load rather than at true rest.
        temperature_std
            Prior standard deviation of cell temperature in kelvin, for models
            carrying it as a state. Ignored by the others.
        """
        direction = np.asarray(model.soc_direction()).reshape(-1)
        floor = (gradient_std_fraction * soc_std * np.max(np.abs(direction))) ** 2
        diagonal = np.full(model.n_states, floor)

        index = getattr(model, "temperature_index", None)
        if index is not None:
            direction = direction.copy()
            direction[index] = 0.0
            diagonal[index] = temperature_std**2

        return soc_std**2 * np.outer(direction, direction) + np.diag(diagonal)

    @staticmethod
    def suggest_process_noise(
        model: CellModel,
        current_std: float = 0.05,
        soc_drift_per_hour: float = 0.01,
        temperature_std: float = 0.5,
    ) -> np.ndarray:
        """Process noise from a current-measurement error and a state-of-charge drift allowance.

        Process noise is hard to set by inspection because its units are those of
        the state, and a physics-based state vector mixes concentrations with modal
        coordinates that have no intuitive scale. Two interpretable quantities are
        combined instead.

        The first is current-measurement error. An error of ``current_std`` amperes
        perturbs the state along the input column ``b``, giving the rank-one term
        :math:`\\sigma_I^{2} b b^{\\!\\top}`. This is exact for the mechanism it
        describes, and it is correctly shaped: a mis-measured ampere cannot produce
        an arbitrary state disturbance, only one proportional to how current enters.

        The second is a drift allowance along
        :meth:`~cellkernel.models.base.CellModel.soc_direction`, which covers what
        the first term does not. Sensor error in practice is dominated by slowly
        varying *bias* rather than white noise, and white noise of realistic
        amplitude averages out to a negligible state-of-charge drift -- a 50 mA
        white error on a 5 Ah cell at 1 Hz drifts under 0.02% per hour. Modelling
        bias properly would mean augmenting the state, which is what
        :class:`~cellkernel.estimators.dual.DualEKF` does for resistance. Short of
        that, an explicit drift term keeps the filter willing to be corrected
        during long stretches where voltage is uninformative.

        A model carrying temperature as a state gets a third term, and the reason
        is worth stating because getting it wrong makes the filter diverge rather
        than merely mistune. The input column of such a model has a nonzero
        temperature entry, because current generates heat, so the rank-one
        current term would place temperature in rigid correlation with the
        diffusion states. It is not: temperature uncertainty comes from the
        ambient and from a heat-transfer coefficient nobody has measured
        accurately, and those have nothing to do with the current sensor. Left
        correlated, a voltage residual is partly absorbed by a temperature
        correction -- and since temperature is only weakly observable from
        terminal voltage, the filter walks it away and takes the concentration
        states with it. The temperature entry is therefore removed from the
        current column and given an independent variance of its own.

        Parameters
        ----------
        model
            Model whose state space the covariance refers to.
        current_std
            Standard deviation of the current measurement, in amperes.
        soc_drift_per_hour
            Additional random-walk allowance on state of charge, per hour.
        temperature_std
            Per-sample thermal model error in kelvin, for models carrying
            temperature as a state. Ignored by the others.
        """
        b = np.asarray(model.input_direction()).reshape(-1)
        direction = np.asarray(model.soc_direction()).reshape(-1)
        samples_per_hour = 3600.0 / model.dt
        drift_variance = soc_drift_per_hour**2 / samples_per_hour

        index = getattr(model, "temperature_index", None)
        thermal = np.zeros((model.n_states, model.n_states))
        if index is not None:
            b = b.copy()
            b[index] = 0.0
            direction = direction.copy()
            direction[index] = 0.0
            thermal[index, index] = temperature_std**2

        return (
            current_std**2 * np.outer(b, b)
            + drift_variance * np.outer(direction, direction)
            + thermal
        )

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
