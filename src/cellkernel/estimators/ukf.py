"""Unscented Kalman filter, specialised to a linear process model."""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, cholesky

from ..models.base import CellModel
from .base import Estimator, EstimatorOutputs, symmetrise

__all__ = ["UKF", "safe_cholesky"]


def safe_cholesky(matrix: np.ndarray, max_attempts: int = 8) -> np.ndarray:
    """Lower-triangular Cholesky factor, adding jitter if needed.

    A covariance that has drifted marginally indefinite through rounding will
    fail to factorise even though it is numerically fine for filtering. Rather
    than abort, escalating multiples of the mean diagonal are added until the
    factorisation succeeds. The amount added is reported by growing from a very
    small value, so a healthy filter never pays anything and a sick one degrades
    instead of crashing.
    """
    matrix = symmetrise(matrix)
    scale = float(np.mean(np.diag(matrix)))
    jitter = 0.0
    for attempt in range(max_attempts):
        try:
            return cholesky(matrix + jitter * np.eye(matrix.shape[0]), lower=True)
        except LinAlgError:
            jitter = max(scale, 1e-300) * (10.0 ** (attempt - 10))
    raise LinAlgError("covariance could not be factorised even with jitter")


class UKF(Estimator):
    """Sigma-point filter with an exact linear prediction step.

    The unscented transform replaces linearisation with propagation of a
    deterministic set of sigma points. Here it is applied only to the *measurement*
    update.

    That is not a shortcut. The process model in this package is exactly linear,
    :math:`x_{k+1} = A x_k + B u_k`, and the unscented transform of an affine map
    reproduces the mean and covariance *exactly*: the sigma points are symmetric
    about the mean, so an affine map carries them to points whose weighted mean
    and covariance are precisely :math:`Ax + Bu` and :math:`APA^{\\top}`. Doing
    the prediction with sigma points would therefore compute the same numbers as
    the matrix update, only more slowly and with more rounding. The test suite
    asserts this equivalence rather than taking it on trust.

    What the sigma points do earn is accuracy in the voltage measurement, which is
    genuinely nonlinear through both the open-circuit potential and the ``asinh``
    kinetics. Where the open-circuit potential has strong curvature -- the knees at
    each end of the range, and the staging features of graphite -- a linearised
    measurement Jacobian misrepresents how a spread of concentrations maps onto a
    spread of voltages, and the filter becomes over-confident.

    Parameters
    ----------
    model
        Cell model to filter.
    process_noise, measurement_noise, initial_covariance
        As for :class:`~cellkernel.estimators.ekf.EKF`.
    alpha
        Spread of the sigma points. Small values cluster them near the mean, which
        makes the transform behave more like a linearisation; ``1e-3`` is the usual
        default from the literature.
    beta
        Incorporates prior knowledge of the distribution; ``2`` is optimal for a
        Gaussian.
    kappa
        Secondary scaling. ``0`` is a common choice for state estimation, and
        ``3 - n`` minimises fourth-order error but can make weights negative.
    """

    def __init__(
        self,
        model: CellModel,
        process_noise: float | np.ndarray = 1.0,
        measurement_noise: float = 1e-6,
        initial_covariance: float | np.ndarray = 1e4,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(model, process_noise, measurement_noise, initial_covariance)
        n = model.n_states
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self._lambda = self.alpha**2 * (n + self.kappa) - n
        self._scale = n + self._lambda
        if abs(self._scale) < 1e-12:
            raise ValueError("degenerate sigma-point scaling; adjust alpha or kappa")

        self.weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * self._scale))
        self.weights_cov = self.weights_mean.copy()
        self.weights_mean[0] = self._lambda / self._scale
        self.weights_cov[0] = self._lambda / self._scale + (1.0 - self.alpha**2 + self.beta)
        self._A = model.state_jacobian(model.initial_state(0.5), 0.0)

    def sigma_points(self) -> np.ndarray:
        """Sigma points for the current mean and covariance, shape ``(2n+1, n)``."""
        n = self.model.n_states
        factor = safe_cholesky(self._scale * self.P)
        points = np.empty((2 * n + 1, n))
        points[0] = self.x
        for i in range(n):
            points[1 + i] = self.x + factor[:, i]
            points[1 + n + i] = self.x - factor[:, i]
        return points

    def update(self, current: float, voltage: float) -> EstimatorOutputs:
        if not self._initialised:
            self.initialise_from_voltage(voltage)

        model = self.model
        points = self.sigma_points()
        predicted = np.array([model.voltage(p, current) for p in points])
        mean_v = float(self.weights_mean @ predicted)

        dv = predicted - mean_v
        dx = points - self.x
        var_v = float(self.weights_cov @ (dv * dv)) + self.R
        cov_xv = (self.weights_cov * dv) @ dx

        innovation = float(voltage) - mean_v
        gain = cov_xv / var_v

        self.x = self.x + gain * innovation
        self.P = symmetrise(self.P - np.outer(gain, gain) * var_v)

        corrected = model.outputs(self.x, current)
        soc_std = self.soc_std()

        self.x = model.step(self.x, current)
        self.P = symmetrise(self._A @ self.P @ self._A.T + self.Q)

        return EstimatorOutputs(
            soc=corrected.soc,
            voltage=corrected.voltage,
            innovation=innovation,
            innovation_variance=var_v,
            soc_std=soc_std,
        )
