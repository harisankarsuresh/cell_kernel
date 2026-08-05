"""Extended Kalman filter for physics-based cell models."""

from __future__ import annotations

import numpy as np

from ..models.base import CellModel
from .base import Estimator, EstimatorOutputs, symmetrise

__all__ = ["EKF"]


class EKF(Estimator):
    """Extended Kalman filter with a Joseph-form covariance update.

    One sample proceeds as correct-then-predict:

    .. math::

        \\begin{aligned}
        e_k &= V_k^{\\text{meas}} - h(x_k, I_k), \\\\
        S_k &= H_k P_k H_k^{\\!\\top} + R, \\qquad
        K_k = P_k H_k^{\\!\\top} S_k^{-1}, \\\\
        x_k^{+} &= x_k + K_k e_k, \\\\
        P_k^{+} &= (I - K_k H_k) P_k (I - K_k H_k)^{\\!\\top}
                   + K_k R K_k^{\\!\\top}, \\\\
        x_{k+1} &= A x_k^{+} + B I_k, \\qquad
        P_{k+1} = A P_k^{+} A^{\\!\\top} + Q .
        \\end{aligned}

    Two choices are worth explaining.

    **Joseph form.** The covariance correction is written as
    :math:`(I-KH)P(I-KH)^{\\top} + KRK^{\\top}` rather than the shorter
    :math:`(I-KH)P`. The two are equal in exact arithmetic, but only the Joseph
    form is a sum of two positive-semidefinite terms, so it stays
    positive-semidefinite under rounding. The short form subtracts, and on a
    32-bit target with a sharply observable state it can produce a covariance with
    a small negative eigenvalue; from there the filter either diverges or dies on
    a square root. The extra cost is one matrix multiply, which is the cheapest
    insurance in the whole algorithm.

    **Exact prediction.** For the models in this package the process update is
    linear, so :math:`A` is the true Jacobian rather than a linearisation. The
    prediction step introduces no linearisation error at all, and the usual
    complaint about extended Kalman filters -- that repeated linearisation of the
    dynamics corrupts the covariance -- does not apply. Only :math:`H`, the
    voltage gradient, is approximate.

    Parameters
    ----------
    model
        Cell model to filter.
    process_noise
        Per-state process variance. For concentration states the natural scale is
        the square of the concentration drift you are willing to tolerate per
        sample; see :meth:`suggest_process_noise`.
    measurement_noise
        Voltage measurement variance in V2.
    initial_covariance
        Initial state covariance.
    iterations
        Gauss-Newton iterations in the measurement update. ``1`` is the textbook
        extended Kalman filter; larger values give the *iterated* filter. See the
        notes.

    Notes
    -----
    **Why iterate.** A single-shot correction linearises the voltage once, at the
    prior. That is fine when the prior error is small, and it fails badly when it
    is not, because the open-circuit voltage is strongly curved. Seeding a
    nickel-manganese-cobalt cell at 90% when it is really at 75% is a concrete
    example: the local slope ``dOCV/dSOC`` there is about 0.22 V, but the actual
    voltage difference across that 15% gap corresponds to an average slope near
    1.1 V. The filter therefore believes a given voltage residual implies about
    five times more charge error than it does, and the first correction overshoots
    by the same factor. It then overshoots back from a region with a different
    slope, and the estimate oscillates instead of settling.

    Iterating the update re-linearises at the improved estimate each time, which is
    Gauss-Newton on the measurement residual and converges in a handful of steps.
    The extra cost is one voltage evaluation and one Jacobian per iteration; the
    expensive parts of the filter -- the covariance triple products -- are done once
    regardless. The default remains ``1`` so that the class behaves as its name
    says, but ``3`` to ``5`` is the better choice whenever the filter may be seeded
    a long way from the truth, which in the field means every cold start.

    The filter reports state of charge from the *bulk* concentration, which is
    driven by the structurally exact mass balance. A voltage correction moves the
    bulk and surface states together through the Kalman gain, so the estimate
    combines coulomb counting with voltage feedback without either one being
    bolted on.
    """

    def __init__(
        self,
        model: CellModel,
        process_noise: float | np.ndarray = 1.0,
        measurement_noise: float = 1e-6,
        initial_covariance: float | np.ndarray = 1e4,
        iterations: int = 1,
    ) -> None:
        super().__init__(model, process_noise, measurement_noise, initial_covariance)
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.iterations = int(iterations)
        self._A = model.state_jacobian(model.initial_state(0.5), 0.0)

    def update(self, current: float, voltage: float) -> EstimatorOutputs:
        if not self._initialised:
            self.initialise_from_voltage(voltage)

        model = self.model
        prior = self.x.copy()
        innovation = float(voltage) - model.voltage(prior, current)

        estimate = prior
        H = model.voltage_jacobian(prior, current).reshape(1, -1)
        PH = self.P @ H.T
        S = float((H @ PH).item()) + self.R
        K = PH / S

        for _ in range(self.iterations):
            H = model.voltage_jacobian(estimate, current).reshape(1, -1)
            PH = self.P @ H.T
            S = float((H @ PH).item()) + self.R
            K = PH / S
            # Gauss-Newton step measured from the prior, not from the current
            # iterate, so the prior's information is not counted repeatedly.
            residual = (
                float(voltage)
                - model.voltage(estimate, current)
                - float((H @ (prior - estimate)).item())
            )
            estimate = prior + (K * residual).reshape(-1)

        self.x = estimate
        identity = np.eye(model.n_states)
        KH = K @ H
        self.P = symmetrise((identity - KH) @ self.P @ (identity - KH).T + K @ K.T * self.R)

        corrected = model.outputs(self.x, current)
        soc_std = self.soc_std()

        # Propagate to the next sample. A is exact for these models.
        self.x = model.step(self.x, current)
        self.P = symmetrise(self._A @ self.P @ self._A.T + self.Q)

        return EstimatorOutputs(
            soc=corrected.soc,
            voltage=corrected.voltage,
            innovation=innovation,
            innovation_variance=S,
            soc_std=soc_std,
        )
