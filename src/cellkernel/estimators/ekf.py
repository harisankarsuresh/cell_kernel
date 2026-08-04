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

    @staticmethod
    def suggest_initial_covariance(
        model: CellModel, soc_std: float = 0.1, gradient_std_fraction: float = 0.02
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

        Parameters
        ----------
        model
            Model whose state space the covariance refers to.
        soc_std
            Prior standard deviation of state of charge.
        gradient_std_fraction
            Isotropic floor as a fraction of the dominant prior scale. Raise it if
            the cell may be seeded shortly after a load rather than at true rest.
        """
        direction = np.asarray(model.soc_direction()).reshape(-1)
        floor = (gradient_std_fraction * soc_std * np.max(np.abs(direction))) ** 2
        return soc_std**2 * np.outer(direction, direction) + floor * np.eye(model.n_states)

    @staticmethod
    def suggest_process_noise(
        model: CellModel,
        current_std: float = 0.05,
        soc_drift_per_hour: float = 0.01,
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

        Parameters
        ----------
        model
            Model whose state space the covariance refers to.
        current_std
            Standard deviation of the current measurement, in amperes.
        soc_drift_per_hour
            Additional random-walk allowance on state of charge, per hour.
        """
        b = np.asarray(model.input_direction()).reshape(-1)
        direction = np.asarray(model.soc_direction()).reshape(-1)
        samples_per_hour = 3600.0 / model.dt
        drift_variance = soc_drift_per_hour**2 / samples_per_hour
        return current_std**2 * np.outer(b, b) + drift_variance * np.outer(direction, direction)
