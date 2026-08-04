"""Recursive state and health estimators.

:class:`EKF` is the workhorse and the one the code generator emits.
:class:`UKF` trades cost for accuracy where the voltage curve is sharply
nonlinear. :class:`DualEKF` adds capacity retention and resistance growth to the
state vector.

All three exploit the same structural fact: the process models in this package
are exactly linear, so the prediction step is exact and only the voltage
measurement needs approximating.
"""

from .base import Estimator, EstimatorOutputs
from .dual import DualEKF, HealthEstimate
from .ekf import EKF
from .ukf import UKF, safe_cholesky

__all__ = [
    "DualEKF",
    "EKF",
    "Estimator",
    "EstimatorOutputs",
    "HealthEstimate",
    "UKF",
    "safe_cholesky",
]
