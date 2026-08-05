"""Cell-level models.

:class:`SPM` is the physics-based model: two particles, reduced-order solid
diffusion, and Butler-Volmer kinetics. :class:`ThermalSPM` adds cell temperature
as a state, with the reduced-order matrices gain-scheduled over temperature.
:class:`ECM` is the equivalent-circuit baseline the other two are meant to
displace.

All three present the same interface
(:class:`~cellkernel.models.base.CellModel`), so estimators, the code generator
and the verification harness treat them identically.

Choosing between :class:`SPM` and :class:`ThermalSPM` is a real decision rather
than a matter of fidelity for its own sake. The isothermal model has exactly
linear state dynamics, which means an extended Kalman filter on it has no
linearisation error in its prediction step; adding a temperature state gives that
up, because heat generation is quadratic in current and the transition matrix
starts depending on a state. Take the isothermal model when the cell stays near
the temperature it was calibrated at, and the thermal one when it does not.
"""

from .base import CellModel, ModelOutputs
from .ecm import ECM
from .spm import SPM
from .thermal import ThermalSPM

__all__ = ["CellModel", "ECM", "ModelOutputs", "SPM", "ThermalSPM"]
