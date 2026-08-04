"""Cell-level models.

:class:`SPM` is the physics-based model: two particles, reduced-order solid
diffusion, and Butler-Volmer kinetics. :class:`ECM` is the equivalent-circuit
baseline it is meant to displace.

Both present the same interface (:class:`~cellkernel.models.base.CellModel`), so
estimators, the code generator and the verification harness treat them
identically.
"""

from .base import CellModel, ModelOutputs
from .ecm import ECM
from .spm import SPM

__all__ = ["CellModel", "ECM", "ModelOutputs", "SPM"]
