"""cellkernel: physics-based lithium-ion state estimation, from PDE to embedded C.

The package is organised as a pipeline. Each stage is usable on its own.

``cellkernel.rom``
    Reduced-order models of solid-phase diffusion in a spherical particle,
    delivered as discrete-time linear state-space systems.
``cellkernel.params``
    Cell parameter sets and open-circuit potential functions.
``cellkernel.models``
    Cell-level models -- single particle, single particle with electrolyte,
    equivalent circuit -- with optional lumped thermal coupling.
``cellkernel.estimators``
    Extended and unscented Kalman filters, and a dual filter that tracks
    capacity and resistance for state of health.
``cellkernel.codegen``
    Emits a self-contained C99 estimator with static allocation and no
    dynamic memory, plus a flash, RAM and cycle-count budget.
``cellkernel.verify``
    Compiles the generated C, replays the same input through it and through the
    Python reference, and reports the divergence.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
