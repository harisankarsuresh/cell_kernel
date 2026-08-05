"""cellkernel: physics-based lithium-ion state estimation, from PDE to embedded C.

The package is organised as a pipeline. Each stage is usable on its own.

``cellkernel.rom``
    Reduced-order models of solid-phase diffusion in a spherical particle,
    delivered as discrete-time linear state-space systems.
``cellkernel.params``
    Cell parameter sets and open-circuit potential functions.
``cellkernel.models``
    Cell-level models: single particle, single particle with resolved
    electrolyte, coupled electro-thermal, and an equivalent-circuit baseline.
``cellkernel.estimators``
    Extended and unscented Kalman filters, and a dual filter that tracks
    capacity and resistance for state of health.
``cellkernel.degradation``
    Interphase growth and lithium plating: predicts capacity fade and reports
    plating margin, rather than tracking what has already happened.
``cellkernel.codegen``
    Emits a self-contained C99 estimator with static allocation and no
    dynamic memory, plus a flash, RAM and cycle-count budget. Optionally
    temperature-scheduled, valid across a range rather than at one point.
``cellkernel.verify``
    Compiles the generated C, replays the same input through it and through the
    Python reference, and reports the divergence.
``cellkernel.data``
    Synthetic current profiles and cycler-file loading.

Choosing a model
----------------
:class:`~cellkernel.models.SPM` unless something forces otherwise. It has the
fewest states and exactly linear dynamics, which is what makes an extended Kalman
filter well behaved on it. Reach for :class:`~cellkernel.models.SPMe` when the
duty cycle spends real time above 1C, and
:class:`~cellkernel.models.ThermalSPM` when the cell will not stay near the
temperature it was calibrated at. Both cost states, and the thermal one also
costs the linear-dynamics property.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
