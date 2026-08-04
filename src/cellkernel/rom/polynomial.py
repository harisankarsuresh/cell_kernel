"""Two-state polynomial (volume-averaged) solid-diffusion model."""

from __future__ import annotations

import numpy as np

from .base import DiffusionROM

__all__ = ["PolynomialDiffusion"]


class PolynomialDiffusion(DiffusionROM):
    """Subramanian three-parameter reduced-order diffusion model.

    Assumes a fourth-order polynomial concentration profile in the radial
    coordinate and tracks only two moments: the volume-averaged concentration
    :math:`\\bar{c}` and the volume-averaged concentration flux :math:`q`. The
    resulting system is

    .. math::

        \\frac{d\\bar{c}}{dt} = \\frac{3N}{R},
        \\qquad
        \\frac{dq}{dt} = -\\frac{30 D}{R^{2}} q + \\frac{45}{2} \\frac{N}{R^{2}},

    .. math::

        c_{\\text{surf}} = \\bar{c} + \\frac{8R}{35} q + \\frac{R}{35 D} N .

    Despite having only two states it reproduces the exact steady-state surface
    offset: substituting :math:`q \\to 3N/4D` gives

    .. math::

        c_{\\text{surf}} - \\bar{c}
            \\to \\frac{8R}{35}\\frac{3N}{4D} + \\frac{RN}{35D}
             = \\frac{6RN}{35D} + \\frac{RN}{35D}
             = \\frac{RN}{5D},

    matching the :math:`R/5D` term of the exact low-frequency expansion
    identically rather than approximately. The test suite asserts this to
    machine precision.

    This is the cheapest model that is still quantitatively trustworthy under
    slowly varying load, which makes it the natural default for a
    resource-constrained target. Its weakness is short, sharp transients: with a
    single relaxation mode at :math:`30D/R^{2}` it cannot represent the
    :math:`\\sqrt{s}` spread of timescales that a real particle exhibits, so
    surface concentration is under-damped during fast pulses. Use
    :class:`~cellkernel.rom.pade.PadeDiffusion` or
    :class:`~cellkernel.rom.spectral.SpectralDiffusion` when pulse power
    accuracy matters.

    References
    ----------
    Subramanian, Diwakar and Tapriyal (2005), *Efficient macro-micro scale
    coupled modeling of batteries*, J. Electrochem. Soc. 152(10) A2002.
    """

    name = "poly"

    def __init__(self, radius: float, diffusivity: float) -> None:
        super().__init__(radius, diffusivity)

    @property
    def n_states(self) -> int:
        return 2

    def continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = self.radius
        D = self.diffusivity
        A = np.array([[0.0, 0.0], [0.0, -30.0 * D / R**2]])
        B = np.array([[3.0 / R], [22.5 / R**2]])
        C = np.array([[1.0, 8.0 * R / 35.0], [1.0, 0.0]])
        Df = np.array([[R / (35.0 * D)], [0.0]])
        return A, B, C, Df

    def uniform_state(self) -> np.ndarray:
        return np.array([[1.0], [0.0]])
