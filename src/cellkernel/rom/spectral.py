"""Eigenfunction (spectral) reduced-order model of spherical solid diffusion."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .base import DiffusionROM

__all__ = ["SpectralDiffusion", "zero_flux_eigenvalues"]


def zero_flux_eigenvalues(count: int) -> np.ndarray:
    """First ``count`` positive roots of :math:`\\tan \\lambda = \\lambda`.

    These are the eigenvalues of the spherical Laplacian under a zero-gradient
    condition at the particle surface. The root equation is solved in the form
    :math:`\\sin \\lambda - \\lambda \\cos \\lambda = 0`, which is entire, rather
    than :math:`\\tan \\lambda - \\lambda`, which has poles inside every
    bracketing interval.

    Root ``k`` lies in :math:`(k\\pi, (k + 1/2)\\pi)` for :math:`k \\ge 1`; the
    residual changes sign across that interval, so bisection is guaranteed to
    converge.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    def residual(x: float) -> float:
        return np.sin(x) - x * np.cos(x)

    roots = np.empty(count)
    for k in range(1, count + 1):
        lo = k * np.pi + 1e-12
        hi = (k + 0.5) * np.pi - 1e-12
        roots[k - 1] = brentq(residual, lo, hi, xtol=1e-14, rtol=1e-15)
    return roots


class SpectralDiffusion(DiffusionROM):
    """Modal truncation of spherical diffusion onto Laplacian eigenfunctions.

    Expanding the concentration field as

    .. math::

        c(\\rho, t) = \\bar{c}(t)
            + \\sum_{k \\ge 1} \\alpha_k(t) \\phi_k(\\rho),
        \\qquad
        \\phi_k(\\rho) = \\frac{\\sin(\\lambda_k \\rho)}{\\lambda_k \\rho},

    with :math:`\\tan \\lambda_k = \\lambda_k`, and projecting the PDE onto the
    :math:`\\rho^{2}`-weighted inner product gives a set of *decoupled* scalar
    equations:

    .. math::

        \\frac{d\\bar{c}}{dt} = \\frac{3N}{R},
        \\qquad
        \\frac{d\\alpha_k}{dt}
            = -\\frac{D \\lambda_k^{2}}{R^{2}} \\alpha_k
              + \\frac{2}{R \\cos \\lambda_k} N,
        \\qquad
        c_{\\text{surf}} = \\bar{c} + \\sum_k \\alpha_k \\cos \\lambda_k .

    The modes are orthogonal to the constant mode, so the mean and the shape
    dynamics never mix and the mass balance is exact. Because ``A`` is diagonal,
    the discrete-time update is ``n`` independent scalar multiply-accumulates,
    which is the cheapest possible structure on a microcontroller and needs no
    matrix at all.

    Parameters
    ----------
    radius, diffusivity
        Particle geometry and transport property.
    n_modes
        Number of retained eigenmodes. Total state count is ``n_modes + 1``.
    residualise
        Whether to fold the static gain of the discarded modes into a direct
        feedthrough term. Default ``True``. See the notes below.

    Notes
    -----
    Plain modal truncation is systematically biased at steady state. The exact
    surface offset under constant flux is :math:`RN/5D`, which the full modal sum
    reproduces through the identity :math:`\\sum_{k \\ge 1} \\lambda_k^{-2} =
    1/10`. Keeping only ``n_modes`` terms recovers just part of that sum, and the
    series converges slowly because :math:`\\lambda_k \\to (k + 1/2)\\pi`: one
    mode recovers 49.5% of the offset, three modes 74.7%, and even eight modes
    are 11% short. That deficit appears directly as a surface-concentration
    error, and hence as an open-circuit-potential error, at any sustained rate.

    With ``residualise=True`` the missing static gain is added as an
    instantaneous term,

    .. math::

        D_{\\!f} = \\frac{2R}{D}
                   \\left( \\frac{1}{10}
                   - \\sum_{k=1}^{n} \\lambda_k^{-2} \\right),

    which is static condensation of the truncated modes: rather than discarding
    them, it treats them as infinitely fast. That is the better approximation,
    because the modes being dropped genuinely are the fast ones -- their time
    constants fall off as :math:`\\lambda_k^{-2} \\sim (k\\pi)^{-2}`. The model
    becomes exact in the DC limit at every order for the cost of one extra
    multiply. Measured over :math:`\\omega R^{2}/D \\le 10`, residualisation
    improves worst-case error by one to two orders of magnitude (at three modes,
    from 1.3e-1 to 2.7e-3), and two residualised modes beat five plain ones by
    more than tenfold.

    The trade-off is at the very top of the frequency range. A non-zero
    feedthrough makes :math:`G \\to D_{\\!f}` as :math:`s \\to \\infty`, whereas
    the true response rolls off as :math:`s^{-1/2}`. Beyond about
    :math:`\\omega R^{2}/D = 10^{3}` the residualised model is therefore *worse*
    than plain truncation. That regime is milliseconds for a typical particle,
    well below the bandwidth of any current sensor a battery-management unit
    has, so the default is ``True``; set it to ``False`` if the model is being
    driven by high-frequency ripple.
    """

    name = "spectral"

    def __init__(
        self,
        radius: float,
        diffusivity: float,
        n_modes: int = 3,
        residualise: bool = True,
    ) -> None:
        super().__init__(radius, diffusivity)
        if n_modes < 0:
            raise ValueError("n_modes must be non-negative")
        self.n_modes = int(n_modes)
        self.residualise = bool(residualise)
        self.eigenvalues = zero_flux_eigenvalues(self.n_modes)

    @property
    def n_states(self) -> int:
        return self.n_modes + 1

    def continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_modes
        lam = self.eigenvalues
        cos_lam = np.cos(lam)

        A = np.zeros((n + 1, n + 1))
        B = np.zeros((n + 1, 1))
        B[0, 0] = 3.0 / self.radius
        if n:
            A[1:, 1:] = np.diag(-self.diffusivity * lam**2 / self.radius**2)
            B[1:, 0] = 2.0 / (self.radius * cos_lam)

        C = np.zeros((2, n + 1))
        C[0, 0] = 1.0
        if n:
            C[0, 1:] = cos_lam
        C[1, 0] = 1.0
        D = np.zeros((2, 1))
        D[0, 0] = self.residual_gain
        return A, B, C, D

    def uniform_state(self) -> np.ndarray:
        x0 = np.zeros((self.n_modes + 1, 1))
        x0[0, 0] = 1.0
        return x0

    def steady_state_offset_factor(self) -> float:
        """Truncated value of :math:`\\sum_k \\lambda_k^{-2}`, exactly ``1/10`` in the limit."""
        if self.n_modes == 0:
            return 0.0
        return float(np.sum(1.0 / self.eigenvalues**2))

    @property
    def residual_gain(self) -> float:
        """Static feedthrough compensating the truncated modes.

        Zero when ``residualise`` is ``False``. Otherwise
        :math:`(2R/D)(1/10 - \\sum_{k \\le n} \\lambda_k^{-2})`, which is the
        exact static contribution of every mode that was dropped.
        """
        if not self.residualise:
            return 0.0
        return 2.0 * self.radius * (0.1 - self.steady_state_offset_factor()) / self.diffusivity
