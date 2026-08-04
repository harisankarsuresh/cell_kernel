"""Finite-volume discretisation of spherical solid diffusion."""

from __future__ import annotations

import numpy as np

from .base import DiffusionROM

__all__ = ["FiniteVolumeDiffusion"]


class FiniteVolumeDiffusion(DiffusionROM):
    """Conservative finite-volume model of diffusion in a sphere.

    The particle is divided into ``n_shells`` concentric control volumes of
    equal radial thickness. Integrating the diffusion equation over shell
    :math:`i` and applying the divergence theorem gives

    .. math::

        V_i \\frac{dc_i}{dt} = A_i q_i - A_{i+1} q_{i+1},
        \\qquad
        q_k = -D \\frac{c_k - c_{k-1}}{\\Delta r},

    with :math:`A_0 = 0` at the centre and :math:`q_N = -N` imposed by the
    surface flux condition. Because interior fluxes appear once with each sign,
    they telescope when the shells are summed, so total lithium is conserved to
    machine precision and :math:`d\\bar{c}/dt = 3N/R` holds exactly for any
    number of shells.

    The surface concentration is recovered by extrapolating from the outermost
    cell centre along the known boundary gradient,

    .. math::

        c_{\\text{surf}} = c_{N-1} + \\frac{\\Delta r}{2} \\frac{N}{D},

    which is second-order accurate and introduces a direct feedthrough term.
    Using the cell-centre value alone instead would be first-order and would
    systematically under-predict surface depletion at high rate, biasing the
    open-circuit potential and hence the estimated state of charge.

    Notes
    -----
    This is the structure that a hand-written embedded solver would normally
    step with an explicit or Crank-Nicolson update. Explicit stepping is stable
    only for :math:`\\Delta t < \\Delta r^{2} / 2D`, which for a 5 um particle
    with :math:`D = 10^{-14}` forces a step far below the 100 ms budget of a
    typical battery-management task, and Crank-Nicolson costs a tridiagonal
    solve every step. :meth:`~cellkernel.rom.base.DiffusionROM.discretise`
    sidesteps both by exponentiating the generator once, offline: the online
    cost becomes a single dense matrix-vector product that is unconditionally
    stable and exact for piecewise-constant current.
    """

    name = "fv"

    def __init__(self, radius: float, diffusivity: float, n_shells: int = 10) -> None:
        super().__init__(radius, diffusivity)
        if n_shells < 2:
            raise ValueError("n_shells must be at least 2")
        self.n_shells = int(n_shells)

    @property
    def n_states(self) -> int:
        return self.n_shells

    @property
    def dr(self) -> float:
        """Radial thickness of one shell."""
        return self.radius / self.n_shells

    def _geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """Face areas and shell volumes, both with the common ``4 pi`` dropped."""
        n = self.n_shells
        edges = np.arange(n + 1) * self.dr
        face = edges**2
        volume = edges[1:] ** 3 - edges[:-1] ** 3
        return face, volume

    def continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_shells
        dr = self.dr
        D = self.diffusivity
        face, volume = self._geometry()

        A = np.zeros((n, n))
        B = np.zeros((n, 1))
        # Coefficient of the flux across the face between cells i-1 and i.
        coeff = 3.0 * D / dr
        for i in range(n):
            if i > 0:
                g = coeff * face[i] / volume[i]
                A[i, i] -= g
                A[i, i - 1] += g
            if i < n - 1:
                g = coeff * face[i + 1] / volume[i]
                A[i, i] -= g
                A[i, i + 1] += g
        B[n - 1, 0] = 3.0 * face[n] / volume[n - 1]

        C = np.zeros((2, n))
        C[0, n - 1] = 1.0
        C[1, :] = volume / volume.sum()
        Df = np.zeros((2, 1))
        Df[0, 0] = 0.5 * dr / D
        return A, B, C, Df

    def uniform_state(self) -> np.ndarray:
        return np.ones((self.n_shells, 1))
