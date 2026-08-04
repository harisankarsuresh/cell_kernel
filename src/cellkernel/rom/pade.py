"""Pade approximation of the spherical solid-diffusion transfer function."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import tf2ss

from .base import DiffusionROM, normalised_series_coefficients

__all__ = ["PadeDiffusion", "pade_coefficients"]


def pade_coefficients(m: int, n: int) -> tuple[list[Fraction], list[Fraction]]:
    """Exact ``[m/n]`` Pade approximant of the normalised response.

    Finds polynomials :math:`N` of degree ``m`` and :math:`M` of degree ``n``
    with :math:`M(0) = 1` such that

    .. math::

        \\hat{H}(a) M(a) - N(a) = \\mathcal{O}(a^{m+n+1}).

    Matching coefficients of :math:`a^{k}` for :math:`k > m` gives a linear
    system in the ``n`` unknowns :math:`M_1 \\ldots M_n`, which is solved by
    Gaussian elimination over :class:`~fractions.Fraction`. The numerator then
    follows by direct convolution.

    Exact arithmetic is used deliberately. The Pade linear system inherits the
    conditioning of a Hankel matrix built from the series coefficients, whose
    magnitudes span many orders (``1``, ``1/15``, ``-1/525``, ``2/23625``,
    ``-37/9095625``, ...). Solving it in double precision starts to lose
    significant digits around ``n = 5`` and is unusable by ``n = 8``, whereas
    the rational solve is exact at any order and is performed once, offline.

    Returns
    -------
    numerator, denominator
        Coefficient lists in *ascending* powers of ``a``. ``denominator[0]`` is
        exactly ``1``.
    """
    if m < 0 or n < 1:
        raise ValueError("require m >= 0 and n >= 1")
    h = normalised_series_coefficients(m + n)

    # Rows k = m+1 .. m+n of  sum_j M_j h_{k-j} = 0, unknowns M_1..M_n.
    rows: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for k in range(m + 1, m + n + 1):
        row = [h[k - j] if 0 <= k - j <= m + n else Fraction(0) for j in range(1, n + 1)]
        rows.append(row)
        rhs.append(-h[k])

    denom = [Fraction(1)] + _solve_exact(rows, rhs)
    numer = [
        sum(denom[j] * h[k - j] for j in range(0, min(k, n) + 1)) for k in range(m + 1)
    ]
    return numer, denom


def _solve_exact(rows: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Gauss-Jordan elimination with partial pivoting over the rationals."""
    n = len(rhs)
    aug = [list(rows[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise np.linalg.LinAlgError(
                "Pade system is singular; the requested order is not attainable "
                "for this function"
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col], strict=False)]
    return [aug[i][n] for i in range(n)]


class PadeDiffusion(DiffusionROM):
    """Pade reduced-order model of solid diffusion.

    The exact transfer function is factored as an integrator times an analytic
    remainder, :math:`G(s) = 3/(Rs) \\cdot \\hat{H}(R^{2}s/D)`, and only the
    remainder is approximated. The integrator is retained exactly and realised
    as the first state, so the volume-averaged concentration obeys
    :math:`d\\bar{c}/dt = 3N/R` to machine precision regardless of ``order``.

    This matters for state estimation: coulomb counting is the one part of the
    model that must not drift, and here it is structurally exact rather than
    an artefact of the approximation.

    Parameters
    ----------
    radius, diffusivity
        Particle geometry and transport property.
    order
        Number of states. ``order = 1`` is pure coulomb counting with the
        correct ``R/5D`` steady-state offset absent; ``order = 3`` is the common
        choice in the literature and resolves the diffusion response over
        several decades; ``order >= 5`` approaches the exact response over the
        full frequency range of interest for automotive duty cycles.
    """

    name = "pade"

    def __init__(self, radius: float, diffusivity: float, order: int = 3) -> None:
        super().__init__(radius, diffusivity)
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = int(order)
        self._build()

    def _build(self) -> None:
        n = self.order - 1
        theta = self.time_constant
        if n == 0:
            # Hhat approximated by its DC value: the plain mass balance.
            self._Ah = np.zeros((0, 0))
            self._Bh = np.zeros((0, 1))
            self._Ch = np.zeros((1, 0))
            self._Dh = 1.0
            return

        # Diagonal [n/n] Pade in a; substituting a = theta * s makes the overall
        # system strictly proper of order n + 1, matching the G ~ s^-1/2
        # high-frequency roll-off of the true system.
        numer_a, denom_a = pade_coefficients(n, n)
        num_s = [float(c) * theta**i for i, c in enumerate(numer_a)][::-1]
        den_s = [float(c) * theta**i for i, c in enumerate(denom_a)][::-1]
        Ah, Bh, Ch, Dh = tf2ss(num_s, den_s)
        self._Ah = np.asarray(Ah, dtype=float)
        self._Bh = np.asarray(Bh, dtype=float).reshape(-1, 1)
        self._Ch = np.asarray(Ch, dtype=float).reshape(1, -1)
        self._Dh = float(np.asarray(Dh).reshape(-1)[0])

    @property
    def n_states(self) -> int:
        return self.order

    def continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self._Ah.shape[0]
        size = n + 1
        A = np.zeros((size, size))
        B = np.zeros((size, 1))
        # State 0 is c_bar, driven directly by the exact mass balance.
        B[0, 0] = 3.0 / self.radius
        if n:
            A[1:, 0:1] = self._Bh
            A[1:, 1:] = self._Ah
        C = np.zeros((2, size))
        C[0, 0] = self._Dh
        if n:
            C[0, 1:] = self._Ch
        C[1, 0] = 1.0
        D = np.zeros((2, 1))
        return A, B, C, D

    def uniform_state(self) -> np.ndarray:
        n = self._Ah.shape[0]
        x0 = np.zeros((n + 1, 1))
        x0[0, 0] = 1.0
        if n:
            # Equilibrium of the filter states when c_bar is held at c_init,
            # which enforces c_surf == c_init at rest since Hhat(0) = 1.
            x0[1:, 0:1] = -np.linalg.solve(self._Ah, self._Bh)
        return x0
