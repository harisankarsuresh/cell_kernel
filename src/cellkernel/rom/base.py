"""Reduced-order models for solid-phase diffusion in a spherical particle.

All reduced-order models in this package approximate the same physical system:
Fickian diffusion of intercalated lithium in a sphere of radius ``R`` with
diffusivity ``D``, driven by a molar flux applied at the surface.

.. math::

    \\frac{\\partial c}{\\partial t}
        = \\frac{D}{r^{2}} \\frac{\\partial}{\\partial r}
          \\left( r^{2} \\frac{\\partial c}{\\partial r} \\right),
    \\qquad
    \\left. \\frac{\\partial c}{\\partial r} \\right|_{r=0} = 0,
    \\qquad
    \\left. D \\frac{\\partial c}{\\partial r} \\right|_{r=R} = N,

where ``N`` is the molar influx density (mol m-2 s-1), positive when lithium
enters the particle.

Two scalar outputs matter for cell-level models:

``c_surf``
    Surface concentration, which sets the open-circuit potential and the
    exchange current density.
``c_bar``
    Volume-averaged concentration, which is the particle state of charge and is
    exactly conserved by the mass balance :math:`d\\bar{c}/dt = 3N/R`.

Every model is exposed as a *discrete-time linear state-space system* obtained
by zero-order-hold sampling at a fixed step ``dt``:

.. math::

    x_{k+1} = A x_k + B N_k,
    \\qquad
    \\begin{bmatrix} c_{\\text{surf}} \\\\ \\bar{c} \\end{bmatrix}_k
        = C x_k + D_{\\!f} N_k .

This is the key design decision of :mod:`cellkernel`. The expensive and
numerically delicate work -- discretising a parabolic PDE, choosing a stable
time integrator, approximating a transcendental transfer function -- is done
once, offline, in double precision. What is shipped to the microcontroller is a
small dense matrix and a matrix-vector product, which is unconditionally stable
for any ``dt`` and exact for piecewise-constant current.

The analytic transfer function of the exact system, used throughout for
verification, is

.. math::

    G(s) = \\frac{\\tilde{c}_{\\text{surf}}(s)}{\\tilde{N}(s)}
         = \\frac{R}{D}\\,
           \\frac{\\sinh \\xi}{\\xi \\cosh \\xi - \\sinh \\xi},
    \\qquad \\xi = R \\sqrt{s / D},

whose low-frequency expansion is

.. math::

    G(s) = \\frac{3}{R s} + \\frac{R}{5 D}
           - \\frac{R^{3} s}{175 D^{2}} + \\mathcal{O}(s^{2}).

The leading term is the mass balance; the constant :math:`R / 5D` is the
steady-state surface-to-average concentration offset. Both are reproduced
exactly by the spectral and polynomial models and to the requested order by the
Pade model, and these identities are asserted in the test suite.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial

import numpy as np
from scipy.signal import cont2discrete


@lru_cache(maxsize=32)
def normalised_series_coefficients(order: int) -> tuple[Fraction, ...]:
    """Exact Taylor coefficients of the normalised diffusion response.

    The exact surface transfer function factorises as

    .. math::

        G(s) = \\frac{3}{R s}\\, \\hat{H}(a),
        \\qquad a = \\frac{R^{2} s}{D},

    where the integrator ``3 / (R s)`` carries the mass balance and
    :math:`\\hat{H}` is analytic at the origin with :math:`\\hat{H}(0) = 1`.
    Writing :math:`\\xi = \\sqrt{a}` and using

    .. math::

        \\sinh \\xi = \\xi \\sum_{k \\ge 0} \\frac{a^{k}}{(2k+1)!},
        \\qquad
        \\xi \\cosh \\xi - \\sinh \\xi
            = \\xi^{3} \\sum_{m \\ge 0} \\frac{2(m+1)}{(2m+3)!} a^{m},

    gives :math:`\\hat{H} = P / (3Q)` as a ratio of two power series with
    rational coefficients. This routine performs the division in exact
    arithmetic, which matters because the coefficients alternate in sign and
    shrink by orders of magnitude, so floating-point division of the series
    loses accuracy well before the Pade solve does.

    Returns
    -------
    tuple of Fraction
        Coefficients :math:`\\hat{h}_0 \\ldots \\hat{h}_{\\text{order}}`. The
        first few are ``1``, ``1/15``, ``-1/525``, ``2/23625``.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    n = order + 1
    numer = [Fraction(1, factorial(2 * k + 1)) for k in range(n)]
    denom = [Fraction(3 * 2 * (m + 1), factorial(2 * m + 3)) for m in range(n)]
    # denom[0] == 1 by construction, so the division below is well posed.
    out: list[Fraction] = []
    for i in range(n):
        acc = numer[i] - sum(denom[j] * out[i - j] for j in range(1, i + 1))
        out.append(acc)
    return tuple(out)


def low_frequency_series(s: complex, radius: float, diffusivity: float, order: int = 5) -> complex:
    """Cancellation-free evaluation of ``G(s)`` for small ``|s|``.

    Sums :math:`3 \\hat{h}_i R^{2i-1} s^{i-1} / D^{i}`, i.e. the expansion
    :math:`3/(Rs) + R/5D - R^{3}s/175D^{2} + 2R^{5}s^{2}/7875D^{3} - \\ldots`
    """
    coeffs = normalised_series_coefficients(order)
    total = 0j
    for i, h in enumerate(coeffs):
        total += 3.0 * float(h) * radius ** (2 * i - 1) * s ** (i - 1) / diffusivity**i
    return total


@dataclass(frozen=True)
class DiscreteStateSpace:
    """Zero-order-hold discretisation of a diffusion reduced-order model.

    Attributes
    ----------
    A, B
        State update matrices, shapes ``(n, n)`` and ``(n, 1)``.
    C, D
        Output matrices, shapes ``(2, n)`` and ``(2, 1)``. Output row 0 is the
        surface concentration and row 1 the volume-averaged concentration.
    dt
        Sample period in seconds.
    x0_from_uniform
        Column vector mapping a spatially uniform initial concentration onto the
        state vector, so ``x0 = x0_from_uniform * c_init``.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    dt: float
    x0_from_uniform: np.ndarray

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    def initial_state(self, c_init: float) -> np.ndarray:
        """State vector representing a particle uniformly at ``c_init``."""
        return self.x0_from_uniform.reshape(-1) * float(c_init)

    def step(self, x: np.ndarray, flux: float) -> np.ndarray:
        """Advance the state one sample under constant molar influx ``flux``."""
        return self.A @ x + self.B.reshape(-1) * float(flux)

    def outputs(self, x: np.ndarray, flux: float) -> tuple[float, float]:
        """Return ``(c_surf, c_bar)`` for state ``x`` and influx ``flux``."""
        y = self.C @ x + self.D.reshape(-1) * float(flux)
        return float(y[0]), float(y[1])

    def simulate(self, c_init: float, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replay a flux sequence and return ``(c_surf, c_bar)`` trajectories.

        The returned arrays have the same length as ``flux``; entry ``k`` is the
        output evaluated with state ``x_k`` and input ``flux_k``, before the
        state update to ``k + 1``.
        """
        flux = np.asarray(flux, dtype=float).reshape(-1)
        x = self.initial_state(c_init)
        c_surf = np.empty(flux.size)
        c_bar = np.empty(flux.size)
        for k, u in enumerate(flux):
            c_surf[k], c_bar[k] = self.outputs(x, u)
            x = self.step(x, u)
        return c_surf, c_bar


class DiffusionROM(abc.ABC):
    """Base class for spherical solid-diffusion reduced-order models.

    Parameters
    ----------
    radius
        Particle radius in metres.
    diffusivity
        Solid-phase diffusion coefficient in m2 s-1.
    """

    #: Short identifier used in generated C code and reports.
    name: str = "rom"

    def __init__(self, radius: float, diffusivity: float) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if diffusivity <= 0.0:
            raise ValueError("diffusivity must be positive")
        self.radius = float(radius)
        self.diffusivity = float(diffusivity)

    @property
    def time_constant(self) -> float:
        """Diffusion time constant ``R^2 / D`` in seconds."""
        return self.radius**2 / self.diffusivity

    @property
    @abc.abstractmethod
    def n_states(self) -> int:
        """Number of states in the reduced model."""

    @abc.abstractmethod
    def continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return continuous-time ``(A, B, C, D)`` with outputs ``[c_surf, c_bar]``."""

    @abc.abstractmethod
    def uniform_state(self) -> np.ndarray:
        """Column map from a uniform concentration to the state vector."""

    def state_scaling(self) -> np.ndarray:
        """Diagonal similarity transform putting every state in concentration units.

        The raw realisations are badly scaled. A Pade realisation of a 6 um
        particle holds a bulk concentration near ``1e4`` alongside filter
        coordinates near ``1e10``, and the polynomial model's flux moment reaches
        ``1e8``; the numbers come out of the transfer-function algebra and have no
        physical magnitude of their own. Nothing in a pure simulation minds, but
        two things downstream mind a great deal.

        A Kalman filter is specified by covariance matrices in state units. With a
        state vector spanning six orders of magnitude, a diagonal covariance is
        meaningless: whatever value is chosen is simultaneously far too tight for
        one state and far too loose for another, and the filter either ignores the
        measurement or diverges. Single-precision arithmetic on a microcontroller
        minds even more, having only about seven decimal digits to spend.

        The transform implemented here is ``x = T x'`` with ``T`` diagonal, chosen
        so that each state equals *its own additive contribution to the surface
        concentration*. Every state then carries units of mol m-3 and a magnitude
        set by the physics rather than by the algebra. The first state is left
        untouched so that the exact mass balance and the volume-averaged output
        keep their meaning, and states that do not feed the surface output at all
        -- the interior cells of the finite-volume model, which are already
        concentrations -- are also left alone. The finite-volume model therefore
        gets the identity, correctly.
        """
        _, _, C, _ = self.continuous()
        scale = np.ones(self.n_states)
        for i in range(1, self.n_states):
            weight = abs(float(C[0, i]))
            if weight > 0.0:
                scale[i] = 1.0 / weight
        return scale

    def discretise(self, dt: float, balance: bool = True) -> DiscreteStateSpace:
        """Zero-order-hold discretisation at sample period ``dt``.

        Uses the exact matrix exponential rather than a finite-difference
        integrator, so the result is stable and exact for piecewise-constant
        flux at any ``dt``.

        Parameters
        ----------
        dt
            Sample period in seconds.
        balance
            Apply :meth:`state_scaling` so that all states share concentration
            units. Leave enabled unless you specifically need the raw realisation.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        A, B, C, D = self.continuous()
        x0 = np.asarray(self.uniform_state(), dtype=float).reshape(-1, 1)
        if balance:
            t = self.state_scaling()
            inverse = 1.0 / t
            # Similarity transform: A -> T^-1 A T, B -> T^-1 B, C -> C T.
            A = inverse[:, None] * A * t[None, :]
            B = inverse[:, None] * B
            C = C * t[None, :]
            x0 = inverse[:, None] * x0
        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="zoh")
        Ad, Bd = self._project_conservation(Ad, Bd, Cd, x0.reshape(-1), dt)
        return DiscreteStateSpace(
            A=np.ascontiguousarray(Ad, dtype=float),
            B=np.ascontiguousarray(Bd, dtype=float).reshape(-1, 1),
            C=np.ascontiguousarray(Cd, dtype=float),
            D=np.ascontiguousarray(Dd, dtype=float).reshape(-1, 1),
            dt=float(dt),
            x0_from_uniform=np.ascontiguousarray(x0, dtype=float).reshape(-1, 1),
        )

    def _project_conservation(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Cd: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restore the two conservation invariants exactly after discretisation.

        Two properties hold for the continuous system by construction, and the
        whole package leans on them:

        ``A u = 0``
            A uniformly loaded particle carrying no flux is a fixed point -- a
            rested cell does not drift.
        ``w A = 0`` and ``w B = 3/R``
            where ``w`` is the row of ``C`` extracting the volume average. This is
            the mass balance, and it is why state of charge is exact coulomb
            counting rather than something that accumulates discretisation error.

        After zero-order-hold sampling they become ``A_d u = u``, ``w A_d = w`` and
        ``w B_d = 3 dt / R``. They should survive the matrix exponential, and at
        any sane step they do, to a few machine epsilons.

        They do not always survive an *insane* step. At ``dt = 100 R^2/D`` the
        generator spans fifteen orders of magnitude between the conserved mode and
        the fastest decayed one, and scaling-and-squaring loses accuracy on a
        matrix that stiff. Measured against SciPy 1.15, ``A_d[0, 0]`` for a Pade
        model came out as 0.99988 on one platform and 1.000065 on another instead
        of exactly 1, and a rested particle grew by 3% over 500 steps. SciPy 1.18
        gets the same case right. Depending on the linear-algebra backend to
        preserve a property this fundamental is not a reasonable position for the
        library to take.

        So the invariants are imposed rather than hoped for. Both are enforced by
        rank-one corrections, applied right-eigenvector first and then
        left-eigenvector; because ``w u = 1``, the second correction leaves the
        first intact:

        .. math::

            A_d \\leftarrow A_d - (A_d u - u) w,
            \\qquad
            A_d \\leftarrow A_d - u (w A_d - w).

        At a realistic step this is a no-op to within rounding -- the corrections
        are of order 1e-16 -- so nothing is being papered over. At an extreme step
        it guarantees that the conserved quantity stays conserved even where the
        rest of the matrix has lost accuracy, which is the honest failure mode:
        degrade the shape dynamics, never the lithium count.
        """
        w = np.asarray(Cd, dtype=float)[1].copy()
        u = np.asarray(u, dtype=float).reshape(-1)
        overlap = float(w @ u)
        if not np.isfinite(overlap) or abs(overlap) < 1e-12:
            # No usable conserved direction; leave the discretisation untouched.
            return Ad, Bd
        u = u / overlap

        Ad = np.array(Ad, dtype=float, copy=True)
        Bd = np.array(Bd, dtype=float, copy=True).reshape(-1)

        Ad -= np.outer(Ad @ u - u, w)
        Ad -= np.outer(u, w @ Ad - w)
        Bd -= u * (float(w @ Bd) - 3.0 * dt / self.radius)
        return Ad, Bd.reshape(-1, 1)

    def transfer_function(self, s: complex) -> complex:
        """Reduced-model transfer function from molar influx to ``c_surf``."""
        A, B, C, D = self.continuous()
        n = A.shape[0]
        resolvent = np.linalg.solve(s * np.eye(n) - A, B)
        return complex((C @ resolvent + D)[0, 0])

    def exact_transfer_function(self, s: complex) -> complex:
        """Exact transfer function of the full PDE, for error assessment."""
        return exact_surface_transfer_function(s, self.radius, self.diffusivity)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(radius={self.radius:.4g}, "
            f"diffusivity={self.diffusivity:.4g}, n_states={self.n_states})"
        )


def exact_surface_transfer_function(s: complex, radius: float, diffusivity: float) -> complex:
    """Exact ``c_surf / N`` transfer function of spherical diffusion.

    With :math:`\\xi = R \\sqrt{s/D}` the textbook form is

    .. math::

        G(s) = \\frac{R}{D}
               \\frac{\\sinh \\xi}{\\xi \\cosh \\xi - \\sinh \\xi},

    which is unusable as written at both ends of the frequency range. Two
    reformulations are applied.

    At low frequency the denominator is a difference of two quantities that both
    tend to :math:`\\xi` while their difference tends to :math:`\\xi^{3}/3`, so
    roughly ``2 log10(1/|xi|)`` significant digits are lost. Below
    ``|xi| = 0.1`` the exact rational series is summed instead.

    At high frequency :math:`\\sinh` and :math:`\\cosh` both overflow a double
    near ``|xi| = 710`` even though their ratio is well behaved. Dividing
    through by :math:`\\sinh \\xi` gives the algebraically identical

    .. math::

        G(s) = \\frac{R}{D} \\frac{1}{\\xi \\coth \\xi - 1},

    and ``tanh`` saturates smoothly to one instead of overflowing. This matters
    in practice: a 1 Hz current ripple on a particle with ``R^2/D`` of about
    1000 s already sits at ``|xi| ~ 80``, and pulse content an order of
    magnitude faster would overflow the naive expression outright.

    Beyond ``Re(xi) > 20`` even ``tanh`` warns while returning the right answer,
    so ``coth`` is replaced by one; the neglected term is
    :math:`2 e^{-2\\xi}`, below ``1e-17``. For ``s`` in the closed right half
    plane the principal square root keeps ``|arg(xi)| <= 45 deg``, so a large
    ``|xi|`` always implies a large ``Re(xi)`` and the test is sound.
    """
    s = complex(s)
    if s == 0:
        return complex(np.inf)
    theta = radius**2 / diffusivity
    xi = np.sqrt(s * theta)
    if abs(xi) < 0.1:
        return low_frequency_series(s, radius, diffusivity, order=8)
    coth = 1.0 + 0.0j if xi.real > 20.0 else 1.0 / np.tanh(xi)
    return (radius / diffusivity) / (xi * coth - 1.0)
