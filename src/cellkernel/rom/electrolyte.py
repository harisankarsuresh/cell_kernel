"""Salt transport across the electrolyte, as a conservative linear system.

The solid-diffusion models in this package reduce a PDE in a spherical particle.
This one reduces a PDE across the cell sandwich: negative coating, separator,
positive coating, in the through-plane direction.

.. math::

    \\varepsilon_k \\frac{\\partial c_e}{\\partial t}
        = \\frac{\\partial}{\\partial x}
          \\left( D_e^{\\text{eff}} \\frac{\\partial c_e}{\\partial x} \\right)
        + \\frac{(1 - t_+)}{F} a_k j_k ,

with zero flux at both current collectors. The source term is what couples it to
the electrochemistry: lithium leaving a particle enters the electrolyte, so the
negative coating is a source on discharge and the positive coating a sink.

Two structural facts make this tractable, and both are exploited rather than
merely noted.

**The source integrates to zero.** Over the negative coating the volumetric source
integrates to :math:`(1 - t_+) I / F A`, and over the positive to exactly minus
that. Total salt in the cell is therefore conserved for any current, which means
this system has the same character as solid diffusion: one conserved mode and a
set of decaying shape modes. The conservation is imposed explicitly after
discretisation for the same reason it is there -- see
:meth:`~cellkernel.rom.base.DiffusionROM.discretise`.

**It is linear in current.** Neither the transport coefficients nor the source
depend on concentration, once the usual constant-property assumptions are made,
so the whole thing is a linear system driven by cell current and can be
discretised offline exactly like everything else here.

What is deliberately not modelled: concentration-dependent diffusivity,
conductivity and transference number. All three vary appreciably over the
concentration range a cell actually visits, and including them would make the
system nonlinear and destroy the offline discretisation that the rest of the
package is built on. The error that buys is discussed in
:class:`~cellkernel.models.spme.SPMe`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete

__all__ = ["ElectrolyteDiffusion", "ElectrolyteStateSpace"]


@dataclass(frozen=True)
class ElectrolyteStateSpace:
    """Discrete-time salt transport, driven by cell current in amperes.

    Outputs are the volume-averaged electrolyte concentration in each coating,
    which is what a single particle model with electrolyte needs: the negative
    and positive averages set the concentration overpotential between them, and
    each sets the local exchange current density in its own electrode.
    """

    A: np.ndarray
    B: np.ndarray
    #: Row 0 is the negative-coating average, row 1 the positive-coating average.
    C: np.ndarray
    dt: float
    #: Maps a uniform electrolyte concentration onto the state vector.
    uniform: np.ndarray

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    def initial_state(self, concentration: float) -> np.ndarray:
        return self.uniform * float(concentration)

    def step(self, x: np.ndarray, current: float) -> np.ndarray:
        return self.A @ np.asarray(x, dtype=float).reshape(-1) + self.B.reshape(-1) * float(current)

    def averages(self, x: np.ndarray) -> tuple[float, float]:
        """``(negative, positive)`` coating-average concentrations, mol m-3."""
        y = self.C @ np.asarray(x, dtype=float).reshape(-1)
        return float(y[0]), float(y[1])


class ElectrolyteDiffusion:
    """Finite-volume salt transport across the cell sandwich.

    Parameters
    ----------
    thickness_negative, thickness_separator, thickness_positive
        Layer thicknesses in metres.
    porosity_negative, porosity_separator, porosity_positive
        Electrolyte volume fractions.
    diffusivity
        Bulk salt diffusivity in m2 s-1, before the Bruggeman correction.
    transference_number
        Cation transference number.
    electrode_area
        Geometric electrode area in m2.
    bruggeman
        Exponent in :math:`D^{\\text{eff}} = \\varepsilon^{b} D`.
    cells_negative, cells_separator, cells_positive
        Control volumes per region.

    Notes
    -----
    Interface conductance between control volumes uses the harmonic mean of the
    two effective diffusivities weighted by half-widths, rather than an
    arithmetic mean. That matters at the coating-separator boundaries, where
    porosity jumps by a factor of two or more: an arithmetic mean of the
    diffusivities there overestimates transport through the interface, because
    the resistance of a series pair is dominated by the *worse* of the two, not
    their average. The harmonic form is the one that reproduces a series
    resistance correctly, and it is the reason a coarse grid still lands on the
    right steady-state gradient.
    """

    def __init__(
        self,
        thickness_negative: float,
        thickness_separator: float,
        thickness_positive: float,
        porosity_negative: float,
        porosity_separator: float,
        porosity_positive: float,
        diffusivity: float,
        transference_number: float,
        electrode_area: float,
        bruggeman: float = 1.5,
        cells_negative: int = 4,
        cells_separator: int = 3,
        cells_positive: int = 4,
    ) -> None:
        for name, value in (
            ("thickness_negative", thickness_negative),
            ("thickness_separator", thickness_separator),
            ("thickness_positive", thickness_positive),
            ("diffusivity", diffusivity),
            ("electrode_area", electrode_area),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("porosity_negative", porosity_negative),
            ("porosity_separator", porosity_separator),
            ("porosity_positive", porosity_positive),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not 0.0 <= transference_number < 1.0:
            raise ValueError("transference_number must lie in [0, 1)")
        for name, value in (
            ("cells_negative", cells_negative),
            ("cells_separator", cells_separator),
            ("cells_positive", cells_positive),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")

        self.thickness = (thickness_negative, thickness_separator, thickness_positive)
        self.porosity = (porosity_negative, porosity_separator, porosity_positive)
        self.diffusivity = float(diffusivity)
        self.transference_number = float(transference_number)
        self.electrode_area = float(electrode_area)
        self.bruggeman = float(bruggeman)
        self.counts = (int(cells_negative), int(cells_separator), int(cells_positive))

    # ------------------------------------------------------------------ shape

    @property
    def n_states(self) -> int:
        return sum(self.counts)

    @property
    def region_index(self) -> np.ndarray:
        """Region label per control volume: 0 negative, 1 separator, 2 positive."""
        return np.concatenate([np.full(n, k) for k, n in enumerate(self.counts)])

    @property
    def widths(self) -> np.ndarray:
        """Control-volume widths in metres."""
        return np.concatenate(
            [np.full(n, self.thickness[k] / n) for k, n in enumerate(self.counts)]
        )

    @property
    def effective_diffusivity(self) -> np.ndarray:
        """Bruggeman-corrected diffusivity per control volume."""
        eps = np.array([self.porosity[k] for k in self.region_index])
        return self.diffusivity * eps**self.bruggeman

    @property
    def time_constant(self) -> float:
        """Diffusion time across the whole sandwich, in seconds.

        Uses the total thickness and the harmonic-mean effective diffusivity,
        which is the transport-relevant average for layers in series.
        """
        total = sum(self.thickness)
        widths = self.widths
        resistance = float(np.sum(widths / self.effective_diffusivity))
        return total * resistance

    # ------------------------------------------------------------ state space

    def continuous(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(A, B, C)`` with cell current as input."""
        n = self.n_states
        widths = self.widths
        regions = self.region_index
        eps = np.array([self.porosity[k] for k in regions])
        d_eff = self.effective_diffusivity
        capacity = eps * widths  # accumulation coefficient per control volume

        A = np.zeros((n, n))
        for face in range(1, n):
            left, right = face - 1, face
            # Series resistance of the two half-cells meeting at this face.
            resistance = 0.5 * widths[left] / d_eff[left] + 0.5 * widths[right] / d_eff[right]
            conductance = 1.0 / resistance
            A[left, left] -= conductance / capacity[left]
            A[left, right] += conductance / capacity[left]
            A[right, right] -= conductance / capacity[right]
            A[right, left] += conductance / capacity[right]

        # Source: (1 - t+) I / (F A L_k) per unit volume in each coating, with the
        # sign set by which electrode is releasing lithium. Positive current is
        # discharge, so the negative coating is the source.
        faraday = 96485.33212
        gain = (1.0 - self.transference_number) / (faraday * self.electrode_area)
        B = np.zeros((n, 1))
        for i in range(n):
            if regions[i] == 0:
                B[i, 0] = +gain / self.thickness[0] / eps[i]
            elif regions[i] == 2:
                B[i, 0] = -gain / self.thickness[2] / eps[i]

        C = np.zeros((2, n))
        for target, region in ((0, 0), (1, 2)):
            mask = regions == region
            weights = widths[mask]
            C[target, mask] = weights / weights.sum()
        return A, B, C

    def discretise(self, dt: float) -> ElectrolyteStateSpace:
        """Zero-order-hold discretisation, with salt conservation imposed exactly.

        The continuous system conserves total salt because the source integrates
        to zero and the boundaries are closed. The discretisation should inherit
        that and, at any sane step, does. It is imposed anyway, by the same
        rank-one projection used for solid diffusion and for the same reason: the
        property is structural, and leaving it to depend on the accuracy of a
        matrix exponential over a stiff generator is not a position worth taking.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        A, B, C = self.continuous()
        Ad, Bd, Cd, _, _ = cont2discrete((A, B, C, np.zeros((2, 1))), dt, method="zoh")
        Ad = np.array(Ad, dtype=float)
        Bd = np.array(Bd, dtype=float).reshape(-1)

        widths = self.widths
        eps = np.array([self.porosity[k] for k in self.region_index])
        capacity = eps * widths
        total = capacity / capacity.sum()  # w, so that w @ c is the mean concentration
        uniform = np.ones(self.n_states)

        # w A = w (the mean is unchanged), A u = u (a uniform field is a fixed
        # point), and w B = 0 (no net salt is created).
        overlap = float(total @ uniform)
        u = uniform / overlap
        Ad -= np.outer(Ad @ u - u, total)
        Ad -= np.outer(u, total @ Ad - total)
        Bd -= u * float(total @ Bd)

        return ElectrolyteStateSpace(
            A=np.ascontiguousarray(Ad),
            B=np.ascontiguousarray(Bd.reshape(-1, 1)),
            C=np.ascontiguousarray(Cd, dtype=float),
            dt=float(dt),
            uniform=uniform,
        )

    # ------------------------------------------------------------- analytics

    def steady_state_split(self, current: float) -> tuple[float, float]:
        """Coating-average concentrations at steady state under constant current.

        Solved from the continuous system directly rather than by simulating to
        convergence, which makes it usable as an independent check on the
        discretised model. The mean is pinned to zero, so the returned values are
        deviations from the initial uniform concentration.
        """
        A, B, C = self.continuous()
        widths = self.widths
        eps = np.array([self.porosity[k] for k in self.region_index])
        weights = eps * widths
        weights = weights / weights.sum()

        # A is singular by construction (the mean is conserved), so pin the mean
        # to zero and solve the resulting nonsingular system.
        M = np.vstack([A[:-1, :], weights])
        rhs = np.concatenate([-B.reshape(-1)[:-1] * float(current), [0.0]])
        profile = np.linalg.solve(M, rhs)
        y = C @ profile
        return float(y[0]), float(y[1])

    def ohmic_resistance(self, conductivity: float) -> float:
        """Electrolyte ohmic resistance of the sandwich, in ohms.

        .. math::

            R_e = \\frac{1}{A}\\left(
                \\frac{L_n}{3 \\kappa_n^{\\text{eff}}}
                + \\frac{L_s}{\\kappa_s^{\\text{eff}}}
                + \\frac{L_p}{3 \\kappa_p^{\\text{eff}}} \\right)

        The factors of three on the coatings are not arbitrary. Ionic current in
        a coating falls linearly from the separator face to the current
        collector, because it is progressively handed over to the solid phase, so
        the average squared current density -- which is what sets dissipation --
        is a third of what a uniform current would give. The separator carries
        the full current everywhere and gets no such factor.
        """
        if conductivity <= 0.0:
            raise ValueError("conductivity must be positive")
        kappa = [conductivity * self.porosity[k] ** self.bruggeman for k in range(3)]
        return (
            self.thickness[0] / (3.0 * kappa[0])
            + self.thickness[1] / kappa[1]
            + self.thickness[2] / (3.0 * kappa[2])
        ) / self.electrode_area

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ElectrolyteDiffusion(n_states={self.n_states}, tau={self.time_constant:.1f}s)"
