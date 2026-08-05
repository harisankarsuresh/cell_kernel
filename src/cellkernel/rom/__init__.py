"""Reduced-order models for solid-phase lithium diffusion in a spherical particle.

Four families are provided, all reducible to a discrete-time linear state-space
system with outputs ``[c_surf, c_bar]``:

=========================================  ======  ===========================================
Model                                      States  Character
=========================================  ======  ===========================================
:class:`PolynomialDiffusion`               2       Cheapest; exact steady-state offset
:class:`PadeDiffusion`                     ``k``   Best accuracy per state; dense
:class:`SpectralDiffusion`                 ``k+1``  Diagonal, so cheapest per state
:class:`FiniteVolumeDiffusion`             ``k``   Resolves the interior profile
=========================================  ======  ===========================================

Only :class:`FiniteVolumeDiffusion` retains a spatially resolved concentration
field; the others track modal or moment coordinates and reconstruct the two
scalars a cell model actually needs.
"""

from .base import (
    DiffusionROM,
    DiscreteStateSpace,
    exact_surface_transfer_function,
    low_frequency_series,
    normalised_series_coefficients,
)
from .finite_volume import FiniteVolumeDiffusion
from .pade import PadeDiffusion, pade_coefficients
from .polynomial import PolynomialDiffusion
from .schedule import ScheduledStateSpace, schedule_over_temperature
from .spectral import SpectralDiffusion, zero_flux_eigenvalues

__all__ = [
    "DiffusionROM",
    "DiscreteStateSpace",
    "FiniteVolumeDiffusion",
    "PadeDiffusion",
    "PolynomialDiffusion",
    "ScheduledStateSpace",
    "SpectralDiffusion",
    "exact_surface_transfer_function",
    "low_frequency_series",
    "make_rom",
    "normalised_series_coefficients",
    "pade_coefficients",
    "schedule_over_temperature",
    "zero_flux_eigenvalues",
]


def make_rom(
    kind: str,
    radius: float,
    diffusivity: float,
    order: int = 3,
    residualise: bool = True,
) -> DiffusionROM:
    """Construct a diffusion reduced-order model by name.

    Parameters
    ----------
    kind
        One of ``"pade"``, ``"spectral"``, ``"fv"``, ``"poly"``.
    radius, diffusivity
        Particle radius (m) and solid diffusivity (m2 s-1).
    order
        Requested number of states. Ignored by ``"poly"``, which is always two.
    residualise
        Passed to :class:`SpectralDiffusion`; ignored by the other models.
    """
    key = kind.lower().replace("-", "_")
    if key == "pade":
        return PadeDiffusion(radius, diffusivity, order=order)
    if key == "spectral":
        return SpectralDiffusion(
            radius, diffusivity, n_modes=max(order - 1, 0), residualise=residualise
        )
    if key in {"fv", "finite_volume"}:
        return FiniteVolumeDiffusion(radius, diffusivity, n_shells=max(order, 2))
    if key in {"poly", "polynomial"}:
        return PolynomialDiffusion(radius, diffusivity)
    raise ValueError(
        f"unknown reduced-order model {kind!r}; expected one of 'pade', 'spectral', 'fv', 'poly'"
    )
