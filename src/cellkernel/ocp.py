"""Open-circuit potential functions and tabulated interpolants.

An open-circuit potential (OCP) is the equilibrium potential of an electrode
against lithium as a function of stoichiometry :math:`x = c_{\\text{surf}} /
c_{\\max}`. It enters a physics-based model twice: as the dominant term in the
terminal voltage, and through its derivative, which sets how strongly a voltage
measurement constrains the concentration state. A filter is only as good as
``dU/dx``; where the OCP is flat, voltage carries almost no information about
state of charge, which is the well-known difficulty with lithium iron phosphate.

Two representations are supported. Analytic fits are differentiable in closed
form and are what the built-in parameter sets use. Tabulated OCPs, wrapped by
:class:`TabulatedOCP`, come from laboratory pseudo-open-circuit measurements and
are the common case in industry; they are interpolated with a monotone cubic
spline so that the derivative used by the filter is continuous and does not
introduce the sawtooth artefacts that piecewise-linear interpolation would.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = [
    "OCPFunction",
    "OCPTable",
    "TabulatedOCP",
    "derivative_of",
    "graphite_chen2020",
    "graphite_chen2020_derivative",
    "lfp_prada2013",
    "nmc811_chen2020",
    "nmc811_chen2020_derivative",
    "numerical_derivative",
    "tabulate",
]

#: An OCP is any callable mapping stoichiometry to potential in volts.
OCPFunction = Callable[[np.ndarray], np.ndarray]

#: Analytic fits are evaluated no closer than this to the ends of ``[0, 1]``.
#:
#: Every published OCP fit is a regression over measured data and none is valid
#: outside the sampled range. Taken literally at the endpoints these expressions
#: misbehave badly rather than gracefully: the graphite fit's ``exp(-39.4 x)``
#: overflows for negative ``x``, and the lithium iron phosphate fit raises
#: ``1 - x`` to a fractional power, giving NaN. NaN is far worse than a clamped
#: value here, because it propagates silently into a parameter solve or a filter
#: covariance and turns a recoverable inaccuracy into a dead process.
STOICH_EPS = 1e-6


def _clip_stoichiometry(x: np.ndarray) -> np.ndarray:
    """Clamp stoichiometry into the open interval where the fits are defined."""
    return np.clip(np.asarray(x, dtype=float), STOICH_EPS, 1.0 - STOICH_EPS)


def graphite_chen2020(x: np.ndarray) -> np.ndarray:
    """Graphite negative-electrode OCP, volts against lithium.

    Fit reported for the LG M50 cell by Chen et al. (2020). The exponential term
    captures the steep rise below 5% lithiation and the three ``tanh`` terms
    reproduce the staging plateaux at roughly 0.12, 0.28 and 0.61.
    """
    x = _clip_stoichiometry(x)
    return (
        1.9793 * np.exp(-39.3631 * x)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
    )


def nmc811_chen2020(x: np.ndarray) -> np.ndarray:
    """NMC811 positive-electrode OCP, volts against lithium.

    Fit reported for the LG M50 cell by Chen et al. (2020). The two large,
    nearly cancelling ``tanh`` terms near ``x = 0.312`` together represent a
    narrow feature; they are kept in the published form rather than simplified,
    because their difference is what shapes the curve there.
    """
    x = _clip_stoichiometry(x)
    return (
        -0.8090 * x
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (x - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (x - 0.3120))
    )


def lfp_prada2013(x: np.ndarray) -> np.ndarray:
    """Lithium iron phosphate positive-electrode OCP, volts against lithium.

    Fit reported by Prada et al. (2013). The curve is extremely flat between
    roughly 20% and 90% lithiation -- about 40 mV across the whole plateau -- so
    voltage alone barely constrains state of charge in that range. This is the
    case where a physics-based estimator earns its keep: the surface-to-bulk
    concentration gradient still responds to load even where the equilibrium
    potential does not.

    The two exponential terms in this fit reach magnitudes near 2000 V and
    almost entirely cancel as ``x`` approaches zero, which is the fully
    delithiated limit. That cancellation is intrinsic to the published form, so
    the fit should be trusted only inside its calibration range; the built-in
    parameter set keeps the positive stoichiometry above a few percent for this
    reason.
    """
    x = _clip_stoichiometry(x)
    y = 1.0 - x
    return (
        3.4323
        - 0.8428 * np.exp(-80.2493 * np.power(y, 1.3198))
        - 3.2474e-6 * np.exp(20.2645 * np.power(y, 3.8003))
        + 3.2482e-6 * np.exp(20.2646 * np.power(y, 3.7995))
    )


def graphite_chen2020_derivative(x: np.ndarray) -> np.ndarray:
    """Closed-form ``dU/dx`` of :func:`graphite_chen2020`."""
    x = np.asarray(x, dtype=float)
    return (
        -1.9793 * 39.3631 * np.exp(-39.3631 * x)
        - 0.0909 * 29.8538 / np.cosh(29.8538 * (x - 0.1234)) ** 2
        - 0.04478 * 14.9159 / np.cosh(14.9159 * (x - 0.2769)) ** 2
        - 0.0205 * 30.4444 / np.cosh(30.4444 * (x - 0.6103)) ** 2
    )


def nmc811_chen2020_derivative(x: np.ndarray) -> np.ndarray:
    """Closed-form ``dU/dx`` of :func:`nmc811_chen2020`."""
    x = np.asarray(x, dtype=float)
    return (
        -0.8090
        - 0.0428 * 18.5138 / np.cosh(18.5138 * (x - 0.5542)) ** 2
        - 17.7326 * 15.7890 / np.cosh(15.7890 * (x - 0.3117)) ** 2
        + 17.5842 * 15.9308 / np.cosh(15.9308 * (x - 0.3120)) ** 2
    )


#: Analytic derivatives for the built-in fits, consulted by :func:`derivative_of`.
_ANALYTIC_DERIVATIVES: dict[int, OCPFunction] = {
    id(graphite_chen2020): graphite_chen2020_derivative,
    id(nmc811_chen2020): nmc811_chen2020_derivative,
}


def derivative_of(fn: OCPFunction) -> OCPFunction:
    """Return the best available ``dU/dx`` for ``fn``.

    Prefers, in order: a registered closed-form derivative for the built-in
    fits; a ``derivative`` method, as :class:`TabulatedOCP` provides; and finally
    a central difference.

    The ordering matters more than it appears. ``dU/dx`` multiplies the
    innovation in every Kalman update in this package, so error in it does not
    merely perturb the estimate, it mis-scales the correction. A central
    difference of a ``tanh`` fit with an argument gain near 30 -- as graphite has
    -- retains only about eight significant digits, and near a plateau edge,
    where the curvature is largest and the derivative is changing fastest, that
    is the term the filter is most sensitive to.
    """
    analytic = _ANALYTIC_DERIVATIVES.get(id(fn))
    if analytic is not None:
        return analytic
    method = getattr(fn, "derivative", None)
    if callable(method):
        return method

    def fallback(x: np.ndarray) -> np.ndarray:
        return numerical_derivative(fn, x)

    return fallback


def numerical_derivative(fn: OCPFunction, x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference derivative of an OCP function.

    Used as a fallback when an analytic derivative is unavailable. The step is a
    compromise: truncation error grows as ``step^2`` while cancellation error
    grows as ``eps/step``, putting the optimum for a smooth O(1) function near
    ``eps^(1/3)``, about 6e-6. Stoichiometry is clipped into ``[0, 1]`` so the
    stencil never evaluates the fit outside its range of validity, where the
    exponential terms in these fits diverge violently.
    """
    x = np.asarray(x, dtype=float)
    hi = np.clip(x + step, 0.0, 1.0)
    lo = np.clip(x - step, 0.0, 1.0)
    span = hi - lo
    span = np.where(span > 0.0, span, step)
    return (np.asarray(fn(hi)) - np.asarray(fn(lo))) / span


@dataclass(frozen=True)
class OCPTable:
    """A uniformly sampled OCP, in the exact form the generated C evaluates.

    Attributes
    ----------
    sto_min, sto_max
        Endpoints of the sampled stoichiometry range.
    values
        Potentials in volts at ``n`` equally spaced stoichiometries.
    max_abs_error
        Largest absolute difference, in volts, between linear interpolation of
        this table and the underlying function, measured on a dense grid.

    Notes
    -----
    The generated estimator evaluates the potential from a table rather than
    re-evaluating the fit, for three reasons. A uniform grid locates a segment
    with one multiply and one truncation, so execution time does not depend on
    state of charge -- which matters when the task has a hard deadline. It avoids
    ``exp`` and ``tanh`` calls, which on a core without hardware transcendentals
    cost hundreds of cycles each. And it sidesteps a genuine precision problem:
    :func:`nmc811_chen2020` contains two ``tanh`` terms of magnitude 17.6 whose
    difference is a few tens of millivolts, so evaluating it in single precision
    loses roughly three digits to cancellation, whereas a table stores the
    already-cancelled result.

    Because the table is an approximation in its own right, ``max_abs_error``
    is reported and folded into the generated code's error budget rather than
    left implicit.
    """

    sto_min: float
    sto_max: float
    values: np.ndarray
    max_abs_error: float

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.values.size)

    @property
    def step(self) -> float:
        """Stoichiometry increment between samples."""
        return (self.sto_max - self.sto_min) / (self.n - 1)

    def interpolate(self, sto: float) -> float:
        """Evaluate bit-for-bit as the generated C does, for cross-checking."""
        pos = (float(sto) - self.sto_min) / self.step
        pos = min(max(pos, 0.0), float(self.n - 1))
        i = min(int(pos), self.n - 2)
        frac = pos - i
        return float(self.values[i] + frac * (self.values[i + 1] - self.values[i]))


def tabulate(
    fn: OCPFunction, n: int = 257, sto_min: float = 0.0, sto_max: float = 1.0
) -> OCPTable:
    """Sample ``fn`` onto a uniform grid and measure the interpolation error.

    Parameters
    ----------
    fn
        Any OCP callable.
    n
        Number of samples. Error falls as ``n**-2``, so doubling the table
        quarters the error.
    sto_min, sto_max
        Range to cover. Restricting this to the stoichiometry window the
        electrode actually visits, rather than the full ``[0, 1]``, spends the
        available points where they are needed -- worth a factor of two or three
        on an electrode with a narrow operating window.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    grid = np.linspace(sto_min, sto_max, n)
    values = np.asarray(fn(grid), dtype=float).reshape(-1)
    probe = OCPTable(sto_min, sto_max, values, 0.0)
    dense = np.linspace(sto_min, sto_max, 20 * n + 1)
    exact = np.asarray(fn(dense), dtype=float).reshape(-1)
    approx = np.array([probe.interpolate(s) for s in dense])
    return OCPTable(
        sto_min=sto_min,
        sto_max=sto_max,
        values=values,
        max_abs_error=float(np.max(np.abs(approx - exact))),
    )


@dataclass
class TabulatedOCP:
    """Monotone cubic interpolant of a measured open-circuit potential.

    Parameters
    ----------
    stoichiometry
        Strictly increasing sample points in ``[0, 1]``.
    potential
        Measured potential in volts at each sample point.

    Notes
    -----
    A PCHIP interpolant is used rather than a natural cubic spline. Both are
    :math:`C^{1}`, which is what the filter needs, but PCHIP is shape
    preserving: it will not overshoot between samples. On a lithium iron
    phosphate plateau a natural spline can ring by several millivolts and even
    produce a locally positive ``dU/dx``, which flips the sign of the Kalman
    gain and drives the filter away from the truth. Guarding against that is
    worth the slightly lower formal order of accuracy.
    """

    stoichiometry: np.ndarray
    potential: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.stoichiometry, dtype=float).reshape(-1)
        u = np.asarray(self.potential, dtype=float).reshape(-1)
        if x.size != u.size:
            raise ValueError("stoichiometry and potential must have equal length")
        if x.size < 2:
            raise ValueError("need at least two samples")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("stoichiometry must be strictly increasing")
        self.stoichiometry = x
        self.potential = u
        self._spline = PchipInterpolator(x, u, extrapolate=True)
        self._derivative = self._spline.derivative()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Potential at stoichiometry ``x``, clamped to the sampled range."""
        xc = np.clip(np.asarray(x, dtype=float), self.stoichiometry[0], self.stoichiometry[-1])
        return np.asarray(self._spline(xc))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """``dU/dx`` at stoichiometry ``x``, clamped to the sampled range."""
        xc = np.clip(np.asarray(x, dtype=float), self.stoichiometry[0], self.stoichiometry[-1])
        return np.asarray(self._derivative(xc))

    def resample(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Uniform resampling onto ``count`` points, for embedded lookup tables.

        A uniform grid lets the generated C locate a segment by one
        multiplication and a truncation instead of a binary search, which keeps
        the estimator's execution time constant regardless of state of charge.
        """
        if count < 2:
            raise ValueError("count must be at least 2")
        grid = np.linspace(self.stoichiometry[0], self.stoichiometry[-1], count)
        return grid, self(grid)
