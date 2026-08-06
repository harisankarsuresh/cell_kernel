"""Fitting a physics-based model to measurements, and saying what was learned.

A literature parameter set describes a cell design, not the cell on the bench.
Re-solving the stoichiometry window against a measured open-circuit curve fixes
the static part of that -- see
:func:`~cellkernel.params.fit_stoichiometry_window` -- but leaves the behaviour
under load, which is kinetics and transport. This module fits those.

Fitting them is easy. Knowing which of them the data actually determined is the
part that matters, and the part most fitting code omits. Six physical parameters
against one discharge curve is comfortably over-parameterised: raising the
negative reaction rate and raising the positive one have nearly the same effect
on terminal voltage, so a solver will happily trade one against the other and
report a confident answer to a question the measurement never asked.

:class:`IdentificationReport` therefore carries more than the fitted values. It
reports, per parameter, how much the residual actually responded, which
parameters moved together, and which ones ran into their bounds -- because a
parameter resting on a bound is the clearest possible statement that the data
wanted something the model could not provide, and that whatever came out is not a
measurement of it.

Scope
-----
This is a few hundred lines of least squares, not a replacement for
`PyBOP <https://github.com/pybop-team/PyBOP>`_, which does parameter
identification properly -- multiple optimisers, sampling, priors, uncertainty
quantification. Use PyBOP when the answer matters. Use this when what you want is
a cell parameter set good enough to generate an estimator from, and an honest
account of how far to trust it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares

from .params import CellParameters

__all__ = [
    "Knob",
    "IdentificationReport",
    "KINETIC_KNOBS",
    "TRANSPORT_KNOBS",
    "DEFAULT_KNOBS",
    "anchor_series_resistance",
    "identify",
]


@dataclass(frozen=True)
class Knob:
    """One fitted quantity, and how to put it back into a parameter set.

    Scale parameters are fitted as powers of ten. Diffusivities span orders of
    magnitude and a solver working on them linearly spends its whole budget in
    the first decade; working on the exponent makes the search well conditioned
    and makes the bounds mean something a person can state -- "within a factor of
    thirty" rather than "between 4e-15 and 1.2e-13".
    """

    name: str
    apply: Callable[[CellParameters, float], CellParameters]
    #: Bounds on the fitted variable, in decades for log knobs.
    low: float = -1.5
    high: float = 1.5
    #: ``True`` if the variable is a base-ten exponent multiplying the original.
    logarithmic: bool = True
    units: str = ""

    def describe(self, value: float) -> str:
        if self.logarithmic:
            return f"x{10.0**value:.3f}"
        return f"{value:.6g} {self.units}".strip()


def _scale(field_name: str, side: str | None = None) -> Callable:
    def setter(cell: CellParameters, value: float) -> CellParameters:
        factor = 10.0**value
        if side is None:
            return replace(cell, **{field_name: getattr(cell, field_name) * factor})
        electrode = getattr(cell, side)
        return replace(
            cell,
            **{side: replace(electrode, **{field_name: getattr(electrode, field_name) * factor})},
        )

    return setter


#: Reaction rate prefactors. These set the charge-transfer overpotential, so they
#: dominate the immediate voltage step when current is applied.
KINETIC_KNOBS = (
    Knob("reaction_rate_negative", _scale("reaction_rate", "negative"), units=""),
    Knob("reaction_rate_positive", _scale("reaction_rate", "positive"), units=""),
)

#: Solid and salt diffusivities, which set how the voltage sags over seconds to
#: minutes rather than how far it steps immediately.
TRANSPORT_KNOBS = (
    Knob("diffusivity_negative", _scale("diffusivity", "negative")),
    Knob("diffusivity_positive", _scale("diffusivity", "positive")),
    Knob("electrolyte_diffusivity", _scale("electrolyte_diffusivity")),
)

#: A series resistance for everything not otherwise modelled: tabs, current
#: collectors, weld interfaces. Fitted linearly because zero is a meaningful
#: value for it and a logarithm cannot reach zero.
_CONTACT = Knob(
    "contact_resistance",
    lambda cell, value: replace(cell, contact_resistance=max(0.0, value)),
    low=0.0,
    high=0.05,
    logarithmic=False,
    units="ohm",
)

DEFAULT_KNOBS = (*KINETIC_KNOBS, *TRANSPORT_KNOBS, _CONTACT)


@dataclass
class IdentificationReport:
    """What the fit produced, and how much of it the data supports."""

    cell: CellParameters
    knobs: tuple[Knob, ...]
    values: np.ndarray
    #: Root-mean-square voltage residual before and after, in volts.
    rmse_before: float
    rmse_after: float
    #: Per-segment root-mean-square residual after fitting, in volts.
    per_segment: dict[str, float] = field(default_factory=dict)
    #: Relative sensitivity of the residual norm to each parameter, from the
    #: Jacobian at the solution. Small means the data barely constrained it.
    sensitivity: np.ndarray | None = None
    #: Correlation matrix of the parameter estimates.
    correlation: np.ndarray | None = None
    #: Names of parameters resting on a bound.
    at_bounds: tuple[str, ...] = ()

    def poorly_identified(self, threshold: float = 0.05) -> tuple[str, ...]:
        """Parameters the data barely constrained, relative to the best-determined one."""
        if self.sensitivity is None:
            return ()
        largest = float(np.max(self.sensitivity)) or 1.0
        return tuple(
            knob.name
            for knob, value in zip(self.knobs, self.sensitivity, strict=True)
            if value / largest < threshold
        )

    def correlated_pairs(self, threshold: float = 0.9) -> tuple[tuple[str, str, float], ...]:
        """Parameter pairs the fit could trade against each other."""
        if self.correlation is None:
            return ()
        found = []
        for i in range(len(self.knobs)):
            for j in range(i + 1, len(self.knobs)):
                rho = float(self.correlation[i, j])
                if abs(rho) >= threshold:
                    found.append((self.knobs[i].name, self.knobs[j].name, rho))
        return tuple(found)

    def summary(self) -> str:
        lines = [
            f"residual {1e3 * self.rmse_before:.2f} mV -> {1e3 * self.rmse_after:.2f} mV",
            "",
            f"  {'parameter':26s} {'fitted':>12s} {'sensitivity':>12s}",
        ]
        largest = (float(np.max(self.sensitivity)) if self.sensitivity is not None else 1.0) or 1.0
        for index, knob in enumerate(self.knobs):
            relative = (
                self.sensitivity[index] / largest if self.sensitivity is not None else float("nan")
            )
            flag = "  (at bound)" if knob.name in self.at_bounds else ""
            lines.append(
                f"  {knob.name:26s} {knob.describe(self.values[index]):>12s} {relative:11.3f}{flag}"
            )
        if self.per_segment:
            lines.append("")
            for name, value in self.per_segment.items():
                lines.append(f"  {name:26s} {1e3 * value:8.2f} mV")
        weak = self.poorly_identified()
        if weak:
            lines += ["", "  barely constrained by this data: " + ", ".join(weak)]
        pairs = self.correlated_pairs()
        if pairs:
            lines.append("  traded against each other:")
            lines += [f"    {a} and {b}, rho {r:+.3f}" for a, b, r in pairs]
        if self.at_bounds:
            lines += [
                "",
                "  At a bound means the data wanted something the model could not",
                "  supply. Whatever came out is not a measurement of that parameter.",
            ]
        return "\n".join(lines)


def anchor_series_resistance(
    cell: CellParameters,
    build: Callable[[CellParameters], object],
    measured_resistance: float,
    current: float,
    soc: float = 0.5,
) -> CellParameters:
    """Set ``contact_resistance`` from a measured pulse edge, before fitting anything.

    Do this first. It is the difference between a fit that works and one that
    does not, and the reason is that the instantaneous voltage step is the only
    part of the response *every* parameter can imitate. Left free, the solver
    spends its whole budget reproducing that step -- and since a reaction rate
    and a resistance both move it, the two become interchangeable and everything
    slower gets fitted with whatever is left over.

    Measured on the reference cell: fitting transport to pulses with the
    resistance free left a residual of 87 mV. Anchoring it first, and then
    fitting the same transport parameters to what remains, gave 31 mV -- and the
    solid diffusivities went from being swamped to being the best-determined
    quantities in the fit.

    The value assigned is the *shortfall*: the measured resistance less whatever
    instantaneous drop the model already produces from its own kinetics and
    geometry. Assigning the whole measured value instead would count the
    charge-transfer step twice.

    Parameters
    ----------
    cell
        Parameter set to modify.
    build
        Model constructor, as for :func:`identify`.
    measured_resistance
        Ohms, from :attr:`~cellkernel.data.reference.PulseSegment.series_resistance`.
        Use the median over several states of charge, and over several cells if
        you have them -- they scatter by a few percent.
    current
        Pulse current in amperes, positive.
    soc
        State of charge to evaluate the model's own drop at. Mid-window, where
        the exchange current density is least sensitive to composition.
    """
    if current <= 0.0:
        raise ValueError("current must be positive")
    if measured_resistance < 0.0:
        raise ValueError("measured_resistance must be non-negative")
    probe = build(replace(cell, contact_resistance=0.0))
    rested = probe.initial_state(soc)
    intrinsic = (probe.voltage(rested, 0.0) - probe.voltage(rested, current)) / current
    return replace(cell, contact_resistance=max(0.0, measured_resistance - intrinsic))


def identify(
    cell: CellParameters,
    segments: Sequence[tuple[str, np.ndarray, np.ndarray]],
    build: Callable[[CellParameters], object],
    knobs: Sequence[Knob] = DEFAULT_KNOBS,
    soc0: float = 1.0,
    voltage_floor: float = 2.7,
    max_points: int = 400,
) -> IdentificationReport:
    """Fit ``knobs`` so that ``build(cell)`` reproduces the measured segments.

    Parameters
    ----------
    cell
        Starting parameter set. Fit the stoichiometry window first: this routine
        adjusts kinetics and transport, and will otherwise spend them
        compensating for an electrode balance that is simply wrong.
    segments
        ``(name, current, voltage)`` triples, or ``(name, current, voltage,
        soc0)`` quadruples where each segment starts from its own state of
        charge -- which pulses do, since each is taken at a different level.
        All must be sampled at the model's step.

        Pass **pulses if you have them**. Constant-current discharges, however
        many rates, cannot separate an ohmic drop from a charge-transfer
        overpotential from a diffusion limitation: over a whole discharge all
        three look like a voltage that is too low, and the report will say as
        much. A pulse separates them by timescale instead, which is a difference
        of a hundredfold rather than a difference of degree.
    build
        Turns a parameter set into a model. Usually ``lambda p: SPM(p, dt=1.0)``.
    knobs
        What to fit. Fewer is better: every extra one is another direction the
        solver can hide error in.
    voltage_floor
        Ignore samples below this, where the model is leaving its valid range and
        the measurement is dominated by the cut-off.
    max_points
        Residual samples per segment. A slow discharge recorded at 1 Hz has tens
        of thousands of points that carry almost no independent information about
        a handful of smooth parameters; decimating costs nothing and makes the
        fit tractable.
    """
    if not segments:
        raise ValueError("need at least one measured segment")
    if not knobs:
        raise ValueError("need at least one parameter to fit")
    knobs = tuple(knobs)

    prepared = []
    for entry in segments:
        if len(entry) == 4:
            name, current, voltage, start = entry
        else:
            name, current, voltage = entry
            start = soc0
        current = np.asarray(current, dtype=float).reshape(-1)
        voltage = np.asarray(voltage, dtype=float).reshape(-1)
        if current.size != voltage.size:
            raise ValueError(f"{name}: current and voltage differ in length")
        keep = voltage > voltage_floor
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            raise ValueError(f"{name}: nothing above the voltage floor")
        step = max(1, indices.size // max_points)
        prepared.append((name, current, voltage, indices[::step], float(start)))

    def rebuild(theta: np.ndarray) -> CellParameters:
        trial = cell
        for knob, value in zip(knobs, theta, strict=True):
            trial = knob.apply(trial, float(value))
        return trial

    def residual(theta: np.ndarray) -> np.ndarray:
        model = build(rebuild(theta))
        chunks = []
        for _name, current, voltage, sample, start in prepared:
            run = model.simulate(current, soc0=start)
            chunks.append(run["voltage"][sample] - voltage[sample])
        return np.concatenate(chunks)

    start = np.array(
        [0.0 if knob.logarithmic else getattr(cell, "contact_resistance", 0.0) for knob in knobs]
    )
    start = np.clip(start, [k.low for k in knobs], [k.high for k in knobs])
    before = residual(start)

    solved = least_squares(
        residual,
        start,
        bounds=([k.low for k in knobs], [k.high for k in knobs]),
        xtol=1e-10,
        ftol=1e-12,
        diff_step=1e-3,
    )
    after = solved.fun

    # Sensitivity and correlation from a Jacobian computed here, not the one the
    # solver returns. Under bounds, `least_squares` in its trust-region mode
    # hands back a *modified* Jacobian with columns rescaled by the distance to
    # the bound -- which is right for its own step calculation and wrong as a
    # statement about the data, because a parameter sitting on a bound gets a
    # column scaled towards zero or blown up depending on which way it is
    # pushing. Reading sensitivity off it gave answers that flipped between runs.
    #
    # This is still only the linearised picture and says nothing about multiple
    # minima, but it does catch the common failure: two parameters that move the
    # residual the same way, so their sum is determined and their difference is
    # not.
    step = 1e-3
    columns = []
    for index in range(len(knobs)):
        forward = np.array(solved.x, dtype=float)
        backward = np.array(solved.x, dtype=float)
        forward[index] = min(forward[index] + step, knobs[index].high)
        backward[index] = max(backward[index] - step, knobs[index].low)
        width = forward[index] - backward[index]
        if width <= 0.0:  # pragma: no cover - degenerate bounds
            columns.append(np.zeros_like(after))
            continue
        columns.append((residual(forward) - residual(backward)) / width)
    jacobian = np.column_stack(columns)
    sensitivity = np.linalg.norm(jacobian, axis=0)
    correlation = None
    gram = jacobian.T @ jacobian
    try:
        covariance = np.linalg.inv(gram)
        scale = np.sqrt(np.clip(np.diag(covariance), 1e-300, None))
        correlation = covariance / np.outer(scale, scale)
    except np.linalg.LinAlgError:  # pragma: no cover - singular, nothing to report
        correlation = None

    tolerance = 1e-6
    at_bounds = tuple(
        knob.name
        for knob, value in zip(knobs, solved.x, strict=True)
        if value <= knob.low + tolerance or value >= knob.high - tolerance
    )

    fitted = rebuild(solved.x)
    model = build(fitted)
    per_segment = {}
    for name, current, voltage, sample, start in prepared:
        run = model.simulate(current, soc0=start)
        error = run["voltage"][sample] - voltage[sample]
        per_segment[name] = float(np.sqrt(np.mean(error**2)))

    return IdentificationReport(
        cell=fitted,
        knobs=knobs,
        values=np.asarray(solved.x, dtype=float),
        rmse_before=float(np.sqrt(np.mean(before**2))),
        rmse_after=float(np.sqrt(np.mean(after**2))),
        per_segment=per_segment,
        sensitivity=sensitivity,
        correlation=correlation,
        at_bounds=at_bounds,
    )
