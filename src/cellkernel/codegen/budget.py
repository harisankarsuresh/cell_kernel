"""Static resource accounting for a generated estimator.

Answers the three questions a firmware engineer asks before agreeing to
integrate anything: how much flash, how much RAM, and how long does it take.

All figures are derived by counting the emitted data structures and arithmetic
operations, not measured. Flash and RAM counts are exact for the data; code size
is excluded because it depends on the compiler and options, though it is small
next to the tables. The cycle estimate is a model, and its assumptions are stated
so it can be corrected for a specific core rather than trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import EstimatorSpec

__all__ = ["ResourceBudget", "estimate_budget"]


@dataclass(frozen=True)
class ResourceBudget:
    """Static resource estimate for one generated estimator instance."""

    precision: str
    word_bytes: int
    n_states: int
    #: Read-only tables: matrices, output rows and potential lookups.
    flash_bytes: int
    #: Per-instance mutable state: the state vector and its covariance.
    ram_bytes: int
    #: Stack high-water mark inside :c:func:`ck_predict`, its scratch matrix.
    stack_bytes: int
    multiplies_per_step: int
    adds_per_step: int
    divides_per_step: int
    sqrts_per_step: int
    logs_per_step: int
    #: Modelled cycles per step on the reference core described in the notes.
    estimated_cycles: int
    table_points: tuple[int, int]
    #: Largest potential-table interpolation error, in volts.
    table_error_volts: float

    @property
    def estimated_microseconds_at(self) -> dict[int, float]:
        """Estimated execution time in microseconds at a few clock rates."""
        return {mhz: self.estimated_cycles / mhz for mhz in (48, 80, 120, 180)}

    def summary(self) -> str:
        """A short human-readable report."""
        us = self.estimated_microseconds_at
        lines = [
            f"precision            {self.precision} ({self.word_bytes} bytes/word)",
            f"states               {self.n_states}",
            f"flash (tables)       {self.flash_bytes} B",
            f"RAM (per instance)   {self.ram_bytes} B",
            f"stack (predict)      {self.stack_bytes} B",
            f"arithmetic/step      {self.multiplies_per_step} mul, "
            f"{self.adds_per_step} add, {self.divides_per_step} div, "
            f"{self.sqrts_per_step} sqrt, {self.logs_per_step} log",
            f"modelled cycles      {self.estimated_cycles}",
            "estimated time       "
            + ", ".join(f"{mhz} MHz: {t:.1f} us" for mhz, t in us.items()),
            f"OCP table points     {self.table_points[0]} / {self.table_points[1]}",
            f"OCP table error      {self.table_error_volts * 1e3:.3f} mV",
        ]
        return "\n".join(lines)


def estimate_budget(spec: EstimatorSpec, precision: str = "double") -> ResourceBudget:
    """Count the resources a generated estimator needs.

    Notes
    -----
    **Flash** counts every ``static const`` array: three ``n^2`` matrices (the
    transition and the two covariances), the input vector, the two uniform-loading
    maps, a surface and a bulk output row per electrode, and the two potential
    tables. The tables usually dominate -- 257 single-precision points per
    electrode is about 2 kB -- so reducing ``table_points`` is the first lever if
    flash is tight, at a cost that :attr:`~ResourceBudget.table_error_volts` makes
    explicit.

    **RAM** is the state vector plus the covariance, ``n + n^2`` words per
    instance. The covariance dominates, and it is the reason state count matters:
    going from 4 states to 20 is not five times the memory but roughly twenty-five.
    A 4-state estimator in single precision needs 80 bytes.

    **Cycles** are modelled on a Cortex-M4F, assuming one cycle for a
    single-precision multiply or add, 14 for a divide, 14 for a square root, and
    roughly 100 for a ``log`` from a typical libm. Two costs are added that the
    operation counts alone miss: loop and addressing overhead, taken as 40% on the
    covariance triple loops, which is realistic for compiled C that is not
    hand-unrolled; and a fixed 60-cycle allowance for call overhead. In double
    precision on a core with only single-precision hardware, every figure should
    be multiplied by roughly 15 -- which is the argument for generating ``float``.

    Treat the cycle number as an order-of-magnitude planning figure. The
    structural claims are the reliable ones: cost is quadratic in state count and
    dominated by the covariance propagation, there is no iteration, and the worst
    case equals the typical case.
    """
    n = spec.n_states
    word = 4 if precision == "float" else 8

    n_table = spec.negative.ocp_table.n + spec.positive.ocp_table.n
    flash_words = (
        n * n  # A
        + n  # B
        + n * n  # process noise covariance
        + n * n  # initial covariance
        + spec.n_negative  # uniform map, negative
        + spec.n_positive  # uniform map, positive
        + 2 * spec.n_negative  # surface and bulk rows, negative
        + 2 * spec.n_positive  # surface and bulk rows, positive
        + n_table
    )
    ram_words = n + n * n
    stack_words = n * n + 3 * n  # scratch AP, plus h, ph and gain

    # Arithmetic per full predict-and-correct step.
    mul_state = n * n + n  # A x + B I
    mul_cov = 2 * n * n * n  # (A P) A'
    mul_meas = 3 * n + 2 * n * n  # jacobian, P h, gain application, P update
    multiplies = mul_state + mul_cov + mul_meas
    adds = mul_cov + n * n + 3 * n + n * n
    divides = n + 6  # Kalman gain, plus the concentration and asinh quotients
    sqrts = 6  # two exchange currents and their derivatives, plus asinh
    logs = 2

    cycles = (
        multiplies
        + adds
        + 14 * divides
        + 14 * sqrts
        + 100 * logs
        + int(0.4 * (mul_cov + adds))
        + 60
    )
    if precision != "float":
        cycles *= 15

    return ResourceBudget(
        precision=precision,
        word_bytes=word,
        n_states=n,
        flash_bytes=flash_words * word,
        ram_bytes=ram_words * word,
        stack_bytes=stack_words * word,
        multiplies_per_step=multiplies,
        adds_per_step=adds,
        divides_per_step=divides,
        sqrts_per_step=sqrts,
        logs_per_step=logs,
        estimated_cycles=int(cycles),
        table_points=(spec.negative.ocp_table.n, spec.positive.ocp_table.n),
        table_error_volts=max(
            spec.negative.ocp_table.max_abs_error, spec.positive.ocp_table.max_abs_error
        ),
    )
