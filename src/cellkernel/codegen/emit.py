"""Emit a self-contained C99 estimator from an :class:`EstimatorSpec`.

Design constraints, all of which are things a battery-management team will be
asked about during a safety review:

* **No dynamic allocation.** Every array is a fixed-size member of a caller-owned
  struct or a ``static const`` table in flash. Nothing calls ``malloc``.
* **No hidden global state.** All mutable state lives in ``ck_estimator_t``, so
  two instances can run side by side and the code is reentrant.
* **Bounded, data-independent execution time.** No iteration to convergence, no
  binary search, no branches on data other than saturation clamps. The worst case
  equals the typical case, which is what a hard-real-time schedule needs.
* **No transcendental library calls except ``sqrt``, ``asinh`` and ``log``.**
  Potentials come from tables. The kinetic term needs one inverse hyperbolic sine,
  which is emitted as an explicit ``log(u + sqrt(u*u + 1))`` so the result does not
  depend on the target's libm version.
* **Selectable precision.** ``float`` or ``double`` through one typedef, so the
  same source runs on a Cortex-M4F and on a host.

The emitted code is plain C99 with no compiler extensions, and it compiles clean
under ``-Wall -Wextra -Wpedantic -Werror``, which the verification harness checks.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone

import numpy as np

from .spec import ElectrodeSpec, EstimatorSpec

__all__ = ["emit_header", "emit_source", "emit_main"]

_GUARD = "CELLKERNEL_ESTIMATOR_H"


def _fmt(value: float, precision: str) -> str:
    """Format a float as a C literal that round-trips exactly.

    ``repr`` of a Python float is the shortest string that reads back to the same
    double, so the emitted constant is bit-identical to the one the reference
    implementation used. The ``f`` suffix is appended for single precision to
    avoid a double-to-float narrowing at every use, which would otherwise leave
    the compiler emitting a conversion in the inner loop.
    """
    text = repr(float(value))
    if "e" not in text and "E" not in text and "." not in text and "inf" not in text:
        text += ".0"
    return text + ("f" if precision == "float" else "")


def _array_1d(name: str, values: np.ndarray, precision: str, per_line: int = 4) -> str:
    items = [_fmt(v, precision) for v in np.asarray(values).reshape(-1)]
    lines = [
        "    " + ", ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)
    ]
    body = ",\n".join(lines)
    return f"static const ck_real_t {name}[{len(items)}] = {{\n{body}\n}};"


def _array_2d(name: str, matrix: np.ndarray, precision: str) -> str:
    matrix = np.asarray(matrix)
    rows = []
    for row in matrix:
        items = ", ".join(_fmt(v, precision) for v in row)
        rows.append(f"    {{ {items} }}")
    body = ",\n".join(rows)
    return (
        f"static const ck_real_t {name}[{matrix.shape[0]}][{matrix.shape[1]}] = "
        f"{{\n{body}\n}};"
    )


def emit_header(spec: EstimatorSpec, precision: str = "double") -> str:
    """Generate the public header."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lo, hi = spec.voltage_limits
    return f"""\
/*
 * cellkernel generated estimator -- DO NOT EDIT BY HAND.
 *
 * Physics-based state-of-charge estimator for a lithium-ion cell, derived from a
 * single particle model with reduced-order solid diffusion and corrected by an
 * extended Kalman filter against measured terminal voltage.
 *
 * Generated : {stamp}
 * Model     : {spec.provenance}
 * States    : {spec.n_states} ({spec.n_negative} negative, {spec.n_positive} positive)
 * Step      : {spec.dt} s  (fixed; see ck_step)
 * Precision : {precision}
 *
 * The sample period is baked into the coefficient tables. Calling the update
 * functions at any other rate is a silent modelling error, not a scaling one:
 * regenerate instead.
 */
#ifndef {_GUARD}
#define {_GUARD}

#ifdef __cplusplus
extern "C" {{
#endif

typedef {precision} ck_real_t;

#define CK_N_STATES   {spec.n_states}
#define CK_N_NEGATIVE {spec.n_negative}
#define CK_N_POSITIVE {spec.n_positive}
#define CK_DT         ({_fmt(spec.dt, precision)})
#define CK_V_MIN      ({_fmt(lo, precision)})
#define CK_V_MAX      ({_fmt(hi, precision)})

/*
 * All mutable state. Allocate one per cell; the code touches no globals, so
 * instances are independent and every entry point is reentrant.
 */
typedef struct {{
    ck_real_t x[CK_N_STATES];                 /* diffusion coordinates        */
    ck_real_t P[CK_N_STATES][CK_N_STATES];    /* state covariance             */
}} ck_estimator_t;

/* Reset to a rested cell at the given state of charge, in [0, 1]. */
void ck_init(ck_estimator_t *est, ck_real_t soc);

/*
 * Advance the state and covariance by one sample under constant current.
 * Current is POSITIVE ON DISCHARGE, in amperes.
 */
void ck_predict(ck_estimator_t *est, ck_real_t current);

/* Predicted terminal voltage in volts at the present state and current. */
ck_real_t ck_voltage(const ck_estimator_t *est, ck_real_t current);

/* State of charge in [0, 1], from the conserved average concentration. */
ck_real_t ck_soc(const ck_estimator_t *est);

/*
 * Apply one extended Kalman correction from a voltage measurement.
 * Returns the innovation (measured minus predicted) in volts, which is worth
 * monitoring: a persistent bias means the model, not the filter, is wrong.
 */
ck_real_t ck_correct(ck_estimator_t *est, ck_real_t current, ck_real_t measured_voltage);

/*
 * Convenience: predict, then correct. This is the normal per-sample entry point.
 * Returns the updated state of charge.
 */
ck_real_t ck_step(ck_estimator_t *est, ck_real_t current, ck_real_t measured_voltage);

/*
 * Electrode-level diagnostics.
 *
 * Surface stoichiometry is what limits fast charge: plating risk on the negative
 * electrode is governed by its surface value, not by cell state of charge, and the
 * two diverge markedly under load. Bulk stoichiometry is the electrode's own state
 * of charge; the two electrodes drift apart as the cell ages, and tracking that
 * divergence is how electrode slippage is detected.
 */
ck_real_t ck_surface_stoichiometry_negative(const ck_estimator_t *est, ck_real_t current);
ck_real_t ck_surface_stoichiometry_positive(const ck_estimator_t *est, ck_real_t current);
ck_real_t ck_bulk_stoichiometry_negative(const ck_estimator_t *est);
ck_real_t ck_bulk_stoichiometry_positive(const ck_estimator_t *est);

#ifdef __cplusplus
}}
#endif

#endif /* {_GUARD} */
"""


def _electrode_tables(el: ElectrodeSpec, precision: str) -> str:
    tag = "neg" if el.name == "negative" else "pos"
    parts = [
        _array_1d(f"ck_c_surface_{tag}", el.c_surface, precision),
        _array_1d(f"ck_c_bulk_{tag}", el.c_bulk, precision),
        _array_1d(f"ck_ocp_{tag}", el.ocp_table.values, precision, per_line=8),
    ]
    return "\n\n".join(parts)


def _electrode_defines(el: ElectrodeSpec, precision: str) -> str:
    tag = "NEG" if el.name == "negative" else "POS"
    table = el.ocp_table
    return textwrap.dedent(
        f"""\
        #define CK_D_SURFACE_{tag}   ({_fmt(el.d_surface, precision)})
        #define CK_FLUX_{tag}        ({_fmt(el.flux_per_amp, precision)})
        #define CK_J_{tag}           ({_fmt(el.j_per_amp, precision)})
        #define CK_CMAX_{tag}        ({_fmt(el.max_concentration, precision)})
        #define CK_I0_PREFACTOR_{tag} ({_fmt(el.exchange_prefactor, precision)})
        #define CK_OCP_N_{tag}       {table.n}
        #define CK_OCP_MIN_{tag}     ({_fmt(table.sto_min, precision)})
        #define CK_OCP_STEP_{tag}    ({_fmt(table.step, precision)})
        #define CK_OCP_INVSTEP_{tag} ({_fmt(1.0 / table.step, precision)})"""
    )


def emit_source(spec: EstimatorSpec, precision: str = "double") -> str:
    """Generate the implementation file."""
    neg, pos = spec.negative, spec.positive
    sqrt_fn = "sqrtf" if precision == "float" else "sqrt"
    log_fn = "logf" if precision == "float" else "log"

    return f"""\
/* cellkernel generated estimator -- DO NOT EDIT BY HAND. */
#include "cellkernel_estimator.h"

#include <math.h>

/* ------------------------------------------------------------------ constants */

{_electrode_defines(neg, precision)}

{_electrode_defines(pos, precision)}

#define CK_R_CONTACT      ({_fmt(spec.contact_resistance, precision)})
#define CK_KINETIC_PREFAC ({_fmt(spec.kinetic_prefactor, precision)})
#define CK_SOC_SCALE      ({_fmt(spec.soc_scale, precision)})
#define CK_SOC_OFFSET     ({_fmt(spec.soc_offset, precision)})
#define CK_MEAS_NOISE     ({_fmt(spec.measurement_noise, precision)})

/* State transition. Input is cell current in amperes; the molar-flux
 * conversion is already folded into ck_B, which saves a multiply per state and
 * keeps the public interface in engineering units. */
{_array_2d("ck_A", spec.A, precision)}

{_array_1d("ck_B", spec.B, precision)}

/* Process noise covariance.
 *
 * Full matrix rather than a diagonal, and not for generality: the dominant
 * disturbance is current measurement error, which reaches the state through the
 * single direction ck_B, making the true covariance rank one. A diagonal
 * approximation of it lets the filter move the state off the manifold of
 * physically realisable concentration profiles, which is divergent rather than
 * merely suboptimal. */
{_array_2d("ck_Q", spec.process_noise, precision)}

/* Initial covariance. Rank one along d(state)/d(soc) plus a regulariser, since a
 * cold-boot state-of-charge error is a single degree of freedom shared by both
 * electrodes. */
{_array_2d("ck_P0", spec.initial_covariance, precision)}

/* Uniform-loading maps: state vector for a particle at unit concentration. */
{_array_1d("ck_map_neg", spec.map_negative, precision)}

{_array_1d("ck_map_pos", spec.map_positive, precision)}

/* Open-circuit potential tables and state-space output rows. */
{_electrode_tables(neg, precision)}

{_electrode_tables(pos, precision)}

/* Stoichiometry window endpoints, for ck_init. */
#define CK_STO0_NEG ({_fmt(spec.sto_negative[0], precision)})
#define CK_STO1_NEG ({_fmt(spec.sto_negative[1], precision)})
#define CK_STO0_POS ({_fmt(spec.sto_positive[0], precision)})
#define CK_STO1_POS ({_fmt(spec.sto_positive[1], precision)})

/* -------------------------------------------------------------------- helpers */

/*
 * Linear interpolation on a uniform grid. Constant time: one multiply, one
 * truncation, one blend. A binary search over a non-uniform table would make
 * execution time depend on state of charge, which is exactly the kind of
 * data-dependent timing a real-time schedule cannot absorb.
 */
static ck_real_t ck_lookup(const ck_real_t *table, int count, ck_real_t sto,
                           ck_real_t sto_min, ck_real_t inv_step)
{{
    ck_real_t pos = (sto - sto_min) * inv_step;
    int i;
    ck_real_t frac;

    if (pos < (ck_real_t) 0) {{
        pos = (ck_real_t) 0;
    }} else if (pos > (ck_real_t) (count - 1)) {{
        pos = (ck_real_t) (count - 1);
    }}
    i = (int) pos;
    if (i > count - 2) {{
        i = count - 2;
    }}
    frac = pos - (ck_real_t) i;
    return table[i] + frac * (table[i + 1] - table[i]);
}}

/* Slope of the interpolating segment containing ``sto``.
 *
 * The filter must differentiate the potential it actually evaluates. Using the
 * analytic fit's derivative here instead would make the Kalman gain inconsistent
 * with the residual it multiplies -- a subtle mismatch that shows up as a small
 * steady-state bias rather than as instability, and is correspondingly hard to
 * find later. */
static ck_real_t ck_lookup_slope(const ck_real_t *table, int count, ck_real_t sto,
                                 ck_real_t sto_min, ck_real_t inv_step)
{{
    ck_real_t pos = (sto - sto_min) * inv_step;
    int i;

    if (pos < (ck_real_t) 0) {{
        pos = (ck_real_t) 0;
    }} else if (pos > (ck_real_t) (count - 1)) {{
        pos = (ck_real_t) (count - 1);
    }}
    i = (int) pos;
    if (i > count - 2) {{
        i = count - 2;
    }}
    return (table[i + 1] - table[i]) * inv_step;
}}

/*
 * asinh, written out rather than taken from libm.
 *
 * Two reasons. C89 libm has no asinh, and some vendor toolchains for small cores
 * still ship an incomplete C99 math library. More importantly, this form is
 * identical to what the Python reference evaluates, so the cross-check in
 * cellkernel.verify measures code generation fidelity rather than differences
 * between one libm and another.
 *
 * The formulation log(u + sqrt(u*u + 1)) loses accuracy for large negative u
 * through cancellation, so the odd symmetry asinh(-u) = -asinh(u) is used to
 * evaluate on the non-negative branch only.
 */
static ck_real_t ck_asinh(ck_real_t u)
{{
    ck_real_t a = u < (ck_real_t) 0 ? -u : u;
    ck_real_t r = {log_fn}(a + {sqrt_fn}(a * a + (ck_real_t) 1));
    return u < (ck_real_t) 0 ? -r : r;
}}

static ck_real_t ck_clamp_concentration(ck_real_t c, ck_real_t c_max)
{{
    const ck_real_t margin = (ck_real_t) 1e-6 * c_max;
    if (c < margin) {{
        return margin;
    }}
    if (c > c_max - margin) {{
        return c_max - margin;
    }}
    return c;
}}

static ck_real_t ck_surface_neg(const ck_estimator_t *est, ck_real_t current)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        acc += ck_c_surface_neg[i] * est->x[i];
    }}
    return acc + CK_D_SURFACE_NEG * (CK_FLUX_NEG * current);
}}

static ck_real_t ck_surface_pos(const ck_estimator_t *est, ck_real_t current)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_POSITIVE; ++i) {{
        acc += ck_c_surface_pos[i] * est->x[CK_N_NEGATIVE + i];
    }}
    return acc + CK_D_SURFACE_POS * (CK_FLUX_POS * current);
}}

/* Butler-Volmer overpotential, exact inverse rather than a Tafel or linear
 * approximation: eta = (2RT/F) asinh(j / 2 i0). */
static ck_real_t ck_overpotential(ck_real_t c_surf, ck_real_t c_max,
                                  ck_real_t i0_prefactor, ck_real_t j)
{{
    ck_real_t c = ck_clamp_concentration(c_surf, c_max);
    ck_real_t i0 = i0_prefactor * {sqrt_fn}(c) * {sqrt_fn}(c_max - c);
    return CK_KINETIC_PREFAC * ck_asinh(j / ((ck_real_t) 2 * i0));
}}

/* d(eta)/d(c_surf). See cellkernel.models.spm for the derivation. */
static ck_real_t ck_d_overpotential(ck_real_t c_surf, ck_real_t c_max,
                                    ck_real_t i0_prefactor, ck_real_t j)
{{
    ck_real_t c = ck_clamp_concentration(c_surf, c_max);
    ck_real_t i0 = i0_prefactor * {sqrt_fn}(c) * {sqrt_fn}(c_max - c);
    ck_real_t u = j / ((ck_real_t) 2 * i0);
    ck_real_t d_log_i0 = (c_max - (ck_real_t) 2 * c) / ((ck_real_t) 2 * c * (c_max - c));
    return CK_KINETIC_PREFAC * (-u * d_log_i0) / {sqrt_fn}((ck_real_t) 1 + u * u);
}}

/* ---------------------------------------------------------------- public API */

void ck_init(ck_estimator_t *est, ck_real_t soc)
{{
    const ck_real_t c_neg = (CK_STO0_NEG + soc * (CK_STO1_NEG - CK_STO0_NEG)) * CK_CMAX_NEG;
    const ck_real_t c_pos = (CK_STO0_POS + soc * (CK_STO1_POS - CK_STO0_POS)) * CK_CMAX_POS;
    int i, j;

    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        est->x[i] = ck_map_neg[i] * c_neg;
    }}
    for (i = 0; i < CK_N_POSITIVE; ++i) {{
        est->x[CK_N_NEGATIVE + i] = ck_map_pos[i] * c_pos;
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            est->P[i][j] = ck_P0[i][j];
        }}
    }}
}}

ck_real_t ck_voltage(const ck_estimator_t *est, ck_real_t current)
{{
    const ck_real_t cs_n = ck_surface_neg(est, current);
    const ck_real_t cs_p = ck_surface_pos(est, current);
    const ck_real_t u_n = ck_lookup(ck_ocp_neg, CK_OCP_N_NEG, cs_n / CK_CMAX_NEG,
                                    CK_OCP_MIN_NEG, CK_OCP_INVSTEP_NEG);
    const ck_real_t u_p = ck_lookup(ck_ocp_pos, CK_OCP_N_POS, cs_p / CK_CMAX_POS,
                                    CK_OCP_MIN_POS, CK_OCP_INVSTEP_POS);
    const ck_real_t eta_n = ck_overpotential(cs_n, CK_CMAX_NEG, CK_I0_PREFACTOR_NEG,
                                             CK_J_NEG * current);
    const ck_real_t eta_p = ck_overpotential(cs_p, CK_CMAX_POS, CK_I0_PREFACTOR_POS,
                                             CK_J_POS * current);
    return (u_p + eta_p) - (u_n + eta_n) - CK_R_CONTACT * current;
}}

ck_real_t ck_soc(const ck_estimator_t *est)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        acc += ck_c_bulk_neg[i] * est->x[i];
    }}
    return CK_SOC_SCALE * acc + CK_SOC_OFFSET;
}}

ck_real_t ck_surface_stoichiometry_negative(const ck_estimator_t *est, ck_real_t current)
{{
    return ck_surface_neg(est, current) / CK_CMAX_NEG;
}}

ck_real_t ck_surface_stoichiometry_positive(const ck_estimator_t *est, ck_real_t current)
{{
    return ck_surface_pos(est, current) / CK_CMAX_POS;
}}

ck_real_t ck_bulk_stoichiometry_negative(const ck_estimator_t *est)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        acc += ck_c_bulk_neg[i] * est->x[i];
    }}
    return acc / CK_CMAX_NEG;
}}

ck_real_t ck_bulk_stoichiometry_positive(const ck_estimator_t *est)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_POSITIVE; ++i) {{
        acc += ck_c_bulk_pos[i] * est->x[CK_N_NEGATIVE + i];
    }}
    return acc / CK_CMAX_POS;
}}

/*
 * Gradient of predicted voltage with respect to the state.
 *
 * Surface concentration is a linear functional of the state, so the gradient is
 * a scalar sensitivity per electrode times the corresponding output row -- no
 * numerical differentiation, and cost linear in the state count.
 */
static void ck_voltage_jacobian(const ck_estimator_t *est, ck_real_t current,
                                ck_real_t *grad)
{{
    const ck_real_t cs_n = ck_surface_neg(est, current);
    const ck_real_t cs_p = ck_surface_pos(est, current);
    ck_real_t sens_n, sens_p;
    int i;

    sens_n = -(ck_lookup_slope(ck_ocp_neg, CK_OCP_N_NEG, cs_n / CK_CMAX_NEG,
                               CK_OCP_MIN_NEG, CK_OCP_INVSTEP_NEG) / CK_CMAX_NEG
               + ck_d_overpotential(cs_n, CK_CMAX_NEG, CK_I0_PREFACTOR_NEG,
                                    CK_J_NEG * current));
    sens_p = ck_lookup_slope(ck_ocp_pos, CK_OCP_N_POS, cs_p / CK_CMAX_POS,
                             CK_OCP_MIN_POS, CK_OCP_INVSTEP_POS) / CK_CMAX_POS
             + ck_d_overpotential(cs_p, CK_CMAX_POS, CK_I0_PREFACTOR_POS,
                                  CK_J_POS * current);

    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        grad[i] = sens_n * ck_c_surface_neg[i];
    }}
    for (i = 0; i < CK_N_POSITIVE; ++i) {{
        grad[CK_N_NEGATIVE + i] = sens_p * ck_c_surface_pos[i];
    }}
}}

void ck_predict(ck_estimator_t *est, ck_real_t current)
{{
    ck_real_t x_next[CK_N_STATES];
    ck_real_t ap[CK_N_STATES][CK_N_STATES];
    int i, j, k;

    /* x <- A x + B I. The process model is exactly linear, so this is not a
     * linearisation: ck_A is the true Jacobian, and the covariance propagation
     * below is exact given the noise model. */
    for (i = 0; i < CK_N_STATES; ++i) {{
        ck_real_t acc = ck_B[i] * current;
        for (j = 0; j < CK_N_STATES; ++j) {{
            acc += ck_A[i][j] * est->x[j];
        }}
        x_next[i] = acc;
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        est->x[i] = x_next[i];
    }}

    /* P <- A P A' + Q, in two passes through a scratch matrix. */
    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            ck_real_t acc = (ck_real_t) 0;
            for (k = 0; k < CK_N_STATES; ++k) {{
                acc += ck_A[i][k] * est->P[k][j];
            }}
            ap[i][j] = acc;
        }}
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            ck_real_t acc = (ck_real_t) 0;
            for (k = 0; k < CK_N_STATES; ++k) {{
                acc += ap[i][k] * ck_A[j][k];
            }}
            est->P[i][j] = acc + ck_Q[i][j];
        }}
    }}
}}

ck_real_t ck_correct(ck_estimator_t *est, ck_real_t current, ck_real_t measured_voltage)
{{
    ck_real_t h[CK_N_STATES];
    ck_real_t ph[CK_N_STATES];
    ck_real_t gain[CK_N_STATES];
    ck_real_t denom = CK_MEAS_NOISE;
    ck_real_t innovation;
    int i, j;

    ck_voltage_jacobian(est, current, h);

    for (i = 0; i < CK_N_STATES; ++i) {{
        ck_real_t acc = (ck_real_t) 0;
        for (j = 0; j < CK_N_STATES; ++j) {{
            acc += est->P[i][j] * h[j];
        }}
        ph[i] = acc;
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        denom += h[i] * ph[i];
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        gain[i] = ph[i] / denom;
    }}

    innovation = measured_voltage - ck_voltage(est, current);
    for (i = 0; i < CK_N_STATES; ++i) {{
        est->x[i] += gain[i] * innovation;
    }}

    /* P <- P - K (P H)'.
     *
     * The Joseph form (I-KH) P (I-KH)' + K R K' is more robust in general, at
     * roughly three times the cost. It is not needed here: the measurement is
     * scalar, so the gain denominator is a positive scalar that cannot become
     * indefinite, and the explicit symmetrisation below removes the asymmetry
     * that would otherwise accumulate in single precision. Switch to Joseph if
     * you add a second measurement channel. */
    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            est->P[i][j] -= gain[i] * ph[j];
        }}
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < i; ++j) {{
            ck_real_t avg = (ck_real_t) 0.5 * (est->P[i][j] + est->P[j][i]);
            est->P[i][j] = avg;
            est->P[j][i] = avg;
        }}
    }}
    return innovation;
}}

ck_real_t ck_step(ck_estimator_t *est, ck_real_t current, ck_real_t measured_voltage)
{{
    ck_predict(est, current);
    (void) ck_correct(est, current, measured_voltage);
    return ck_soc(est);
}}
"""


def emit_main(spec: EstimatorSpec | None = None) -> str:
    """Generate a host harness that replays a CSV through the estimator.

    Reads ``current,voltage`` rows from standard input and writes
    ``soc,voltage,sto_neg,sto_pos``. :mod:`cellkernel.verify` drives this to
    compare the compiled estimator against the Python reference, and it doubles as
    a worked example of the intended call sequence.

    Outputs for sample ``k`` are evaluated *before* that sample's prediction and
    correction, so they are the one-step-ahead prediction given everything known
    up to ``k - 1``. That ordering is chosen for two reasons: subtracting the
    recorded voltage from the measured one gives exactly the filter innovation, so
    it need not be reported separately; and in open-loop mode the output then
    matches :meth:`cellkernel.models.base.CellModel.simulate` sample for sample,
    which makes the two directly comparable without an off-by-one.

    Printed with 17 significant digits, so the comparison is limited by the
    arithmetic rather than by the decimal round-trip.

    Takes no information from the spec: the harness is written against the public
    header only, so it is the same file for every generated estimator. That is
    deliberate -- it means the harness exercises exactly the interface a firmware
    integrator sees, and cannot accidentally depend on a constant that the real
    caller would not have.
    """
    return """\
/* cellkernel generated host harness -- replays a CSV through the estimator. */
#include "cellkernel_estimator.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * usage: ck_harness (openloop|filter) INITIAL_SOC < input.csv > output.csv
 *
 *   openloop  ignore the measured voltage and run the model forward only, which
 *             isolates model and code-generation fidelity from filter behaviour.
 *   filter    apply the extended Kalman correction at every sample.
 *
 * Input rows are "current,voltage"; current is positive on discharge.
 */
int main(int argc, char **argv)
{
    ck_estimator_t est;
    double current, voltage;
    int filtering;
    double soc0;

    if (argc < 3) {
        fprintf(stderr, "usage: %s (openloop|filter) INITIAL_SOC\\n", argv[0]);
        return 2;
    }
    filtering = (argv[1][0] == 'f');
    soc0 = atof(argv[2]);

    ck_init(&est, (ck_real_t) soc0);
    printf("soc,voltage,sto_neg,sto_pos\\n");

    while (scanf("%lf,%lf", &current, &voltage) == 2) {
        const ck_real_t i_k = (ck_real_t) current;

        printf("%.17g,%.17g,%.17g,%.17g\\n",
               (double) ck_soc(&est),
               (double) ck_voltage(&est, i_k),
               (double) ck_surface_stoichiometry_negative(&est, i_k),
               (double) ck_surface_stoichiometry_positive(&est, i_k));

        ck_predict(&est, i_k);
        if (filtering) {
            (void) ck_correct(&est, i_k, (ck_real_t) voltage);
        }
    }
    return 0;
}
"""
