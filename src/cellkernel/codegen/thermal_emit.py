"""Emit a temperature-scheduled C99 estimator."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .emit import _array_1d, _array_2d, _fmt
from .thermal_spec import ScheduledElectrodeSpec, ThermalEstimatorSpec

__all__ = ["emit_thermal_header", "emit_thermal_source", "emit_thermal_main"]

_GUARD = "CELLKERNEL_SCHEDULED_ESTIMATOR_H"


def _array_3d(name: str, tensor: np.ndarray, precision: str) -> str:
    """Emit a ``[grid][row][col]`` table of matrices."""
    tensor = np.asarray(tensor)
    blocks = []
    for plane in tensor:
        rows = ",\n".join(
            "        { " + ", ".join(_fmt(v, precision) for v in row) + " }" for row in plane
        )
        blocks.append("    {\n" + rows + "\n    }")
    body = ",\n".join(blocks)
    return (
        f"static const ck_real_t {name}"
        f"[{tensor.shape[0]}][{tensor.shape[1]}][{tensor.shape[2]}] = "
        f"{{\n{body}\n}};"
    )


def _tag(electrode: ScheduledElectrodeSpec) -> str:
    return "neg" if electrode.name == "negative" else "pos"


def emit_thermal_header(spec: ThermalEstimatorSpec, precision: str = "double") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lo, hi = spec.voltage_limits
    grid = spec.temperature_grid
    return f"""\
/*
 * cellkernel generated temperature-scheduled estimator -- DO NOT EDIT BY HAND.
 *
 * Physics-based state-of-charge estimator for a lithium-ion cell, valid across a
 * temperature range rather than at a single operating point.
 *
 * Generated   : {stamp}
 * Model       : {spec.provenance}
 * States      : {spec.n_states} ({spec.n_negative} negative, {spec.n_positive} positive)
 * Step        : {spec.dt} s (fixed; baked into the coefficient tables)
 * Temperature : {grid[0]:.2f} K to {grid[-1]:.2f} K, {spec.n_temperatures} grid points
 * Precision   : {precision}
 *
 * TEMPERATURE IS AN INPUT, NOT A STATE.
 *
 * Every entry point takes the measured cell temperature in kelvin. That is a
 * deliberate choice: any pack worth running this on has thermistors, so
 * temperature is a measurement rather than an unknown, and treating it as one
 * keeps the covariance the size it was and keeps heat generation -- along with
 * the poorly identified thermal parameters it needs -- out of this file
 * entirely. Temperature is also only weakly observable from terminal voltage,
 * so a filter that inferred it would do worse than the thermistor anyway.
 *
 * Temperatures outside the grid are clamped to the nearest endpoint rather than
 * extrapolated. An Arrhenius law extrapolated past its calibration is not a
 * useful thing to trust.
 */
#ifndef {_GUARD}
#define {_GUARD}

#ifdef __cplusplus
extern "C" {{
#endif

typedef {precision} ck_real_t;

#define CK_N_STATES       {spec.n_states}
#define CK_N_NEGATIVE     {spec.n_negative}
#define CK_N_POSITIVE     {spec.n_positive}
#define CK_N_TEMPERATURES {spec.n_temperatures}
#define CK_DT             ({_fmt(spec.dt, precision)})
#define CK_T_MIN          ({_fmt(grid[0], precision)})
#define CK_T_MAX          ({_fmt(grid[-1], precision)})
#define CK_V_MIN          ({_fmt(lo, precision)})
#define CK_V_MAX          ({_fmt(hi, precision)})

/*
 * Bisection steps in ck_max_charge_current, fixed so the routine costs the same
 * on every call, and the largest current the potential tables were sized for.
 */
#define CK_LIMIT_ITERATIONS 24
#define CK_LIMIT_CEILING  ({_fmt(spec.current_ceiling, precision)})

/*
 * Worst-case error of each potential table against the analytic fit, in volts.
 * ck_plating_potential inherits the negative electrode's figure directly, so a
 * plating margin below it is measuring the table rather than the cell. Graphite
 * tabulates far worse than a layered oxide because of its stage transitions.
 */
#define CK_OCP_ERROR_NEG  ({_fmt(spec.negative.ocp_table.max_abs_error, precision)})
#define CK_OCP_ERROR_POS  ({_fmt(spec.positive.ocp_table.max_abs_error, precision)})

/*
 * All mutable state. The blended matrices are cached here rather than
 * recomputed per call: a predict-and-correct cycle touches them four times, and
 * blending once per step costs one pass over the tables instead of four.
 */
typedef struct {{
    ck_real_t x[CK_N_STATES];                  /* diffusion coordinates       */
    ck_real_t P[CK_N_STATES][CK_N_STATES];     /* state covariance            */

    ck_real_t cached_temperature;              /* temperature of the blend    */
    ck_real_t A[CK_N_STATES][CK_N_STATES];     /* blended transition          */
    ck_real_t B[CK_N_STATES];                  /* blended input column        */
    ck_real_t c_surface_neg[CK_N_NEGATIVE];
    ck_real_t c_surface_pos[CK_N_POSITIVE];
    ck_real_t d_surface_neg;
    ck_real_t d_surface_pos;
    ck_real_t i0_prefactor_neg;                /* Arrhenius-corrected         */
    ck_real_t i0_prefactor_pos;
    ck_real_t kinetic_prefactor;               /* 2 R T / F                   */
}} ck_estimator_t;

/* Reset to a rested cell at a given state of charge and temperature. */
void ck_init(ck_estimator_t *est, ck_real_t soc, ck_real_t temperature);

/*
 * Rebuild the cached coefficients for a temperature. Called automatically by the
 * functions below when the temperature has changed, so there is normally no need
 * to call it directly.
 */
void ck_refresh(ck_estimator_t *est, ck_real_t temperature);

/* Advance state and covariance one sample. Current is POSITIVE ON DISCHARGE. */
void ck_predict(ck_estimator_t *est, ck_real_t current, ck_real_t temperature);

/* Predicted terminal voltage in volts. */
ck_real_t ck_voltage(ck_estimator_t *est, ck_real_t current, ck_real_t temperature);

/* State of charge in [0, 1], from the conserved average concentration. */
ck_real_t ck_soc(const ck_estimator_t *est);

/* One extended Kalman correction. Returns the innovation in volts. */
ck_real_t ck_correct(ck_estimator_t *est, ck_real_t current, ck_real_t temperature,
                     ck_real_t measured_voltage);

/* Predict then correct. The normal per-sample entry point. */
ck_real_t ck_step(ck_estimator_t *est, ck_real_t current, ck_real_t temperature,
                  ck_real_t measured_voltage);

/* Surface stoichiometry, in [0, 1]. Diagnostic: this, not state of charge, is
 * what limits fast charge. */
ck_real_t ck_surface_stoichiometry_negative(ck_estimator_t *est, ck_real_t current,
                                            ck_real_t temperature);
ck_real_t ck_surface_stoichiometry_positive(ck_estimator_t *est, ck_real_t current,
                                            ck_real_t temperature);

/* Bulk stoichiometry of each electrode: its own state of charge. The two drift
 * apart as the cell ages, and tracking that divergence is how electrode slippage
 * is detected. Temperature independent, since the conserved functional is. */
ck_real_t ck_bulk_stoichiometry_negative(const ck_estimator_t *est);
ck_real_t ck_bulk_stoichiometry_positive(const ck_estimator_t *est);

/*
 * Negative electrode potential against lithium metal, in volts. Below zero the
 * cell is depositing metal rather than intercalating it.
 *
 * This is the version of the plating check worth having, because plating is a
 * cold-weather failure and this estimator is the one that knows the temperature.
 * The isothermal generator can only answer for the single point it was built at.
 */
ck_real_t ck_plating_potential(ck_estimator_t *est, ck_real_t current,
                               ck_real_t temperature);

/*
 * Largest charging current, as a positive magnitude in amperes, that keeps the
 * negative electrode at least `margin` volts above the plating onset at the
 * measured temperature. Zero means no charging current is safe, which is a real
 * answer at high state of charge in the cold.
 *
 * Bisection over a fixed CK_LIMIT_ITERATIONS steps, so execution time is
 * constant. The blend is refreshed once for the whole search rather than on
 * every evaluation, since temperature does not move inside it.
 */
ck_real_t ck_max_charge_current(ck_estimator_t *est, ck_real_t temperature,
                                ck_real_t margin, ck_real_t ceiling);

#ifdef __cplusplus
}}
#endif

#endif /* {_GUARD} */
"""


def _electrode_tables(electrode: ScheduledElectrodeSpec, precision: str) -> str:
    tag = _tag(electrode)
    parts = [
        _array_3d(f"ck_A_{tag}", electrode.A_grid, precision),
        _array_2d(f"ck_B_{tag}", electrode.B_grid, precision),
        _array_2d(f"ck_c_surface_{tag}", electrode.c_surface_grid, precision),
        _array_1d(f"ck_d_surface_{tag}", electrode.d_surface_grid, precision),
        _array_1d(f"ck_c_bulk_{tag}", electrode.c_bulk, precision),
        _array_1d(f"ck_factor_{tag}", electrode.diffusion_factor_grid, precision, per_line=6),
        _array_1d(f"ck_ocp_{tag}", electrode.ocp_table.values, precision, per_line=8),
    ]
    return "\n\n".join(parts)


def _electrode_defines(electrode: ScheduledElectrodeSpec, precision: str) -> str:
    tag = _tag(electrode).upper()
    table = electrode.ocp_table
    return f"""\
#define CK_N_{tag}_STATES    {electrode.n_states}
#define CK_FLUX_{tag}        ({_fmt(electrode.flux_per_amp, precision)})
#define CK_J_{tag}           ({_fmt(electrode.j_per_amp, precision)})
#define CK_CMAX_{tag}        ({_fmt(electrode.max_concentration, precision)})
#define CK_I0_REF_{tag}      ({_fmt(electrode.exchange_prefactor, precision)})
#define CK_EA_DIFF_{tag}     ({_fmt(electrode.diffusion_activation_energy, precision)})
#define CK_EA_REACT_{tag}    ({_fmt(electrode.reaction_activation_energy, precision)})
#define CK_OCP_N_{tag}       {table.n}
#define CK_OCP_MIN_{tag}     ({_fmt(table.sto_min, precision)})
#define CK_OCP_INVSTEP_{tag} ({_fmt(1.0 / table.step, precision)})"""


def emit_thermal_source(spec: ThermalEstimatorSpec, precision: str = "double") -> str:
    neg, pos = spec.negative, spec.positive
    sqrt_fn = "sqrtf" if precision == "float" else "sqrt"
    log_fn = "logf" if precision == "float" else "log"
    exp_fn = "expf" if precision == "float" else "exp"

    return f"""\
/* cellkernel generated temperature-scheduled estimator -- DO NOT EDIT BY HAND. */
#include "cellkernel_scheduled.h"

#include <math.h>

/* ------------------------------------------------------------------ constants */

{_electrode_defines(neg, precision)}

{_electrode_defines(pos, precision)}

#define CK_R_CONTACT   ({_fmt(spec.contact_resistance, precision)})
#define CK_SOC_SCALE   ({_fmt(spec.soc_scale, precision)})
#define CK_SOC_OFFSET  ({_fmt(spec.soc_offset, precision)})
#define CK_MEAS_NOISE  ({_fmt(spec.measurement_noise, precision)})
#define CK_T_REF       ({_fmt(spec.reference_temperature, precision)})
#define CK_INV_T_REF   ({_fmt(1.0 / spec.reference_temperature, precision)})
#define CK_GAS_CONST   ({_fmt(8.31446261815324, precision)})
#define CK_TWO_R_OVER_F ({_fmt(2.0 * 8.31446261815324 / 96485.33212, precision)})

#define CK_STO0_NEG ({_fmt(spec.sto_negative[0], precision)})
#define CK_STO1_NEG ({_fmt(spec.sto_negative[1], precision)})
#define CK_STO0_POS ({_fmt(spec.sto_positive[0], precision)})
#define CK_STO1_POS ({_fmt(spec.sto_positive[1], precision)})

{_array_1d("ck_temperature_grid", spec.temperature_grid, precision, per_line=6)}

{_array_2d("ck_Q", spec.process_noise, precision)}

{_array_2d("ck_P0", spec.initial_covariance, precision)}

{_array_1d("ck_map_neg", neg.uniform_map, precision)}

{_array_1d("ck_map_pos", pos.uniform_map, precision)}

{_electrode_tables(neg, precision)}

{_electrode_tables(pos, precision)}

/* -------------------------------------------------------------------- helpers */

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

/* asinh written out; see the isothermal generator for why it is not taken from
 * libm. Odd symmetry avoids cancellation on the negative branch. */
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

/* Arrhenius factor relative to the reference temperature. */
static ck_real_t ck_arrhenius(ck_real_t activation_energy, ck_real_t temperature)
{{
    if (activation_energy == (ck_real_t) 0) {{
        return (ck_real_t) 1;
    }}
    return {exp_fn}(activation_energy / CK_GAS_CONST
                    * (CK_INV_T_REF - (ck_real_t) 1 / temperature));
}}

/*
 * Locate the bracketing grid interval. Linear scan rather than binary search:
 * the grid has a handful of points, a scan has no branch misprediction penalty
 * at that size, and its execution time is bounded by the grid length rather than
 * varying with the data.
 */
static int ck_bracket(ck_real_t temperature)
{{
    int i;
    for (i = 0; i < CK_N_TEMPERATURES - 2; ++i) {{
        if (temperature < ck_temperature_grid[i + 1]) {{
            return i;
        }}
    }}
    return CK_N_TEMPERATURES - 2;
}}

/*
 * Blend weight for one electrode.
 *
 * Computed on the Arrhenius factor rather than on temperature. At a sample
 * period short against the diffusion time constant the discrete matrix is close
 * to I + A_c(D) dt with A_c proportional to D, so the matrices are nearly affine
 * in diffusivity -- and diffusivity is exponential in 1/T. Interpolating linearly
 * in temperature therefore fits a straight line through an exponential, and does
 * so worst at the cold end: on a nine-point grid that is worth 197 mV of voltage
 * error at -18 C against 1.8 mV for this form.
 */
static ck_real_t ck_blend(int lower, ck_real_t temperature,
                          ck_real_t activation_energy, const ck_real_t *factors)
{{
    ck_real_t low, high, span, weight;

    if (activation_energy == (ck_real_t) 0) {{
        span = ck_temperature_grid[lower + 1] - ck_temperature_grid[lower];
        weight = (temperature - ck_temperature_grid[lower]) / span;
    }} else {{
        low = factors[lower];
        high = factors[lower + 1];
        span = high - low;
        if (span > (ck_real_t) -1e-30 && span < (ck_real_t) 1e-30) {{
            weight = (ck_real_t) 0;
        }} else {{
            weight = (ck_arrhenius(activation_energy, temperature) - low) / span;
        }}
    }}
    if (weight < (ck_real_t) 0) {{
        weight = (ck_real_t) 0;
    }} else if (weight > (ck_real_t) 1) {{
        weight = (ck_real_t) 1;
    }}
    return weight;
}}

void ck_refresh(ck_estimator_t *est, ck_real_t temperature)
{{
    const int lower = ck_bracket(temperature);
    const ck_real_t wn = ck_blend(lower, temperature, CK_EA_DIFF_NEG, ck_factor_neg);
    const ck_real_t wp = ck_blend(lower, temperature, CK_EA_DIFF_POS, ck_factor_pos);
    const ck_real_t vn = (ck_real_t) 1 - wn;
    const ck_real_t vp = (ck_real_t) 1 - wp;
    int i, j;

    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            est->A[i][j] = (ck_real_t) 0;
        }}
    }}
    for (i = 0; i < CK_N_NEG_STATES; ++i) {{
        for (j = 0; j < CK_N_NEG_STATES; ++j) {{
            est->A[i][j] = vn * ck_A_neg[lower][i][j] + wn * ck_A_neg[lower + 1][i][j];
        }}
        est->B[i] = vn * ck_B_neg[lower][i] + wn * ck_B_neg[lower + 1][i];
        est->c_surface_neg[i] =
            vn * ck_c_surface_neg[lower][i] + wn * ck_c_surface_neg[lower + 1][i];
    }}
    for (i = 0; i < CK_N_POS_STATES; ++i) {{
        for (j = 0; j < CK_N_POS_STATES; ++j) {{
            est->A[CK_N_NEGATIVE + i][CK_N_NEGATIVE + j] =
                vp * ck_A_pos[lower][i][j] + wp * ck_A_pos[lower + 1][i][j];
        }}
        est->B[CK_N_NEGATIVE + i] = vp * ck_B_pos[lower][i] + wp * ck_B_pos[lower + 1][i];
        est->c_surface_pos[i] =
            vp * ck_c_surface_pos[lower][i] + wp * ck_c_surface_pos[lower + 1][i];
    }}
    est->d_surface_neg = vn * ck_d_surface_neg[lower] + wn * ck_d_surface_neg[lower + 1];
    est->d_surface_pos = vp * ck_d_surface_pos[lower] + wp * ck_d_surface_pos[lower + 1];

    est->i0_prefactor_neg = CK_I0_REF_NEG * ck_arrhenius(CK_EA_REACT_NEG, temperature);
    est->i0_prefactor_pos = CK_I0_REF_POS * ck_arrhenius(CK_EA_REACT_POS, temperature);
    est->kinetic_prefactor = CK_TWO_R_OVER_F * temperature;
    est->cached_temperature = temperature;
}}

/* Refresh only when the temperature has moved. A battery-management task reads a
 * thermistor far more slowly than it runs this loop, so the cache usually hits. */
static void ck_ensure(ck_estimator_t *est, ck_real_t temperature)
{{
    if (temperature != est->cached_temperature) {{
        ck_refresh(est, temperature);
    }}
}}

static ck_real_t ck_surface_neg(const ck_estimator_t *est, ck_real_t current)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        acc += est->c_surface_neg[i] * est->x[i];
    }}
    return acc + est->d_surface_neg * (CK_FLUX_NEG * current);
}}

static ck_real_t ck_surface_pos(const ck_estimator_t *est, ck_real_t current)
{{
    ck_real_t acc = (ck_real_t) 0;
    int i;
    for (i = 0; i < CK_N_POSITIVE; ++i) {{
        acc += est->c_surface_pos[i] * est->x[CK_N_NEGATIVE + i];
    }}
    return acc + est->d_surface_pos * (CK_FLUX_POS * current);
}}

static ck_real_t ck_overpotential(const ck_estimator_t *est, ck_real_t c_surf,
                                  ck_real_t c_max, ck_real_t prefactor, ck_real_t j)
{{
    ck_real_t c = ck_clamp_concentration(c_surf, c_max);
    ck_real_t i0 = prefactor * {sqrt_fn}(c) * {sqrt_fn}(c_max - c);
    return est->kinetic_prefactor * ck_asinh(j / ((ck_real_t) 2 * i0));
}}

static ck_real_t ck_d_overpotential(const ck_estimator_t *est, ck_real_t c_surf,
                                    ck_real_t c_max, ck_real_t prefactor, ck_real_t j)
{{
    ck_real_t c = ck_clamp_concentration(c_surf, c_max);
    ck_real_t i0 = prefactor * {sqrt_fn}(c) * {sqrt_fn}(c_max - c);
    ck_real_t u = j / ((ck_real_t) 2 * i0);
    ck_real_t d_log_i0 = (c_max - (ck_real_t) 2 * c) / ((ck_real_t) 2 * c * (c_max - c));
    return est->kinetic_prefactor * (-u * d_log_i0) / {sqrt_fn}((ck_real_t) 1 + u * u);
}}

/* ----------------------------------------------------------------- public API */

void ck_init(ck_estimator_t *est, ck_real_t soc, ck_real_t temperature)
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
    ck_refresh(est, temperature);
}}

ck_real_t ck_voltage(ck_estimator_t *est, ck_real_t current, ck_real_t temperature)
{{
    ck_real_t cs_n, cs_p, u_n, u_p, eta_n, eta_p;

    ck_ensure(est, temperature);
    cs_n = ck_surface_neg(est, current);
    cs_p = ck_surface_pos(est, current);
    u_n = ck_lookup(ck_ocp_neg, CK_OCP_N_NEG, cs_n / CK_CMAX_NEG,
                    CK_OCP_MIN_NEG, CK_OCP_INVSTEP_NEG);
    u_p = ck_lookup(ck_ocp_pos, CK_OCP_N_POS, cs_p / CK_CMAX_POS,
                    CK_OCP_MIN_POS, CK_OCP_INVSTEP_POS);
    eta_n = ck_overpotential(est, cs_n, CK_CMAX_NEG, est->i0_prefactor_neg,
                             CK_J_NEG * current);
    eta_p = ck_overpotential(est, cs_p, CK_CMAX_POS, est->i0_prefactor_pos,
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

ck_real_t ck_plating_potential(ck_estimator_t *est, ck_real_t current,
                               ck_real_t temperature)
{{
    ck_real_t cs_n, u_n, eta_n;

    ck_ensure(est, temperature);
    cs_n = ck_surface_neg(est, current);
    u_n = ck_lookup(ck_ocp_neg, CK_OCP_N_NEG, cs_n / CK_CMAX_NEG,
                    CK_OCP_MIN_NEG, CK_OCP_INVSTEP_NEG);
    eta_n = ck_overpotential(est, cs_n, CK_CMAX_NEG, est->i0_prefactor_neg,
                             CK_J_NEG * current);
    return u_n + eta_n;
}}

ck_real_t ck_max_charge_current(ck_estimator_t *est, ck_real_t temperature,
                                ck_real_t margin, ck_real_t ceiling)
{{
    ck_real_t low = (ck_real_t) 0;
    ck_real_t high = ceiling;
    int i;

    if (ceiling <= (ck_real_t) 0) {{
        return (ck_real_t) 0;
    }}
    /* One blend for the whole search: temperature does not move inside it, and
     * ck_ensure would otherwise be paid on all twenty-six evaluations. */
    ck_ensure(est, temperature);
    if (ck_plating_potential(est, -ceiling, temperature) >= margin) {{
        return ceiling;
    }}
    if (ck_plating_potential(est, (ck_real_t) 0, temperature) < margin) {{
        return (ck_real_t) 0;
    }}
    for (i = 0; i < CK_LIMIT_ITERATIONS; ++i) {{
        const ck_real_t middle = (ck_real_t) 0.5 * (low + high);
        if (ck_plating_potential(est, -middle, temperature) >= margin) {{
            low = middle;
        }} else {{
            high = middle;
        }}
    }}
    return low;
}}

ck_real_t ck_surface_stoichiometry_negative(ck_estimator_t *est, ck_real_t current,
                                            ck_real_t temperature)
{{
    ck_ensure(est, temperature);
    return ck_surface_neg(est, current) / CK_CMAX_NEG;
}}

ck_real_t ck_surface_stoichiometry_positive(ck_estimator_t *est, ck_real_t current,
                                            ck_real_t temperature)
{{
    ck_ensure(est, temperature);
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

static void ck_voltage_jacobian(const ck_estimator_t *est, ck_real_t current,
                                ck_real_t *grad)
{{
    const ck_real_t cs_n = ck_surface_neg(est, current);
    const ck_real_t cs_p = ck_surface_pos(est, current);
    ck_real_t sens_n, sens_p;
    int i;

    sens_n = -(ck_lookup_slope(ck_ocp_neg, CK_OCP_N_NEG, cs_n / CK_CMAX_NEG,
                               CK_OCP_MIN_NEG, CK_OCP_INVSTEP_NEG) / CK_CMAX_NEG
               + ck_d_overpotential(est, cs_n, CK_CMAX_NEG, est->i0_prefactor_neg,
                                    CK_J_NEG * current));
    sens_p = ck_lookup_slope(ck_ocp_pos, CK_OCP_N_POS, cs_p / CK_CMAX_POS,
                             CK_OCP_MIN_POS, CK_OCP_INVSTEP_POS) / CK_CMAX_POS
             + ck_d_overpotential(est, cs_p, CK_CMAX_POS, est->i0_prefactor_pos,
                                  CK_J_POS * current);

    for (i = 0; i < CK_N_NEGATIVE; ++i) {{
        grad[i] = sens_n * est->c_surface_neg[i];
    }}
    for (i = 0; i < CK_N_POSITIVE; ++i) {{
        grad[CK_N_NEGATIVE + i] = sens_p * est->c_surface_pos[i];
    }}
}}

void ck_predict(ck_estimator_t *est, ck_real_t current, ck_real_t temperature)
{{
    ck_real_t x_next[CK_N_STATES];
    ck_real_t ap[CK_N_STATES][CK_N_STATES];
    int i, j, k;

    ck_ensure(est, temperature);

    for (i = 0; i < CK_N_STATES; ++i) {{
        ck_real_t acc = est->B[i] * current;
        for (j = 0; j < CK_N_STATES; ++j) {{
            acc += est->A[i][j] * est->x[j];
        }}
        x_next[i] = acc;
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        est->x[i] = x_next[i];
    }}

    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            ck_real_t acc = (ck_real_t) 0;
            for (k = 0; k < CK_N_STATES; ++k) {{
                acc += est->A[i][k] * est->P[k][j];
            }}
            ap[i][j] = acc;
        }}
    }}
    for (i = 0; i < CK_N_STATES; ++i) {{
        for (j = 0; j < CK_N_STATES; ++j) {{
            ck_real_t acc = (ck_real_t) 0;
            for (k = 0; k < CK_N_STATES; ++k) {{
                acc += ap[i][k] * est->A[j][k];
            }}
            est->P[i][j] = acc + ck_Q[i][j];
        }}
    }}
}}

ck_real_t ck_correct(ck_estimator_t *est, ck_real_t current, ck_real_t temperature,
                     ck_real_t measured_voltage)
{{
    ck_real_t h[CK_N_STATES];
    ck_real_t ph[CK_N_STATES];
    ck_real_t gain[CK_N_STATES];
    ck_real_t denom = CK_MEAS_NOISE;
    ck_real_t innovation;
    int i, j;

    ck_ensure(est, temperature);
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

    innovation = measured_voltage - ck_voltage(est, current, temperature);
    for (i = 0; i < CK_N_STATES; ++i) {{
        est->x[i] += gain[i] * innovation;
    }}
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

ck_real_t ck_step(ck_estimator_t *est, ck_real_t current, ck_real_t temperature,
                  ck_real_t measured_voltage)
{{
    ck_predict(est, current, temperature);
    (void) ck_correct(est, current, temperature, measured_voltage);
    return ck_soc(est);
}}
"""


def emit_thermal_main() -> str:
    """Host harness: replays ``current,temperature,voltage`` rows."""
    return """\
/* cellkernel generated host harness -- replays a CSV through the estimator. */
#include "cellkernel_scheduled.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * usage: ck_harness (openloop|filter) INITIAL_SOC < input.csv > output.csv
 *
 * Input rows are "current,temperature,voltage"; current positive on discharge,
 * temperature in kelvin. Outputs for sample k are evaluated before that sample's
 * prediction, so subtracting the reported voltage from the measured one gives
 * exactly the filter innovation.
 */
int main(int argc, char **argv)
{
    ck_estimator_t est;
    double current, temperature, voltage;
    int filtering;
    double soc0;
    int first = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s (openloop|filter) INITIAL_SOC\\n", argv[0]);
        return 2;
    }
    filtering = (argv[1][0] == 'f');
    soc0 = atof(argv[2]);

    printf("soc,voltage,sto_neg,sto_pos,plating,charge_limit\\n");
    while (scanf("%lf,%lf,%lf", &current, &temperature, &voltage) == 3) {
        const ck_real_t i_k = (ck_real_t) current;
        const ck_real_t t_k = (ck_real_t) temperature;

        if (first) {
            ck_init(&est, (ck_real_t) soc0, t_k);
            first = 0;
        }

        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\\n",
               (double) ck_soc(&est),
               (double) ck_voltage(&est, i_k, t_k),
               (double) ck_surface_stoichiometry_negative(&est, i_k, t_k),
               (double) ck_surface_stoichiometry_positive(&est, i_k, t_k),
               (double) ck_plating_potential(&est, i_k, t_k),
               (double) ck_max_charge_current(&est, t_k, (ck_real_t) 0.01,
                                              (ck_real_t) CK_LIMIT_CEILING));

        ck_predict(&est, i_k, t_k);
        if (filtering) {
            (void) ck_correct(&est, i_k, t_k, (ck_real_t) voltage);
        }
    }
    return 0;
}
"""
