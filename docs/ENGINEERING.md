# Engineering reference

[Overview](../README.md) · [Calibration procedure](SOP.md) · [ML benchmark](BENCHMARK.md)

This page describes the model assumptions and implementation choices. Commands
run from the repository root. Measured results and toolchain details are in the
[validation record](VALIDATION.md).

## Model conventions

- Current is in amperes, **positive on discharge**.
- State of charge is a fraction from 0 to 1; the demo displays percentages.
- Temperature is in kelvin; terminal voltage is in volts.
- Simulated outputs at sample `k` use the state and current at `k`, before the state advances.
- The built-in parameters describe a cell design. An individual cell needs calibration.

## Reduced solid diffusion

Solid-phase diffusion is represented as a small discrete state-space system.
For a spherical particle, the surface response has transfer function

```math
G(s) = \frac{R}{D}\frac{\sinh \xi}{\xi\cosh\xi-\sinh\xi},
\qquad \xi = R\sqrt{s/D}.
```

The implementation separates the integrator carrying mass balance from the
remaining dynamics. All four approximations preserve the volume-averaged
concentration as an exact state:

| Method | Main tradeoff |
|---|---|
| Padé | Approximates the transfer function with few states. Coefficients are solved using exact rational arithmetic to avoid an ill-conditioned floating-point solve. |
| Spectral | Uses diffusion modes and a diagonal continuous-time state matrix. Truncation determines the retained timescales. |
| Finite volume | Resolves the interior concentration profile. Accuracy depends on spatial resolution. |
| Polynomial | Two-state approximation with the correct steady-state surface offset. |

Discretisation uses a matrix exponential offline. The resulting online update
is exact for the reduced linear system under piecewise-constant current; it
does not eliminate error introduced by reducing the original diffusion model.

See [the implementation](../src/cellkernel/rom),
[closed-form tests](../tests/test_rom.py), and
[the comparison example](../examples/01_compare_reduced_order_models.py).

## Cell parameters and voltage

Electrode stoichiometry limits must be charge balanced: both electrodes must
transfer the same usable capacity between empty and full charge.
`balanced_stoichiometry_window` solves for limits consistent with capacity and
the selected voltage window. `fit_stoichiometry_window` calibrates those limits
against measured pseudo-open-circuit voltage while constraining capacity.

For the single-particle model (SPM), terminal voltage is

```math
V = U_p(x_p^s) + \eta_p - U_n(x_n^s) - \eta_n - I R_c,
```

where the electrode potentials depend on surface concentration and the kinetic
overpotentials use symmetric Butler–Volmer kinetics. Solid diffusion has linear
state dynamics; the voltage measurement is nonlinear.

The equivalent-circuit model (ECM) provides a simpler baseline. SPMe adds
electrolyte transport, concentration polarisation and electrolyte resistance.
SPMe's transport coefficients are held constant; use its `validity()` result
to inspect whether a state is within the model's intended regime.

See [cell models](../src/cellkernel/models), [parameter handling](../src/cellkernel/params.py),
and [electrolyte tests](../tests/test_electrolyte.py).

## State estimation

The extended Kalman filter (EKF) uses the exact linear transition of the SPM for
prediction and linearises its voltage measurement for correction. Optional
iterations repeat the nonlinear measurement update from the same prior; they
do not count the same measurement as independent observations.

Initial and process covariance matter. Uncertainty in a rested cell's charge
should follow the uniform-concentration direction, while current measurement
uncertainty follows the input direction. Independent uncertainty on every state
does not represent either situation well.

SPMe electrolyte states that follow a known initial profile and current history
are propagated without a measurement correction. This relies on the stated
initial-condition and model assumptions; it does not establish that electrolyte
states in a real cell are known without uncertainty.

The UKF provides a sigma-point alternative. `DualEKF` additionally estimates
capacity retention and resistance growth under its documented degradation
assumptions. Check innovation and uncertainty diagnostics as well as SoC error.

The [charge-estimation example](../examples/02_estimate_state_of_charge.py)
compares these filters. The [demo](../src/cellkernel/demo.py) uses a three-iteration
EKF and a correctly time-aligned current-counting baseline.

## Temperature, ageing and charging

`ThermalSPM` couples a lumped thermal state to temperature-dependent diffusion
and kinetics. Temperature-scheduled C generation precomputes model matrices and
interpolates them using the Arrhenius factor. It takes measured temperature as
an input; it does not export the Python temperature estimator.

The degradation module represents interphase growth and lithium plating.
`plating_limited_current` searches for a current satisfying the model's electrode
potential margin. The C limiter uses a fixed iteration count, so its loop bounds
do not depend on convergence. This is a model constraint, not a complete charging
safety system or certification.

The built-in thermal and ageing parameters are not calibrated to every cell.
Use these features to investigate mechanisms and trends; quantitative life or
temperature predictions need the corresponding measured data.

Examples: [thermal coupling](../examples/05_thermal_coupling.py),
[degradation](../examples/07_degradation.py), [charging](../examples/08_fast_charge.py).

## C export and verification

Supported exports are the isothermal SPM and temperature-scheduled SPM. SPMe C
export is not implemented and raises an explicit error. The measured-data ML
correction is an offline benchmark and is not exported to C.

The generated code uses fixed-size arrays, no dynamic allocation and no mutable
global state. Its EKF uses one measurement correction per update. The Python
demo's three-iteration recovery results therefore describe a different filter
configuration.

Verification separates three sources of disagreement:

1. **Generated C versus its NumPy mirror:** arithmetic and code-generation fidelity.
2. **Mirror versus the table-backed model:** consistency of the reference implementations in open-loop mode.
3. **Table-backed versus analytic voltage:** deliberate lookup-table approximation.

The filter replay compares C with the mirror. It does not compare a corrected
filter with an uncorrected open-loop model as if they were equivalent.

`BUDGET.txt` counts data structures and estimates operations. Its table bytes
exclude executable code, its per-instance RAM excludes stack, and its timing
estimate is not a hardware measurement. Use `cellkernel measure` to cross-compile
and inspect the target image and emulated instruction count. Physical-board
timing still requires measurement.

```bash
cellkernel verify build/verified --precision float
cellkernel measure build/arm --precision float
```

See [the generator](../src/cellkernel/codegen), [replay checks](../src/cellkernel/verify),
and [current measurements](VALIDATION.md).

## Validating against measured cells

Agreement with [PyBaMM](https://github.com/pybamm-team/PyBaMM) checks an independent
implementation of the equations. Agreement with measured cell voltage checks
the combined model, parameters and data handling. These answer different questions.

Calibration should separate electrode balance, resistance and transport as far
as the measurements permit. Constant-current discharges alone can leave physical
parameters poorly identified even when fitted voltage error decreases. The
identification report includes sensitivities, correlations and active bounds to
make that limitation visible.

Inspect current/voltage alignment and pulse-edge resampling before increasing
model complexity. A more flexible model can fit a measurement artefact without
improving the underlying prediction.

See the [measured-cell example](../examples/10_against_a_real_cell.py),
[identification example](../examples/11_identify_from_data.py), and
[calibration procedure](SOP.md). The [ML benchmark](BENCHMARK.md) uses its own
fixed calibration and curve-level split; its numbers should not be substituted
for those of experiments using different parameter sets or records.

## Remaining limitations

- The lumped thermal model does not resolve internal temperature gradients.
- Temperature is weakly observable from voltage alone; measured temperature is preferable when available.
- The electrolyte model uses constant transport coefficients and loses accuracy under severe depletion.
- Some state uncertainty can be underestimated during long rests. A small reported covariance does not by itself establish accuracy.
- Lookup-table error varies across stoichiometry. Check the emitted error bounds when choosing table resolution and model margins.
- Capacity retention represents loss of active material rather than all possible ageing mechanisms.
- The ML benchmark uses one cell dataset at one ambient temperature. It does not demonstrate generalisation to unseen cells.
- Execution time and interference with other firmware tasks have not been measured on a physical board.
