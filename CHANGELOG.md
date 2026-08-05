# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends
to follow [semantic versioning](https://semver.org/spec/v2.0.0.html) from 1.0
onwards. Before then, minor versions may break the API.

## Unreleased

### Fixed

- **The filter prior no longer assumes every state carries the same units.** The
  isotropic floor in `suggest_initial_covariance` is scaled from the largest
  entry of the state-of-charge direction — a concentration of order 1e5 mol m⁻³
  — which is a sensible few hundred mol m⁻³ for a diffusion state and several
  hundred *kelvin* for a temperature one. The filter put the cell at 430 K on its
  first update, at −14 K on its fifth, and never recovered. Temperature now gets
  a prior of its own.
- **Process noise no longer ties temperature to the current sensor.** The input
  column of a thermal model has a nonzero temperature entry because current
  generates heat, so the rank-one current term placed temperature in rigid
  correlation with the diffusion states. Temperature uncertainty comes from the
  ambient and from a heat-transfer coefficient nobody has measured accurately.
- **Arrhenius factors are clamped to a physical temperature range.** An unscented
  filter carrying temperature as a state places sigma points outside it during a
  transient, and `1/T` then overflows the exponential and puts infinities in the
  covariance.
- `ThermalSPM` gained the `soc_jacobian` the other models have. Without it the
  estimators fell back to a zero gradient and reported a state-of-charge
  uncertainty of exactly zero, which is worse than reporting none.
- `input_direction` accepts an operating point, and documents why it is a
  one-sided difference: dissipation is even in current, so a central difference
  about zero cancels resistive heating exactly.

### Changed

- `suggest_process_noise` and `suggest_initial_covariance` moved from `EKF` to the
  base `Estimator`. They depend only on the model, and reaching them from one
  filter only was an accident of which was written first.

### Added

- **`cellkernel.protocols`** — charging protocols that invert the plating
  criterion rather than guessing a safe rate.
- **The plating limiter in the generated C.** `ck_plating_potential` and
  `ck_max_charge_current` put the charge-rate decision on the microcontroller
  that sets the current. The potential is free — `ck_voltage` already formed it
  and threw it away — and the limiter is bisection over a fixed 24 steps, so
  execution time is constant. The generated bisection is bit-identical to its
  Python mirror. The scheduled estimator exposes the same pair taking measured
  temperature, which is the version that matters — plating is a cold-weather
  failure and the isothermal generator answers only for one point.
- **`CK_OCP_ERROR_NEG` and `CK_OCP_ERROR_POS`** in the generated header. The
  plating potential inherits the negative electrode's table error, and graphite
  tabulates badly: 4.95 mV at the default 257 points against 0.13 mV for the
  oxide. A margin below that figure is measuring the table, not the cell, and
  the figure was previously visible only in a build log.
- `cellkernel charge` and `cellkernel age` subcommands.
- **Validation against PyBaMM**, on PyBaMM's own Chen2020 parameter set and
  started from identical stoichiometries so the comparison measures the physics
  rather than each package's state-of-charge bookkeeping. Two independent
  implementations of the single particle model agree to 0.2 mV RMSE at 0.5C, and
  resolving the electrolyte roughly halves the distance to a full
  Doyle–Fuller–Newman solution at every rate below 3C. An optional dependency,
  running as its own CI job.
- Cross-model integration tests: every estimator against every model, plus the
  interface contract each model must satisfy.
- Test coverage for the CLI (was 0%) and cycler-file loading (was 18%).

## 0.2.0

### Added

- **`SPMe`** — single particle model with resolved salt transport across the
  sandwich. Adds the electrolyte ohmic drop, computed from geometry and
  conductivity rather than fitted, and the concentration overpotential that
  builds as salt is driven from one coating to the other. Reports its own
  validity through `depletion()` and `validity()`, because a linear transport
  model will happily predict negative concentration if pushed.
- **`ThermalSPM`** — cell temperature as a state, with Bernardi heat generation,
  an exactly integrated lumped thermal node, and reduced-order matrices gain
  scheduled over temperature.
- **`cellkernel.degradation`** — interphase growth and lithium plating.
  Predicts capacity fade and reports plating margin in volts. Total ageing comes
  out U-shaped in temperature, with the cold arm owned by plating and the hot arm
  by interphase growth.
- **`generate_scheduled()`** — emits a temperature-scheduled C estimator, valid
  across a range rather than at one operating point. Temperature is an input, not
  a state; see the module docstring for why.
- **`ScheduledStateSpace`** — gain scheduling of a discretised diffusion model
  over temperature, blending on the Arrhenius factor rather than on temperature.
- **`CellParameters.with_activation_energies()`** — supplies Arrhenius energies,
  which the built-in parameter sets deliberately leave at zero.
- Electrolyte transport parameters on `CellParameters`, and `porosity` on
  `ElectrodeParameters`.
- Examples 05 through 07, covering thermal coupling, the electrolyte, and
  degradation.

### Fixed

- **Conservation identities are now imposed after discretisation** rather than
  assumed to survive it. At `dt = 100 R²/D` the generator spans fifteen orders of
  magnitude and SciPy 1.15 returned `A[0,0]` as 0.99988 on Linux and 1.000065 on
  macOS instead of exactly 1, with a rested particle growing 3% over 500 steps.
  The claim that lithium conservation is structural should not depend on which
  LAPACK wheel pip installed.
- **Plating uses Butler-Volmer, not Tafel.** A bare Tafel term has no reverse
  branch and predicts deposition at every potential including rest, which
  integrated over a month of storage plated out an entire cell.
- **Plated and dead lithium are separate inventories.** With one running total,
  stripping made the same lithium available to strip again on the next sample.
- **Film resistance is no longer divided by electrode area twice**, which had
  made the film's effect on the side reactions vanish.
- **The solvent-diffusion branch of interphase growth is temperature dependent.**
  Growth becomes diffusion limited within a week, so without it the model
  predicted identical ageing at 10 °C and 45 °C.
- **Potential tables for scheduled estimators are sized at the coldest grid
  point**, where the surface excursion is largest. Sized at the reference
  temperature they saturate in the cold, silently.
- README maths now renders on GitHub. `\operatorname` is not in the allowed macro
  set, single-line `$$…$$` was not being recognised at all, and markdown was
  eating backslash-punctuation inside inline maths, so `100\,R^2/D` reached the
  renderer as `100,R^2/D`.

### Changed

- Temperature schedules blend on the Arrhenius factor rather than linearly in
  temperature. The matrices are nearly affine in diffusivity and diffusivity is
  exponential in `1/T`, so linear-in-temperature interpolation fits a straight
  line through an exponential and is worst at the cold end: 197 mV of error at
  −18 °C on a nine-point grid, against 1.8 mV for factor blending.
- `ThermalSPM` caches its blended systems one deep, taking roughly half the
  runtime out of the model — six blends per sample became one.
- Built-in parameter sets carry Chen2020-like porosities.
- `actions/checkout` and `actions/setup-python` moved to the Node 24 majors.

## 0.1.0

Initial release. Reduced-order solid diffusion in four families, cell parameters
with solved electrode balancing, single particle and equivalent-circuit models,
extended, unscented and dual Kalman filters, C99 code generation, and a
verification harness that compiles the generated code and compares it against a
Python mirror.
