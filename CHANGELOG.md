# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends
to follow [semantic versioning](https://semver.org/spec/v2.0.0.html) from 1.0
onwards. Before then, minor versions may break the API.

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
