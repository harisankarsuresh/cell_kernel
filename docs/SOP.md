# Bringing a new cell from the bench to embedded C

A working procedure: what to measure, what to extract from it, in what order to
calibrate, what error to accept at each stage, and how to get from there to a C
estimator you can defend in a design review.

Every number quoted as achievable in this document was measured during the
development of this package, on an LG M50 against [PyBaMM](https://github.com/pybamm-team/PyBaMM)
and against a real cycler dataset. Where a figure is a recommendation rather than
a measurement, it says so.

**The single most important idea here** is that "model error" is three different
quantities with three different remedies, and mixing them up wastes weeks:

| kind | what it is | typical size | how you fix it |
|---|---|---|---|
| implementation error | your code does not solve the equations you wrote | **µV** | debugging, and it is a bug |
| parameter error | the equations are right, the numbers are another cell's | **10–50 mV** | characterisation and fitting |
| structural error | the equations do not describe this phenomenon | **10–100 mV** | a different model |

A 40 mV disagreement is not a coding problem and no amount of debugging will
move it. A 40 µV disagreement is not a parameter problem and no amount of fitting
will move it. Stage 6 exists to keep them apart.

---

## Stage 0 — Decide what accuracy you actually need

Do this first, because it determines how much testing you must pay for. The
requirement flows from the *application*, not from a wish for accuracy.

| application | governing quantity | implies |
|---|---|---|
| State-of-charge display | dOCV/dSOC and voltage error | 1% SoC on a graphite/NMC cell needs roughly 10 mV of voltage accuracy in the flat region; on LFP the same 1% needs about 1.5 mV, which is at the noise floor of a good front end |
| Power limits | series resistance and its temperature dependence | resistance to ~10%, over the whole temperature range |
| Fast charge / plating protection | negative electrode surface potential | the *margin* must exceed the total model error; see Stage 6 |
| Life prediction | degradation parameters | not achievable from cell-level electrical data alone; expect trends, not years |

Write this number down before you start. If you skip this you will over-test for
a display and under-test for a charge controller.

---

## Stage 1 — Characterisation test matrix

### 1.1 What you cannot skip

| # | test | conditions | gives you |
|---|---|---|---|
| **T1** | Capacity | C/20 discharge, 25 °C, full window, ×3 | usable capacity, and its repeatability |
| **T2** | Pseudo-OCV | C/20 charge **and** discharge, average the pair | OCV vs SoC, with hysteresis cancelled |
| **T3** | Rate test | 0.2 / 0.5 / 1 / 2 / 3C discharge, 25 °C | validation data — **do not fit to this**, see 3.4 |
| **T4** | **Pulse / HPPC** | 10 % SoC steps, pulses of 1 s, 10 s and 60 s, both directions, 2 C | the only test that separates ohmic from kinetic from diffusive |
| **T5** | Temperature repeat | T2 and T4 at −10, 0, 25, 45 °C | activation energies |

**T4 is the one people skip and the one that matters most.** This package's
own identification run against four constant-current discharges came back with
a residual of 37 mV and a sensitivity analysis saying only the *series
resistance* had been determined — every physical parameter was a few percent as
influential, and the positive reaction rate was correlated with resistance at
ρ = +0.95. A constant current is a single steady excitation, and one excitation
cannot separate three loss mechanisms that all look like "the voltage is too
low". Pulses separate them by timescale: ohmic responds in microseconds, charge
transfer in milliseconds to seconds, diffusion in tens to hundreds of seconds.

If you take one thing from this document: **fitting to discharge curves produces
confident numbers that are not measurements.**

A pulse also hands you one parameter with no fitting at all. On the reference
cell a 1.5C pulse drops the terminal voltage 240 mV the instant it is applied and
a further 35 mV over the next ten seconds; the first divided by the current is
the series resistance, and it comes out with the shape it should — flat through
the middle of the charge window and rising at both ends, and strongly temperature
dependent:

| SoC | 0 °C | 10 °C | 25 °C | 45 °C |
|---|---|---|---|---|
| 10% | 54.9 mΩ | 55.7 mΩ | 36.9 mΩ | 29.3 mΩ |
| 50% | 42.0 mΩ | 36.2 mΩ | 31.5 mΩ | 27.0 mΩ |
| 90% | 47.3 mΩ | 38.1 mΩ | 32.7 mΩ | 26.5 mΩ |

`reference.load_pulse(...).series_resistance` does this. Note that the literature
parameter set for this cell has essentially no series resistance at all, so this
one measurement is worth more than everything the discharge fit produced.

### 1.5 Run more than one cell, and find out what your ceiling is

Six nominally identical cells from the reference dataset, same pulse, same
temperature:

| ambient | mean R₀ | spread | scatter |
|---|---|---|---|
| 0 °C | 43.0 mΩ | 41.9 – 45.8 mΩ | 3.0% |
| 25 °C | 29.3 mΩ | 27.0 – 31.9 mΩ | **6.2%** |
| 45 °C | 29.1 mΩ | 25.0 – 41.8 mΩ | 20.4% |

**This is the floor on any accuracy target you set.** A model tuned to one of
these cells begins 6% away from the next one off the same line. At 31 mΩ and 5 A
that is around 10 mV — so a model matching a single cell to better than about
10 mV is fitting that cell's individuality, not its chemistry, and will not
transfer.

Two consequences. Characterise at least three cells and fit to the median, or
accept that your parameters describe one unit. And do not set an acceptance
threshold below the scatter, however good your instrumentation is.

The 45 °C row is also a reminder to look at the spread and not just the mean:
one cell there reads 41.8 mΩ against a group average of 29. Fitting to that cell
would have produced a confident and thoroughly misleading parameter set.

### 1.2 Worth having if you can

| # | test | gives you |
|---|---|---|
| T6 | GITT | solid diffusivity *as a function of SoC*, which pulses give only coarsely |
| T7 | EIS at several SoC and temperature | cleanest separation of ohmic from charge-transfer |
| T8 | Half-cell coin cells from harvested electrodes | the electrode OCP curves themselves — removes the largest source of parameter error |
| T9 | Teardown | thicknesses, porosity, particle size, active fractions, electrode area |
| T10 | Entropic coefficient (potentiometric, dU/dT) | reversible heat; matters above ~1C |
| T11 | Calorimetry or instrumented thermal soak | heat capacity and heat transfer coefficient |

Without T8 and T9 you are using literature values for another cell and should
expect the parameter error in the table at the top of this document.

### 1.3 Distrust your own data handling before you distrust the model

The most expensive error in this whole exercise was not in the physics. Against
the reference pulses the model scored 31 mV, which looked like a structural
limitation and prompted a hunt for missing dynamics. It was not. **All but 3 mV
of it came from two artefacts in how the log was read**, and once they were
handled the same model — with no additional fitting — came out at 4.6 mV.

Both are generic to cycler data and worth checking in yours:

- **Non-uniform logging.** The instrument ran at 10 Hz through the pulse and then
  left a gap of about a second across the falling edge, which is exactly where
  the voltage jumps a quarter of a volt. Resampling onto a uniform grid puts
  points inside that gap, and linear interpolation across a step manufactures
  values the cell never had. Resample **current** with a zero-order hold, never
  interpolation, and mark any voltage sample that is interpolation rather than
  measurement.
- **Channel skew at transitions.** The current and voltage channels are recorded
  about one sample apart, so at every edge there is a record showing current
  already at zero with the voltage still at its loaded value. No cell does that.
  A model reading that current correctly predicts the recovered voltage and is
  scored 230 mV wrong for being right. **Discard one sample either side of every
  current transition.**

`PulseSegment.measured()` applies both. The general lesson is the ordering: when
a model disagrees with measurement, look at the measurement handling *before*
reaching for a richer model, because the richer model will absorb the artefact
and you will never find it.

### 1.4 Instrumentation minimums

- Voltage: ≤ 1 mV resolution, ≤ 2 mV absolute accuracy. Your model can never be
  demonstrated better than this.
- Current: 0.1% of range. Current error integrates directly into SoC error.
- Temperature: at least three surface points. A single thermocouple on a
  cylindrical cell misses tab-to-mid gradients that reach 10 K at 2C — measured
  on the reference dataset.
- Sampling: 10 Hz or faster **during pulses**. A 1 s pulse sampled at 1 Hz
  cannot resolve the ohmic step, which is the point of the test.
- Rest before every OCV point: ≥ 2 h at 25 °C, ≥ 4 h below 10 °C.

---

## Stage 2 — Parameter inventory

Every field the model needs, and where it comes from. `?` marks the ones you can
leave at a default without much regret.

### Per electrode (`ElectrodeParameters`)

| parameter | source | fittable? |
|---|---|---|
| `thickness` | T9 teardown, or datasheet | no — measure it |
| `particle_radius` | T9 (SEM or laser diffraction) | weakly; trades against diffusivity |
| `active_fraction` | T9 | no |
| `porosity` | T9 | no |
| `max_concentration` | crystallography for the chemistry (literature) | no |
| `ocp` | T8 half cells; else literature for the chemistry | no — this is the backbone |
| `stoich_at_0_soc`, `stoich_at_100_soc` | **fit** to T2 | **yes — always fit these** |
| `diffusivity` | T6 GITT, else fit to T4 | yes, from pulses |
| `reaction_rate` | T7 EIS, else fit to T4 | yes, from pulses |
| `diffusion_activation_energy` | fit T5 | yes |
| `reaction_activation_energy` | fit T5 | yes |
| `entropic_coefficient` | T10 | ? default 0 below 1C |

### Cell level (`CellParameters`)

| parameter | source | fittable? |
|---|---|---|
| `electrode_area` | T9 | no |
| `separator_thickness`, `separator_porosity` | T9 | no |
| `nominal_capacity` | **T1** | no — pin it, see 3.2 |
| `voltage_limits` | datasheet | no |
| `contact_resistance` | fit to the 1 s point of T4 | yes |
| `electrolyte_concentration` | electrolyte spec | ? 1000 mol m⁻³ |
| `electrolyte_diffusivity` | literature for the electrolyte | ? but get it right — see the warning below |
| `ionic_conductivity` | literature for the electrolyte | ? |
| `transference_number` | literature | ? 0.26 |
| `bruggeman` | ? 1.5 | ? |
| `thermal.*` | T11 | yes, to measured temperature rise |

> **A warning paid for in this project.** The default salt diffusivity here was
> once set nearly 3× too high because the depletion the correct value produced
> "looked implausible". It was not implausible; it was right, and the wrong value
> cost 7× accuracy against a full-order solution. Take transport properties from
> a source for *your electrolyte*, and do not overrule them with intuition.

### Which model to target

| use | model | states | when |
|---|---|---|---|
| default | `SPM` | 6 | duty cycle mostly below 1C |
| high rate | `SPMe` | 17 | real time above 1C; halves the error against a full solution |
| wide temperature | `ThermalSPM` | 7 | cell leaves ±10 °C of its calibration point |
| baseline | `ECM` | 3 | you need something to beat |

---

## Stage 3 — Calibration, in this order

The order is not arbitrary. Each stage removes a degree of freedom the next one
would otherwise absorb incorrectly.

### 3.1 Balance the electrodes first

```python
from cellkernel.params import balanced_stoichiometry_window
```

Loadings and stoichiometry limits taken independently from a datasheet almost
always leave a percent-level charge imbalance, which then contaminates whatever
transport parameter is fitted next. The built-in parameter sets solve this rather
than assert it.

### 3.2 Fit the stoichiometry window to T2 — with capacity pinned

```python
capacity = np.trapezoid(slow.current, slow.time) / 3600.0     # from T1
cell = fit_stoichiometry_window(cell, soc, measured_ocv, capacity=capacity)
```

Measured: this took the open-circuit error on a real LG M50 from **41.5 mV to
8.5 mV**.

> **Pin the capacity.** Fitting the OCV curve with capacity free is degenerate —
> the solver improves the curve by stretching the SoC axis. Left free it moved
> capacity 10%, improved the OCV fit *further*, and made every discharge roughly
> twice as bad, because they then ran on a mis-scaled clock. `capacity_weight`
> defaults to enforcing this and there is a regression test asserting the
> unconstrained version still misbehaves.

### 3.3 Reconcile the resistance if you are moving to `SPMe`

```python
cell = SPMe.reconcile(cell)
```

A contact resistance fitted with a model that had no electrolyte *already
contains* the electrolyte loss. `SPMe` computes that loss from geometry, so the
same ohms get counted twice and the better model performs worse than the simpler
one. This was caught in this project's own CI, inverting a result by 30 mV.

### 3.4 Anchor the series resistance before fitting anything else

```python
measured = np.median([p.series_resistance for p in pulses])      # from T4 edges
cell = anchor_series_resistance(cell, build, measured, current=7.5)
```

**Do this before 3.5, not as part of it.** The instantaneous voltage step is the
one feature of the response that *every* parameter can imitate. Left free, the
solver spends its whole budget reproducing it — and since a reaction rate and a
resistance both move it, the two become interchangeable and everything slower
gets whatever is left over.

Measured on the reference cell: fitting transport to pulses with the resistance
free left an 87 mV residual. Anchoring it first and fitting the same parameters
to what remained gave **31 mV**, and the solid diffusivities went from swamped to
being the best-determined quantities in the fit.

What gets assigned is the *shortfall* — the measured resistance less whatever
instantaneous drop the model already produces from its own kinetics. On the
reference cell that was 31.8 mΩ measured against 15.6 mΩ intrinsic, so 16.2 mΩ of
genuinely unmodelled resistance. Assigning the whole measured value would count
charge transfer twice.

### 3.5 Fit transport to the relaxation, using **T4** and not T3

```python
report = identify(cell, pulse_segments, lambda p: SPM(p, dt=0.1), knobs=TRANSPORT_KNOBS)
print(report.summary())
```

Then **read the sensitivity column before you read the residual.** Reject the fit
if any parameter you care about shows:

- sensitivity below ~5% of the largest — the data did not constrain it;
- correlation above 0.9 with another — the two were traded, only their combination
  is determined;
- a value resting on a bound — the data wanted something the model cannot supply,
  and the number is not a measurement.

Fit at a coarse step. Constant-current and slow-pulse segments carry no
information above ~0.1 Hz, and the zero-order-hold discretisation is exact for
piecewise-constant current, so a 10 s step is 20× faster for a few percent on the
fitted values. Use a fine step only for the sub-second part of pulses.

### 3.6 Activation energies from T5

```python
cell = cell.with_activation_energies(diffusion_negative=..., reaction_negative=...)
```

Deliberately never defaulted to a non-zero value: published values scatter widely
and a default would be a guess wearing the clothes of a measurement. Typical
range 20–50 kJ mol⁻¹.

### 3.7 Thermal parameters from T11

Fit `heat_transfer_coefficient` and `heat_capacity` to a measured temperature
rise. For scale, the reference cell rises **33 K on a 2C discharge from 25 °C and
42 K from 0 °C** — colder is worse, because sluggish transport dissipates more.

### 3.8 Validate on T3, which you did not fit to

Held-out validation. If the rate test error is much worse than the pulse-fit
residual, the fit has absorbed something that does not generalise.

---

## Stage 4 — Acceptance criteria

Recommended limits, with what this project actually achieved beside them.

### 4.1 Model against measurement

| check | recommended | measured here |
|---|---|---|
| OCV, 5–95% SoC, RMSE | **< 10 mV** | 8.5 mV after 3.2 |
| OCV at the extremes | < 50 mV | 372 mV before fitting; the ends are always worst |
| Series resistance from pulses | **within cell-to-cell scatter** | 6.2% scatter at 25 °C |
| **Pulse response, RMSE** | **< 10 mV** | **4.6 mV**, no transport fitting |
| Rate test, held out, RMSE | **< 25 mV** to 2C | 37 mV fitting CC only |
| Capacity | **< 1%** | 0.03% with capacity pinned |
| Temperature rise | < 20% of the rise | — (needs T11) |

> **Do not set these tighter than section 1.5.** Six identical cells scatter 6%
> in series resistance, which is about 10 mV at 5 A. A voltage target below that
> is unreachable in any transferable sense no matter how good the model is, and
> pursuing it produces a parameter set fitted to one cell's individuality.

If you cannot reach the OCV target, the problem is the OCP curves — get T8 done.
If OCV is fine and the pulse residual is not, the problem is the model structure:
try `SPMe`, then `ThermalSPM`.

### 4.2 Model against a reference implementation

Do this once. It separates implementation error from everything else and is the
only way to know your remaining error is physics.

| check | recommended | measured here |
|---|---|---|
| your SPM vs PyBaMM SPM, 0.5C | **< 1 mV** | 0.26 mV |
| same, 2C | < 5 mV | 2.63 mV |
| your SPMe vs PyBaMM DFN, 1C | < 20 mV | 6.7 mV |

Start both from **identical stoichiometries**, not from a nominal SoC — each
package has its own SoC bookkeeping and comparing that tells you nothing about
the physics. This project's first attempt at the comparison got this wrong and
produced 100 mV of purely definitional disagreement.

### 4.3 Estimator

| check | recommended |
|---|---|
| SoC error after convergence, known-good data | < 2% |
| Convergence from a 15% seeded error | < 10 min at ≥ 0.5C |
| Covariance | symmetric, positive definite at every step |
| Reported σ vs actual error | within 3×; expect optimism during long rests |

### 4.4 Code generation — a different order of magnitude

These are arithmetic, not physics. Anything larger is a bug.

| check | recommended | measured here |
|---|---|---|
| Generated C vs NumPy mirror, double | **< 1e-12 V** | 8.9e-16 V |
| Generated C vs NumPy mirror, single | **< 100 µV** | 7 µV |
| OCP table vs analytic fit | **< 1/3 of your voltage budget** | 4.95 mV at 257 points (graphite); 1.3 mV at 513 |
| Compiles `-Wall -Wextra -Wpedantic -Werror` | no warnings | clean, gcc and clang |
| Undefined-behaviour sanitiser | clean | clean |

> **Size the table against what you will use the model for.** Graphite tabulates
> far worse than a layered oxide — its stage transitions cost 4.95 mV at the
> default 257 points against 0.13 mV for the oxide opposite it. If you are using
> `ck_plating_potential` with a 10 mV margin, a 5 mV table error is half your
> margin: raise `--table-points` to 513. The generated header emits
> `CK_OCP_ERROR_NEG` so this trade is explicit.
>
> **Doing Stage 3.2 first makes the table much cheaper**, which is not obvious.
> The table only has to span the stoichiometry the electrode actually reaches,
> and almost all of graphite's interpolation error lives in the steep exponential
> rise below about 4% lithiation. Fitting the window moved the negative
> electrode's lower limit from 0.026 to 0.051, out of that region — and the
> end-to-end table error in the worked example came out at **0.02 mV**, some sixty
> times better than the uncalibrated figure, at the same point count. Calibrate
> before you pay for a bigger table.

---

## Stage 5 — Estimator design

1. Choose the filter. `EKF` with `iterations=3` is the default recommendation —
   it beat the unscented filter in this project's own comparison at a fraction of
   the cost, because the process is exactly linear and only the measurement is
   not. A single iteration is *not* enough: seeded 15% wrong it overshoots and
   ends worse than coulomb counting.
2. Shape the priors. Use `suggest_initial_covariance` and
   `suggest_process_noise` rather than diagonal matrices. An isotropic prior
   sends corrections into gradient states instead of bulk concentration and the
   filter wanders during rests.
3. If the model carries temperature, pass `temperature_std` — do not let the
   concentration-scaled floor apply to a state measured in kelvin. That bug put a
   cell at 430 K on the first update here.
4. Tune `measurement_noise` to your front end, typically (1–3 mV)².

---

## Stage 6 — Generate, verify, measure

```bash
cellkernel generate  build/est --precision float --order 3 --table-points 513
cellkernel verify    build/est --precision float
cellkernel measure   build/est --precision float --order 3
```

**Verification is three legs on purpose.** Generated C against a NumPy mirror
with identical loop order; the mirror against a table-backed Python model; that
against the full analytic model. The first leg measures the generator, the last
is the price of the lookup table. Aggregating them hides which you are looking
at — a 3 mV gap is unremarkable if it is table resolution and alarming if it is
arithmetic. This localised a 138 mV defect to a table domain in minutes.

**Measure the footprint, do not model it.** For scale, a 6-state float estimator
on a Cortex-M4F at `-Os`: **4636 B flash (2124 code + 2512 tables), 168 B RAM per
cell, zero static RAM, 5086 instructions per step.** Note that the code is about
as large as the tables — a flash figure counting tables only, which is what the
built-in budget reports, is short by 45%. And instructions are not cycles: QEMU
models no pipeline or flash wait states, so real silicon takes at least that many
and generally more. Budget 2–3× the instruction count for a task period.

### Integration checklist

- [ ] One `ck_estimator_t` per cell. There are no globals — verified by `.bss`
      and `.data` measuring zero — so instances are independent and reentrant.
- [ ] Current sign: **positive on discharge**. Getting this backwards produces an
      estimate that moves confidently in the wrong direction.
- [ ] Feed the *measured* temperature to a scheduled estimator. It takes
      temperature as an input, not a state, deliberately.
- [ ] Watch the innovation returned by `ck_correct`. A persistent bias means the
      model is wrong, not the filter.
- [ ] If using `ck_max_charge_current`, confirm your margin exceeds
      `CK_OCP_ERROR_NEG`.
- [ ] Run the generated harness against a recorded drive cycle on the target
      before trusting it in the loop.

---

## Stage 7 — Sign-off

| | evidence |
|---|---|
| Accuracy requirement stated before testing began | Stage 0 |
| T1, T2, T3, T4, T5 complete | test reports |
| Parameters traceable to a measurement or a cited source | parameter sheet |
| Capacity pinned during the OCV fit | fit log |
| Series resistance anchored from pulse edges before fitting | Stage 3.4 |
| Sensitivity analysis reviewed; unidentified parameters listed | `report.summary()` |
| Validation on held-out data | Stage 3.8 |
| Compared against an independent implementation | Stage 4.2 |
| Three-leg verification passed | `cellkernel verify` |
| Footprint and timing measured on the target architecture | `cellkernel measure` |
| Table error smaller than any margin that depends on it | header `CK_OCP_ERROR_*` |

---

## A worked run

`examples/12_new_cell_walkthrough.py` executes every stage above against the
measured LG M50 and prints the acceptance check after each. Its output is the
shape a real report should take — including the failures, since the dataset has
no pulse test:

```
STAGE 3  Calibration  (starting from PyBaMM Chen2020)
  3.0  literature OCV error                        41.53 mV
  3.2  fitted the window with capacity pinned
  PASS  OCV, 5-95% SoC                                   8.52 mV  (limit 10)
  PASS  capacity                                         0.03 %   (limit 1)
  3.4  fitting kinetics and transport
       residual 90.2 -> 37.2 mV
       best determined  : contact_resistance
       not determined   : reaction_rate_negative, reaction_rate_positive, ...
  FAIL  REJECT this fit. Only one parameter was identified.

STAGE 4  Acceptance against held-out data
  FAIL  worst rate-test RMSE                            65.23 mV  (limit 25)

STAGE 6  Generate, verify, measure
  PASS  generated C vs mirror                            0.01 mV  (limit 0.1)
  PASS  OCP table error (budget 3 mV from stage 0)       0.02 mV  (limit 3)
  PASS  end to end                                       0.02 mV  (limit 3)
       Cortex-M4F -Os flash                             6692 B
       static RAM (must be zero)                           0 B

VERDICT
  Implementation  : verified to machine precision.
  Parameters      : window calibrated; kinetics NOT identified.
  Blocking gap    : no pulse test. Stage 1.1 T4.
```

Note the shape of that result, because it is the normal one: **every
implementation check passes and the parameter checks fail.** The arithmetic is
exact to a hundredth of a millivolt while the model is 65 mV from the cell. Those
are the two columns of the table at the top of this document, and a procedure
that reported a single "accuracy" number would have hidden which one was the
problem.

## What this procedure will not give you

- **Life prediction in years.** Interphase rate constants in the literature span
  orders of magnitude. Expect the shape of the fade curve, not its timescale.
- **Identified kinetics from discharge curves.** Stage 1.1, again, because it is
  the most expensive mistake available here.
- **Accuracy better than your voltage sensor.** If the front end is ±2 mV, no
  amount of modelling demonstrates better than that.
- **Accuracy better than the cells differ from each other.** Six identical cells
  scatter 6% in series resistance here. Below roughly 10 mV you are modelling one
  unit, not a design.
- **A model valid outside the conditions you tested.** Particularly cold: a
  thermal design validated at 25 °C is not validated, since the same discharge
  heats the cell 27% more from freezing.
