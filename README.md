# cellkernel

**Physics-based lithium-ion state estimation, from reduced-order electrochemistry to verified embedded C.**

`cellkernel` takes a physics-based cell model, wraps it in a Kalman filter, and emits a self-contained C99 estimator that runs on a battery-management microcontroller — then compiles that C, replays it against the Python original, and tells you exactly how far apart they are.

Cross-compiled for a Cortex-M4F at `-Os` and run on an emulated core, a 6-state single particle model with an extended Kalman filter costs **4.6 kB of flash, 168 bytes of RAM per cell, no static RAM at all, and 5,086 instructions per step**. Those are measured, not modelled — see below, because the modelled figures were wrong. The generated C agrees with the Python reference to **9e-16 V in double precision** and **7 µV in single precision**.

The models are checked against [PyBaMM](https://github.com/pybamm-team/PyBaMM) rather than only against themselves — two independent implementations of the single particle model agree to **0.2 mV** at 0.5C. And the generated estimator answers the question a charger actually needs: `ck_max_charge_current` returns the fastest rate that will not plate lithium, in bounded time, on the microcontroller.

```
verification PASS  (double, 6 states, 900 samples, openloop, gcc)

  code generation fidelity
    generated C vs NumPy mirror, voltage   8.882e-16 V
    generated C vs NumPy mirror, SoC       0.000e+00
    mirror vs table-backed model, voltage  3.997e-15 V

  deliberate approximation
    lookup table vs analytic fit, voltage  6.120e-05 V  (0.061 mV)

  end to end
    generated C vs full Python model       6.120e-05 V  (0.061 mV)
```

## Why this exists

The open-source battery modelling ecosystem is strong and getting stronger. [PyBaMM](https://github.com/pybamm-team/PyBaMM) simulates continuum models beautifully. [PyBOP](https://github.com/pybop-team/PyBOP) identifies their parameters. [cellpy](https://github.com/jepegit/cellpy) and [BEEP](https://github.com/TRI-AMDD/beep) read cycler files. All of them stop at the Python boundary.

But a battery-management unit does not run Python. Getting a physics-based estimator onto the microcontroller that actually controls a pack is, today, a hand translation: someone reads the equations, writes C, and hopes. That translation is where the errors go in, it is unverifiable after the fact, and it is the single biggest reason production systems still ship equivalent-circuit models with coulomb counting while the electrochemistry stays in a research notebook.

`cellkernel` closes that gap and, just as importantly, *measures* it.

## Install

```bash
pip install cellkernel            # runtime: numpy, scipy
pip install cellkernel[dev]       # plus pytest, ruff, matplotlib
```

Python 3.10 or newer. A C compiler (gcc or clang) is needed only to *verify* generated code, not to generate it.

## Five minutes

Compare the reduced-order diffusion models against the exact solution of the sphere problem:

```bash
cellkernel roms
```

```
  model          states       <=1e0       <=1e1       <=1e2
  ---------------------------------------------------------
  pade                3    2.12e-09    1.30e-04    1.06e-01
  pade                5    3.82e-14    1.98e-10    5.72e-04
  spectral            5    1.80e-05    1.39e-03    4.87e-02
  fv                  5    4.41e-03    4.00e-02    4.33e-01
  poly                2    2.10e-05    1.37e-02    2.56e-01
```

Generate an estimator, compile it, and check it:

```bash
cellkernel verify build/estimator --precision float
```

Ask how fast the cell can be charged at −10 °C without plating lithium, and what
three hundred cycles will cost it at each temperature:

```bash
cellkernel charge --temperature 263.15 298.15
cellkernel age --cycles 300
```

Or from Python:

```python
from cellkernel.codegen import generate
from cellkernel.data import synthetic_drive_cycle
from cellkernel.estimators import EKF
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite
from cellkernel.verify import verify

cell = chen2020_nmc811_graphite()
model = SPM(cell, dt=1.0, rom="pade", order=3)

# Filter a record.
current = synthetic_drive_cycle(cell.nominal_capacity, duration=1800.0)
truth = model.simulate(current, soc0=0.8)

ekf = EKF(
    model,
    process_noise=EKF.suggest_process_noise(model, current_std=0.05),
    measurement_noise=1e-6,
    initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.1),
    iterations=3,
)
ekf.initialise(0.65)                       # deliberately wrong by 15%
result = ekf.run(current, truth["voltage"])

# Ship it.
project = generate(model, "build/estimator", precision="float")
print(verify(project, model, current).summary())
```

## How it works

Five layers, each usable alone.

### 1. Solid diffusion, reduced

Lithium diffusion in a spherical particle has the exact surface transfer function

```math
G(s) = \frac{R}{D} \cdot \frac{\sinh\xi}{\xi\cosh\xi - \sinh\xi},
\qquad \xi = R\sqrt{s/D}
```

which factors into an integrator carrying the mass balance and an analytic remainder:

```math
G(s) = \frac{3}{Rs} \cdot \hat{H}\left(\frac{R^2 s}{D}\right),
\qquad
\hat{H}(a) = 1 + \frac{a}{15} - \frac{a^2}{525} + \frac{2a^3}{23625} - \cdots
```

Four families approximate it. All are delivered as discrete-time state-space systems, all keep the volume-averaged concentration as an exact state, and all are compared against closed-form results in the test suite:

| Model | States | Character |
|---|---|---|
| `PadeDiffusion` | *k* | Best accuracy per state. Coefficients solved in exact rational arithmetic. |
| `SpectralDiffusion` | *k*+1 | Diagonal state matrix, so the cheapest to evaluate. |
| `FiniteVolumeDiffusion` | *k* | The only one that keeps a resolved interior profile. |
| `PolynomialDiffusion` | 2 | Cheapest that still gets the steady-state surface offset exactly. |

![Surface concentration response against the exact PDE](rom_comparison.png)

The lower panel is the one that matters. A 6-state Padé model tracks the exact transfer function at machine precision out to $\omega R^2/D \approx 100$, roughly a 10 second timescale for this particle, while a 10-shell finite-volume discretisation is already at 0.1% error in the quasi-static limit — its error is set by spatial resolution, not by bandwidth, so it does not improve as the excitation slows. Reproduce with `python examples/01_compare_reduced_order_models.py`.

Two details do most of the work:

**Discretisation happens offline, by matrix exponential.** A hand-written embedded diffusion solver usually steps explicitly, which is stable only for $\Delta t \lt \Delta r^2 / 2D$ — for a 6 µm particle that is milliseconds, far below a battery-management task period. Exponentiating the generator once, at build time, makes the online update a single dense matrix-vector product that is *unconditionally stable and exact for piecewise-constant current*. All the hard numerics move to the host.

**Padé coefficients are solved in exact rationals.** The Padé linear system is a Hankel matrix built from series coefficients spanning many orders of magnitude ($1$, $\tfrac1{15}$, $-\tfrac1{525}$, $\tfrac2{23625}$, $-\tfrac{37}{9095625}$, …). Solved in double precision it loses significant digits by order 5 and is unusable by order 8. Solved over `fractions.Fraction` it is exact at any order, and it happens once.

### 2. Cell parameters that are actually consistent

A parameter set is only physically meaningful if the electrodes are **charge balanced**: sweeping state of charge from 0 to 1 must move the same lithium out of one electrode as into the other. Quoting electrode loadings and stoichiometry limits independently — as data sheets and papers do — almost always leaves a percent-level imbalance, which then contaminates any transport parameter fitted against the same data.

`balanced_stoichiometry_window` solves four unknowns against four constraints (both electrodes pass exactly the rated capacity; the open-circuit voltage hits both limits) as a bounded least-squares problem. The built-in sets come out balanced to machine precision:

```
nmc811-graphite-5Ah
   usable neg 5.00000 Ah   pos 5.00000 Ah   balance err 0.00e+00
   OCV(0)=2.50000  OCV(1)=4.20000
```

### 3. Cell models with exactly linear dynamics

`SPM` is two particles with reduced-order diffusion and symmetric Butler-Volmer kinetics:

```math
V = U_p(x_p) + \eta_p - U_n(x_n) - \eta_n - I R_c
```

```math
\eta_k = \frac{2RT}{F} \cdot \mathrm{asinh}\left(\frac{j_k}{2 i_{0,k}}\right)
```

`ECM` is the equivalent-circuit baseline it exists to displace, with each RC branch discretised exactly rather than by forward Euler.

The structural point: **the state dynamics are exactly linear.** Diffusion is linear in flux and flux is linear in current, so all nonlinearity lives in the voltage measurement. An extended Kalman filter on this model has *no linearisation error in its prediction step at all* — the usual complaint about EKFs does not apply, and the covariance propagation stays well behaved indefinitely.

### 4. Checked against somebody else's code

Closed-form tests catch a great deal, but they cannot catch a misunderstanding shared between a model and the test written by the same person. So the models are also compared against [PyBaMM](https://github.com/pybamm-team/PyBaMM), on PyBaMM's own Chen2020 parameter set, started from *identical* stoichiometries so the comparison measures the physics rather than each package's state-of-charge bookkeeping.

| discharge | our SPM vs PyBaMM SPM | our SPM vs DFN | our SPMe vs DFN |
|---|---|---|---|
| 0.5C | **0.26 mV** | 25.6 mV | **3.2 mV** |
| 1.0C | **0.83 mV** | 52.9 mV | **6.7 mV** |
| 2.0C | **2.6 mV** | 132.2 mV | **14.7 mV** |
| 3.0C | 5.5 mV | 330.0 mV | 141.3 mV |

Two independent implementations of the same model agree to a quarter of a millivolt at 0.5C, which is about the strongest form this check can take. And a 17-state `SPMe` reproduces a full Doyle–Fuller–Newman solution — a discretised system of coupled PDEs — to **6.7 mV at 1C and 14.7 mV at 2C**, roughly an order of magnitude better than the single particle model it extends. At 3C the linear electrolyte gives up, as documented, and `validity()` says so before the voltage does.

**This comparison paid for itself immediately.** Getting these numbers meant finding three defects that self-consistent testing could never have caught, all of them in the PyBaMM bridge and all of them silent:

- The reaction rate was **hardcoded at 1e-6** rather than read, wrong by 1.5× on graphite and 5.3× on the oxide. It got there because PyBaMM's parameter functions return expression nodes rather than numbers, so `float()` raised and a bare fallback swallowed it. That one defect *was* the 23 mV I had previously written off as "a model difference, cause unidentified".
- Electrolyte transport was not imported at all. The salt diffusivity in use was **2.8× PyBaMM's** — a default I had raised myself, earlier in this project, because the depletion it produced looked implausibly strong. It was not implausible; it was right, and the independent reference is what showed the intuition was wrong.
- The float coercion issue above affected every callable parameter, not just kinetics, so the bridge would silently substitute defaults for anything PyBaMM expressed as a function.

What remains is small and honestly labelled: refinement now *does* reduce the gap, which it did not before, but it plateaus around 2.5 mV at 2C rather than going to zero. A residual model difference of a couple of millivolt is still there, an order of magnitude below where it started and below what a measurement front end would resolve.

`from_pybamm` re-solves the stoichiometry window rather than taking the published limits verbatim — those leave a percent-level charge imbalance that would otherwise be absorbed by whatever transport parameter is fitted next.

PyBaMM is an optional dependency; the comparison runs as its own CI job. Reproduce with `pip install pybamm && python examples/09_validate_against_pybamm.py`.

### 5. When the electrolyte stops being a resistor

`SPMe` resolves salt transport across the sandwich — negative coating, separator, positive coating — instead of lumping it into a fitted series resistance. It adds the electrolyte ohmic drop, computed from geometry and conductivity rather than fitted, and the concentration overpotential that builds as salt is driven from one coating to the other.

Below about 1C this buys nothing. A single particle model with a fitted resistance matches it to a few millivolts, has 6 states instead of 17, and is exactly linear. Use the simpler one.

The reason it exists is that the electrolyte term is **not** a resistance. On a 2C step from rest:

| time | concentration overpotential | as an equivalent resistance |
|---|---|---|
| 2 s | −4.5 mV | 0.45 mΩ |
| 10 s | −18.7 mV | 1.87 mΩ |
| 40 s | −51.3 mV | 5.13 mΩ |
| 160 s | −80.1 mV | 8.01 mΩ |

A fitted resistance has to pick one row of that table, and here the top and bottom differ by a factor of eighteen. Calibrate on a ten-second pulse and it badly under-predicts sustained discharge; calibrate on the settled value and it over-predicts every transient. Fitting harder does not help, because the thing being fitted has dynamics of its own — a timescale set by the sandwich thickness, not the particle.

Salt is conserved exactly (the source integrates to zero by construction and the projection enforces it), the steady-state split is verified two independent ways — iterating the discretisation, and solving the singular continuous system with the mean pinned — and they agree to 1e-8. The state transition stays exactly linear, so this model keeps the property that makes an extended Kalman filter well behaved on it.

Transport coefficients are held at their bulk values, which keeps the system linear and discretisable offline but means the model degrades as the electrolyte empties. It reports that rather than hiding it: `depletion()` returns the lowest coating concentration as a fraction of nominal, and `validity()` turns it into `good`, `degraded` or `extrapolating`. Those thresholds are **calibrated against the DFN comparison above**, not guessed — the model holds up considerably further into depletion than intuition suggests (still 14.7 mV at 2C, where the salt has fallen to a quarter of nominal) and then fails abruptly once linear extrapolation drives a coating concentration through zero.

Reproduce with `python examples/06_electrolyte.py`.

### 6. Temperature, when it cannot be ignored

`ThermalSPM` adds cell temperature as a state: Bernardi heat generation, a lumped thermal node integrated exactly rather than by forward Euler, and Arrhenius feedback into both the kinetics and solid diffusion.

Whether that is worth its cost is a real question, so here is the measurement. Same 2C discharge, once self-heating and once held isothermal:

| ambient | temperature rise | peak voltage difference |
|---|---|---|
| −15 °C | 27.1 K | **736 mV** |
| 0 °C | 23.4 K | 288 mV |
| 25 °C | 18.0 K | 101 mV |
| 40 °C | 15.2 K | 60 mV |

Diffusivity moves roughly fifty-fold from −20 °C to 60 °C. An isothermal model freezes that, and the error it makes is largest exactly where a physics-based model is most wanted — in the cold, where surface depletion sets the plating limit.

The awkward part is that diffusivity enters through the matrix exponential, which a microcontroller cannot evaluate. So the matrices are precomputed across a temperature grid and blended online. **The blend is on the Arrhenius factor, not on temperature**, and that detail is worth more than it sounds: at a sample period short against the diffusion time constant the discrete matrix is close to $A \approx I + A_c(D)\Delta t$ with $A_c$ proportional to $D$, so the matrices are nearly affine in diffusivity, and diffusivity is exponential in $1/T$. Interpolating linearly in temperature fits a straight line through an exponential and is worst at the cold end. On a nine-point grid that costs **197 mV at −18 °C**; blending on the factor instead costs **1.8 mV**, for one extra exponential per electrode per step.

| grid points | −18 °C | −3 °C | 22 °C | 47 °C |
|---|---|---|---|---|
| 5 | 5.55 mV | 0.13 mV | 0.21 mV | 0.28 mV |
| 9 | 1.79 mV | 0.07 mV | 0.08 mV | 0.06 mV |
| 17 | 0.58 mV | 0.02 mV | 0.03 mV | 0.02 mV |

Activation energies are deliberately **not** shipped with the parameter sets — they are seldom reported alongside the transport properties they modify, and published values scatter widely enough that a default would be a guess wearing the clothes of a measurement. Supply them with `CellParameters.with_activation_energies`.

Reproduce with `python examples/05_thermal_coupling.py`.

`generate_scheduled()` emits this as embedded C, with one deliberate change: **temperature becomes an input rather than a state.** Any pack worth running a physics-based estimator on has thermistors, so temperature is a measurement, not an unknown. Treating it as one keeps the covariance the size it was, keeps heat generation and its poorly-identified thermal parameters out of the firmware entirely, and leaves the Kalman structure identical to the isothermal case — what is generated is the same estimator with temperature-dependent coefficients, not a differently shaped one. It is also the better estimate: temperature is only weakly observable from terminal voltage, so a filter that infers it loses to a thermistor costing a few cents.

The schedule costs less than it sounds. Against the isothermal generator, on a 6-state model with a 9-point grid in single precision: **1.4× the flash and 2.3× the RAM** — 3.6 kB and 384 B — because the potential tables dominate flash and are shared across the grid. Blended coefficients are cached and rebuilt only when the measured temperature changes, which for a thermistor read far more slowly than the control loop means the cache usually hits. Generated C agrees with its NumPy mirror to 8.9e-16 V in double precision, including while temperature ramps across grid boundaries mid-run.

### 7. Predicting ageing, not just tracking it

`cellkernel.degradation` models the two mechanisms a graphite cell spends most of its life limited by. Interphase growth consumes cyclable lithium continuously and builds the film that throttles its own growth, so loss bends from linear to a square root within the first week. Lithium plating deposits metal instead of intercalating whenever the negative electrode potential falls below that of lithium metal.

They are treated differently on purpose. Growth is integrated as a slow state. Plating is reported as a **margin in volts**, because what a controller needs from it is not a life prediction but an answer to "may I keep charging at this rate", now:

| state of charge | 0.5C | 1.0C | 2.0C | 3.0C |
|---|---|---|---|---|
| 0.30 | +0.108 | +0.068 | +0.024 | +0.001 |
| 0.60 | +0.082 | +0.047 | **−0.012** | **−0.043** |
| 0.80 | +0.039 | +0.006 | **−0.032** | **−0.057** |
| 0.95 | +0.028 | **−0.008** | **−0.054** | **−0.099** |

Negative means depositing metal. This table is why fast charge tapers, and it is not the cell voltage or the bulk state of charge that sets it — it is the potential at the particle surface, which an equivalent circuit does not have and therefore cannot protect against.

The result worth having is that **ageing is U-shaped in temperature**. Interphase growth is Arrhenius and worsens with heat; plating is driven by sluggish transport and worsens with cold. Over 300 cycles at 1C:

| temperature | interphase loss | dead metal | retention | dominant |
|---|---|---|---|---|
| −10 °C | 1.3 mAh | 449 mAh | 0.910 | plating |
| 10 °C | 4.5 mAh | 272 mAh | 0.945 | plating |
| 25 °C | 8.9 mAh | 0 | **0.998** | interphase |
| 55 °C | 23.1 mAh | 0 | 0.995 | interphase |

Neither extreme is safe, for opposite reasons, and the optimum moves upward as charging gets faster. That is why thermal management targets a band rather than a ceiling. Both the U-shape and the attribution of each arm to the right mechanism are asserted as tests.

Plating is modelled with Butler-Volmer rather than Tafel, which matters more than it sounds: a bare Tafel term has no reverse branch and predicts deposition at *every* potential including rest, which integrated over a month of storage plates out an entire cell. The two branches must cancel exactly at the onset. Plated and dead lithium are tracked as separate inventories for a similar reason — with one running total, stripping makes the same lithium available to strip again on the next sample.

Parameters are representative, not fitted, and the module says so: reported interphase rate constants span several orders of magnitude because they absorb whatever the fit could not otherwise explain. The shape of the curve is meaningful; the number of years is not.

Reproduce with `python examples/07_degradation.py`.

Or from the command line, without writing any Python:

```bash
cellkernel charge --temperature 263.15 313.15   # safe charge rate, by state of charge
cellkernel age --cycles 300                     # capacity fade, by temperature
```

### 8. Charging as fast as the physics allows

`cellkernel.protocols` inverts the plating criterion: given a state and a temperature, `plating_limited_current` returns the largest charging current that keeps the electrode a stated margin above the onset. Feeding that back as the setpoint gives a charge that is aggressive where it can be and cautious where it must be.

The safe rate, in C:

| soc | −10 °C | 0 °C | 10 °C | 25 °C | 40 °C |
|---|---|---|---|---|---|
| 0.10 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 |
| 0.50 | 0.53 | 0.85 | 1.34 | 2.49 | 3.00 |
| 0.90 | 0.17 | 0.28 | 0.45 | 0.87 | 1.57 |

A fixed-rate charger has to sit under the worst cell in that grid. Charging from 10% with two hours available:

| | to 80% | min electrode potential | time spent plating |
|---|---|---|---|
| **−5 °C** CCCV 1C | 78 min | −8.1 mV | 8.9 min |
| **−5 °C** CCCV 3C | 72 min | −46.7 mV | 12.3 min |
| **−5 °C** plating-limited | 82 min | **+2.2 mV** | **0** |
| **25 °C** CCCV 3C | 26 min | +2.8 mV | 0 |
| **25 °C** plating-limited | 39 min | +8.4 mV | 0 |

At −5 °C the conventional charge deposits metal at *every* rate offered, including 1C, for about ten minutes of each charge — and nothing in its terminal measurements tells it so. The guarded protocol gets to a comparable state of charge in a comparable time and never crosses the onset.

At 25 °C the trade reverses: nothing plates and the guarded protocol is slower than simply charging at 3C. That is the honest cost of a guarantee — it is paid exactly when it was not needed. Whether it is worth paying depends on how much of the year the pack spends cold.

**And it runs in the firmware.** The generated C exposes both:

```c
ck_real_t phi   = ck_plating_potential(&est, current);
ck_real_t setpt = ck_max_charge_current(&est, 0.01f, CK_LIMIT_CEILING);
```

The potential costs nothing extra — `ck_voltage` already forms it as a sub-expression and discards it. The limiter is bisection over a fixed 24 steps, so execution time is constant and known: no convergence test, no loop that might not terminate. The generated bisection is **bit-identical** to its Python mirror, which is what a fixed iteration count with no tolerance test should give, and the plating potential agrees to 1e-12 V in double precision.

The scheduled estimator exposes the same pair taking measured temperature, and that is the version that earns its keep — plating is a cold-weather failure, and the isothermal generator can only answer for the single point it was built at. At 70% state of charge the safe rate falls from 1.32C at 25 °C to 0.26C at −10 °C, a factor of five, and the embedded answer reproduces the full Python model's table to two decimal places.

One caveat the header now states rather than burying in a build log: `ck_plating_potential` reads the negative electrode's lookup table, so it inherits that table's error. Graphite tabulates badly — its stage transitions cost **4.95 mV at the default 257 points**, against 0.13 mV for the layered oxide opposite it. A 10 mV plating margin is therefore only twice the table error. `CK_OCP_ERROR_NEG` and `CK_OCP_ERROR_POS` are emitted so a margin can be sized against them; `--table-points 513` brings the negative electrode to 1.3 mV.

That closes the loop the package exists to close. The quantity that limits charging is not measurable at the terminals, so a controller either models it or guesses — and this is that model, compiled, on the microcontroller that sets the current.

Reproduce with `python examples/08_fast_charge.py`.

### 9. Estimators

`EKF` (Joseph-form covariance, optional Gauss-Newton iteration), `UKF` (sigma points on the measurement only — for an affine map the unscented transform is exact, so propagating them through the linear process would compute the same numbers more slowly and *less* accurately), and `DualEKF` (adds capacity retention and resistance growth).

Two things here are not decoration:

**Priors are shaped, not isotropic.** "I do not know the state of charge" does not mean every state is independently uncertain — a rested cell has no concentration gradient however full it is. `suggest_initial_covariance` returns a rank-one covariance along the state-of-charge direction plus a small floor; `suggest_process_noise` places a rank-one term along the input column, because a mis-measured ampere cannot produce an arbitrary state disturbance. With an isotropic prior instead, corrections are absorbed by the gradient coordinates rather than the bulk concentration, and the filter fits the voltage while leaving its charge error untouched. In development this showed up as a 4% wander during an hour of rest — exactly when open-circuit voltage should be pinning the estimate down.

**Iterating the measurement update matters more than the filter choice.** Seed a nickel-manganese-cobalt cell at 90% when it is really at 75%: the local `dOCV/dSOC` there is about 0.22 V, but the average slope across the gap is near 1.1 V. A single linearised correction therefore overshoots roughly fivefold and the estimate oscillates instead of settling. Measured on a 40-minute drive cycle:

| Filter | Final state-of-charge error |
|---|---|
| EKF, 1 iteration | 0.241 — diverged |
| EKF, 2 iterations | 0.004 |
| EKF, 3 iterations | 0.0015 |
| UKF | 0.0026 |

The iterated EKF beats the unscented filter here, for a fraction of the cost.

![State-of-charge estimation on a synthetic drive cycle](soc_estimation.png)

The middle panel shows what a single linearised correction actually does when seeded 15% wrong: it drives the estimate *above 100%* state of charge and then takes the whole cycle to crawl back, ending worse than open-loop coulomb counting. That is the failure mode the table above quantifies, and it is the reason `iterations` defaults to more than one. Reproduce with `python examples/02_estimate_state_of_charge.py`.

### 10. Code generation, and evidence

`generate()` emits `cellkernel_estimator.{h,c}` — no dynamic allocation, no global mutable state, fixed-size arrays, bounded execution time, worst case equal to typical case. It compiles under `-std=c99 -Wall -Wextra -Wpedantic -Werror` with no warnings.

The filter path contains no loop whose trip count depends on data. The one routine that iterates, `ck_max_charge_current`, does a fixed 24 bisection steps with no convergence test and no early exit, which costs it two or three wasted halvings and buys the same property: the time it takes does not depend on what the cell is doing. CI checks that the early exit has not crept back in. Alongside it come a host harness, a `Makefile`, a `CMakeLists.txt`, and a `BUDGET.txt`:

```
precision            float (4 bytes/word)
states               6
flash (tables)       2584 B
RAM (per instance)   168 B
stack (predict)      216 B
arithmetic/step      564 mul, 522 add, 12 div, 6 sqrt, 2 log
modelled cycles      1979
estimated time       48 MHz: 41.2 us, 80 MHz: 24.7 us, 120 MHz: 16.5 us, 180 MHz: 11.0 us
```

`verify()` then compiles it and splits the error into three legs, because the aggregate number cannot tell you which one you are looking at:

1. **Generated C against a NumPy mirror** with identical loop order and table evaluation. Any disagreement beyond round-off is a code-generation defect. This is the leg that validates the generator.
2. **Mirror against a table-backed Python model.** Zero by construction; catches a mirror that has drifted.
3. **Table-backed against the full analytic model.** The price of a lookup table — a modelling choice, reported in millivolts rather than left implicit.

A 3 mV gap between generated C and a reference is unremarkable if it is table resolution and alarming if it is arithmetic. Separating the legs is what makes the difference visible. During development this immediately localised a 138 mV discrepancy to a lookup table whose domain was too narrow to cover surface excursion under load.

### Measured, not modelled

That `BUDGET.txt` is arithmetic on the emitted data structures. `cellkernel measure` cross-compiles for a real Cortex-M and reads the linker's own accounting:

```
        target   opt     flash      code    tables
  --------------------------------------------------
    cortex-m4f   -Os     4636 B     2124 B     2512 B
    cortex-m4f   -O2     5492 B     2980 B     2512 B
 cortex-m0plus   -Os     4820 B     2308 B     2512 B

  Instructions retired per filter step, on an emulated Cortex-M4:
     -Os  5,086 instructions   (modelled 1,979 cycles)
```

**Two of those columns say the model was wrong.** The table count is good — within 3% of what the linker reports, which it should be, since it is exact arithmetic. But the budget reported *flash* as tables only, justified by a note claiming code was small beside them. Code is 2124 bytes against 2512 of tables, so the headline flash figure was short by 45%. And the modelled cycle count is optimistic by a factor of two and a half.

The instruction count is itself measured rather than assumed twice over. Fixed startup cost is removed by differencing two step counts. The tick-to-instruction conversion is calibrated in the same run, by timing a block of exactly *n* assembler `nop`s at two values of *n* so the loop overhead cancels — that comes out at exactly 40 instructions per tick, agreeing with the board's documented 25 MHz, but agreeing with a datasheet is a check rather than a substitute. The first attempt at that calibration wrapped the nops in a C loop and came out three times low.

Instructions are still not cycles: QEMU models no pipeline, no flash wait states and no memory system, so real silicon takes at least this many and generally more. `168 B` of RAM per cell is exact, and `.bss` and `.data` are both zero — the estimator has no globals, which is what makes one instance per cell in a pack safe.

## Known limitations

Stated plainly, because a tool that hides these is worse than one that does not exist.

- **The thermal model is a single node.** One lumped node is what can actually be identified from a battery-management unit's own measurements, which are surface temperature at best. Radial gradients inside a cylindrical cell reach tens of kelvin at high rate and are not represented.
- **Generated code takes temperature as an input, and does not estimate it.** `generate_scheduled()` emits an estimator valid across a temperature range, but it must be given a thermistor reading; the Python `ThermalSPM` can infer temperature from voltage, and that capability does not cross to C. This is a deliberate division rather than an omission — see below — but if you have no temperature sensor, the generated path is not for you.
- **Temperature is weakly observable from voltage alone.** The filter infers it through its effect on polarisation, which is indirect and slow. Measured state-of-charge error on a cold 1.5C discharge settles near 3.5% with a temperature state against 0.15% for the isothermal model on a comparable run. If a thermistor is available, use it; a measured cell temperature is worth more than any amount of filter tuning here.
- **The electrolyte model holds its transport coefficients constant.** Diffusivity, conductivity and transference number all vary appreciably across the concentration range a cell visits at high rate, and `SPMe` uses bulk values for all three — which is what keeps the system linear and discretisable offline. It reports the consequence rather than hiding it: `validity()` returns `good`, `degraded` or `extrapolating`, and on this cell 5C is already degraded.
- **Degradation parameters are representative, not fitted.** Interphase rate constants in the literature span several orders of magnitude, because they absorb whatever the fitting procedure could not otherwise explain. The shape of the predicted curve is meaningful; the number of years is not. Loss of active material through particle cracking, transition-metal dissolution and positive-electrode side reactions are not modelled at all.
- **The posterior covariance is optimistic at open circuit.** One voltage measurement cannot separate the two electrodes; there is a direction in state space that voltage never observes, and only current integration couples them. The reported standard deviation comes out several times smaller than the true error during long rests. This is asserted as a test rather than hidden, so it will be noticed when fixed.
- **Lookup-table error is not uniform.** Worst-case interpolation error for the built-in graphite fit is 4.95 mV at 257 points, but it is confined entirely below 4% stoichiometry — the steep exponential rise, 3% of the table domain and below the operating window. Median error over the domain is 0.002 mV. Raise `table_points` if the extremes matter; `BUDGET.txt` reports the figure so the trade is explicit.
- **Capacity retention is modelled as loss of active material** (flux scaling), not loss of lithium inventory. That makes it partially observable from pulse transients, which is correct for that mechanism and wrong for the other. Distinguishing them needs a second parameter.
- **The modelled cycle count is optimistic by about 2.5×, and instructions are not cycles.** `estimate_budget` reports 1,979 cycles per step; measured on an emulated Cortex-M4F the same code retires 5,086 instructions. And QEMU models no pipeline, no flash wait states and no memory system, so real silicon will take at least that many cycles and generally more. Use `cellkernel measure` and treat the modelled figure as a lower bound for early sizing only.
- **No measurement on real silicon.** Everything above comes from a cross-compiler and an emulator. Neither models a flash accelerator, a cache, or contention with the rest of a firmware image.
- **A ~2.5 mV residual against PyBaMM remains unexplained.** Down from 23 mV once three bridge defects were fixed, and it now responds to refinement rather than being a fixed offset — but it plateaus rather than vanishing, so some difference between the two implementations is still there.

## Testing

```bash
pytest                      # 602 tests
pytest -m "not compiler"    # skip tests needing a C compiler
```

The test suite is anchored to closed-form results wherever possible rather than to previous outputs. Among the things it asserts:

- the exact series coefficients $1, \tfrac1{15}, -\tfrac1{525}, \tfrac2{23625}, -\tfrac{37}{9095625}$;
- $\sum_k \lambda_k^{-2} = 1/10$ over the roots of $\tan\lambda = \lambda$, *and* that the residual shrinks as $1/\pi^2 k$;
- the steady-state surface offset $RN/5D$ — exactly for the polynomial and residualised spectral models, converging at second order for finite volume;
- that the mass balance is structurally exact for every model at every order;
- that zero-order-hold discretisation is stable at $\Delta t = 100 R^2/D$;
- that a rested cell's terminal voltage equals its open-circuit voltage to 1e-9 V;
- that exchange current density lands in the physically plausible 0.1–30 A m⁻² band, which is the guard that caught a spurious Faraday constant collapsing the kinetic overpotential to microvolts;
- that the unscented transform of the linear process reproduces $APA^\top$, and that tight sigma points lose precision doing so;
- that the generated C matches its mirror to machine precision.

Where a finite-difference reference is used, the step size is chosen adaptively: with a state vector spanning `1e4` to `1e10` a fixed relative step makes the *reference* less accurate than the derivative it is checking, which is a uniquely misleading failure.

Closed-form tests cannot catch a misunderstanding shared between a model and the test written by the same person, so a separate suite compares against PyBaMM. It needs `pip install pybamm` and skips without it.

```bash
pytest tests/test_pybamm_validation.py
```

Sixteen CI jobs run on every push: the suite on three operating systems and three Python versions, linting, coverage with a floor, every example end to end, the PyBaMM comparison, and the generated C compiled by both gcc and clang under conversion and shadowing warnings and then run under the undefined-behaviour and address sanitisers.

## Citing

If this is useful in published work, please cite it. See [`paper/paper.md`](paper/paper.md).

## Licence

BSD 3-Clause. See [`LICENSE`](LICENSE).
