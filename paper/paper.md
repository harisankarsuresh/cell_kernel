---
title: "cellkernel: verified embedded state estimators from physics-based lithium-ion models"
tags:
  - Python
  - C
  - lithium-ion batteries
  - battery management systems
  - state estimation
  - reduced-order modelling
  - code generation
authors:
  - name: Harisankar Suresh
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher, Bengaluru, India
    index: 1
date: 4 August 2026
bibliography: paper.bib
---

# Summary

`cellkernel` closes the gap between physics-based lithium-ion cell modelling and
the embedded software that runs battery packs. It reduces solid-phase diffusion to
a small linear state-space system, wraps a single particle model in an extended or
unscented Kalman filter, and generates a self-contained C99 estimator with static
allocation and bounded execution time. It then compiles that C, replays it against
the Python original, and reports the divergence, separating code-generation error
from deliberate approximation.

A six-state estimator occupies 168 bytes of RAM and 2.6 kB of flash and executes
in roughly 16 microseconds at 120 MHz. Generated code agrees with the Python
reference to machine precision in double precision and to about 7 microvolts in
single precision — three orders of magnitude below the noise floor of a
production measurement front end.

Optional layers resolve salt transport across the electrode sandwich, couple a
lumped thermal node with the reduced-order matrices gain-scheduled over
temperature, and predict capacity fade from interphase growth and lithium
plating. Generated estimators can be scheduled across a temperature range, taking
a measured cell temperature as an input rather than estimating it.

# Statement of need

The open-source battery modelling ecosystem is mature at the simulation and
parameterisation stages. PyBaMM [@Sulzer2021] solves continuum electrochemical
models; PyBOP [@Courtier2025] identifies their parameters; liionpack
[@Tranter2022] extends simulation to packs; cellpy [@Wind2024] and BEEP
[@Herring2020] ingest cycler data. All of these stop at the Python boundary.

Battery-management units do not run Python. Deploying a physics-based estimator to
the microcontroller that actually controls a pack is currently a manual
translation: equations are read from a paper and rewritten in C. That translation
is unverifiable after the fact, and it is a principal reason production systems
continue to ship equivalent-circuit models with coulomb counting while
electrochemical models remain in research notebooks. Commercial toolchains can
generate code from block diagrams, but they are proprietary, and their output is
not accompanied by an error budget that distinguishes arithmetic fidelity from
modelling approximation.

`cellkernel` provides that path and, critically, the evidence that it is
faithful.

# Approach

**Reduced-order diffusion.** The exact surface transfer function of a spherical
particle,
$G(s) = (R/D)\sinh\xi / (\xi\cosh\xi - \sinh\xi)$ with $\xi = R\sqrt{s/D}$,
factors into an integrator carrying the mass balance and an analytic remainder,
$G(s) = (3/Rs)\,\hat H(R^2 s/D)$. Four families approximate $\hat H$: Padé,
eigenfunction (spectral) truncation with static condensation of the discarded
modes, conservative finite volume, and the two-state moment closure of
Subramanian et al. [@Subramanian2005]. Padé coefficients are obtained in exact
rational arithmetic; the underlying Hankel system is ill-conditioned enough that a
double-precision solve is unusable beyond order eight.

Each model is delivered as a zero-order-hold discretisation computed by matrix
exponential at build time. This is the central design decision: it moves all
numerically delicate work offline, leaving an online update that is a single dense
matrix-vector product, unconditionally stable at any sample period and exact for
piecewise-constant current. A hand-written explicit integrator, by contrast, is
stable only for $\Delta t < \Delta r^2/2D$, which for a six-micron particle is
milliseconds — far below a realistic task period.

**Exactly linear process dynamics.** Diffusion is linear in flux and flux is
linear in current, so all nonlinearity resides in the voltage measurement. The
prediction step of an extended Kalman filter therefore carries no linearisation
error, and the standard objection to extended filters — that repeated
linearisation of the dynamics corrupts the covariance — does not apply. It also
means the unscented transform of the process is exact, so sigma-point propagation
would compute the same result more slowly and, because of weight cancellation at
small spread parameters, less accurately.

**Shaped priors.** Filter moments are constructed along physically meaningful
directions rather than isotropically. Uncertainty in state of charge lies along
the uniform-concentration direction, because a rested cell has no gradient
regardless of its charge; current-measurement error enters along the input column.
An isotropic prior instead directs corrections into gradient coordinates, and the
filter fits the measured voltage while leaving its charge error largely intact.

**Verification in three legs.** Generated C is compared against a NumPy mirror
with identical loop order and table evaluation, the mirror against a table-backed
Python model, and that against the full analytic model. The first leg measures the
generator; the last is the cost of representing an open-circuit potential as a
lookup table. Aggregating them would obscure which is which: a three-millivolt
discrepancy is unremarkable if it is table resolution and serious if it is
arithmetic.

**Scheduling on the physical variable.** Solid diffusivity is Arrhenius in
temperature and enters the model through a matrix exponential, so
temperature-dependent estimators precompute matrices across a grid and blend them
online. The blend is taken on the Arrhenius factor rather than on temperature.
At a sample period short against the diffusion time constant the discrete matrix
approaches $I + A_c(D)\Delta t$ with $A_c \propto D$, so the matrices are nearly
affine in diffusivity while diffusivity is exponential in $1/T$; interpolating
linearly in temperature fits a straight line through an exponential and errs most
where the curvature is greatest, which is the cold end. On a nine-point grid the
difference is 197 mV of terminal-voltage error at $-18$ °C against 1.8 mV.

# Validation

The test suite is anchored to closed-form results rather than to recorded
outputs. It asserts the exact rational series coefficients of $\hat H$; the
identity $\sum_k\lambda_k^{-2} = 1/10$ over the roots of $\tan\lambda=\lambda$,
together with the $1/\pi^2 k$ decay of its residual; the steady-state surface
offset $RN/5D$, reproduced exactly by two of the four models and at second order
by finite volume; structural exactness of the mass balance at every order;
stability of the discretisation at $\Delta t = 100\,R^2/D$; and agreement between
generated C and its mirror at machine precision.

Several defects found this way are worth recording, since none would have been
caught by a regression test comparing against stored output. A spurious Faraday
constant in the exchange current density inflated it by five orders of magnitude
and collapsed the kinetic overpotential to microvolts, leaving a model that ran
and looked plausible while having no charge-transfer resistance. A lookup table
whose domain covered only the bulk stoichiometry window saturated under load,
producing a 138 millivolt error that the three-leg comparison localised
immediately. A Tafel form for lithium plating, lacking a reverse branch, predicted
deposition at every potential including open circuit, and over a simulated month
of storage plated out an entire cell; Butler-Volmer kinetics, whose branches
cancel exactly at the onset potential, replaced it. And conservation identities
were found to depend on the accuracy of a matrix exponential over a stiff
generator, differing between linear-algebra backends at extreme step sizes, and
are now projected back explicitly after discretisation.

Extending the model set produced one result worth reporting in its own right.
With interphase growth and plating both active, total capacity loss is U-shaped in
temperature: growth is Arrhenius and dominates the hot arm, plating is driven by
sluggish transport and dominates the cold one, and the optimum shifts upward as
charge rate increases. Both the shape and the attribution of each arm to the
correct mechanism are asserted as tests.

# Limitations

The thermal model uses a single lumped node, which is what can be identified from
the surface measurements a battery-management unit actually has; radial gradients
within a cylindrical cell are not resolved. Electrolyte transport coefficients are
held at bulk values, which preserves linearity and offline discretisation at the
cost of accuracy under severe depletion; the model reports its own validity rather
than extrapolating silently. Degradation parameters are representative rather than
fitted, since published interphase rate constants span several orders of
magnitude. Loss of active material through particle cracking, transition-metal
dissolution and positive-electrode side reactions is not modelled. The posterior
covariance is optimistic during long open-circuit rests, because a single voltage
measurement cannot separate the two electrodes; this is asserted as a test rather
than concealed. Cycle-count figures are modelled for a Cortex-M4F rather than
measured on hardware.

# Acknowledgements

This work builds on the open parameter sets and open-circuit potential fits
published by Chen et al. [@Chen2020] and Prada et al. [@Prada2013], and on the
example set by the PyBaMM community for open, reproducible battery modelling.

# References
