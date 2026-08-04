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

# Validation

The test suite is anchored to closed-form results rather than to recorded
outputs. It asserts the exact rational series coefficients of $\hat H$; the
identity $\sum_k\lambda_k^{-2} = 1/10$ over the roots of $\tan\lambda=\lambda$,
together with the $1/\pi^2 k$ decay of its residual; the steady-state surface
offset $RN/5D$, reproduced exactly by two of the four models and at second order
by finite volume; structural exactness of the mass balance at every order;
stability of the discretisation at $\Delta t = 100\,R^2/D$; and agreement between
generated C and its mirror at machine precision.

Two defects found this way are worth recording, since both would have been
invisible to a regression test. A spurious Faraday constant in the exchange
current density inflated it by five orders of magnitude and collapsed the kinetic
overpotential to microvolts, leaving a model that ran and looked plausible while
having no charge-transfer resistance; a plausibility bound on $i_0$ now guards it.
And a lookup table whose domain covered only the bulk stoichiometry window
saturated under load, producing a 138 millivolt error that the three-leg
comparison localised immediately.

# Limitations

The models are isothermal, with temperature handled by rebuilding and
gain-scheduling. Electrolyte dynamics are not represented, so the formulation is a
single particle model with lumped series resistance rather than SPMe. The posterior
covariance is optimistic during long open-circuit rests, because a single voltage
measurement cannot separate the two electrodes; this is asserted as a test rather
than concealed. Capacity fade is modelled as loss of active material only.
Cycle-count figures are modelled for a Cortex-M4F rather than measured on
hardware.

# Acknowledgements

This work builds on the open parameter sets and open-circuit potential fits
published by Chen et al. [@Chen2020] and Prada et al. [@Prada2013], and on the
example set by the PyBaMM community for open, reproducible battery modelling.

# References
