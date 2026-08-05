"""Physics-based degradation: solid-electrolyte interphase growth and lithium plating.

The dual filter in :mod:`cellkernel.estimators` *tracks* capacity and resistance:
it watches them drift and reports where they are now. This module *predicts* them,
from the two mechanisms that dominate the life of a graphite cell.

Growth of the solid-electrolyte interphase is the slow one. It consumes cyclable
lithium continuously, and because the film it deposits is also the barrier that
slows its own growth, the loss follows a square root in time rather than a line.
It runs fastest when the negative electrode is most lithiated -- that is, at high
state of charge -- and it accelerates with temperature.

Lithium plating is the fast one. Metallic lithium deposits on the negative
electrode instead of intercalating, whenever the local electrode potential falls
below that of lithium metal. It is the mechanism that limits charging rate at low
temperature, and unlike interphase growth it can put a cell into thermal runaway
rather than merely wearing it out.

The two are treated differently on purpose. Interphase growth is integrated as a
slow state, sensibly evaluated over hours of simulated operation. Plating is
evaluated instantaneously and reported as a *margin* in volts, because what a
battery-management unit needs from it is not a life prediction but an answer to
"may I keep charging at this rate", now.

Scope
-----
What is not here: loss of active material through particle cracking, transition
metal dissolution, gas generation, electrolyte oxidation at the positive
electrode, and the partial reversibility of plated lithium on subsequent
discharge. Each is real and each would need parameters that are harder to obtain
than the ones below. This module covers the two mechanisms that a
graphite-and-layered-oxide cell in normal automotive service spends most of its
life limited by, and it is explicit about being a model of those two rather than
of ageing in general.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .params import FARADAY, GAS_CONSTANT, CellParameters

__all__ = [
    "SEIParameters",
    "PlatingParameters",
    "DegradationState",
    "DegradationModel",
]

#: Largest exponent passed to ``exp`` in a Tafel term. At 298 K an argument of 40
#: corresponds to roughly 2 V of overpotential, far past anything physical, and
#: clipping there keeps a transient excursion from overflowing to infinity.
_MAX_TAFEL = 40.0


@dataclass(frozen=True)
class SEIParameters:
    """Interphase growth on the negative electrode.

    Attributes
    ----------
    reaction_rate
        Kinetic rate constant in m s-1.
    solvent_concentration
        Bulk solvent concentration available to react, mol m-3.
    solvent_diffusivity
        Solvent diffusivity through the film, m2 s-1. This is what makes growth
        self-limiting.
    equilibrium_potential
        Potential of the interphase-forming reaction against lithium, V.
    molar_volume
        Molar volume of the deposited film, m3 mol-1.
    conductivity
        Ionic conductivity of the film, S m-1, which sets the resistance it adds.
    initial_thickness
        Film thickness on a fresh cell, m. Must be non-zero: the diffusion-limited
        branch divides by it, and a genuinely bare electrode would react at an
        unbounded rate, which is why formation exists.
    activation_energy
        Arrhenius activation energy of the kinetic branch, J mol-1.
    diffusion_activation_energy
        Arrhenius activation energy of solvent transport through the film,
        J mol-1. Needed as much as the kinetic one: growth becomes
        diffusion-limited within weeks, and without this the model predicts that
        a cell ages at the same rate at 10 C and 45 C, which is conspicuously
        untrue.
    electrons
        Electrons transferred per formula unit of film. Two is the usual choice
        for a carbonate-derived interphase.

    Notes
    -----
    Defaults are representative of a graphite electrode in a carbonate
    electrolyte and are **not** fitted to any particular cell. Interphase
    parameters are poorly constrained even in the literature -- reported rate
    constants span several orders of magnitude, because they absorb whatever the
    fitting procedure could not otherwise explain. Treat predictions from the
    defaults as qualitative: the shape of the curve is meaningful, the number of
    years is not.
    """

    reaction_rate: float = 1.0e-16
    solvent_concentration: float = 4541.0
    solvent_diffusivity: float = 2.5e-22
    equilibrium_potential: float = 0.4
    molar_volume: float = 9.586e-5
    conductivity: float = 5.0e-6
    initial_thickness: float = 5.0e-9
    activation_energy: float = 38000.0
    diffusion_activation_energy: float = 45000.0
    electrons: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "reaction_rate",
            "solvent_concentration",
            "solvent_diffusivity",
            "molar_volume",
            "conductivity",
            "initial_thickness",
            "electrons",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PlatingParameters:
    """Metallic lithium deposition on the negative electrode.

    Attributes
    ----------
    reaction_rate
        Kinetic rate constant in m s-1.
    transfer_coefficient
        Cathodic charge-transfer coefficient.
    activation_energy
        Arrhenius activation energy, J mol-1.
    safety_margin
        Overpotential in volts at which a cell is declared at risk. Zero is the
        thermodynamic onset; a real controller holds a margin above it, because
        the potential being compared is a model output with its own error bars
        and because the consequence of being wrong is not symmetric.
    """

    reaction_rate: float = 1.0e-10
    transfer_coefficient: float = 0.5
    activation_energy: float = 37500.0
    safety_margin: float = 0.02
    #: Fraction of deposited lithium that re-intercalates when the electrode
    #: potential rises again. Reported values scatter widely, from almost none
    #: for lithium that has lost electrical contact to most of it for a short
    #: excursion. Set to zero to treat all plating as permanent.
    stripping_efficiency: float = 0.7

    def __post_init__(self) -> None:
        if self.reaction_rate <= 0.0:
            raise ValueError("reaction_rate must be positive")
        if not 0.0 < self.transfer_coefficient < 1.0:
            raise ValueError("transfer_coefficient must lie in (0, 1)")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")
        if not 0.0 <= self.stripping_efficiency <= 1.0:
            raise ValueError("stripping_efficiency must lie in [0, 1]")


@dataclass
class DegradationState:
    """Accumulated damage. Mutable, because it is integrated in place.

    Attributes
    ----------
    film_thickness
        Interphase thickness on the negative electrode, m.
    lithium_lost
        Cyclable lithium consumed by the interphase, in ampere hours.
    lithium_plated
        Metallic lithium *currently* deposited and still strippable, in ampere
        hours. Falls again when the electrode potential rises.
    lithium_dead
        Lithium that was plated and then failed to strip back, in ampere hours.
        Permanent.
    throughput
        Charge passed through the cell, in ampere hours, counting both directions.

    Notes
    -----
    Plated and dead lithium are tracked separately, and the distinction is not
    bookkeeping pedantry. Plating is partly reversible, so a single running total
    would have to be decremented on stripping -- and then the *same* lithium can
    be stripped again on the next sample, and again, until an amount that should
    have been permanent has been recovered several times over. Splitting the
    inventory means metal that has lost contact leaves the strippable pool for
    good, which is what actually happens.
    """

    film_thickness: float
    lithium_lost: float = 0.0
    lithium_plated: float = 0.0
    lithium_dead: float = 0.0
    throughput: float = 0.0

    def copy(self) -> DegradationState:
        return replace(self)


@dataclass(frozen=True)
class DegradationOutputs:
    """What the model reports at one instant."""

    #: Interphase current density on the negative electrode, A m-2.
    sei_current_density: float
    #: Net plating current density, A m-2. Positive deposits metal, negative
    #: strips it; exactly zero at the onset potential.
    plating_current_density: float
    #: Negative electrode potential against lithium metal, V. Plating begins below zero.
    plating_potential: float
    #: ``plating_potential`` less the configured safety margin. Negative means at risk.
    plating_margin: float
    #: Which branch limits interphase growth here: ``"kinetic"`` or ``"diffusion"``.
    sei_limited_by: str
    #: Added series resistance from the film, ohm.
    film_resistance: float
    #: Remaining capacity as a fraction of nominal.
    capacity_retention: float


class DegradationModel:
    """Integrates interphase growth and evaluates plating risk.

    Deliberately not a :class:`~cellkernel.models.base.CellModel`. Ageing runs on
    a timescale of months while the electrochemical states run on seconds, and
    folding a state that moves by one part in a million per step into a Kalman
    filter would be numerically pointless and would slow every estimator in the
    package down for nothing. This is a separate integrator that reads a cell
    model's state and returns damage.

    Parameters
    ----------
    parameters
        The cell being aged.
    sei, plating
        Mechanism parameters. Defaults are representative, not fitted.

    Examples
    --------
    >>> from cellkernel.models import SPM
    >>> from cellkernel.params import chen2020_nmc811_graphite
    >>> cell = chen2020_nmc811_graphite()
    >>> model = SPM(cell, dt=1.0)
    >>> ageing = DegradationModel(cell)
    >>> state = ageing.initial_state()
    >>> x = model.initial_state(0.9)
    >>> out = ageing.evaluate(model, x, current=5.0, state=state)
    >>> out.plating_potential > 0.0  # not plating on discharge
    True
    """

    def __init__(
        self,
        parameters: CellParameters,
        sei: SEIParameters | None = None,
        plating: PlatingParameters | None = None,
    ) -> None:
        self.parameters = parameters
        self.sei = sei or SEIParameters()
        self.plating = plating or PlatingParameters()
        negative = parameters.negative
        #: Interfacial area of the negative electrode, m2. Every side-reaction
        #: current density is multiplied by this to get amperes.
        self.negative_area = negative.specific_area * negative.thickness * parameters.electrode_area

    # ------------------------------------------------------------------- state

    def initial_state(self) -> DegradationState:
        """A fresh cell, with the film thickness left by formation."""
        return DegradationState(film_thickness=self.sei.initial_thickness)

    def capacity_retention(self, state: DegradationState) -> float:
        """Remaining capacity as a fraction of nominal.

        Both mechanisms remove cyclable lithium, so both reduce capacity. This
        assumes the cell is lithium-limited, which a graphite cell in normal
        service is; a cell that has instead lost positive active material would
        need the other bookkeeping.
        """
        lost = state.lithium_lost + state.lithium_dead + state.lithium_plated
        return max(0.0, 1.0 - lost / self.parameters.nominal_capacity)

    def film_resistance(self, state: DegradationState) -> float:
        """Series resistance added by the film, in ohms."""
        return state.film_thickness / (self.sei.conductivity * self.negative_area)

    def film_areal_resistance(self, state: DegradationState) -> float:
        """Film resistance per unit interfacial area, ohm m2.

        This, not :meth:`film_resistance`, is what multiplies an interfacial
        current density to give the potential drop the side reactions see.
        Confusing the two costs a factor of the electrode area -- three and a bit
        here, and rather more on a large-format cell.
        """
        return state.film_thickness / self.sei.conductivity

    # -------------------------------------------------------------- mechanisms

    def _arrhenius(self, activation_energy: float, temperature: float) -> float:
        if activation_energy == 0.0:
            return 1.0
        return float(
            np.exp(
                activation_energy
                / GAS_CONSTANT
                * (1.0 / self.parameters.reference_temperature - 1.0 / temperature)
            )
        )

    def negative_potential(self, model, x: np.ndarray, current: float) -> float:
        """Negative electrode potential against lithium metal, in volts.

        This single number decides whether the cell is plating. It is the
        equilibrium potential at the *surface* stoichiometry plus the reaction
        overpotential -- surface, not bulk, because plating is a surface process
        and the two diverge exactly when it matters, under fast charge.
        """
        outputs = model.outputs(x, current)
        sto_negative = outputs.surface_stoichiometry[0]
        eta_negative = outputs.overpotential[0]
        return float(self.parameters.negative.ocp(sto_negative)) + float(eta_negative)

    def sei_current_density(
        self, potential: float, film_thickness: float, temperature: float
    ) -> tuple[float, str]:
        """Interphase reaction current density and the branch that limits it.

        Two resistances in series. The kinetic branch is Tafel in the driving
        overpotential :math:`\\eta = \\phi - U_{\\text{sei}}`, which is more
        negative -- so faster -- when the electrode is more lithiated, which is
        why a cell stored full ages faster than one stored empty. The diffusion
        branch is the solvent's transport through the film already there, and
        goes as :math:`1/L`, which is what makes growth self-limiting and produces
        the square root in time.

        .. math::

            \\frac{1}{j} = \\frac{1}{j_{\\text{kin}}} + \\frac{1}{j_{\\text{diff}}}

        Whichever is smaller dominates, and reporting which one it was is worth
        the extra return value: a cell whose growth is kinetically limited is
        being driven hard, while one that is diffusion limited is simply old.
        """
        sei = self.sei
        overpotential = potential - sei.equilibrium_potential
        exponent = -0.5 * FARADAY * overpotential / (GAS_CONSTANT * temperature)
        exponent = float(np.clip(exponent, -_MAX_TAFEL, _MAX_TAFEL))
        kinetic = (
            FARADAY
            * sei.reaction_rate
            * sei.solvent_concentration
            * self._arrhenius(sei.activation_energy, temperature)
            * np.exp(exponent)
        )
        diffusive = (
            FARADAY
            * sei.solvent_diffusivity
            * self._arrhenius(sei.diffusion_activation_energy, temperature)
            * sei.solvent_concentration
            / max(film_thickness, 1e-12)
        )
        combined = 1.0 / (1.0 / max(kinetic, 1e-300) + 1.0 / max(diffusive, 1e-300))
        return float(combined), ("kinetic" if kinetic < diffusive else "diffusion")

    def plating_current_density(self, potential: float, temperature: float) -> float:
        """Net lithium deposition current density, A m-2. Negative means stripping.

        Butler-Volmer rather than Tafel, and the difference is not cosmetic. A
        bare Tafel term has no reverse branch, so it predicts a small but nonzero
        deposition rate at *every* potential, including at rest. Integrated over
        a month of storage that spurious current plates out an entire cell --
        which is exactly the failure this implementation had before the reverse
        term was added, and it is the kind of error that looks plausible in a
        single evaluation and is absurd in an integral.

        .. math::

            j = i_0 \\left[
                \\exp\\!\\left(\\frac{-\\alpha_c F \\phi}{R_g T}\\right)
                - \\exp\\!\\left(\\frac{(1 - \\alpha_c) F \\phi}{R_g T}\\right)
            \\right]

        With :math:`\\phi` measured against lithium metal, the two branches
        cancel exactly at :math:`\\phi = 0`, which is the thermodynamic condition
        for plating and the only place it can be. Above it the net reaction is
        stripping, below it deposition, and the crossover is sharp but continuous
        -- hence :attr:`PlatingParameters.safety_margin` rather than a threshold.
        """
        plating = self.plating
        alpha = plating.transfer_coefficient
        scale = FARADAY / (GAS_CONSTANT * temperature)
        cathodic = float(np.clip(-alpha * scale * potential, -_MAX_TAFEL, _MAX_TAFEL))
        anodic = float(np.clip((1.0 - alpha) * scale * potential, -_MAX_TAFEL, _MAX_TAFEL))
        exchange = (
            FARADAY
            * plating.reaction_rate
            * self.parameters.electrolyte_concentration
            * self._arrhenius(plating.activation_energy, temperature)
        )
        return float(exchange * (np.exp(cathodic) - np.exp(anodic)))

    # ------------------------------------------------------------- evaluation

    def evaluate(
        self,
        model,
        x: np.ndarray,
        current: float,
        state: DegradationState,
        temperature: float | None = None,
    ) -> DegradationOutputs:
        """Report both mechanisms at the current operating point, without integrating."""
        if temperature is None:
            temperature = float(model.outputs(x, current).temperature)
        potential = self.negative_potential(model, x, current)
        # The film carries the intercalation current, so it shifts the potential
        # the side reactions actually see. On charge the interfacial current
        # density is cathodic and the drop pushes the potential *down*, making
        # plating more likely on an aged cell -- which is the observed behaviour
        # and the wrong thing to get backwards.
        current_density = self.parameters.interfacial_current_scale("negative") * current
        driving = potential + current_density * self.film_areal_resistance(state)
        sei_current, limited_by = self.sei_current_density(
            driving, state.film_thickness, temperature
        )
        return DegradationOutputs(
            sei_current_density=sei_current,
            plating_current_density=self.plating_current_density(driving, temperature),
            plating_potential=driving,
            plating_margin=driving - self.plating.safety_margin,
            sei_limited_by=limited_by,
            film_resistance=self.film_resistance(state),
            capacity_retention=self.capacity_retention(state),
        )

    def step(
        self,
        model,
        x: np.ndarray,
        current: float,
        state: DegradationState,
        dt: float,
        temperature: float | None = None,
    ) -> DegradationOutputs:
        """Advance the damage state by ``dt`` seconds, in place.

        ``dt`` here is the *ageing* step and need not match the cell model's
        sample period. Damage accumulates far too slowly to be worth integrating
        at 1 Hz over a cell's life, so the intended pattern is to hold an
        operating point, integrate the damage over minutes or hours of it, and
        move on. :meth:`age_over_cycle` does that.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        outputs = self.evaluate(model, x, current, state, temperature)

        # Film grows in proportion to the charge passed by the side reaction.
        growth = (
            outputs.sei_current_density * self.sei.molar_volume / (self.sei.electrons * FARADAY)
        )
        state.film_thickness += growth * dt

        # Both side reactions consume cyclable lithium. Ampere hours, so that the
        # loss is directly comparable to the cell's rated capacity.
        state.lithium_lost += outputs.sei_current_density * self.negative_area * dt / 3600.0

        # Plating is signed: positive deposits, negative strips metal already
        # there. Stripping removes metal from the strippable pool entirely; the
        # fraction that fails to re-intercalate moves to the dead inventory
        # rather than staying available to be stripped again on the next sample.
        plated = outputs.plating_current_density * self.negative_area * dt / 3600.0
        if plated >= 0.0:
            state.lithium_plated += plated
        else:
            removed = min(-plated, state.lithium_plated)
            state.lithium_plated -= removed
            state.lithium_dead += removed * (1.0 - self.plating.stripping_efficiency)
        state.throughput += abs(current) * dt / 3600.0
        return outputs

    def age_over_cycle(
        self,
        model,
        state: DegradationState,
        soc_low: float = 0.1,
        soc_high: float = 0.9,
        c_rate: float = 1.0,
        temperature: float | None = None,
        samples: int = 24,
    ) -> DegradationOutputs:
        """Integrate one full charge and discharge, in place.

        Rather than simulate every second, the cycle is sampled at ``samples``
        states of charge and the damage integrated over the dwell at each. That
        is a quadrature, and it is accurate here because the damage rate varies
        smoothly with state of charge while the electrochemical transients that
        do not are irrelevant to a quantity accumulating over months.

        Returns the outputs from the last sample, which is the most aged point.
        """
        if not 0.0 <= soc_low < soc_high <= 1.0:
            raise ValueError("require 0 <= soc_low < soc_high <= 1")
        if samples < 2:
            raise ValueError("samples must be at least 2")

        capacity = self.parameters.nominal_capacity * self.capacity_retention(state)
        current = c_rate * self.parameters.nominal_capacity
        # Seconds spent in each state-of-charge bin, on each leg of the cycle.
        dwell = (soc_high - soc_low) * capacity * 3600.0 / current / samples

        outputs = None
        for sign in (-1.0, +1.0):  # charge first, then discharge
            for soc in np.linspace(soc_low, soc_high, samples):
                x = model.initial_state(float(soc), temperature)
                outputs = self.step(model, x, sign * current, state, dwell, temperature=temperature)
        assert outputs is not None
        return outputs
