"""Cell parameters, electrode balancing and built-in parameter sets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq, least_squares

from .ocp import (
    OCPFunction,
    TabulatedOCP,
    graphite_chen2020,
    lfp_prada2013,
    nmc811_chen2020,
    numerical_derivative,
)

__all__ = [
    "CellParameters",
    "ElectrodeParameters",
    "FARADAY",
    "GAS_CONSTANT",
    "ThermalParameters",
    "balanced_stoichiometry_window",
    "chen2020_nmc811_graphite",
    "lfp_graphite",
    "from_pybamm",
]

#: Faraday constant, C mol-1.
FARADAY = 96485.33212
#: Universal gas constant, J mol-1 K-1.
GAS_CONSTANT = 8.31446261815324


@dataclass
class ElectrodeParameters:
    """Porous-electrode properties needed by a single-particle model.

    Parameters
    ----------
    thickness
        Coating thickness ``L`` in metres.
    particle_radius
        Representative particle radius ``R`` in metres.
    active_fraction
        Volume fraction of active material, ``eps_s``, dimensionless.
    max_concentration
        Maximum lithium concentration in the solid, ``c_max``, mol m-3.
    diffusivity
        Solid-phase diffusion coefficient at ``reference_temperature``, m2 s-1.
    reaction_rate
        Prefactor ``m`` in the exchange current density
        :math:`i_0 = m \\sqrt{c_e}\\sqrt{c_s}\\sqrt{c_{\\max} - c_s}`, in
        ``(A m-2)(m3 mol-1)^1.5``. The Faraday constant is already folded into
        this prefactor, matching the convention of the reported values.
    ocp
        Open-circuit potential as a function of stoichiometry, in volts.
    stoich_at_0_soc, stoich_at_100_soc
        Stoichiometry at the two ends of the usable window. For the negative
        electrode the second is the larger; for the positive it is the smaller.
    diffusion_activation_energy, reaction_activation_energy
        Arrhenius activation energies in J mol-1. Zero means the property is
        treated as temperature independent.
    entropic_coefficient
        ``dU/dT`` in V K-1, used for reversible heat generation.
    """

    thickness: float
    particle_radius: float
    active_fraction: float
    max_concentration: float
    diffusivity: float
    reaction_rate: float
    ocp: OCPFunction
    stoich_at_0_soc: float
    stoich_at_100_soc: float
    diffusion_activation_energy: float = 0.0
    reaction_activation_energy: float = 0.0
    entropic_coefficient: float = 0.0

    @property
    def specific_area(self) -> float:
        """Volumetric interfacial area ``a = 3 eps_s / R``, m2 m-3."""
        return 3.0 * self.active_fraction / self.particle_radius

    def throughput_capacity(self, electrode_area: float) -> float:
        """Total lithium capacity of the coating, in ampere hours.

        This is the charge held between ``x = 0`` and ``x = 1``, not the usable
        capacity, which is restricted to the stoichiometry window.
        """
        volume = self.active_fraction * self.thickness * electrode_area
        return volume * self.max_concentration * FARADAY / 3600.0

    def usable_capacity(self, electrode_area: float) -> float:
        """Charge available across the stoichiometry window, in ampere hours."""
        span = abs(self.stoich_at_100_soc - self.stoich_at_0_soc)
        return self.throughput_capacity(electrode_area) * span

    def stoichiometry(self, soc: np.ndarray) -> np.ndarray:
        """Map state of charge in ``[0, 1]`` onto stoichiometry."""
        soc = np.asarray(soc, dtype=float)
        return self.stoich_at_0_soc + soc * (self.stoich_at_100_soc - self.stoich_at_0_soc)

    def concentration(self, soc: np.ndarray) -> np.ndarray:
        """Solid concentration at a given state of charge, mol m-3."""
        return self.stoichiometry(soc) * self.max_concentration

    def ocp_derivative(self, x: np.ndarray) -> np.ndarray:
        """``dU/dx``, using the interpolant's analytic derivative when available."""
        if isinstance(self.ocp, TabulatedOCP):
            return self.ocp.derivative(x)
        return numerical_derivative(self.ocp, x)

    def diffusivity_at(self, temperature: float, reference: float) -> float:
        """Arrhenius-corrected solid diffusivity."""
        return self.diffusivity * _arrhenius(
            self.diffusion_activation_energy, temperature, reference
        )

    def reaction_rate_at(self, temperature: float, reference: float) -> float:
        """Arrhenius-corrected reaction rate prefactor."""
        return self.reaction_rate * _arrhenius(
            self.reaction_activation_energy, temperature, reference
        )


def _arrhenius(activation_energy: float, temperature: float, reference: float) -> float:
    """Ratio of a rate at ``temperature`` to its value at ``reference``."""
    if activation_energy == 0.0:
        return 1.0
    return float(np.exp(activation_energy / GAS_CONSTANT * (1.0 / reference - 1.0 / temperature)))


@dataclass
class ThermalParameters:
    """Lumped single-node thermal model of a cell.

    A single node is a deliberate simplification. Radial gradients inside a
    cylindrical cell are real -- tens of kelvin at high rate -- but a
    battery-management unit measures only surface temperature, so a
    single-node model with a measured heat-transfer coefficient is what can
    actually be identified from available data. Distributed thermal models
    belong in offline design work, not in an online estimator.

    Parameters
    ----------
    heat_capacity
        Lumped heat capacity in J K-1.
    surface_area
        Area available for convective exchange, m2.
    heat_transfer_coefficient
        Convective coefficient in W m-2 K-1.
    ambient_temperature
        Default coolant or air temperature in kelvin.
    """

    heat_capacity: float
    surface_area: float
    heat_transfer_coefficient: float
    ambient_temperature: float = 298.15

    @property
    def time_constant(self) -> float:
        """Thermal time constant in seconds."""
        return self.heat_capacity / (self.heat_transfer_coefficient * self.surface_area)


@dataclass
class CellParameters:
    """Complete parameter set for a single lithium-ion cell.

    Notes
    -----
    A parameter set is only physically meaningful if the two electrodes are
    *charge balanced*: sweeping the state of charge from 0 to 1 must move the
    same number of moles of lithium out of one electrode as into the other.
    :meth:`balance_error` reports the mismatch and
    :func:`balanced_stoichiometry_window` can construct a window that satisfies
    it exactly along with the terminal-voltage limits.
    """

    negative: ElectrodeParameters
    positive: ElectrodeParameters
    electrode_area: float
    separator_thickness: float
    electrolyte_concentration: float
    nominal_capacity: float
    voltage_limits: tuple[float, float]
    contact_resistance: float = 0.0
    reference_temperature: float = 298.15
    thermal: ThermalParameters | None = None
    name: str = "cell"
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- capacity

    def usable_capacity(self) -> float:
        """Smaller of the two electrode usable capacities, in ampere hours."""
        return min(
            self.negative.usable_capacity(self.electrode_area),
            self.positive.usable_capacity(self.electrode_area),
        )

    def balance_error(self) -> float:
        """Relative mismatch between the two electrode usable capacities.

        Zero for a perfectly balanced set. Values above roughly 1e-3 mean the
        stoichiometry window and the electrode loadings disagree, which shows up
        as a state-of-charge-dependent voltage error that parameter fitting will
        try, and fail, to absorb into transport parameters.
        """
        qn = self.negative.usable_capacity(self.electrode_area)
        qp = self.positive.usable_capacity(self.electrode_area)
        return abs(qn - qp) / max(qn, qp)

    # ----------------------------------------------------------------- voltage

    def open_circuit_voltage(self, soc: np.ndarray) -> np.ndarray:
        """Equilibrium terminal voltage at a given state of charge."""
        soc = np.asarray(soc, dtype=float)
        return np.asarray(self.positive.ocp(self.positive.stoichiometry(soc))) - np.asarray(
            self.negative.ocp(self.negative.stoichiometry(soc))
        )

    def ocv_derivative(self, soc: np.ndarray) -> np.ndarray:
        """``dOCV/dSOC`` in volts per unit state of charge.

        Obtained by the chain rule through both electrode windows. This quantity
        is the observability of state of charge from a voltage measurement: the
        Kalman gain is proportional to it, so where it approaches zero the filter
        must fall back on current integration.
        """
        soc = np.asarray(soc, dtype=float)
        neg, pos = self.negative, self.positive
        dxn = neg.stoich_at_100_soc - neg.stoich_at_0_soc
        dxp = pos.stoich_at_100_soc - pos.stoich_at_0_soc
        return (
            np.asarray(pos.ocp_derivative(pos.stoichiometry(soc))) * dxp
            - np.asarray(neg.ocp_derivative(neg.stoichiometry(soc))) * dxn
        )

    def soc_from_ocv(self, voltage: float) -> float:
        """Invert the open-circuit voltage curve for state of charge.

        Uses bisection on ``OCV(soc) - voltage``. The OCV of a full cell is
        monotone in state of charge by construction, so the root is unique;
        voltages outside the achievable range are clamped to the endpoints
        rather than raising, because a cold-boot measurement below the 0%
        open-circuit voltage is a routine occurrence in the field.
        """
        lo = float(self.open_circuit_voltage(0.0))
        hi = float(self.open_circuit_voltage(1.0))
        if voltage <= lo:
            return 0.0
        if voltage >= hi:
            return 1.0
        return float(brentq(lambda s: float(self.open_circuit_voltage(s)) - voltage, 0.0, 1.0))

    # -------------------------------------------------------------- convenience

    def interfacial_current_scale(self, electrode: str) -> float:
        """Conversion from cell current in amperes to interfacial current density.

        Returns ``1 / (a L A)`` for the requested electrode so that
        ``j = scale * I``, in A m-2. The sign convention is applied by the model,
        not here.
        """
        el = self._electrode(electrode)
        return 1.0 / (el.specific_area * el.thickness * self.electrode_area)

    def flux_scale(self, electrode: str) -> float:
        """Conversion from cell current in amperes to molar flux at the particle surface.

        Returns ``1 / (F a L A)`` in mol m-2 s-1 per ampere.
        """
        return self.interfacial_current_scale(electrode) / FARADAY

    def _electrode(self, which: str) -> ElectrodeParameters:
        key = which.lower()
        if key in {"negative", "neg", "n", "anode"}:
            return self.negative
        if key in {"positive", "pos", "p", "cathode"}:
            return self.positive
        raise ValueError(f"unknown electrode {which!r}")

    def with_activation_energies(
        self,
        diffusion_negative: float = 0.0,
        diffusion_positive: float = 0.0,
        reaction_negative: float = 0.0,
        reaction_positive: float = 0.0,
    ) -> CellParameters:
        """Return a copy with Arrhenius activation energies set, in J mol-1.

        The built-in parameter sets leave these at zero on purpose: they are not
        reported alongside the transport properties they modify, and published
        values scatter widely enough that a shipped default would be a guess
        wearing the clothes of a measurement.

        They are still needed for anything thermal to mean much.
        :class:`~cellkernel.models.ThermalSPM` will run without them, but only the
        ``2RT/F`` kinetic prefactor then responds to temperature, which is the
        smallest of the three channels; solid diffusivity, the one that actually
        governs surface concentration and hence fast-charge limits, stays frozen.
        This helper is the intended way to supply them.

        Representative ranges from the literature, for orientation rather than
        for citing: graphite solid diffusion 30-42 kJ mol-1, layered-oxide solid
        diffusion 25-80 kJ mol-1, reaction rates 17-40 kJ mol-1. Fit your own
        against rate tests at two or more temperatures if the answer matters.

        >>> cell = chen2020_nmc811_graphite().with_activation_energies(
        ...     diffusion_negative=35_000.0, diffusion_positive=30_000.0,
        ...     reaction_negative=35_000.0, reaction_positive=17_800.0,
        ... )
        """
        for name, value in (
            ("diffusion_negative", diffusion_negative),
            ("diffusion_positive", diffusion_positive),
            ("reaction_negative", reaction_negative),
            ("reaction_positive", reaction_positive),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        return replace(
            self,
            negative=replace(
                self.negative,
                diffusion_activation_energy=float(diffusion_negative),
                reaction_activation_energy=float(reaction_negative),
            ),
            positive=replace(
                self.positive,
                diffusion_activation_energy=float(diffusion_positive),
                reaction_activation_energy=float(reaction_positive),
            ),
        )

    def with_capacity_fade(self, retention: float) -> CellParameters:
        """Return a copy with the stoichiometry windows shrunk to ``retention``.

        A crude but useful ageing knob: it narrows both windows symmetrically
        about their midpoints, which reproduces loss of lithium inventory
        without altering electrode loadings. Real degradation also shifts the
        windows relative to one another, which is electrode slippage; that needs
        two parameters and is what a dual state-of-health filter estimates.
        """
        if not 0.0 < retention <= 1.0:
            raise ValueError("retention must be in (0, 1]")

        def shrink(el: ElectrodeParameters) -> ElectrodeParameters:
            mid = 0.5 * (el.stoich_at_0_soc + el.stoich_at_100_soc)
            half = 0.5 * (el.stoich_at_100_soc - el.stoich_at_0_soc) * retention
            return replace(el, stoich_at_0_soc=mid - half, stoich_at_100_soc=mid + half)

        return replace(
            self,
            negative=shrink(self.negative),
            positive=shrink(self.positive),
            nominal_capacity=self.nominal_capacity * retention,
        )


def balanced_stoichiometry_window(
    negative: ElectrodeParameters,
    positive: ElectrodeParameters,
    electrode_area: float,
    capacity: float,
    voltage_limits: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Solve for stoichiometry windows that are charge balanced and hit the voltage limits.

    Four unknowns -- the negative and positive stoichiometries at 0% and 100%
    state of charge -- are constrained by four equations:

    * the negative electrode passes exactly ``capacity`` ampere hours,
    * the positive electrode passes exactly ``capacity`` ampere hours,
    * the terminal voltage at 100% equals the upper limit,
    * the terminal voltage at 0% equals the lower limit.

    This is the honest way to assemble a parameter set: quoting electrode
    loadings and stoichiometry limits independently, as data sheets and papers
    usually do, almost always leaves a percent-level charge imbalance that then
    contaminates any transport parameter fitted against the same data.

    Solved as a bounded least-squares problem rather than by unconstrained root
    finding. Stoichiometry is physically confined to ``[0, 1]`` and every
    published open-circuit-potential fit degenerates outside it, so an
    unconstrained solver that steps out of range gets a meaningless -- often
    non-finite -- residual and stalls. Bounding the variables keeps every trial
    point inside the region where the fits mean something.

    Raises
    ------
    RuntimeError
        If the requested capacity and voltage limits cannot be met
        simultaneously. The message reports what the solver could achieve, since
        the usual cause is a voltage limit outside the range the electrode pair
        can reach at all.

    Returns
    -------
    ((xn_0, xn_100), (xp_0, xp_100))
    """
    v_min, v_max = voltage_limits
    if v_max <= v_min:
        raise ValueError("voltage_limits must be increasing")
    qn = negative.throughput_capacity(electrode_area)
    qp = positive.throughput_capacity(electrode_area)
    if capacity >= min(qn, qp):
        raise ValueError(
            f"requested capacity {capacity:g} Ah exceeds electrode throughput "
            f"({qn:.3f} Ah negative, {qp:.3f} Ah positive)"
        )

    # Residuals are scaled so capacity and voltage errors are comparable in
    # magnitude; otherwise a 1 Ah error would dominate a 1 V error by three
    # orders of magnitude and the solve would ignore the voltage limits.
    def residual(z: np.ndarray) -> np.ndarray:
        xn0, xn100, xp0, xp100 = z
        return np.array(
            [
                (qn * (xn100 - xn0) - capacity) / capacity,
                (qp * (xp0 - xp100) - capacity) / capacity,
                float(positive.ocp(xp100)) - float(negative.ocp(xn100)) - v_max,
                float(positive.ocp(xp0)) - float(negative.ocp(xn0)) - v_min,
            ]
        )

    lo, hi = 1e-3, 1.0 - 1e-3
    guess = np.clip(np.array([0.03, 0.03 + capacity / qn, 0.95, 0.95 - capacity / qp]), lo, hi)
    solution = least_squares(residual, guess, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    xn0, xn100, xp0, xp100 = (float(v) for v in solution.x)
    achieved = residual(solution.x)
    if np.max(np.abs(achieved)) > 1e-6:
        v_hi = float(positive.ocp(xp100)) - float(negative.ocp(xn100))
        v_lo = float(positive.ocp(xp0)) - float(negative.ocp(xn0))
        raise RuntimeError(
            "could not balance electrodes to the requested targets. "
            f"Closest solution reaches OCV {v_lo:.4f} V to {v_hi:.4f} V "
            f"(requested {v_min:.4f} to {v_max:.4f}) with capacity residuals "
            f"{achieved[0]:+.2e} and {achieved[1]:+.2e} relative. "
            "The voltage limits are most likely outside the range this "
            "electrode pair can reach."
        )
    return (xn0, xn100), (xp0, xp100)


def chen2020_nmc811_graphite() -> CellParameters:
    """A 5 Ah NMC811 / graphite cylindrical cell, in the style of the LG M50.

    Geometry, loadings, transport properties and open-circuit potentials follow
    Chen et al. (2020), *Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models*,
    J. Electrochem. Soc. 167 080534.

    The stoichiometry windows are **not** taken from the paper. They are solved
    by :func:`balanced_stoichiometry_window` so that both electrodes pass
    exactly 5.0 Ah and the open-circuit voltage spans 2.5 V to 4.2 V, making the
    set self-consistent. Quoted windows combined with quoted loadings do not
    generally satisfy charge balance, and the residual imbalance would otherwise
    masquerade as a transport error during fitting. Use :func:`from_pybamm` if
    you need the published values verbatim.

    Activation energies default to zero, so the model is isothermal unless you
    set them. Representative literature values are 35 kJ mol-1 for graphite
    solid diffusion and 17 to 40 kJ mol-1 for the reaction rates; they vary
    enough between studies that shipping a default would be misleading.
    """
    area = 0.1027
    negative = ElectrodeParameters(
        thickness=85.2e-6,
        particle_radius=5.86e-6,
        active_fraction=0.75,
        max_concentration=33133.0,
        diffusivity=3.3e-14,
        reaction_rate=6.48e-7,
        ocp=graphite_chen2020,
        stoich_at_0_soc=0.03,
        stoich_at_100_soc=0.90,
        entropic_coefficient=-6.0e-5,
    )
    positive = ElectrodeParameters(
        thickness=75.6e-6,
        particle_radius=5.22e-6,
        active_fraction=0.665,
        max_concentration=63104.0,
        diffusivity=4.0e-15,
        reaction_rate=3.42e-6,
        ocp=nmc811_chen2020,
        stoich_at_0_soc=0.90,
        stoich_at_100_soc=0.27,
        entropic_coefficient=2.0e-5,
    )
    (xn0, xn100), (xp0, xp100) = balanced_stoichiometry_window(
        negative, positive, area, capacity=5.0, voltage_limits=(2.5, 4.2)
    )
    negative = replace(negative, stoich_at_0_soc=xn0, stoich_at_100_soc=xn100)
    positive = replace(positive, stoich_at_0_soc=xp0, stoich_at_100_soc=xp100)
    return CellParameters(
        negative=negative,
        positive=positive,
        electrode_area=area,
        separator_thickness=12.0e-6,
        electrolyte_concentration=1000.0,
        nominal_capacity=5.0,
        voltage_limits=(2.5, 4.2),
        contact_resistance=0.01,
        thermal=ThermalParameters(
            heat_capacity=66.5,
            surface_area=5.31e-3,
            heat_transfer_coefficient=10.0,
        ),
        name="nmc811-graphite-5Ah",
        notes="Chen 2020 style parameters with a charge-balanced stoichiometry window.",
    )


def lfp_graphite(capacity: float = 20.0) -> CellParameters:
    """A lithium iron phosphate / graphite prismatic cell.

    Intended for exercising the hard case rather than as a validated parameter
    set for any particular product. The point is the open-circuit voltage shape:
    on the plateau ``dOCV/dSOC`` falls to a few millivolts per 10% state of
    charge, so a voltage-only estimator has almost nothing to work with over the
    middle 70% of the range. Compare :meth:`CellParameters.ocv_derivative`
    against the NMC set to see the difference.

    The flatness has a second consequence worth knowing about: the balanced
    window this function solves for is not unique. Matching the upper voltage
    limit pins ``U_p`` to a value the plateau takes over a wide span of
    stoichiometry, so the positive window is only weakly determined. That is a
    property of the chemistry, not of the solver, and it is the same degeneracy
    that makes lithium iron phosphate state-of-charge estimation hard in the
    field.
    """
    area = 0.8
    negative = ElectrodeParameters(
        thickness=90.0e-6,
        particle_radius=6.0e-6,
        active_fraction=0.70,
        max_concentration=31370.0,
        diffusivity=3.0e-14,
        reaction_rate=6.0e-7,
        ocp=graphite_chen2020,
        stoich_at_0_soc=0.03,
        stoich_at_100_soc=0.85,
        entropic_coefficient=-6.0e-5,
    )
    positive = ElectrodeParameters(
        thickness=100.0e-6,
        particle_radius=1.5e-6,
        active_fraction=0.55,
        max_concentration=22806.0,
        diffusivity=1.2e-16,
        reaction_rate=1.0e-6,
        ocp=lfp_prada2013,
        stoich_at_0_soc=0.95,
        stoich_at_100_soc=0.05,
        entropic_coefficient=-1.0e-5,
    )
    limits = (2.5, 3.30)
    (xn0, xn100), (xp0, xp100) = balanced_stoichiometry_window(
        negative, positive, area, capacity=capacity, voltage_limits=limits
    )
    negative = replace(negative, stoich_at_0_soc=xn0, stoich_at_100_soc=xn100)
    positive = replace(positive, stoich_at_0_soc=xp0, stoich_at_100_soc=xp100)
    return CellParameters(
        negative=negative,
        positive=positive,
        electrode_area=area,
        separator_thickness=20.0e-6,
        electrolyte_concentration=1200.0,
        nominal_capacity=capacity,
        voltage_limits=limits,
        contact_resistance=0.002,
        thermal=ThermalParameters(
            heat_capacity=550.0,
            surface_area=0.035,
            heat_transfer_coefficient=12.0,
        ),
        name=f"lfp-graphite-{capacity:g}Ah",
        notes="Illustrative LFP set; the flat OCV plateau is the feature of interest.",
    )


def from_pybamm(parameter_values, name: str = "from-pybamm") -> CellParameters:
    """Build a :class:`CellParameters` from a PyBaMM ``ParameterValues`` object.

    PyBaMM is an optional dependency and is not imported by this module; pass an
    already-constructed ``ParameterValues``, for example
    ``pybamm.ParameterValues("Chen2020")``.

    Open-circuit potentials are sampled onto a 501-point grid and wrapped in
    :class:`~cellkernel.ocp.TabulatedOCP` rather than being called through
    directly. PyBaMM's OCP entries may be plain functions, interpolants, or
    symbolic expressions depending on the set, and sampling normalises all three
    into something the code generator can emit as a lookup table.

    Only the subset of parameters a single-particle model needs is read; PyBaMM
    sets carrying electrolyte transport data will have that data ignored.
    """
    get = parameter_values.__getitem__

    def sample(key: str, c_max: float) -> TabulatedOCP:
        grid = np.linspace(1e-4, 1.0 - 1e-4, 501)
        fn = parameter_values[key]
        try:
            values = np.array([float(fn(x * c_max, c_max)) for x in grid])
        except TypeError:
            values = np.array([float(fn(x)) for x in grid])
        return TabulatedOCP(grid, values)

    c_n_max = float(get("Maximum concentration in negative electrode [mol.m-3]"))
    c_p_max = float(get("Maximum concentration in positive electrode [mol.m-3]"))
    area = float(get("Electrode width [m]")) * float(get("Electrode height [m]"))

    negative = ElectrodeParameters(
        thickness=float(get("Negative electrode thickness [m]")),
        particle_radius=float(get("Negative particle radius [m]")),
        active_fraction=float(get("Negative electrode active material volume fraction")),
        max_concentration=c_n_max,
        diffusivity=_scalar_or_call(get("Negative particle diffusivity [m2.s-1]")),
        reaction_rate=1.0e-6,
        ocp=sample("Negative electrode OCP [V]", c_n_max),
        stoich_at_0_soc=0.03,
        stoich_at_100_soc=0.90,
    )
    positive = ElectrodeParameters(
        thickness=float(get("Positive electrode thickness [m]")),
        particle_radius=float(get("Positive particle radius [m]")),
        active_fraction=float(get("Positive electrode active material volume fraction")),
        max_concentration=c_p_max,
        diffusivity=_scalar_or_call(get("Positive particle diffusivity [m2.s-1]")),
        reaction_rate=1.0e-6,
        ocp=sample("Positive electrode OCP [V]", c_p_max),
        stoich_at_0_soc=0.90,
        stoich_at_100_soc=0.27,
    )
    v_min = float(get("Lower voltage cut-off [V]"))
    v_max = float(get("Upper voltage cut-off [V]"))
    capacity = float(get("Nominal cell capacity [A.h]"))
    (xn0, xn100), (xp0, xp100) = balanced_stoichiometry_window(
        negative, positive, area, capacity=capacity, voltage_limits=(v_min, v_max)
    )
    return CellParameters(
        negative=replace(negative, stoich_at_0_soc=xn0, stoich_at_100_soc=xn100),
        positive=replace(positive, stoich_at_0_soc=xp0, stoich_at_100_soc=xp100),
        electrode_area=area,
        separator_thickness=float(get("Separator thickness [m]")),
        electrolyte_concentration=float(get("Initial concentration in electrolyte [mol.m-3]")),
        nominal_capacity=capacity,
        voltage_limits=(v_min, v_max),
        name=name,
        notes="Imported from PyBaMM; stoichiometry window re-balanced.",
    )


def _scalar_or_call(value):
    """Reduce a PyBaMM parameter that may be a constant or a function of state."""
    if callable(value):
        try:
            return float(value(0.5))
        except TypeError:
            return float(value(0.5, 298.15))
    return float(value)
