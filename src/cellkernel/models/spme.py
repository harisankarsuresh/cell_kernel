"""Single particle model with electrolyte."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..ocp import derivative_of
from ..params import FARADAY, GAS_CONSTANT, CellParameters
from ..rom import DiffusionROM, make_rom
from ..rom.electrolyte import ElectrolyteDiffusion
from .base import CellModel, ModelOutputs

__all__ = ["SPMe"]


class SPMe(CellModel):
    """Two particles plus resolved salt transport across the sandwich.

    Adds to :class:`~cellkernel.models.spm.SPM` the two things a single particle
    model lumps into a fitted series resistance: the ohmic drop through the
    electrolyte, and the concentration overpotential that builds as salt is
    driven from one coating to the other.

    .. math::

        V = U_p(x_p^{s}) - U_n(x_n^{s}) + \\eta_p - \\eta_n
            + \\eta_c - I R_e - I R_c

    with the concentration overpotential

    .. math::

        \\eta_c = \\frac{2 R_g T}{F} (1 - t_+) (1 + \\tfrac{d\\ln f}{d\\ln c})
                  \\ln \\frac{\\bar{c}_{e,p}}{\\bar{c}_{e,n}} .

    Why bother
    ----------
    Below roughly 1C the electrolyte contributes a nearly constant resistance,
    and a single particle model with a fitted ``contact_resistance`` absorbs it
    completely. The two models are then indistinguishable, and the simpler one
    is the better choice.

    Above that the electrolyte stops behaving like a resistor. Salt depletes at
    one end of the sandwich and accumulates at the other, on a timescale of tens
    of seconds set by the sandwich thickness rather than by the particle, and the
    exchange current density falls where the salt has gone. A fitted resistance
    cannot reproduce a term with its own dynamics, so it is fitted to whatever
    rate the calibration happened to use and is wrong at every other rate. That
    is the failure this model exists to remove, and
    ``examples/06_electrolyte.py`` measures it.

    What is not modelled
    --------------------
    Concentration-dependent diffusivity, conductivity and transference number.
    All three vary appreciably across the concentration range a cell visits at
    high rate, and all three are held at their bulk values here. This is not an
    oversight but the price of keeping the transport system linear, which is what
    allows it to be discretised once, offline, exactly as the solid diffusion is.
    The consequence is that this model is good where the electrolyte is
    *perturbed* and progressively optimistic where it is *severely depleted* --
    conductivity falls steeply below about a third of nominal concentration, and
    nothing here represents that. Treat it as trustworthy to a few C and
    increasingly indicative beyond.

    Parameters
    ----------
    parameters
        Cell parameter set. Electrolyte transport properties come from its
        ``electrolyte_diffusivity``, ``transference_number``,
        ``ionic_conductivity``, ``bruggeman`` and porosity fields.
    dt
        Sample period in seconds.
    rom
        Solid diffusion reduced-order model family.
    order
        Solid diffusion states per electrode.
    temperature
        Isothermal operating temperature in kelvin.
    electrolyte_cells
        Control volumes ``(negative, separator, positive)`` across the sandwich.
    """

    def __init__(
        self,
        parameters: CellParameters,
        dt: float = 1.0,
        rom: str = "pade",
        order: int = 3,
        temperature: float | None = None,
        electrolyte_cells: tuple[int, int, int] = (4, 3, 4),
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.parameters = parameters
        self.dt = float(dt)
        self.temperature = float(
            temperature if temperature is not None else parameters.reference_temperature
        )
        reference = parameters.reference_temperature

        self.rate_neg = parameters.negative.reaction_rate_at(self.temperature, reference)
        self.rate_pos = parameters.positive.reaction_rate_at(self.temperature, reference)
        self.rom_neg: DiffusionROM = make_rom(
            rom,
            parameters.negative.particle_radius,
            parameters.negative.diffusivity_at(self.temperature, reference),
            order=order,
        )
        self.rom_pos: DiffusionROM = make_rom(
            rom,
            parameters.positive.particle_radius,
            parameters.positive.diffusivity_at(self.temperature, reference),
            order=order,
        )
        self.ss_neg = self.rom_neg.discretise(self.dt)
        self.ss_pos = self.rom_pos.discretise(self.dt)

        self.electrolyte = ElectrolyteDiffusion(
            thickness_negative=parameters.negative.thickness,
            thickness_separator=parameters.separator_thickness,
            thickness_positive=parameters.positive.thickness,
            porosity_negative=parameters.negative.porosity,
            porosity_separator=parameters.separator_porosity,
            porosity_positive=parameters.positive.porosity,
            diffusivity=parameters.electrolyte_diffusivity,
            transference_number=parameters.transference_number,
            electrode_area=parameters.electrode_area,
            bruggeman=parameters.bruggeman,
            cells_negative=electrolyte_cells[0],
            cells_separator=electrolyte_cells[1],
            cells_positive=electrolyte_cells[2],
        )
        self.ss_electrolyte = self.electrolyte.discretise(self.dt)

        self._n_neg = self.ss_neg.n_states
        self._n_pos = self.ss_pos.n_states
        self._n_elec = self.ss_electrolyte.n_states
        self._i_elec = self._n_neg + self._n_pos

        self._flux_neg = -parameters.flux_scale("negative")
        self._flux_pos = +parameters.flux_scale("positive")
        self._j_neg = +parameters.interfacial_current_scale("negative")
        self._j_pos = -parameters.interfacial_current_scale("positive")
        self._kinetic_prefactor = 2.0 * GAS_CONSTANT * self.temperature / FARADAY
        self.electrolyte_resistance = self.electrolyte.ohmic_resistance(
            parameters.ionic_conductivity
        )
        self._concentration_prefactor = (
            self._kinetic_prefactor
            * (1.0 - parameters.transference_number)
            * parameters.thermodynamic_factor
        )

    @staticmethod
    def reconcile(parameters: CellParameters, **kwargs) -> CellParameters:
        """Remove the electrolyte from a lumped contact resistance.

        The single easiest mistake to make with this model, and one made in its
        own test suite before this existed. A parameter set fitted with a model
        that did not resolve the electrolyte carries a contact resistance which
        *already contains* the electrolyte loss. Hand it to ``SPMe``, which
        computes that loss from geometry, and the same ohms are counted twice --
        so the more detailed model performs worse than the simpler one it was
        meant to improve on, and the symptom is a plausible voltage that is
        merely too low.

        This returns the same cell with ``contact_resistance`` reduced by the
        electrolyte resistance the model computes, floored at zero.

        Deliberately a function rather than a warning inside ``__init__``. The
        two cases are not distinguishable from the numbers: a cell may
        legitimately have more tab and current-collector resistance than
        electrolyte resistance, and a check that fired on that would be wrong
        often enough to be switched off, at which point it protects nobody.

        >>> from cellkernel.params import chen2020_nmc811_graphite
        >>> cell = chen2020_nmc811_graphite()
        >>> tidy = SPMe.reconcile(cell)
        >>> tidy.contact_resistance < cell.contact_resistance
        True
        """
        probe = SPMe(parameters, **kwargs)
        return replace(
            parameters,
            contact_resistance=max(
                0.0, parameters.contact_resistance - probe.electrolyte_resistance
            ),
        )

    # ------------------------------------------------------------------- shape

    @property
    def n_states(self) -> int:
        return self._n_neg + self._n_pos + self._n_elec

    @property
    def state_names(self) -> tuple[str, ...]:
        return (
            tuple(f"neg_{i}" for i in range(self._n_neg))
            + tuple(f"pos_{i}" for i in range(self._n_pos))
            + tuple(f"elyte_{i}" for i in range(self._n_elec))
        )

    @property
    def deterministic_states(self) -> tuple[int, ...]:
        """The electrolyte block, which is a known function of the current history.

        Salt transport is driven by current alone and does not depend on the
        solid states, and a rested cell starts from a uniform profile that is
        known rather than estimated. So the electrolyte concentration at any
        moment follows from the current that has been applied, and a voltage
        measurement has nothing to add.

        Measured over a drive cycle where the electrolyte spanned 73 to
        2256 mol m-3, a filter allowed to correct these states moved them by at
        most 5.4 mol m-3, half a percent, and the settled state-of-charge error
        was the same either way. Leaving them out of the covariance makes the
        update eight times cheaper on a typical configuration, which is the
        difference between this model being plausible on a microcontroller and
        not.
        """
        return tuple(range(self._i_elec, self.n_states))

    @property
    def series_resistance(self) -> float:
        """Total ohmic resistance actually applied, in ohms.

        The electrolyte contribution is computed from geometry and conductivity
        rather than fitted, so ``contact_resistance`` should be *reduced*
        accordingly when moving a parameter set from
        :class:`~cellkernel.models.spm.SPM` to this model. Leaving it unchanged
        double-counts the electrolyte and shows up as a voltage that is too low
        at every rate.
        """
        return self.electrolyte_resistance + self.parameters.contact_resistance

    # ------------------------------------------------------------------- state

    def initial_state(self, soc: float, temperature: float | None = None) -> np.ndarray:
        return np.concatenate(
            [
                self.ss_neg.initial_state(float(self.parameters.negative.concentration(soc))),
                self.ss_pos.initial_state(float(self.parameters.positive.concentration(soc))),
                self.ss_electrolyte.initial_state(float(self.parameters.electrolyte_concentration)),
            ]
        )

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        return x[: self._n_neg], x[self._n_neg : self._i_elec], x[self._i_elec :]

    def step(self, x: np.ndarray, current: float) -> np.ndarray:
        xn, xp, xe = self._split(x)
        current = float(current)
        return np.concatenate(
            [
                self.ss_neg.step(xn, self._flux_neg * current),
                self.ss_pos.step(xp, self._flux_pos * current),
                self.ss_electrolyte.step(xe, current),
            ]
        )

    def electrolyte_concentrations(self, x: np.ndarray) -> tuple[float, float]:
        """``(negative, positive)`` coating-average salt concentration, mol m-3."""
        _, _, xe = self._split(x)
        return self.ss_electrolyte.averages(xe)

    # ----------------------------------------------------------------- outputs

    def _exchange_current(self, c_surf: float, c_e: float, side: str) -> float:
        electrode = self.parameters._electrode(side)
        rate = self.rate_neg if side.startswith("n") else self.rate_pos
        margin = 1e-6 * electrode.max_concentration
        c = min(max(c_surf, margin), electrode.max_concentration - margin)
        # Local salt concentration, floored well below anything physical so a
        # transient excursion produces a large overpotential rather than a NaN.
        ce = max(c_e, 1e-3 * self.parameters.electrolyte_concentration)
        return rate * np.sqrt(ce) * np.sqrt(c) * np.sqrt(electrode.max_concentration - c)

    def _terms(self, x: np.ndarray, current: float) -> dict[str, float]:
        xn, xp, xe = self._split(x)
        current = float(current)
        cs_n, cb_n = self.ss_neg.outputs(xn, self._flux_neg * current)
        cs_p, cb_p = self.ss_pos.outputs(xp, self._flux_pos * current)
        ce_n, ce_p = self.ss_electrolyte.averages(xe)

        neg, pos = self.parameters.negative, self.parameters.positive
        x_n = cs_n / neg.max_concentration
        x_p = cs_p / pos.max_concentration

        i0_n = self._exchange_current(cs_n, ce_n, "negative")
        i0_p = self._exchange_current(cs_p, ce_p, "positive")
        eta_n = self._kinetic_prefactor * float(np.arcsinh(self._j_neg * current / (2.0 * i0_n)))
        eta_p = self._kinetic_prefactor * float(np.arcsinh(self._j_pos * current / (2.0 * i0_p)))
        floor = 1e-3 * self.parameters.electrolyte_concentration
        eta_c = self._concentration_prefactor * float(np.log(max(ce_p, floor) / max(ce_n, floor)))
        ohmic = self.series_resistance * current
        voltage = float(pos.ocp(x_p)) - float(neg.ocp(x_n)) + eta_p - eta_n + eta_c - ohmic
        span = neg.stoich_at_100_soc - neg.stoich_at_0_soc
        return {
            "c_surf_neg": cs_n,
            "c_surf_pos": cs_p,
            "x_n": x_n,
            "x_p": x_p,
            "ce_n": ce_n,
            "ce_p": ce_p,
            "eta_n": eta_n,
            "eta_p": eta_p,
            "eta_c": eta_c,
            "ohmic": ohmic,
            "voltage": voltage,
            "soc": (cb_n / neg.max_concentration - neg.stoich_at_0_soc) / span,
        }

    def outputs(self, x: np.ndarray, current: float) -> ModelOutputs:
        terms = self._terms(x, current)
        return ModelOutputs(
            voltage=terms["voltage"],
            soc=terms["soc"],
            temperature=self.temperature,
            surface_stoichiometry=(terms["x_n"], terms["x_p"]),
            overpotential=(terms["eta_n"], terms["eta_p"]),
        )

    def depletion(self, x: np.ndarray) -> float:
        """Lowest coating-average salt concentration, as a fraction of nominal.

        A validity gauge, and worth watching. The transport model here is linear,
        so nothing stops it predicting a concentration below zero if the current
        is high enough for long enough -- and it will, because the real mechanism
        that prevents that is conductivity and diffusivity collapsing as the salt
        runs out, which a constant-property model does not have.

        The thresholds :meth:`validity` applies to it are calibrated against a
        full Doyle-Fuller-Newman solution rather than guessed. On the reference
        cell, discharging until the salt profile settles:

        ===========  ==========  ==========================
        rate         depletion   error against DFN
        ===========  ==========  ==========================
        0.5C         0.81        3.2 mV
        1.0C         0.62        6.7 mV
        2.0C         0.23        14.7 mV
        3.0C         below zero  141 mV
        ===========  ==========  ==========================

        So the model holds up considerably further into depletion than intuition
        suggests, and then fails abruptly once the linear extrapolation drives a
        coating concentration through zero -- which is the point at which it has
        stopped representing anything. An earlier version of this docstring put
        the boundary at 0.6 on the strength of a guess; the measurement moved it.
        """
        ce_n, ce_p = self.electrolyte_concentrations(x)
        return float(min(ce_n, ce_p) / self.parameters.electrolyte_concentration)

    def validity(self, x: np.ndarray) -> str:
        """``"good"``, ``"degraded"`` or ``"extrapolating"`` from :meth:`depletion`.

        Boundaries at 0.2 and 0.05, from the comparison tabulated in
        :meth:`depletion`. Read ``"good"`` as agreeing with a full solution to
        better than about 15 mV, ``"degraded"`` as directionally right and
        progressively optimistic, and ``"extrapolating"`` as approaching or past
        the concentration going negative, where the voltage means nothing.
        """
        fraction = self.depletion(x)
        if fraction >= 0.2:
            return "good"
        if fraction >= 0.05:
            return "degraded"
        return "extrapolating"

    def decompose(self, x: np.ndarray, current: float) -> dict[str, float]:
        """Break the terminal voltage into its physical contributions.

        The electrolyte terms are reported separately from the kinetic ones,
        which is the diagnostic reason to run this model rather than a single
        particle model with a fitted resistance: it distinguishes a cell that is
        kinetically limited from one that is running out of salt, and those call
        for different remedies.
        """
        terms = self._terms(x, current)
        return {
            "voltage": terms["voltage"],
            "kinetic_overpotential": terms["eta_p"] - terms["eta_n"],
            "concentration_overpotential": terms["eta_c"],
            "electrolyte_ohmic": -self.electrolyte_resistance * float(current),
            "contact_ohmic": -self.parameters.contact_resistance * float(current),
            "electrolyte_negative": terms["ce_n"],
            "electrolyte_positive": terms["ce_p"],
            "electrolyte_ratio": terms["ce_p"] / terms["ce_n"],
            "depletion": self.depletion(x),
            "validity": self.validity(x),
        }

    # --------------------------------------------------------------- Jacobians

    def state_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Block diagonal and constant: every transport process here is linear.

        Solid diffusion is linear in flux, salt transport is linear in current,
        and neither depends on the other's state. The coupling between them is
        entirely in the measurement, so this model keeps the property that makes
        an extended Kalman filter well behaved on it -- no linearisation error in
        the prediction step.
        """
        n = self.n_states
        jac = np.zeros((n, n))
        jac[: self._n_neg, : self._n_neg] = self.ss_neg.A
        jac[self._n_neg : self._i_elec, self._n_neg : self._i_elec] = self.ss_pos.A
        jac[self._i_elec :, self._i_elec :] = self.ss_electrolyte.A
        return jac

    def _d_overpotential_d_solid(self, c_surf: float, c_e: float, j: float, side: str) -> float:
        electrode = self.parameters._electrode(side)
        c_max = electrode.max_concentration
        margin = 1e-6 * c_max
        c = min(max(c_surf, margin), c_max - margin)
        i0 = self._exchange_current(c, c_e, side)
        u = j / (2.0 * i0)
        d_log_i0 = (c_max - 2.0 * c) / (2.0 * c * (c_max - c))
        return self._kinetic_prefactor * (-u * d_log_i0) / np.sqrt(1.0 + u * u)

    def _d_overpotential_d_salt(self, c_surf: float, c_e: float, j: float, side: str) -> float:
        """``d(eta)/d(c_e)`` through the exchange current only.

        The exchange current goes as the square root of salt concentration, so
        ``d ln i0 / d c_e = 1 / 2 c_e``. Depleting the electrolyte therefore
        raises the magnitude of the overpotential, which is the mechanism by
        which the electrolyte limits high-rate performance beyond its ohmic
        contribution.

        Returns zero where the concentration floor is active, because there the
        voltage genuinely stops depending on the state. Reporting the unclamped
        slope instead hands a filter a gradient the measurement function does not
        have, and a Jacobian that disagrees with its own output is worse than a
        crude one because nothing downstream can detect it.
        """
        floor = 1e-3 * self.parameters.electrolyte_concentration
        if c_e <= floor:
            return 0.0
        i0 = self._exchange_current(c_surf, c_e, side)
        u = j / (2.0 * i0)
        return self._kinetic_prefactor * (-u * 0.5 / c_e) / np.sqrt(1.0 + u * u)

    def voltage_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        current = float(current)
        xn, xp, xe = self._split(x)
        cs_n, _ = self.ss_neg.outputs(xn, self._flux_neg * current)
        cs_p, _ = self.ss_pos.outputs(xp, self._flux_pos * current)
        ce_n, ce_p = self.ss_electrolyte.averages(xe)
        neg, pos = self.parameters.negative, self.parameters.positive
        j_n = self._j_neg * current
        j_p = self._j_pos * current

        dv_dcn = (
            -float(derivative_of(neg.ocp)(cs_n / neg.max_concentration)) / neg.max_concentration
        )
        dv_dcn -= self._d_overpotential_d_solid(cs_n, ce_n, j_n, "negative")
        dv_dcp = float(derivative_of(pos.ocp)(cs_p / pos.max_concentration)) / pos.max_concentration
        dv_dcp += self._d_overpotential_d_solid(cs_p, ce_p, j_p, "positive")

        grad = np.zeros(self.n_states)
        grad[: self._n_neg] = dv_dcn * self.ss_neg.C[0]
        grad[self._n_neg : self._i_elec] = dv_dcp * self.ss_pos.C[0]

        # Salt affects voltage twice: through each exchange current, and through
        # the logarithm of the concentration ratio. Both contributions vanish
        # where the concentration floor is active, so that the gradient stays
        # consistent with the voltage the model actually reports.
        floor = 1e-3 * self.parameters.electrolyte_concentration
        dv_dce_n = -self._d_overpotential_d_salt(cs_n, ce_n, j_n, "negative")
        if ce_n > floor:
            dv_dce_n -= self._concentration_prefactor / ce_n
        dv_dce_p = self._d_overpotential_d_salt(cs_p, ce_p, j_p, "positive")
        if ce_p > floor:
            dv_dce_p += self._concentration_prefactor / ce_p

        grad[self._i_elec :] = (
            dv_dce_n * self.ss_electrolyte.C[0] + dv_dce_p * self.ss_electrolyte.C[1]
        )
        return grad

    def soc_jacobian(self) -> np.ndarray:
        """Gradient of reported state of charge with respect to the state."""
        neg = self.parameters.negative
        span = neg.stoich_at_100_soc - neg.stoich_at_0_soc
        grad = np.zeros(self.n_states)
        grad[: self._n_neg] = self.ss_neg.C[1] / (neg.max_concentration * span)
        return grad
