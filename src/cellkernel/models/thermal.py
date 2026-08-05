"""Coupled electro-thermal single particle model."""

from __future__ import annotations

import numpy as np

from ..ocp import derivative_of
from ..params import FARADAY, GAS_CONSTANT, CellParameters
from ..rom import make_rom
from ..rom.schedule import ScheduledStateSpace, schedule_over_temperature
from .base import CellModel, ModelOutputs

__all__ = ["ThermalSPM"]


class ThermalSPM(CellModel):
    """Single particle model with cell temperature as a state.

    Extends :class:`~cellkernel.models.spm.SPM` with a lumped thermal node. The
    state vector is the diffusion coordinates followed by temperature in kelvin.

    Heat generation follows Bernardi's decomposition,

    .. math::

        Q = I\\,(U - V) - I\\,T\\,\\frac{dU}{dT},

    where the first term is the irreversible dissipation -- identically
    :math:`I(\\eta_n - \\eta_p) + I^{2}R_c`, so non-negative for either sign of
    current, as it must be -- and the second is reversible entropic heat, which
    changes sign with the current and can cool a cell on charge.

    The thermal node is integrated exactly rather than by forward Euler. For heat
    generation held constant across a step,

    .. math::

        T_{k+1} = T_\\infty + \\frac{Q}{hA}
                  + \\left(T_k - T_\\infty - \\frac{Q}{hA}\\right) e^{-\\Delta t/\\tau},
        \\qquad \\tau = \\frac{C}{hA},

    which is unconditionally stable and exact at any step. Forward Euler on the
    same node is stable only for :math:`\\Delta t < 2\\tau`; that is rarely
    binding for a cell, whose time constant runs to hundreds of seconds, but it
    costs nothing to avoid and it matters for a small cell under forced cooling.

    Temperature feedback
    --------------------
    Temperature enters the electrochemistry three ways: through the ``2RT/F``
    kinetic prefactor, through the Arrhenius reaction rate inside the exchange
    current, and through solid diffusivity, which reshapes the reduced-order
    matrices themselves. The first two are cheap scalar corrections evaluated
    online. The third is not -- it needs a matrix exponential -- so it is gain
    scheduled: matrices are precomputed across a temperature grid and blended.
    See :class:`~cellkernel.rom.schedule.ScheduledStateSpace`.

    What this costs
    ---------------
    The isothermal model has *exactly linear* state dynamics, which is its most
    useful structural property: the extended Kalman filter built on it has no
    linearisation error in the prediction step at all. Coupling temperature
    destroys that. Heat generation is quadratic in current and depends on the
    diffusion state through the overpotentials, and the transition matrix itself
    now depends on a state. The dynamics are genuinely nonlinear and the filter
    genuinely approximate.

    That is a real price, and it is worth being clear about when it is worth
    paying. Below about 1C in a temperate environment, a cell moves a few kelvin
    and an isothermal model calibrated at the right temperature is adequate.
    Under fast charge, in winter, or in a pack where cells self-heat by tens of
    kelvin, diffusivity moves by a factor of several and an isothermal model
    silently mispredicts surface concentration -- which is exactly the quantity
    fast charge is limited by. Use :class:`~cellkernel.models.spm.SPM` when you
    can and this when you cannot.

    Parameters
    ----------
    parameters
        Cell parameter set. Must carry a
        :class:`~cellkernel.params.ThermalParameters` in ``thermal``.
    dt
        Sample period in seconds.
    rom
        Diffusion reduced-order model family.
    order
        States per electrode.
    temperature_grid
        Temperatures in kelvin at which to precompute matrices. The default spans
        -20 to 60 C in 10 K steps, which keeps interpolation error below roughly
        0.1% for a typical graphite diffusivity; check yours with
        :meth:`scheduling_error`.
    ambient
        Ambient temperature in kelvin. Defaults to the value in the thermal
        parameters.
    """

    def __init__(
        self,
        parameters: CellParameters,
        dt: float = 1.0,
        rom: str = "pade",
        order: int = 3,
        temperature_grid=None,
        ambient: float | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if parameters.thermal is None:
            raise ValueError(
                "ThermalSPM needs parameters.thermal; the built-in parameter sets "
                "provide it, or construct a ThermalParameters yourself"
            )
        self.parameters = parameters
        self.dt = float(dt)
        self.thermal = parameters.thermal
        self.ambient = float(ambient if ambient is not None else self.thermal.ambient_temperature)
        if temperature_grid is None:
            temperature_grid = np.arange(253.15, 333.16, 10.0)
        grid = np.asarray(temperature_grid, dtype=float).reshape(-1)

        reference = parameters.reference_temperature
        self._rom_spec = (rom, order)

        def build(side: str):
            electrode = parameters._electrode(side)

            def at(temperature: float):
                return make_rom(
                    rom,
                    electrode.particle_radius,
                    electrode.diffusivity_at(temperature, reference),
                    order=order,
                )

            return at

        self._build_negative = build("negative")
        self._build_positive = build("positive")
        self.schedule_negative: ScheduledStateSpace = schedule_over_temperature(
            self._build_negative,
            grid,
            self.dt,
            activation_energy=parameters.negative.diffusion_activation_energy,
            reference_temperature=reference,
        )
        self.schedule_positive: ScheduledStateSpace = schedule_over_temperature(
            self._build_positive,
            grid,
            self.dt,
            activation_energy=parameters.positive.diffusion_activation_energy,
            reference_temperature=reference,
        )

        self._n_neg = self.schedule_negative.n_states
        self._n_pos = self.schedule_positive.n_states
        self._i_temp = self._n_neg + self._n_pos

        self._flux_neg = -parameters.flux_scale("negative")
        self._flux_pos = +parameters.flux_scale("positive")
        self._j_neg = +parameters.interfacial_current_scale("negative")
        self._j_pos = -parameters.interfacial_current_scale("positive")
        self._reference = reference

        # Cell-level entropic coefficient, dU/dT in V K-1.
        self.entropic_coefficient = float(
            parameters.positive.entropic_coefficient - parameters.negative.entropic_coefficient
        )
        self._conductance = self.thermal.heat_transfer_coefficient * self.thermal.surface_area
        self._decay = float(np.exp(-self.dt / self.thermal.time_constant))

    # ------------------------------------------------------------------- shape

    @property
    def n_states(self) -> int:
        return self._n_neg + self._n_pos + 1

    @property
    def state_names(self) -> tuple[str, ...]:
        return (
            tuple(f"neg_{i}" for i in range(self._n_neg))
            + tuple(f"pos_{i}" for i in range(self._n_pos))
            + ("temperature",)
        )

    @property
    def temperature_index(self) -> int:
        """Position of the temperature state in the state vector."""
        return self._i_temp

    def scheduling_error(self, samples: int = 41) -> dict[str, float]:
        """Interpolation error of the temperature schedule, per electrode."""
        negative = self.schedule_negative.interpolation_error(self._build_negative, samples)
        positive = self.schedule_positive.interpolation_error(self._build_positive, samples)
        return {
            "negative_A": negative["max_relative_A"],
            "negative_B": negative["max_relative_B"],
            "positive_A": positive["max_relative_A"],
            "positive_B": positive["max_relative_B"],
        }

    # ------------------------------------------------------------------- state

    def initial_state(self, soc: float, temperature: float | None = None) -> np.ndarray:
        temp = float(temperature if temperature is not None else self.ambient)
        neg = self.schedule_negative.at(temp)
        pos = self.schedule_positive.at(temp)
        return np.concatenate(
            [
                neg.initial_state(float(self.parameters.negative.concentration(soc))),
                pos.initial_state(float(self.parameters.positive.concentration(soc))),
                [temp],
            ]
        )

    def _split(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        z = np.asarray(z, dtype=float).reshape(-1)
        return z[: self._n_neg], z[self._n_neg : self._i_temp], float(z[self._i_temp])

    def temperature(self, z: np.ndarray) -> float:
        """Cell temperature in kelvin held in the state vector."""
        return float(np.asarray(z, dtype=float).reshape(-1)[self._i_temp])

    # --------------------------------------------------------------- chemistry

    def _concentrations(self, z: np.ndarray, current: float) -> tuple[float, float, float, float]:
        xn, xp, temp = self._split(z)
        cs_n, cb_n = self.schedule_negative.at(temp).outputs(xn, self._flux_neg * current)
        cs_p, cb_p = self.schedule_positive.at(temp).outputs(xp, self._flux_pos * current)
        return cs_n, cb_n, cs_p, cb_p

    def _exchange_current(self, c_surf: float, side: str, temp: float) -> float:
        electrode = self.parameters._electrode(side)
        rate = electrode.reaction_rate_at(temp, self._reference)
        margin = 1e-6 * electrode.max_concentration
        c = min(max(c_surf, margin), electrode.max_concentration - margin)
        return (
            rate
            * np.sqrt(self.parameters.electrolyte_concentration)
            * np.sqrt(c)
            * np.sqrt(electrode.max_concentration - c)
        )

    def _overpotential(self, c_surf: float, j: float, side: str, temp: float) -> float:
        i0 = self._exchange_current(c_surf, side, temp)
        return (2.0 * GAS_CONSTANT * temp / FARADAY) * float(np.arcsinh(j / (2.0 * i0)))

    def _terms(self, z: np.ndarray, current: float) -> dict[str, float]:
        """Everything voltage and heat generation both need, computed once."""
        cs_n, cb_n, cs_p, cb_p = self._concentrations(z, current)
        temp = self.temperature(z)
        neg, pos = self.parameters.negative, self.parameters.positive
        x_n = cs_n / neg.max_concentration
        x_p = cs_p / pos.max_concentration
        eta_n = self._overpotential(cs_n, self._j_neg * current, "negative", temp)
        eta_p = self._overpotential(cs_p, self._j_pos * current, "positive", temp)
        u_surface = float(pos.ocp(x_p)) - float(neg.ocp(x_n))
        voltage = u_surface + eta_p - eta_n - current * self.parameters.contact_resistance
        span = neg.stoich_at_100_soc - neg.stoich_at_0_soc
        return {
            "c_surf_neg": cs_n,
            "c_surf_pos": cs_p,
            "x_n": x_n,
            "x_p": x_p,
            "eta_n": eta_n,
            "eta_p": eta_p,
            "u_surface": u_surface,
            "voltage": voltage,
            "temperature": temp,
            "soc": (cb_n / neg.max_concentration - neg.stoich_at_0_soc) / span,
        }

    def heat_generation(self, z: np.ndarray, current: float) -> dict[str, float]:
        """Irreversible, reversible and total heat generation in watts.

        The irreversible term is written as ``I(U - V)`` rather than as
        ``I^2 R``: they agree, but the first form stays correct when the
        overpotentials are nonlinear, which Butler-Volmer kinetics are. Writing
        it as a resistance would need that resistance to be current-dependent,
        which is how equivalent-circuit thermal models end up needing a lookup
        table for something the physics already supplies.
        """
        terms = self._terms(z, current)
        irreversible = current * (terms["u_surface"] - terms["voltage"])
        reversible = -current * terms["temperature"] * self.entropic_coefficient
        return {
            "irreversible": irreversible,
            "reversible": reversible,
            "total": irreversible + reversible,
        }

    # ------------------------------------------------------------------ update

    def step(self, z: np.ndarray, current: float) -> np.ndarray:
        xn, xp, temp = self._split(z)
        current = float(current)
        neg = self.schedule_negative.at(temp)
        pos = self.schedule_positive.at(temp)
        heat = self.heat_generation(z, current)["total"]
        rise = heat / self._conductance
        temp_next = self.ambient + rise + (temp - self.ambient - rise) * self._decay
        return np.concatenate(
            [
                neg.step(xn, self._flux_neg * current),
                pos.step(xp, self._flux_pos * current),
                [temp_next],
            ]
        )

    def outputs(self, z: np.ndarray, current: float) -> ModelOutputs:
        terms = self._terms(z, float(current))
        return ModelOutputs(
            voltage=terms["voltage"],
            soc=terms["soc"],
            temperature=terms["temperature"],
            surface_stoichiometry=(terms["x_n"], terms["x_p"]),
            overpotential=(terms["eta_n"], terms["eta_p"]),
        )

    # --------------------------------------------------------------- Jacobians

    def _d_overpotential_d_concentration(
        self, c_surf: float, j: float, side: str, temp: float
    ) -> float:
        electrode = self.parameters._electrode(side)
        c_max = electrode.max_concentration
        margin = 1e-6 * c_max
        c = min(max(c_surf, margin), c_max - margin)
        i0 = self._exchange_current(c, side, temp)
        u = j / (2.0 * i0)
        d_log_i0 = (c_max - 2.0 * c) / (2.0 * c * (c_max - c))
        prefactor = 2.0 * GAS_CONSTANT * temp / FARADAY
        return prefactor * (-u * d_log_i0) / np.sqrt(1.0 + u * u)

    def _d_overpotential_d_temperature(
        self, c_surf: float, j: float, side: str, temp: float
    ) -> float:
        """``d(eta)/dT`` at fixed surface concentration.

        Two channels. The Butler-Volmer prefactor ``2RT/F`` is linear in
        temperature, contributing ``eta / T``. The exchange current is Arrhenius,
        so warming the cell raises ``i0`` and shrinks the overpotential; with
        ``d ln i0 / dT = E_a / (R_g T^2)`` that contributes the second term.
        The two have opposite signs and the Arrhenius channel dominates for any
        realistic activation energy.
        """
        electrode = self.parameters._electrode(side)
        c_max = electrode.max_concentration
        margin = 1e-6 * c_max
        c = min(max(c_surf, margin), c_max - margin)
        i0 = self._exchange_current(c, side, temp)
        u = j / (2.0 * i0)
        prefactor = 2.0 * GAS_CONSTANT * temp / FARADAY
        d_log_i0 = electrode.reaction_activation_energy / (GAS_CONSTANT * temp * temp)
        from_prefactor = (2.0 * GAS_CONSTANT / FARADAY) * float(np.arcsinh(u))
        from_kinetics = prefactor * (-u * d_log_i0) / np.sqrt(1.0 + u * u)
        return from_prefactor + from_kinetics

    def voltage_jacobian(self, z: np.ndarray, current: float) -> np.ndarray:
        current = float(current)
        xn, xp, temp = self._split(z)
        cs_n, _, cs_p, _ = self._concentrations(z, current)
        neg, pos = self.parameters.negative, self.parameters.positive
        j_n = self._j_neg * current
        j_p = self._j_pos * current

        dv_dcn = (
            -float(derivative_of(neg.ocp)(cs_n / neg.max_concentration)) / neg.max_concentration
        )
        dv_dcn -= self._d_overpotential_d_concentration(cs_n, j_n, "negative", temp)
        dv_dcp = float(derivative_of(pos.ocp)(cs_p / pos.max_concentration)) / pos.max_concentration
        dv_dcp += self._d_overpotential_d_concentration(cs_p, j_p, "positive", temp)

        neg_ss = self.schedule_negative.at(temp)
        pos_ss = self.schedule_positive.at(temp)
        grad = np.zeros(self.n_states)
        grad[: self._n_neg] = dv_dcn * neg_ss.C[0]
        grad[self._n_neg : self._i_temp] = dv_dcp * pos_ss.C[0]

        # Temperature moves the voltage through the kinetics directly, and
        # indirectly by reshaping the output map that produces surface
        # concentration. Both are included; omitting the second is a common
        # shortcut and it under-predicts the sensitivity at low temperature,
        # where the schedule varies fastest.
        d_eta_n = self._d_overpotential_d_temperature(cs_n, j_n, "negative", temp)
        d_eta_p = self._d_overpotential_d_temperature(cs_p, j_p, "positive", temp)
        dcn_dt, dcp_dt = self._d_surface_concentration_d_temperature(z, current)
        grad[self._i_temp] = -d_eta_n + d_eta_p + dv_dcn * dcn_dt + dv_dcp * dcp_dt
        return grad

    def _d_surface_concentration_d_temperature(
        self, z: np.ndarray, current: float
    ) -> tuple[float, float]:
        """Sensitivity of each surface concentration to the scheduled matrices."""
        xn, xp, temp = self._split(z)
        out = []
        for schedule, state, flux in (
            (self.schedule_negative, xn, self._flux_neg * current),
            (self.schedule_positive, xp, self._flux_pos * current),
        ):
            dc, dd = schedule.output_slope(temp)
            out.append(float(dc @ state + dd * flux))
        return out[0], out[1]

    def state_jacobian(self, z: np.ndarray, current: float) -> np.ndarray:
        current = float(current)
        xn, xp, temp = self._split(z)
        n = self.n_states
        jac = np.zeros((n, n))

        neg_ss = self.schedule_negative.at(temp)
        pos_ss = self.schedule_positive.at(temp)
        jac[: self._n_neg, : self._n_neg] = neg_ss.A
        jac[self._n_neg : self._i_temp, self._n_neg : self._i_temp] = pos_ss.A

        # Diffusion states respond to temperature through the schedule slope.
        dA_neg, dB_neg = self.schedule_negative.slope(temp)
        dA_pos, dB_pos = self.schedule_positive.slope(temp)
        jac[: self._n_neg, self._i_temp] = dA_neg @ xn + dB_neg.reshape(-1) * (
            self._flux_neg * current
        )
        jac[self._n_neg : self._i_temp, self._i_temp] = dA_pos @ xp + dB_pos.reshape(-1) * (
            self._flux_pos * current
        )

        # Temperature row: T_next = ambient + Q/(hA) + (T - ambient - Q/(hA)) * decay,
        # so dT_next/dy = (1 - decay)/(hA) * dQ/dy for any state y other than T,
        # and the temperature entry additionally carries the decay itself.
        gain = (1.0 - self._decay) / self._conductance
        dq = self._heat_jacobian(z, current)
        jac[self._i_temp, :] = gain * dq
        jac[self._i_temp, self._i_temp] += self._decay
        return jac

    def _heat_jacobian(self, z: np.ndarray, current: float) -> np.ndarray:
        """Gradient of total heat generation with respect to the full state.

        ``Q = I(eta_n - eta_p) + I^2 R_c - I T dU/dT``, having substituted
        ``U - V = eta_n - eta_p + I R_c``. Note that the open-circuit potential
        cancels out of the irreversible term entirely, so heat generation does
        not depend on the state through the potentials -- only through the
        overpotentials. That is why this gradient is short.
        """
        current = float(current)
        xn, xp, temp = self._split(z)
        cs_n, _, cs_p, _ = self._concentrations(z, current)
        j_n = self._j_neg * current
        j_p = self._j_pos * current

        deta_n_dc = self._d_overpotential_d_concentration(cs_n, j_n, "negative", temp)
        deta_p_dc = self._d_overpotential_d_concentration(cs_p, j_p, "positive", temp)
        neg_ss = self.schedule_negative.at(temp)
        pos_ss = self.schedule_positive.at(temp)

        grad = np.zeros(self.n_states)
        grad[: self._n_neg] = current * deta_n_dc * neg_ss.C[0]
        grad[self._n_neg : self._i_temp] = -current * deta_p_dc * pos_ss.C[0]

        deta_n_dt = self._d_overpotential_d_temperature(cs_n, j_n, "negative", temp)
        deta_p_dt = self._d_overpotential_d_temperature(cs_p, j_p, "positive", temp)
        dcn_dt, dcp_dt = self._d_surface_concentration_d_temperature(z, current)
        grad[self._i_temp] = (
            current * (deta_n_dt - deta_p_dt)
            + current * (deta_n_dc * dcn_dt - deta_p_dc * dcp_dt)
            - current * self.entropic_coefficient
        )
        return grad
