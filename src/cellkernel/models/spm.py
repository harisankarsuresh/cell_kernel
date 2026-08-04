"""Single particle model of a lithium-ion cell."""

from __future__ import annotations

import numpy as np

from ..params import FARADAY, GAS_CONSTANT, CellParameters
from ..rom import DiffusionROM, DiscreteStateSpace, make_rom
from .base import CellModel, ModelOutputs

__all__ = ["SPM"]


class SPM(CellModel):
    """Single particle model with Butler-Volmer kinetics.

    Each electrode is represented by one spherical particle whose solid
    diffusion is handled by a reduced-order model from :mod:`cellkernel.rom`.
    Terminal voltage is

    .. math::

        V = U_p(x_p) + \\eta_p - U_n(x_n) - \\eta_n - I R_c,

    with surface stoichiometries :math:`x_k = c_{s,k} / c_{k,\\max}` and
    symmetric Butler-Volmer overpotentials

    .. math::

        \\eta_k = \\frac{2 R T}{F} \\operatorname{asinh}
                  \\left( \\frac{j_k}{2 i_{0,k}} \\right),
        \\qquad
        i_{0,k} = m_k \\sqrt{c_e}\\sqrt{c_{s,k}}\\sqrt{c_{k,\\max} - c_{s,k}} .

    Note that the exchange-current prefactor :math:`m_k` carries the units
    :math:`(\\mathrm{A\\,m^{-2}})(\\mathrm{m^{3}\\,mol^{-1}})^{3/2}` and already
    absorbs the Faraday constant; there is no separate :math:`F` in
    :math:`i_0`. This is the convention used by the parameter values reported in
    the literature and by PyBaMM. Inserting an extra :math:`F` inflates
    :math:`i_0` by five orders of magnitude and collapses the kinetic
    overpotential to microvolts, which is a silent failure -- the model still
    runs and still looks broadly reasonable, it just has no charge-transfer
    resistance at all.

    The ``asinh`` form is the exact inverse of the symmetric Butler-Volmer
    equation, not the linearised or Tafel approximation. It costs no more to
    evaluate and stays valid from micro-amp rest currents through to several C,
    where the linear form is already tens of millivolts wrong.

    **The state dynamics are exactly linear.** Both reduced-order models are
    linear systems driven by molar flux, which is proportional to current, so the
    state update is a constant matrix-vector product. All nonlinearity lives in
    the voltage measurement. An extended Kalman filter built on this model
    therefore has *no linearisation error in its prediction step at all*; only
    the measurement Jacobian is approximate. This is a structural advantage over
    equivalent-circuit formulations that put state of charge inside a nonlinear
    coulomb-counting term, and it is why the covariance propagation stays well
    behaved over long runs.

    Parameters
    ----------
    parameters
        Cell parameter set.
    dt
        Sample period in seconds.
    rom
        Reduced-order model family, one of ``"pade"``, ``"spectral"``, ``"fv"``,
        ``"poly"``, or a pair of pre-built :class:`~cellkernel.rom.DiffusionROM`
        instances for the negative and positive electrodes.
    order
        Number of states per electrode when ``rom`` is given by name.
    temperature
        Isothermal operating temperature in kelvin.

    Notes
    -----
    The model is isothermal by construction. Temperature enters through the
    ``2RT/F`` kinetic prefactor, through Arrhenius corrections to diffusivity and
    reaction rate, and through the reduced-order matrices, which are rebuilt
    because diffusivity changes. Use :meth:`at_temperature` to obtain a model for
    a different temperature and gain-schedule between them; that is how a
    production estimator handles a wide temperature range, since it moves all
    the matrix exponentials offline.

    One reduced-order family makes online rescheduling genuinely cheap:
    :class:`~cellkernel.rom.SpectralDiffusion` has a diagonal state matrix whose
    entries are :math:`\\exp(-\\lambda_k^{2} D \\Delta t / R^{2})`, so a change in
    diffusivity costs one exponential per mode rather than a matrix exponential.
    """

    def __init__(
        self,
        parameters: CellParameters,
        dt: float = 1.0,
        rom: str | tuple[DiffusionROM, DiffusionROM] = "pade",
        order: int = 3,
        temperature: float | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.parameters = parameters
        self.dt = float(dt)
        self.temperature = float(
            temperature if temperature is not None else parameters.reference_temperature
        )
        self._rom_spec = (rom, order)

        t_ref = parameters.reference_temperature
        d_neg = parameters.negative.diffusivity_at(self.temperature, t_ref)
        d_pos = parameters.positive.diffusivity_at(self.temperature, t_ref)
        self.rate_neg = parameters.negative.reaction_rate_at(self.temperature, t_ref)
        self.rate_pos = parameters.positive.reaction_rate_at(self.temperature, t_ref)

        if isinstance(rom, str):
            self.rom_neg: DiffusionROM = make_rom(
                rom, parameters.negative.particle_radius, d_neg, order=order
            )
            self.rom_pos: DiffusionROM = make_rom(
                rom, parameters.positive.particle_radius, d_pos, order=order
            )
        else:
            self.rom_neg, self.rom_pos = rom

        self.ss_neg: DiscreteStateSpace = self.rom_neg.discretise(self.dt)
        self.ss_pos: DiscreteStateSpace = self.rom_pos.discretise(self.dt)

        # Molar influx per ampere. Discharge removes lithium from the negative
        # particle and inserts it into the positive one, hence the sign split.
        self._flux_neg = -parameters.flux_scale("negative")
        self._flux_pos = +parameters.flux_scale("positive")
        # Interfacial current density per ampere, with the opposite sign split.
        self._j_neg = +parameters.interfacial_current_scale("negative")
        self._j_pos = -parameters.interfacial_current_scale("positive")

        self._n_neg = self.ss_neg.n_states
        self._n_pos = self.ss_pos.n_states
        self._kinetic_prefactor = 2.0 * GAS_CONSTANT * self.temperature / FARADAY

    # ------------------------------------------------------------------- shape

    @property
    def n_states(self) -> int:
        return self._n_neg + self._n_pos

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(
            [f"neg_{i}" for i in range(self._n_neg)]
            + [f"pos_{i}" for i in range(self._n_pos)]
        )

    def at_temperature(self, temperature: float) -> SPM:
        """A model of the same cell rebuilt for a different temperature."""
        rom, order = self._rom_spec
        return SPM(
            self.parameters,
            dt=self.dt,
            rom=rom if isinstance(rom, str) else (self.rom_neg, self.rom_pos),
            order=order,
            temperature=temperature,
        )

    # ------------------------------------------------------------------- state

    def initial_state(self, soc: float, temperature: float | None = None) -> np.ndarray:
        """Uniformly loaded particles corresponding to a rested cell at ``soc``."""
        c_neg = float(self.parameters.negative.concentration(soc))
        c_pos = float(self.parameters.positive.concentration(soc))
        return np.concatenate(
            [self.ss_neg.initial_state(c_neg), self.ss_pos.initial_state(c_pos)]
        )

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        return x[: self._n_neg], x[self._n_neg :]

    def step(self, x: np.ndarray, current: float) -> np.ndarray:
        xn, xp = self._split(x)
        current = float(current)
        return np.concatenate(
            [
                self.ss_neg.step(xn, self._flux_neg * current),
                self.ss_pos.step(xp, self._flux_pos * current),
            ]
        )

    # ----------------------------------------------------------------- outputs

    def _concentrations(
        self, x: np.ndarray, current: float
    ) -> tuple[float, float, float, float]:
        """Return ``(c_surf_neg, c_bar_neg, c_surf_pos, c_bar_pos)``."""
        xn, xp = self._split(x)
        cs_n, cb_n = self.ss_neg.outputs(xn, self._flux_neg * current)
        cs_p, cb_p = self.ss_pos.outputs(xp, self._flux_pos * current)
        return cs_n, cb_n, cs_p, cb_p

    def _exchange_current(self, c_surf: float, electrode: str) -> float:
        el = self.parameters._electrode(electrode)
        rate = self.rate_neg if electrode.startswith("n") else self.rate_pos
        # Clamp inside the physical range: the square roots are undefined outside
        # it, and a filter transient can briefly push the estimate past either
        # end. Returning NaN there would poison the covariance permanently,
        # whereas a small positive exchange current merely produces a large
        # overpotential, which is the physically sensible limit anyway.
        margin = 1e-6 * el.max_concentration
        c = min(max(c_surf, margin), el.max_concentration - margin)
        return (
            rate
            * np.sqrt(self.parameters.electrolyte_concentration)
            * np.sqrt(c)
            * np.sqrt(el.max_concentration - c)
        )

    def _overpotential(self, c_surf: float, j: float, electrode: str) -> float:
        i0 = self._exchange_current(c_surf, electrode)
        return self._kinetic_prefactor * float(np.arcsinh(j / (2.0 * i0)))

    def outputs(self, x: np.ndarray, current: float) -> ModelOutputs:
        current = float(current)
        cs_n, cb_n, cs_p, cb_p = self._concentrations(x, current)
        neg, pos = self.parameters.negative, self.parameters.positive

        x_n = cs_n / neg.max_concentration
        x_p = cs_p / pos.max_concentration
        eta_n = self._overpotential(cs_n, self._j_neg * current, "negative")
        eta_p = self._overpotential(cs_p, self._j_pos * current, "positive")

        voltage = (
            float(pos.ocp(x_p))
            + eta_p
            - float(neg.ocp(x_n))
            - eta_n
            - current * self.parameters.contact_resistance
        )
        span = neg.stoich_at_100_soc - neg.stoich_at_0_soc
        soc = (cb_n / neg.max_concentration - neg.stoich_at_0_soc) / span
        return ModelOutputs(
            voltage=voltage,
            soc=float(soc),
            temperature=self.temperature,
            surface_stoichiometry=(x_n, x_p),
            overpotential=(eta_n, eta_p),
        )

    # --------------------------------------------------------------- Jacobians

    def state_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Exact and constant: the process model is linear.

        ``current`` and ``x`` are accepted for interface compatibility and are
        unused, which is the whole point -- there is nothing to linearise.
        """
        jac = np.zeros((self.n_states, self.n_states))
        jac[: self._n_neg, : self._n_neg] = self.ss_neg.A
        jac[self._n_neg :, self._n_neg :] = self.ss_pos.A
        return jac

    def _d_overpotential_d_concentration(
        self, c_surf: float, j: float, electrode: str
    ) -> float:
        """Analytic ``d(eta)/d(c_surf)``.

        With :math:`u = j / 2 i_0` and
        :math:`i_0 \\propto \\sqrt{c}\\sqrt{c_{\\max} - c}`,

        .. math::

            \\frac{d i_0}{dc} = i_0 \\frac{c_{\\max} - 2c}{2 c (c_{\\max} - c)},
            \\qquad
            \\frac{d\\eta}{dc} = \\frac{2RT}{F}
                \\frac{1}{\\sqrt{1 + u^{2}}}
                \\left( -u \\frac{c_{\\max} - 2c}{2 c (c_{\\max} - c)} \\right).

        The bracket vanishes at ``c = c_max/2``, where the exchange current is
        stationary, and diverges at both ends of the range, which is why the
        concentration is clamped away from them.
        """
        el = self.parameters._electrode(electrode)
        margin = 1e-6 * el.max_concentration
        c = min(max(c_surf, margin), el.max_concentration - margin)
        c_max = el.max_concentration
        i0 = self._exchange_current(c, electrode)
        u = j / (2.0 * i0)
        d_log_i0 = (c_max - 2.0 * c) / (2.0 * c * (c_max - c))
        return self._kinetic_prefactor * (-u * d_log_i0) / np.sqrt(1.0 + u * u)

    def voltage_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Analytic ``dV/dx`` via the chain rule through each electrode.

        Surface concentration is a linear functional of the state,
        ``c_surf = C[0] @ x_rom + D[0] * flux``, so the gradient with respect to
        the full state vector is the scalar sensitivity times that output row.
        """
        current = float(current)
        cs_n, _, cs_p, _ = self._concentrations(x, current)
        neg, pos = self.parameters.negative, self.parameters.positive

        j_n = self._j_neg * current
        j_p = self._j_pos * current
        dv_dcn = -float(neg.ocp_derivative(cs_n / neg.max_concentration)) / neg.max_concentration
        dv_dcn -= self._d_overpotential_d_concentration(cs_n, j_n, "negative")
        dv_dcp = float(pos.ocp_derivative(cs_p / pos.max_concentration)) / pos.max_concentration
        dv_dcp += self._d_overpotential_d_concentration(cs_p, j_p, "positive")

        grad = np.zeros(self.n_states)
        grad[: self._n_neg] = dv_dcn * self.ss_neg.C[0]
        grad[self._n_neg :] = dv_dcp * self.ss_pos.C[0]
        return grad

    def soc_jacobian(self) -> np.ndarray:
        """Gradient of reported state of charge with respect to the state.

        Constant, because bulk concentration is a fixed linear functional of the
        state and the stoichiometry window is fixed.
        """
        neg = self.parameters.negative
        span = neg.stoich_at_100_soc - neg.stoich_at_0_soc
        grad = np.zeros(self.n_states)
        grad[: self._n_neg] = self.ss_neg.C[1] / (neg.max_concentration * span)
        return grad
