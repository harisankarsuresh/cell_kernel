"""Code-generation spec for a temperature-scheduled estimator.

The generated estimator takes cell temperature as an **input**, not as a state.

That is a deliberate departure from :class:`~cellkernel.models.thermal.ThermalSPM`,
which carries temperature as a state and infers it from voltage. Both are useful,
for different reasons.

Any pack worth putting a physics-based estimator into has thermistors, so cell
temperature is a measurement rather than an unknown. Treating it as an input
rather than a state buys three things. The covariance stays the size it was, so
the arithmetic cost is unchanged -- a temperature state would add a row and
column to an ``n^2`` propagation. Heat generation disappears from the generated
code entirely, along with the thermal parameters it needs, which are the least
well identified numbers in any parameter set. And the Kalman structure becomes
identical to the isothermal case, so what is generated here is the same estimator
with temperature-dependent coefficients rather than a second, differently-shaped
one.

It is also the better estimate. Temperature is only weakly observable from
terminal voltage -- it acts through polarisation, indirectly and slowly -- so a
filter that infers it will do worse than a thermistor costing a few cents.
Estimating temperature is what you do when you cannot measure it, and on a real
pack you can.

What temperature changes
------------------------
Three things, and they are handled differently because they cost differently.

The Butler-Volmer prefactor ``2 R_g T / F`` is linear in temperature: evaluated
online, free.

The exchange current is Arrhenius in temperature: one exponential per electrode
per step.

Solid diffusivity is Arrhenius too, but it reaches the model through a matrix
exponential, which cannot be evaluated on the target. Those matrices are
precomputed across a temperature grid and blended online, on the Arrhenius factor
rather than on temperature -- see
:class:`~cellkernel.rom.schedule.ScheduledStateSpace` for why that distinction is
worth 197 mV at the cold end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ocp import OCPTable, tabulate

__all__ = [
    "ScheduledElectrodeSpec",
    "ThermalEstimatorSpec",
    "ThermalReferenceEstimator",
    "spec_from_thermal_spm",
]

GAS_CONSTANT = 8.31446261815324
FARADAY = 96485.33212


@dataclass(frozen=True)
class ScheduledElectrodeSpec:
    """Per-electrode constants, with the temperature-dependent ones as grids."""

    name: str
    n_states: int
    #: ``(n_temperatures, n_states, n_states)`` transition blocks.
    A_grid: np.ndarray
    #: ``(n_temperatures, n_states)`` input columns, molar flux already folded in.
    B_grid: np.ndarray
    #: ``(n_temperatures, n_states)`` surface-concentration output rows.
    c_surface_grid: np.ndarray
    #: ``(n_temperatures,)`` surface feedthrough terms.
    d_surface_grid: np.ndarray
    #: Bulk output row. Temperature independent for every model here, because the
    #: conserved functional does not depend on diffusivity.
    c_bulk: np.ndarray
    #: ``(n_temperatures,)`` Arrhenius factor of diffusivity at each grid point,
    #: precomputed so the online blend needs one exponential rather than three.
    diffusion_factor_grid: np.ndarray
    diffusion_activation_energy: float
    reaction_activation_energy: float
    flux_per_amp: float
    j_per_amp: float
    max_concentration: float
    #: Exchange-current prefactor at the reference temperature.
    exchange_prefactor: float
    ocp_table: OCPTable
    voltage_sign: float
    #: Maps a uniform particle concentration onto the reduced state vector.
    uniform_map: np.ndarray


@dataclass(frozen=True)
class ThermalEstimatorSpec:
    """Everything needed to generate a temperature-scheduled estimator."""

    name: str
    dt: float
    n_states: int
    temperature_grid: np.ndarray
    reference_temperature: float
    negative: ScheduledElectrodeSpec
    positive: ScheduledElectrodeSpec
    contact_resistance: float
    soc_scale: float
    soc_offset: float
    voltage_limits: tuple[float, float]
    process_noise: np.ndarray
    measurement_noise: float
    initial_covariance: np.ndarray
    sto_negative: tuple[float, float]
    sto_positive: tuple[float, float]
    provenance: str = ""

    @property
    def n_negative(self) -> int:
        return self.negative.n_states

    @property
    def n_positive(self) -> int:
        return self.positive.n_states

    @property
    def n_temperatures(self) -> int:
        return int(self.temperature_grid.size)

    # -------------------------------------------------------------- blending

    def blend(
        self, electrode: ScheduledElectrodeSpec, temperature: float
    ) -> tuple[int, int, float]:
        """Bracket index pair and blend weight, mirroring the generated C exactly."""
        grid = self.temperature_grid
        upper = int(np.searchsorted(grid, float(temperature), side="right"))
        upper = min(max(upper, 1), grid.size - 1)
        lower = upper - 1
        if electrode.diffusion_activation_energy == 0.0:
            weight = (float(temperature) - grid[lower]) / (grid[upper] - grid[lower])
        else:
            low = electrode.diffusion_factor_grid[lower]
            high = electrode.diffusion_factor_grid[upper]
            span = high - low
            weight = (
                0.0
                if abs(span) < 1e-300
                else (self.diffusion_factor(electrode, temperature) - low) / span
            )
        return lower, upper, float(min(max(weight, 0.0), 1.0))

    def diffusion_factor(self, electrode: ScheduledElectrodeSpec, temperature: float) -> float:
        if electrode.diffusion_activation_energy == 0.0:
            return 1.0
        return float(
            np.exp(
                electrode.diffusion_activation_energy
                / GAS_CONSTANT
                * (1.0 / self.reference_temperature - 1.0 / float(temperature))
            )
        )

    def reaction_factor(self, electrode: ScheduledElectrodeSpec, temperature: float) -> float:
        if electrode.reaction_activation_energy == 0.0:
            return 1.0
        return float(
            np.exp(
                electrode.reaction_activation_energy
                / GAS_CONSTANT
                * (1.0 / self.reference_temperature - 1.0 / float(temperature))
            )
        )

    def matrices(self, electrode: ScheduledElectrodeSpec, temperature: float):
        """Blended ``(A, B, c_surface, d_surface)`` at a temperature."""
        lower, upper, weight = self.blend(electrode, temperature)
        one = 1.0 - weight
        return (
            one * electrode.A_grid[lower] + weight * electrode.A_grid[upper],
            one * electrode.B_grid[lower] + weight * electrode.B_grid[upper],
            one * electrode.c_surface_grid[lower] + weight * electrode.c_surface_grid[upper],
            float(one * electrode.d_surface_grid[lower] + weight * electrode.d_surface_grid[upper]),
        )

    def initial_state(self, soc: float, temperature: float) -> np.ndarray:
        """State vector for a rested cell, mirroring the generated ``ck_init``."""
        state = np.zeros(self.n_states)
        s0n, s1n = self.sto_negative
        s0p, s1p = self.sto_positive
        c_neg = (s0n + soc * (s1n - s0n)) * self.negative.max_concentration
        c_pos = (s0p + soc * (s1p - s0p)) * self.positive.max_concentration
        state[: self.n_negative] = self.negative.uniform_map * c_neg
        state[self.n_negative :] = self.positive.uniform_map * c_pos
        return state


def scheduled_table_domain(
    model, side: str, max_c_rate: float = 3.0, safety: float = 1.5
) -> tuple[float, float]:
    """Stoichiometry range a potential table must cover, across the whole grid.

    Same reasoning as :func:`cellkernel.codegen.spec.table_domain` -- the table
    must span every stoichiometry the particle *surface* can reach, not just the
    bulk window, because a lookup table clamps rather than extrapolating -- with
    one addition that matters here.

    The steady-state surface excursion is :math:`(R/5D)|N|`, so it is largest
    where diffusivity is smallest, which is the coldest point on the schedule. A
    table sized at the reference temperature would be comfortably adequate at
    25 C and saturate at -20 C, where diffusivity is roughly a twelfth of its
    reference value and the excursion correspondingly twelve times larger. The
    domain is therefore computed at the coldest grid temperature.
    """
    electrode = model.parameters._electrode(side)
    schedule = model.schedule_negative if side.startswith("n") else model.schedule_positive
    coldest = float(schedule.temperatures[0])
    diffusivity = electrode.diffusivity_at(coldest, model.parameters.reference_temperature)
    flux_per_amp = model._flux_neg if side.startswith("n") else model._flux_pos
    current = max_c_rate * model.parameters.nominal_capacity
    offset = electrode.particle_radius / (5.0 * diffusivity) * abs(flux_per_amp * current)
    margin = safety * offset / electrode.max_concentration
    lo, hi = sorted((electrode.stoich_at_0_soc, electrode.stoich_at_100_soc))
    return max(lo - margin, 0.0), min(hi + margin, 1.0)


def spec_from_thermal_spm(
    model,
    table_points: int = 257,
    measurement_noise: float = 1e-4,
    initial_soc_uncertainty: float = 0.05,
    max_c_rate: float = 3.0,
    current_std: float = 0.05,
    name: str = "thermal_spm",
) -> ThermalEstimatorSpec:
    """Extract a scheduled spec from a :class:`~cellkernel.models.thermal.ThermalSPM`.

    The thermal *node* is discarded: the generated estimator is given temperature
    rather than predicting it, so heat capacity, heat-transfer coefficient and the
    entropic coefficient play no part. Everything else -- the schedule, the
    Arrhenius factors, the potentials -- comes across.
    """
    cell = model.parameters
    grid = np.asarray(model.schedule_negative.temperatures, dtype=float)
    reference = cell.reference_temperature

    def electrode_spec(side: str) -> ScheduledElectrodeSpec:
        electrode = cell._electrode(side)
        schedule = model.schedule_negative if side == "negative" else model.schedule_positive
        flux = model._flux_neg if side == "negative" else model._flux_pos
        j_per_amp = model._j_neg if side == "negative" else model._j_pos
        systems = schedule.systems
        n_e = schedule.n_states

        A_grid = np.stack([s.A for s in systems])
        B_grid = np.stack([s.B.reshape(-1) * flux for s in systems])
        c_surface_grid = np.stack([s.C[0] for s in systems])
        d_surface_grid = np.array([s.D[0, 0] for s in systems])

        # The bulk functional must not depend on temperature: it is the conserved
        # quantity, and a temperature-dependent definition of "how much lithium is
        # in this particle" would be a contradiction. Assert rather than assume.
        bulk = np.stack([s.C[1] for s in systems])
        spread = float(np.max(np.abs(bulk - bulk[0])))
        if spread > 1e-12 * max(float(np.max(np.abs(bulk))), 1.0):
            raise ValueError(
                f"{side} bulk output row varies with temperature by {spread:.3e}; "
                "the conserved functional must be temperature independent"
            )

        factors = np.array(
            [
                np.exp(
                    electrode.diffusion_activation_energy
                    / GAS_CONSTANT
                    * (1.0 / reference - 1.0 / float(t))
                )
                if electrode.diffusion_activation_energy != 0.0
                else 1.0
                for t in grid
            ]
        )
        return ScheduledElectrodeSpec(
            name=side,
            n_states=n_e,
            A_grid=A_grid,
            B_grid=B_grid,
            c_surface_grid=c_surface_grid,
            d_surface_grid=d_surface_grid,
            c_bulk=np.ascontiguousarray(bulk[0]),
            diffusion_factor_grid=factors,
            diffusion_activation_energy=float(electrode.diffusion_activation_energy),
            reaction_activation_energy=float(electrode.reaction_activation_energy),
            flux_per_amp=float(flux),
            j_per_amp=float(j_per_amp),
            max_concentration=float(electrode.max_concentration),
            exchange_prefactor=float(
                electrode.reaction_rate * np.sqrt(cell.electrolyte_concentration)
            ),
            ocp_table=tabulate(
                electrode.ocp,
                table_points,
                *scheduled_table_domain(model, side, max_c_rate),
            ),
            voltage_sign=-1.0 if side == "negative" else +1.0,
            uniform_map=np.ascontiguousarray(systems[0].x0_from_uniform.reshape(-1).copy()),
        )

    negative = electrode_spec("negative")
    positive = electrode_spec("positive")
    n = negative.n_states + positive.n_states

    neg, pos = cell.negative, cell.positive
    span = neg.stoich_at_100_soc - neg.stoich_at_0_soc

    # Noise shaping, as in the isothermal spec: rank one along the direction a
    # state-of-charge error displaces the state, and along the input column.
    # Both are evaluated at the reference temperature, which is adequate because
    # they set uncertainty rather than dynamics.
    reference_index = int(np.argmin(np.abs(grid - reference)))
    scale = np.abs(
        np.concatenate(
            [
                negative.uniform_map * float(neg.concentration(0.5)),
                positive.uniform_map * float(pos.concentration(0.5)),
            ]
        )
    )
    scale = np.maximum(scale, 1e-6 * max(scale.max(), 1.0))
    soc_direction = np.concatenate(
        [
            negative.uniform_map * neg.max_concentration * span,
            positive.uniform_map
            * pos.max_concentration
            * (pos.stoich_at_100_soc - pos.stoich_at_0_soc),
        ]
    )
    p0 = initial_soc_uncertainty**2 * np.outer(soc_direction, soc_direction)
    p0 += np.diag((1e-3 * scale) ** 2)

    b_reference = np.concatenate(
        [negative.B_grid[reference_index], positive.B_grid[reference_index]]
    )
    q = current_std**2 * np.outer(b_reference, b_reference) + np.diag((1e-5 * scale) ** 2)

    return ThermalEstimatorSpec(
        name=name,
        dt=float(model.dt),
        n_states=n,
        temperature_grid=grid,
        reference_temperature=float(reference),
        negative=negative,
        positive=positive,
        contact_resistance=float(cell.contact_resistance),
        soc_scale=float(1.0 / (neg.max_concentration * span)),
        soc_offset=float(-neg.stoich_at_0_soc / span),
        voltage_limits=tuple(cell.voltage_limits),
        process_noise=q,
        measurement_noise=float(measurement_noise),
        initial_covariance=p0,
        sto_negative=(float(neg.stoich_at_0_soc), float(neg.stoich_at_100_soc)),
        sto_positive=(float(pos.stoich_at_0_soc), float(pos.stoich_at_100_soc)),
        provenance=(
            f"cell={cell.name!r} thermal schedule "
            f"{grid.size} points {grid[0]:.1f}-{grid[-1]:.1f}K dt={model.dt}s"
        ),
    )


class ThermalReferenceEstimator:
    """NumPy mirror of the generated scheduled estimator.

    Written as scalar loops in the same order as the emitted C, for the same
    reason as :class:`~cellkernel.codegen.spec.ReferenceEstimator`: vectorised
    NumPy reductions use pairwise summation and are *more* accurate than a C
    loop, which would mask a genuine code-generation difference beneath a
    summation-order one.
    """

    def __init__(self, spec: ThermalEstimatorSpec) -> None:
        self.spec = spec
        self.x = np.zeros(spec.n_states)
        self.P = spec.initial_covariance.copy()

    def init(self, soc: float, temperature: float) -> None:
        self.x = self.spec.initial_state(soc, temperature)
        self.P = self.spec.initial_covariance.copy()

    # ------------------------------------------------------------- chemistry

    def _surface(self, side: str, current: float, temperature: float) -> float:
        spec = self.spec
        electrode = spec.negative if side == "negative" else spec.positive
        offset = 0 if side == "negative" else spec.n_negative
        _, _, c_row, d_term = spec.matrices(electrode, temperature)
        acc = 0.0
        for i in range(electrode.n_states):
            acc += c_row[i] * self.x[offset + i]
        return acc + d_term * electrode.flux_per_amp * current

    def _bulk_negative(self) -> float:
        electrode = self.spec.negative
        acc = 0.0
        for i in range(electrode.n_states):
            acc += electrode.c_bulk[i] * self.x[i]
        return acc

    def _exchange_current(
        self, electrode: ScheduledElectrodeSpec, c_surf: float, temperature: float
    ) -> float:
        margin = 1e-6 * electrode.max_concentration
        c = min(max(c_surf, margin), electrode.max_concentration - margin)
        prefactor = electrode.exchange_prefactor * self.spec.reaction_factor(electrode, temperature)
        return prefactor * np.sqrt(c) * np.sqrt(electrode.max_concentration - c)

    def _overpotential(
        self,
        electrode: ScheduledElectrodeSpec,
        c_surf: float,
        current: float,
        temperature: float,
    ) -> float:
        i0 = self._exchange_current(electrode, c_surf, temperature)
        u = electrode.j_per_amp * current / (2.0 * i0)
        return (2.0 * GAS_CONSTANT * temperature / FARADAY) * np.arcsinh(u)

    def _d_overpotential(
        self,
        electrode: ScheduledElectrodeSpec,
        c_surf: float,
        current: float,
        temperature: float,
    ) -> float:
        c_max = electrode.max_concentration
        margin = 1e-6 * c_max
        c = min(max(c_surf, margin), c_max - margin)
        i0 = self._exchange_current(electrode, c, temperature)
        u = electrode.j_per_amp * current / (2.0 * i0)
        d_log_i0 = (c_max - 2.0 * c) / (2.0 * c * (c_max - c))
        prefactor = 2.0 * GAS_CONSTANT * temperature / FARADAY
        return prefactor * (-u * d_log_i0) / np.sqrt(1.0 + u * u)

    def voltage(self, current: float, temperature: float) -> float:
        spec = self.spec
        total = -spec.contact_resistance * current
        for side, electrode in (("negative", spec.negative), ("positive", spec.positive)):
            c_surf = self._surface(side, current, temperature)
            sto = c_surf / electrode.max_concentration
            total += electrode.voltage_sign * (
                electrode.ocp_table.interpolate(sto)
                + self._overpotential(electrode, c_surf, current, temperature)
            )
        return float(total)

    def soc(self) -> float:
        return float(self.spec.soc_scale * self._bulk_negative() + self.spec.soc_offset)

    def voltage_jacobian(self, current: float, temperature: float) -> np.ndarray:
        spec = self.spec
        grad = np.zeros(spec.n_states)
        for side, electrode, offset in (
            ("negative", spec.negative, 0),
            ("positive", spec.positive, spec.n_negative),
        ):
            c_surf = self._surface(side, current, temperature)
            table = electrode.ocp_table
            position = (c_surf / electrode.max_concentration - table.sto_min) / table.step
            position = min(max(position, 0.0), float(table.n - 1))
            index = min(int(position), table.n - 2)
            slope = (table.values[index + 1] - table.values[index]) / table.step
            sensitivity = electrode.voltage_sign * (
                slope / electrode.max_concentration
                + self._d_overpotential(electrode, c_surf, current, temperature)
            )
            _, _, c_row, _ = spec.matrices(electrode, temperature)
            for k in range(electrode.n_states):
                grad[offset + k] = sensitivity * c_row[k]
        return grad

    # -------------------------------------------------------------- filtering

    def _transition(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        spec = self.spec
        n = spec.n_states
        A = np.zeros((n, n))
        B = np.zeros(n)
        a_neg, b_neg, _, _ = spec.matrices(spec.negative, temperature)
        a_pos, b_pos, _, _ = spec.matrices(spec.positive, temperature)
        A[: spec.n_negative, : spec.n_negative] = a_neg
        A[spec.n_negative :, spec.n_negative :] = a_pos
        B[: spec.n_negative] = b_neg
        B[spec.n_negative :] = b_pos
        return A, B

    def predict(self, current: float, temperature: float) -> None:
        spec = self.spec
        n = spec.n_states
        A, B = self._transition(temperature)
        x_new = np.zeros(n)
        for i in range(n):
            acc = B[i] * current
            for j in range(n):
                acc += A[i, j] * self.x[j]
            x_new[i] = acc
        self.x = x_new

        ap = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += A[i, k] * self.P[k, j]
                ap[i, j] = acc
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += ap[i, k] * A[j, k]
                self.P[i, j] = acc + spec.process_noise[i, j]

    def update(self, current: float, temperature: float, measured_voltage: float) -> float:
        spec = self.spec
        n = spec.n_states
        h = self.voltage_jacobian(current, temperature)
        ph = np.zeros(n)
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += self.P[i, j] * h[j]
            ph[i] = acc
        denom = spec.measurement_noise
        for i in range(n):
            denom += h[i] * ph[i]
        gain = ph / denom
        innovation = measured_voltage - self.voltage(current, temperature)
        for i in range(n):
            self.x[i] += gain[i] * innovation
        for i in range(n):
            for j in range(n):
                self.P[i, j] -= gain[i] * ph[j]
        for i in range(n):
            for j in range(i):
                average = 0.5 * (self.P[i, j] + self.P[j, i])
                self.P[i, j] = average
                self.P[j, i] = average
        return float(innovation)
