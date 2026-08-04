"""The intermediate representation between a Python model and generated C.

An :class:`EstimatorSpec` is a flat bundle of numbers -- matrices, lookup tables
and scalar constants -- that fully determines a physics-based state estimator. It
contains no Python callables and no model objects.

That indirection is the point. Emitting C directly from a
:class:`~cellkernel.models.spm.SPM` would couple the generator to that class's
internals, so every refactor upstream would break code generation. Here, exactly
one function -- :func:`spec_from_spm` -- knows how to read a model, and
everything downstream depends only on the spec. It also means a spec can be
serialised, diffed between releases, or hand-edited by someone calibrating a
parameter without re-running the Python stack.

The spec additionally carries :class:`ReferenceEstimator`, a plain-NumPy
implementation that mirrors the emitted C operation for operation. Cross-checking
generated C against the *full* Python model would conflate two different things:
mistakes in code generation, and the approximations the generated code makes
deliberately (a lookup table instead of a transcendental fit, single instead of
double precision). Comparing against a mirror isolates the first, and
:mod:`cellkernel.verify` then reports the second separately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..ocp import OCPTable, tabulate

__all__ = ["EstimatorSpec", "ReferenceEstimator", "spec_from_spm", "table_backed_model"]


@dataclass(frozen=True)
class ElectrodeSpec:
    """Per-electrode constants needed by the estimator."""

    name: str
    n_states: int
    #: Row mapping the reduced state onto surface concentration.
    c_surface: np.ndarray
    #: Direct feedthrough from molar influx onto surface concentration.
    d_surface: float
    #: Row mapping the reduced state onto volume-averaged concentration.
    c_bulk: np.ndarray
    #: Molar influx per ampere of cell current.
    flux_per_amp: float
    #: Interfacial current density per ampere of cell current, A m-2.
    j_per_amp: float
    max_concentration: float
    #: ``rate * F * sqrt(c_e)``, so that ``i0 = prefactor * sqrt(c (cmax - c))``.
    exchange_prefactor: float
    ocp_table: OCPTable
    #: Sign with which this electrode's potential enters the terminal voltage.
    voltage_sign: float


@dataclass(frozen=True)
class EstimatorSpec:
    """Everything needed to generate and to reproduce an estimator."""

    name: str
    dt: float
    n_states: int
    #: Combined block-diagonal state transition, cell current as input.
    A: np.ndarray
    B: np.ndarray
    negative: ElectrodeSpec
    positive: ElectrodeSpec
    contact_resistance: float
    #: ``2 R_g T / F``, the Butler-Volmer voltage scale.
    kinetic_prefactor: float
    #: ``soc = soc_scale * c_bulk_negative + soc_offset``.
    soc_scale: float
    soc_offset: float
    temperature: float
    voltage_limits: tuple[float, float]
    #: Extended Kalman filter tuning. Both covariances are full ``(n, n)``
    #: matrices, not diagonals: see the discussion in :func:`spec_from_spm`.
    process_noise: np.ndarray
    measurement_noise: float
    initial_covariance: np.ndarray
    #: Maps a uniform particle concentration onto the reduced state vector.
    map_negative: np.ndarray
    map_positive: np.ndarray
    #: ``(stoichiometry at 0% SoC, stoichiometry at 100% SoC)`` per electrode.
    sto_negative: tuple[float, float]
    sto_positive: tuple[float, float]
    #: Provenance, emitted into the generated header.
    provenance: str = ""

    @property
    def n_negative(self) -> int:
        return self.negative.n_states

    @property
    def n_positive(self) -> int:
        return self.positive.n_states

    def initial_state(self, soc: float) -> np.ndarray:
        """State vector for a rested cell, mirroring the generated ``ck_init``.

        A rested particle is spatially uniform, and each reduced model expresses
        that condition differently -- the spectral and polynomial models put the
        concentration in their conserved coordinate and zero the rest, while the
        Pade model needs its filter states at the equilibrium consistent with
        that loading. Both are captured by the precomputed ``map_*`` vectors, so
        the generated C reduces to one scale-and-copy per electrode.
        """
        x = np.zeros(self.n_states)
        s0n, s1n = self.sto_negative
        s0p, s1p = self.sto_positive
        c_neg = (s0n + soc * (s1n - s0n)) * self.negative.max_concentration
        c_pos = (s0p + soc * (s1p - s0p)) * self.positive.max_concentration
        x[: self.n_negative] = self.map_negative * c_neg
        x[self.n_negative :] = self.map_positive * c_pos
        return x


class ReferenceEstimator:
    """NumPy mirror of the generated C, in double precision.

    Deliberately written as scalar loops over the same arrays in the same order
    as the emitted code, rather than in idiomatic vectorised NumPy. Vectorised
    reductions use pairwise summation, which is *more* accurate than the
    sequential summation a C loop performs; matching the loop order keeps the two
    implementations comparable at the 1e-15 level so that a genuine code
    generation bug cannot hide beneath a summation-order difference.
    """

    def __init__(self, spec: EstimatorSpec) -> None:
        self.spec = spec
        self.x = np.zeros(spec.n_states)
        self.P = spec.initial_covariance.copy()

    # ------------------------------------------------------------------ set-up

    def init(self, soc: float) -> None:
        """Reset to a rested cell at ``soc`` with the default covariance."""
        self.x = self.spec.initial_state(soc)
        self.P = self.spec.initial_covariance.copy()

    # ------------------------------------------------------------- kinematics

    def _surface_concentration(self, electrode: str, current: float) -> float:
        spec = self.spec
        el = spec.negative if electrode == "negative" else spec.positive
        offset = 0 if electrode == "negative" else spec.n_negative
        acc = 0.0
        for i in range(el.n_states):
            acc += el.c_surface[i] * self.x[offset + i]
        return acc + el.d_surface * el.flux_per_amp * current

    def _bulk_concentration(self, electrode: str) -> float:
        spec = self.spec
        el = spec.negative if electrode == "negative" else spec.positive
        offset = 0 if electrode == "negative" else spec.n_negative
        acc = 0.0
        for i in range(el.n_states):
            acc += el.c_bulk[i] * self.x[offset + i]
        return acc

    def _overpotential(self, el: ElectrodeSpec, c_surf: float, current: float) -> float:
        margin = 1e-6 * el.max_concentration
        c = min(max(c_surf, margin), el.max_concentration - margin)
        i0 = el.exchange_prefactor * np.sqrt(c) * np.sqrt(el.max_concentration - c)
        u = el.j_per_amp * current / (2.0 * i0)
        return self.spec.kinetic_prefactor * np.arcsinh(u)

    def _d_overpotential(self, el: ElectrodeSpec, c_surf: float, current: float) -> float:
        margin = 1e-6 * el.max_concentration
        c = min(max(c_surf, margin), el.max_concentration - margin)
        c_max = el.max_concentration
        i0 = el.exchange_prefactor * np.sqrt(c) * np.sqrt(c_max - c)
        u = el.j_per_amp * current / (2.0 * i0)
        d_log_i0 = (c_max - 2.0 * c) / (2.0 * c * (c_max - c))
        return self.spec.kinetic_prefactor * (-u * d_log_i0) / np.sqrt(1.0 + u * u)

    def voltage(self, current: float) -> float:
        spec = self.spec
        total = -spec.contact_resistance * current
        for electrode, el in (("negative", spec.negative), ("positive", spec.positive)):
            c_surf = self._surface_concentration(electrode, current)
            sto = c_surf / el.max_concentration
            total += el.voltage_sign * (
                el.ocp_table.interpolate(sto) + self._overpotential(el, c_surf, current)
            )
        return float(total)

    def soc(self) -> float:
        return float(
            self.spec.soc_scale * self._bulk_concentration("negative") + self.spec.soc_offset
        )

    def voltage_jacobian(self, current: float) -> np.ndarray:
        spec = self.spec
        grad = np.zeros(spec.n_states)
        for electrode, el, offset in (
            ("negative", spec.negative, 0),
            ("positive", spec.positive, spec.n_negative),
        ):
            c_surf = self._surface_concentration(electrode, current)
            table = el.ocp_table
            # Slope of the interpolating segment, which is what the generated
            # code differentiates. Using the analytic fit's slope instead would
            # make the Jacobian inconsistent with the potential the filter
            # actually evaluates.
            pos = (c_surf / el.max_concentration - table.sto_min) / table.step
            pos = min(max(pos, 0.0), float(table.n - 1))
            i = min(int(pos), table.n - 2)
            du_dsto = (table.values[i + 1] - table.values[i]) / table.step
            sens = el.voltage_sign * (
                du_dsto / el.max_concentration + self._d_overpotential(el, c_surf, current)
            )
            for k in range(el.n_states):
                grad[offset + k] = sens * el.c_surface[k]
        return grad

    # ---------------------------------------------------------------- filtering

    def predict(self, current: float) -> None:
        spec = self.spec
        n = spec.n_states
        x_new = np.zeros(n)
        for i in range(n):
            acc = spec.B[i] * current
            for j in range(n):
                acc += spec.A[i, j] * self.x[j]
            x_new[i] = acc
        self.x = x_new

        # P = A P A' + Q, formed as (A P) A' to keep the loop order identical
        # to the generated code.
        ap = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += spec.A[i, k] * self.P[k, j]
                ap[i, j] = acc
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += ap[i, k] * spec.A[j, k]
                self.P[i, j] = acc + spec.process_noise[i, j]

    def update(self, current: float, measured_voltage: float) -> float:
        """Apply one extended Kalman correction and return the innovation."""
        spec = self.spec
        n = spec.n_states
        h = self.voltage_jacobian(current)
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
        innovation = measured_voltage - self.voltage(current)
        for i in range(n):
            self.x[i] += gain[i] * innovation
        # Symmetric Joseph-free form: P <- P - K (P H)'. Adequate here because
        # the covariance stays well conditioned; see the generated code comment.
        for i in range(n):
            for j in range(n):
                self.P[i, j] -= gain[i] * ph[j]
        for i in range(n):
            for j in range(i):
                avg = 0.5 * (self.P[i, j] + self.P[j, i])
                self.P[i, j] = avg
                self.P[j, i] = avg
        return float(innovation)


def table_domain(
    model, side: str, max_c_rate: float = 3.0, safety: float = 1.5
) -> tuple[float, float]:
    """Stoichiometry range a potential table must cover for one electrode.

    The table has to span every stoichiometry the *surface* can reach, not merely
    the bulk window between 0% and 100% state of charge. Those are different
    ranges, and the gap is not small. Under load the surface runs ahead of the
    bulk by

    .. math:: \\Delta c = \\frac{R}{5 D} \\, |N|,

    the steady-state offset of spherical diffusion, where :math:`N` is the molar
    influx at the current of interest. For the positive electrode of a typical
    layered-oxide cell, whose solid diffusivity is an order of magnitude below the
    graphite value, that offset reaches roughly 20% of full stoichiometry at 3C.

    Getting this wrong is a nasty failure mode, because a lookup table clamps
    instead of extrapolating. Sizing the domain to the bulk window plus a token
    margin leaves the table saturated during exactly the high-rate transients that
    matter most, and a saturated graphite potential is badly wrong -- its
    equilibrium potential rises by hundreds of millivolts over the last few
    percent of lithiation. In this package's own development that mistake produced
    a 138 mV discrepancy between generated C and the Python model, which the
    three-way comparison in :mod:`cellkernel.verify` localised immediately.

    Parameters
    ----------
    max_c_rate
        Largest C-rate the estimator must remain valid at.
    safety
        Multiplier on the computed excursion, covering transient overshoot beyond
        the steady-state offset.
    """
    el = model.parameters._electrode(side)
    rom = model.rom_neg if side.startswith("n") else model.rom_pos
    flux_per_amp = model._flux_neg if side.startswith("n") else model._flux_pos
    current = max_c_rate * model.parameters.nominal_capacity
    offset = rom.radius / (5.0 * rom.diffusivity) * abs(flux_per_amp * current)
    margin = safety * offset / el.max_concentration
    lo, hi = sorted((el.stoich_at_0_soc, el.stoich_at_100_soc))
    return max(lo - margin, 0.0), min(hi + margin, 1.0)


def ocp_table_for(model, side: str, table_points: int = 257, max_c_rate: float = 3.0) -> OCPTable:
    """Tabulate one electrode's potential over its reachable surface range."""
    el = model.parameters._electrode(side)
    lo, hi = table_domain(model, side, max_c_rate)
    return tabulate(el.ocp, table_points, lo, hi)


def table_backed_model(model, spec: EstimatorSpec):
    """Rebuild ``model`` with its potentials replaced by ``spec``'s lookup tables.

    Used by :mod:`cellkernel.verify` to separate two error sources that would
    otherwise be tangled together: the fidelity of the generated C, and the cost of
    representing a potential as a table. Comparing generated C against *this* model
    isolates the former.

    The tables are taken from the spec rather than recomputed, so the two cannot
    drift apart. Recomputing them would reintroduce exactly the class of mismatch
    this comparison exists to detect.
    """
    from ..models.spm import SPM

    cell = model.parameters
    new_cell = replace(
        cell,
        negative=replace(cell.negative, ocp=_TableCallable(spec.negative.ocp_table)),
        positive=replace(cell.positive, ocp=_TableCallable(spec.positive.ocp_table)),
    )
    rom, order = model._rom_spec
    return SPM(new_cell, dt=model.dt, rom=rom, order=order, temperature=model.temperature)


class _TableCallable:
    """A callable OCP that evaluates exactly as the generated C does."""

    def __init__(self, table: OCPTable) -> None:
        self.table = table

    def __call__(self, x):
        flat = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
        out = np.array([self.table.interpolate(v) for v in flat])
        return out.reshape(np.shape(x)) if np.ndim(x) else float(out[0])

    def derivative(self, x):
        table = self.table
        flat = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
        out = np.empty(flat.size)
        for k, v in enumerate(flat):
            pos = (v - table.sto_min) / table.step
            pos = min(max(pos, 0.0), float(table.n - 1))
            i = min(int(pos), table.n - 2)
            out[k] = (table.values[i + 1] - table.values[i]) / table.step
        return out.reshape(np.shape(x)) if np.ndim(x) else float(out[0])


def _probe_exchange_prefactor(model, side: str, tolerance: float = 1e-10) -> float:
    """Recover ``k`` in ``i0 = k sqrt(c) sqrt(c_max - c)`` by probing the model.

    The exchange-current prefactor is read back from the model rather than
    reconstructed from parameters. Reconstructing it means restating a unit
    convention -- whether the reaction rate already carries the Faraday constant,
    whether the electrolyte term is normalised by a reference concentration --
    in a second place, and the two then have to be kept in agreement by hand.
    During development of this package they silently diverged when the convention
    changed upstream, and the result was a 138 mV error in generated code that
    looked exactly like a code-generation bug.

    Probing also lets the assumed *shape* be checked rather than assumed: ``k`` is
    evaluated at several concentrations, and if the ratios disagree then the
    model's kinetics are no longer of the assumed form and generation stops with
    a useful message instead of emitting plausible, wrong C.
    """
    el = model.parameters._electrode(side)
    c_max = el.max_concentration
    values = []
    for fraction in (0.25, 0.5, 0.75):
        c = fraction * c_max
        i0 = float(model._exchange_current(c, side))
        values.append(i0 / (np.sqrt(c) * np.sqrt(c_max - c)))
    spread = (max(values) - min(values)) / max(abs(v) for v in values)
    if spread > tolerance:
        raise ValueError(
            f"{side} electrode kinetics are not of the form "
            f"i0 = k*sqrt(c)*sqrt(cmax-c): probing gave k = {values} "
            f"(relative spread {spread:.2e}). Code generation assumes that form; "
            "update cellkernel.codegen.emit if the model has changed."
        )
    return float(np.mean(values))


def _probe_kinetic_prefactor(model, side: str, tolerance: float = 1e-9) -> float:
    """Recover ``2 R_g T / F`` in ``eta = prefactor * asinh(j / 2 i0)``.

    Read back from the model for the same reason as the exchange prefactor, and
    checked at two current densities so that a change to the kinetic law -- a
    switch to a Tafel or linearised form, or asymmetric charge transfer -- is
    caught rather than absorbed.
    """
    el = model.parameters._electrode(side)
    c = 0.5 * el.max_concentration
    i0 = float(model._exchange_current(c, side))
    values = []
    for j in (0.3 * i0, 1.7 * i0):
        eta = float(model._overpotential(c, j, side))
        values.append(eta / np.arcsinh(j / (2.0 * i0)))
    spread = abs(values[0] - values[1]) / max(abs(v) for v in values)
    if spread > tolerance:
        raise ValueError(
            f"{side} electrode overpotential is not of the form "
            f"eta = k*asinh(j/2/i0): probing gave k = {values} "
            f"(relative spread {spread:.2e})."
        )
    return float(np.mean(values))


def spec_from_spm(
    model,
    table_points: int = 257,
    process_noise: np.ndarray | float | None = None,
    measurement_noise: float = 1e-4,
    initial_soc_uncertainty: float = 0.05,
    max_c_rate: float = 3.0,
    name: str = "spm",
) -> EstimatorSpec:
    """Extract a code-generation spec from a :class:`~cellkernel.models.spm.SPM`.

    This is the only function in :mod:`cellkernel.codegen` that touches model
    internals.

    Parameters
    ----------
    model
        A constructed single particle model.
    table_points
        Samples per open-circuit-potential table. The domain is chosen by
        :func:`table_domain` to cover the reachable *surface* stoichiometry rather
        than the bulk window, so the resolution goes where the electrode actually
        operates without risking table saturation under load.
    max_c_rate
        Largest C-rate the generated estimator must stay valid at. Widens the
        potential tables to cover the surface excursion that rate produces. Raising
        it costs accuracy at fixed ``table_points``, since the same number of
        samples spans a wider range; setting it too low is worse, because the table
        clamps instead of extrapolating.
    process_noise
        Diagonal of the process noise covariance, in squared state units. When
        omitted, a heuristic is used: the conserved concentration coordinate gets
        a variance corresponding to a 0.2% current-integration error per step,
        and the shape coordinates get a much looser value since they are driven
        strongly by the input and are far less sensitive to mistuning.
    measurement_noise
        Voltage measurement variance in V^2. The default ``1e-4`` corresponds to
        a 10 mV standard deviation, which is a realistic figure for a
        battery-management front end once quantisation, gain error and the
        sampling skew between current and voltage channels are all included --
        the last of these usually dominates and is frequently forgotten.
    initial_soc_uncertainty
        Standard deviation of the initial state of charge, used to seed the
        covariance.
    """
    neg, pos = model.parameters.negative, model.parameters.positive
    ss_neg, ss_pos = model.ss_neg, model.ss_pos
    n_neg, n_pos = ss_neg.n_states, ss_pos.n_states
    n = n_neg + n_pos

    A = np.zeros((n, n))
    A[:n_neg, :n_neg] = ss_neg.A
    A[n_neg:, n_neg:] = ss_pos.A
    B = np.zeros(n)
    B[:n_neg] = ss_neg.B.reshape(-1) * model._flux_neg
    B[n_neg:] = ss_pos.B.reshape(-1) * model._flux_pos

    def electrode_spec(side: str) -> ElectrodeSpec:
        el = neg if side == "negative" else pos
        ss = ss_neg if side == "negative" else ss_pos
        table = ocp_table_for(model, side, table_points, max_c_rate)
        return ElectrodeSpec(
            name=side,
            n_states=ss.n_states,
            c_surface=np.ascontiguousarray(ss.C[0], dtype=float),
            d_surface=float(ss.D[0, 0]),
            c_bulk=np.ascontiguousarray(ss.C[1], dtype=float),
            flux_per_amp=float(model._flux_neg if side == "negative" else model._flux_pos),
            j_per_amp=float(model._j_neg if side == "negative" else model._j_pos),
            max_concentration=float(el.max_concentration),
            exchange_prefactor=_probe_exchange_prefactor(model, side),
            ocp_table=table,
            voltage_sign=-1.0 if side == "negative" else +1.0,
        )

    span = neg.stoich_at_100_soc - neg.stoich_at_0_soc
    soc_scale = 1.0 / (neg.max_concentration * span)
    soc_offset = -neg.stoich_at_0_soc / span

    # ---- noise models, structured rather than diagonal --------------------
    #
    # A diagonal covariance is the obvious choice and it is wrong here, badly
    # enough to make the filter diverge. Two reasons.
    #
    # The state vector is not a set of independent physical quantities. It mixes a
    # conserved concentration of order 1e4 with internal reduced-order coordinates
    # that can reach 1e8, and those coordinates are rigidly related to the
    # concentration profile: only certain combinations correspond to a physically
    # realisable profile. Assigning each coordinate its own independent variance
    # lets the filter move the state off that manifold, producing a surface
    # concentration inconsistent with the bulk. The predicted voltage then no
    # longer explains the measurement, the filter corrects further in the same
    # wrong direction, and it runs away.
    #
    # So both covariances are built from the *directions the uncertainty actually
    # lies along*, each a rank-one outer product plus a small regulariser:
    #
    #   P0  along d(x_init)/d(soc) -- not knowing the state of charge is a single
    #       degree of freedom, a uniform loading level shared by both electrodes.
    #   Q   along B -- the dominant process disturbance is current measurement
    #       error, and current enters the state through exactly one direction.
    #
    # The regulariser is scaled per state by that state's own magnitude, so it
    # stays dimensionally sensible across coordinates of wildly different size.
    scale = np.abs(
        np.concatenate(
            [
                ss_neg.x0_from_uniform.reshape(-1) * float(neg.concentration(0.5)),
                ss_pos.x0_from_uniform.reshape(-1) * float(pos.concentration(0.5)),
            ]
        )
    )
    scale = np.maximum(scale, 1e-6 * max(scale.max(), 1.0))

    # d(x_init)/d(soc): the direction a state-of-charge error displaces the state.
    soc_direction = np.concatenate(
        [
            ss_neg.x0_from_uniform.reshape(-1)
            * neg.max_concentration
            * (neg.stoich_at_100_soc - neg.stoich_at_0_soc),
            ss_pos.x0_from_uniform.reshape(-1)
            * pos.max_concentration
            * (pos.stoich_at_100_soc - pos.stoich_at_0_soc),
        ]
    )
    p0 = initial_soc_uncertainty**2 * np.outer(soc_direction, soc_direction)
    p0 += np.diag((1e-3 * scale) ** 2)

    if process_noise is None:
        # Current measurement error of 0.2% of the 1C current per sample, entering
        # through B, plus a floor that keeps the filter from becoming overconfident
        # about the reduced-order coordinates over long runs.
        sigma_current = 0.002 * model.parameters.nominal_capacity
        q = sigma_current**2 * np.outer(B, B) + np.diag((1e-5 * scale) ** 2)
    else:
        arr = np.asarray(process_noise, dtype=float)
        q = np.diag(np.broadcast_to(arr, (n,)).copy()) if arr.ndim <= 1 else arr

    return EstimatorSpec(
        name=name,
        dt=float(model.dt),
        n_states=n,
        A=A,
        B=B,
        negative=electrode_spec("negative"),
        positive=electrode_spec("positive"),
        contact_resistance=float(model.parameters.contact_resistance),
        kinetic_prefactor=_probe_kinetic_prefactor(model, "negative"),
        soc_scale=float(soc_scale),
        soc_offset=float(soc_offset),
        temperature=float(model.temperature),
        voltage_limits=tuple(model.parameters.voltage_limits),
        process_noise=q,
        measurement_noise=float(measurement_noise),
        initial_covariance=p0,
        map_negative=ss_neg.x0_from_uniform.reshape(-1).copy(),
        map_positive=ss_pos.x0_from_uniform.reshape(-1).copy(),
        sto_negative=(float(neg.stoich_at_0_soc), float(neg.stoich_at_100_soc)),
        sto_positive=(float(pos.stoich_at_0_soc), float(pos.stoich_at_100_soc)),
        provenance=(
            f"cell={model.parameters.name!r} "
            f"rom={type(model.rom_neg).__name__}/{type(model.rom_pos).__name__} "
            f"dt={model.dt}s T={model.temperature:.2f}K"
        ),
    )
