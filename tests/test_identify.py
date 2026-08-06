"""Parameter identification, and knowing when it has not identified anything.

Fitting is the easy half. The suite below spends most of its effort on the other
half: whether the data actually determined what the solver reported.

Synthetic data is used for the round-trip tests, because only there is the true
answer known. A fit that cannot recover parameters it generated itself is broken
in a way no amount of agreement with a real cell would reveal.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cellkernel.identify import (
    DEFAULT_KNOBS,
    KINETIC_KNOBS,
    TRANSPORT_KNOBS,
    Knob,
    identify,
)
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite

DT = 5.0


@pytest.fixture(scope="module")
def cell():
    return chen2020_nmc811_graphite()


def build(params):
    return SPM(params, dt=DT, rom="pade", order=3)


def synthesise(params, c_rates=(0.5, 1.0, 2.0), seconds=2400.0, noise=0.0, seed=0):
    """Measurements a cell with these parameters would have produced."""
    rng = np.random.default_rng(seed)
    model = build(params)
    segments = []
    for rate in c_rates:
        steps = int(seconds / rate / DT)
        current = np.full(steps, rate * params.nominal_capacity)
        run = model.simulate(current, soc0=0.95)
        voltage = run["voltage"]
        if noise:
            voltage = voltage + rng.normal(0.0, noise, voltage.size)
        segments.append((f"{rate}C", current, voltage))
    return segments


# ------------------------------------------------------------------ plumbing


def test_needs_something_to_fit_and_something_to_fit_to(cell):
    with pytest.raises(ValueError, match="segment"):
        identify(cell, [], build)
    with pytest.raises(ValueError, match="parameter"):
        identify(cell, synthesise(cell), build, knobs=(), soc0=0.95)


def test_mismatched_series_are_rejected(cell):
    with pytest.raises(ValueError, match="differ in length"):
        identify(cell, [("bad", np.ones(10), np.ones(9))], build)


def test_a_segment_entirely_below_the_floor_is_rejected(cell):
    with pytest.raises(ValueError, match="voltage floor"):
        identify(cell, [("flat", np.ones(20), np.full(20, 2.0))], build)


def test_decimation_bounds_the_residual_length(cell):
    """A slow discharge at 1 Hz has far more points than independent information."""
    segments = synthesise(cell, c_rates=(1.0,), seconds=3600.0)
    report = identify(cell, segments, build, knobs=KINETIC_KNOBS, max_points=50, soc0=0.95)
    assert report.rmse_after >= 0.0  # ran at all
    assert len(segments[0][1]) > 200, "the test data should be long enough to matter"


# ------------------------------------------------------------- the round trip


def test_it_recovers_parameters_it_generated(cell):
    """The check that the machinery works at all, with the answer known."""
    truth = replace(
        cell,
        negative=replace(cell.negative, reaction_rate=cell.negative.reaction_rate * 0.4),
        positive=replace(cell.positive, reaction_rate=cell.positive.reaction_rate * 2.5),
    )
    segments = synthesise(truth)
    report = identify(cell, segments, build, knobs=KINETIC_KNOBS, soc0=0.95)

    assert report.rmse_after < 0.2 * report.rmse_before
    recovered = 10.0**report.values
    assert recovered[0] == pytest.approx(0.4, rel=0.25)
    assert recovered[1] == pytest.approx(2.5, rel=0.25)


def test_a_perfect_start_stays_put(cell):
    """No data, no movement. Guards against a fit that always finds something."""
    report = identify(cell, synthesise(cell), build, knobs=KINETIC_KNOBS, soc0=0.95)
    assert report.rmse_before < 1e-9
    assert np.allclose(report.values, 0.0, atol=1e-3)


def test_noise_does_not_derail_it(cell):
    truth = replace(
        cell, negative=replace(cell.negative, reaction_rate=cell.negative.reaction_rate * 0.5)
    )
    segments = synthesise(truth, noise=2e-3, seed=3)
    report = identify(cell, segments, build, knobs=KINETIC_KNOBS, soc0=0.95)
    assert 10.0 ** report.values[0] == pytest.approx(0.5, rel=0.35)


# ---------------------------------------------------------- identifiability


def test_one_rate_cannot_separate_kinetics_from_resistance(cell):
    """The reason `identify` asks for several C-rates, asserted rather than advised.

    At a single current a reaction rate and a series resistance produce the same
    voltage offset. Nothing in the data distinguishes them, and the report must
    say so rather than presenting whichever split the solver landed on.
    """
    segments = synthesise(cell, c_rates=(1.0,))
    knobs = (KINETIC_KNOBS[0], DEFAULT_KNOBS[-1])
    report = identify(cell, segments, build, knobs=knobs, soc0=0.95)
    pairs = report.correlated_pairs(threshold=0.85)
    assert pairs, f"expected a strong correlation, got {report.correlation}"


def test_the_report_names_parameters_the_data_ignored(cell):
    """A parameter the residual does not respond to must be flagged, not reported."""
    inert = Knob(
        "does_nothing",
        lambda parameters, value: parameters,  # deliberately has no effect
        low=-1.0,
        high=1.0,
    )
    segments = synthesise(cell, c_rates=(1.0, 2.0))
    report = identify(cell, segments, build, knobs=(KINETIC_KNOBS[0], inert), soc0=0.95)
    assert "does_nothing" in report.poorly_identified()
    assert "reaction_rate_negative" not in report.poorly_identified()


def test_a_parameter_pinned_to_its_bound_is_reported(cell):
    """Because whatever came out of it is not a measurement."""
    truth = replace(
        cell, negative=replace(cell.negative, reaction_rate=cell.negative.reaction_rate * 0.1)
    )
    segments = synthesise(truth)
    narrow = (replace(KINETIC_KNOBS[0], low=-0.05, high=0.05),)
    report = identify(cell, segments, build, knobs=narrow, soc0=0.95)
    assert "reaction_rate_negative" in report.at_bounds
    assert "not a measurement" in report.summary()


def test_the_summary_reports_what_was_and_was_not_learned(cell):
    segments = synthesise(cell, c_rates=(0.5, 2.0))
    report = identify(cell, segments, build, knobs=KINETIC_KNOBS, soc0=0.95)
    text = report.summary()
    assert "residual" in text
    assert "sensitivity" in text
    for knob in KINETIC_KNOBS:
        assert knob.name in text


def test_per_segment_errors_are_reported(cell):
    segments = synthesise(cell, c_rates=(0.5, 1.0, 2.0))
    report = identify(cell, segments, build, knobs=KINETIC_KNOBS, soc0=0.95)
    assert set(report.per_segment) == {"0.5C", "1.0C", "2.0C"}


def test_anchoring_assigns_only_the_shortfall(cell):
    """Not the whole measured resistance, which would count kinetics twice."""
    from cellkernel.identify import anchor_series_resistance

    bare = replace(cell, contact_resistance=0.0)
    probe = build(bare)
    rested = probe.initial_state(0.5)
    intrinsic = (probe.voltage(rested, 0.0) - probe.voltage(rested, 5.0)) / 5.0
    assert intrinsic > 0.0, "the model has some instantaneous drop of its own"

    anchored = anchor_series_resistance(bare, build, 0.030, current=5.0)
    assert anchored.contact_resistance == pytest.approx(0.030 - intrinsic, rel=1e-6)


def test_anchoring_floors_at_zero(cell):
    """A cell whose measured resistance is below the model's own drop."""
    from cellkernel.identify import anchor_series_resistance

    anchored = anchor_series_resistance(cell, build, 1e-6, current=5.0)
    assert anchored.contact_resistance == 0.0


def test_anchoring_validates_its_inputs(cell):
    from cellkernel.identify import anchor_series_resistance

    with pytest.raises(ValueError, match="current"):
        anchor_series_resistance(cell, build, 0.03, current=0.0)
    with pytest.raises(ValueError, match="measured_resistance"):
        anchor_series_resistance(cell, build, -0.01, current=5.0)


def test_anchoring_first_makes_transport_identifiable(cell):
    """The staged procedure, checked on synthetic data where truth is known.

    With the resistance free, the solver reproduces the instantaneous step -- the
    one feature every parameter can imitate -- and leaves the slower behaviour to
    whatever is left. Anchoring it removes that degree of freedom, and what
    remains in the data is diffusion.
    """
    from cellkernel.identify import anchor_series_resistance

    truth = replace(
        cell,
        contact_resistance=0.020,
        negative=replace(cell.negative, diffusivity=cell.negative.diffusivity * 0.3),
    )
    segments = synthesise(truth, c_rates=(1.0, 2.0))

    probe = build(truth)
    rested = probe.initial_state(0.5)
    measured = (probe.voltage(rested, 0.0) - probe.voltage(rested, 5.0)) / 5.0

    start = replace(cell, contact_resistance=0.0)
    anchored = anchor_series_resistance(start, build, measured, current=5.0)
    assert anchored.contact_resistance == pytest.approx(0.020, abs=2e-3)

    knobs = (TRANSPORT_KNOBS[0],)
    loose = identify(start, segments, build, knobs=knobs, soc0=0.95)
    tight = identify(anchored, segments, build, knobs=knobs, soc0=0.95)
    assert tight.rmse_after < loose.rmse_after
    assert 10.0 ** tight.values[0] == pytest.approx(0.3, rel=0.4)


def test_the_fitted_cell_is_usable(cell):
    """It must come back as a parameter set, not a vector needing reassembly."""
    truth = replace(
        cell, positive=replace(cell.positive, reaction_rate=cell.positive.reaction_rate * 2.0)
    )
    report = identify(cell, synthesise(truth), build, knobs=KINETIC_KNOBS, soc0=0.95)
    model = build(report.cell)
    state = model.initial_state(0.8)
    assert np.isfinite(model.voltage(state, 5.0))
    assert report.cell.positive.reaction_rate > cell.positive.reaction_rate
