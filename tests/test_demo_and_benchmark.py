"""Check benchmark leakage, exported evidence and the first-run workflow."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from cellkernel.benchmark import TEST_RATES, TRAIN_RATES, RidgeRegressor, run_benchmark
from cellkernel.cli import main
from cellkernel.data import reference


def test_ridge_handles_constant_features_and_recovers_a_linear_signal():
    x = np.column_stack((np.linspace(-1, 1, 100), np.ones(100)))
    target = 2 + 3 * x[:, 0]
    regressor = RidgeRegressor(alpha=1e-8).fit(x, target, np.ones(100))
    assert np.max(np.abs(regressor.predict(x) - target)) < 1e-6
    assert np.all(np.isfinite(regressor.coefficients))


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_ridge_rejects_invalid_penalties(alpha):
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegressor(alpha)


@pytest.mark.parametrize(
    "x,y,w",
    [
        (np.ones(3), np.ones(3), np.ones(3)),
        (np.ones((3, 2)), np.ones(2), np.ones(3)),
        (np.ones((3, 2)), np.ones(3), np.array([1, 0, 1])),
        (np.ones((3, 2)), np.array([1, np.nan, 1]), np.ones(3)),
    ],
)
def test_ridge_rejects_invalid_training_data(x, y, w):
    with pytest.raises(ValueError):
        RidgeRegressor().fit(x, y, w)


def test_missing_benchmark_data_has_an_actionable_message(tmp_path, capsys):
    assert main(["benchmark", str(tmp_path / "out"), "--data", str(tmp_path / "missing")]) == 2
    assert "python -m cellkernel.data.reference" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_holdout_voltages_cannot_influence_training_or_predictions(tmp_path, monkeypatch):
    """Changing only test labels must change scores, never learned parameters."""
    from cellkernel.params import chen2020_nmc811_graphite

    cell = chen2020_nmc811_graphite()
    test_shift = 0.0
    soc = np.linspace(0, 1, 21)
    monkeypatch.setattr(reference, "load_ocv", lambda **_: (soc, 3.0 + soc))
    # Calibration itself is covered in test_measured_cell. This experiment fixes
    # it so the leakage test does not need any network or external dataset.
    monkeypatch.setattr("cellkernel.benchmark.fit_stoichiometry_window", lambda *a, **k: cell)

    def discharge(ambient, rate, dt=1.0, cache=None):
        time = np.arange(50) * dt
        current = np.full(50, reference.RATES[rate] * 5)
        voltage = 4.1 - time * 1e-4 - current * 0.01
        if rate in TEST_RATES:
            voltage += test_shift
        return reference.DischargeSegment(
            time, current, voltage, None, 298.15, reference.RATES[rate]
        )

    monkeypatch.setattr(reference, "load_discharge", discharge)
    for url in (reference.OCV_URL, reference.RATE_TEST_URL):
        (tmp_path / url.rsplit("/", 1)[-1]).write_bytes(b"test fixture")
    assert main(["benchmark", str(tmp_path / "first"), "--data", str(tmp_path)]) == 0
    first = json.loads((tmp_path / "first/metrics.json").read_text(encoding="utf-8"))
    test_shift = 0.5
    second = run_benchmark(tmp_path / "second", cache=tmp_path)
    assert set(TRAIN_RATES).isdisjoint(TEST_RATES)
    assert first["models"] == second["models"]
    assert first["capacity_Ah_from_training_curve"] == second["capacity_Ah_from_training_curve"]
    assert first["scores"][:2] == second["scores"][:2]
    assert first["scores"][2:] != second["scores"][2:]
    with (tmp_path / "first/predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (tmp_path / "second/predictions.csv").open(newline="") as handle:
        other = list(csv.DictReader(handle))
    for a, b in zip(rows, other, strict=True):
        for key in ("physics_V", "data_only_V", "hybrid_V"):
            assert a[key] == b[key]


def test_demo_exports_real_metrics_and_time_aligned_baseline(tmp_path, capsys, monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    out = tmp_path / "demo"
    assert main(["demo", str(out), "--open"]) == 0
    assert opened == [(out / "index.html").resolve().as_uri()]
    assert "Synthetic data" in capsys.readouterr().out
    data = json.loads((out / "demo-data.json").read_text(encoding="utf-8"))
    assert (out / "estimator/cellkernel_estimator.c").is_file()
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "__DEMO_DATA__" not in html
    assert "<script src=" not in html  # can be shared as a standalone offline file
    for scenario in data["scenarios"]:
        truth = np.asarray(scenario["true_soc_pct"])
        estimate = np.asarray(scenario["estimated_soc_pct"])
        baseline = np.asarray(scenario["counting_soc_pct"])
        assert baseline[0] == pytest.approx(scenario["initial_soc_pct"])
        assert np.allclose(baseline - truth, scenario["initial_error_pp"], atol=1e-8)
        rmse = np.sqrt(np.mean((estimate[-600:] - truth[-600:]) ** 2))
        assert rmse == pytest.approx(scenario["settled_rmse_pp"])
        assert rmse < 2  # recovery with the documented wrong-start/noise settings
    with (out / "trajectories.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3 * 1800


def test_spme_export_fails_explicitly_instead_of_dropping_electrolyte(tmp_path):
    from cellkernel.codegen import generate
    from cellkernel.models import SPMe
    from cellkernel.params import chen2020_nmc811_graphite

    model = SPMe(chen2020_nmc811_graphite(), dt=1.0)
    with pytest.raises(NotImplementedError, match="electrolyte export is unfinished"):
        generate(model, tmp_path)
