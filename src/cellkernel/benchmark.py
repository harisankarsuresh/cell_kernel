"""A fixed, curve-held-out voltage prediction experiment on the measured LG M50.

Compare a physical model, ridge regression, and ridge correction of that model.
The split is by complete discharge rate, never by randomly shuffled time samples.
This is an offline voltage benchmark, not a state-of-charge estimator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from . import __version__
from .data import reference
from .models import SPM
from .params import chen2020_nmc811_graphite, fit_stoichiometry_window

TRAIN_RATES = ("cRate_0p1C", "cRate_1C")
TEST_RATES = ("cRate_0p5C", "cRate_2C")
FEATURES = ("soc", "soc_squared", "current_A", "current_times_soc", "current_times_soc_squared")


class RidgeRegressor:
    """Small weighted ridge baseline with scaling learned from training inputs only."""

    def __init__(self, alpha: float = 0.01) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> RidgeRegressor:
        x, y, weights = (np.asarray(a, dtype=float) for a in (x, y, weights))
        if x.ndim != 2 or y.shape != (len(x),) or weights.shape != y.shape or not len(x):
            raise ValueError("expected a feature matrix and matching target and weight vectors")
        if not all(np.all(np.isfinite(a)) for a in (x, y, weights)) or np.any(weights <= 0):
            raise ValueError("inputs must be finite and weights positive")
        weights = weights / weights.sum()
        self.mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x - self.mean) ** 2, axis=0, weights=weights)
        self.scale = np.where(variance > 1e-16, np.sqrt(variance), 1.0)
        self.intercept = float(np.average(y, weights=weights))
        z = (x - self.mean) / self.scale
        self.coefficients = np.linalg.solve(
            z.T @ (weights[:, None] * z) + self.alpha * np.eye(x.shape[1]),
            z.T @ (weights * (y - self.intercept)),
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + ((x - self.mean) / self.scale) @ self.coefficients

    def record(self) -> dict:
        return {
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
        }


def _features(soc: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.column_stack((soc, soc**2, current, current * soc, current * soc**2))


def run_benchmark(output: str | Path, cache: str | Path | None = None) -> dict:
    """Write scores, fitted baselines and full trajectories; never download implicitly.

    OCV calibration uses the separate pseudo-OCV record. Capacity comes only
    from the 0.1C training discharge. The held-out voltages enter scoring alone.
    Each training curve has equal total weight despite different durations.
    """
    directory = Path(cache) if cache is not None else reference.default_cache()
    soc, ocv = reference.load_ocv(cache=directory)
    slow = reference.load_discharge("T25", TRAIN_RATES[0], cache=directory)
    capacity = float(trapezoid(slow.current, slow.time) / 3600.0)
    cell = fit_stoichiometry_window(chen2020_nmc811_graphite(), soc, ocv, capacity=capacity)
    dt = 10.0
    model = SPM(cell, dt=dt, rom="pade", order=5)
    records = {}
    for rate in (*TRAIN_RATES, *TEST_RATES):
        segment = reference.load_discharge("T25", rate, dt=dt, cache=directory)
        physics = model.simulate(segment.current, soc0=1.0)
        records[rate] = (segment, physics, _features(physics["soc"], segment.current))

    x = np.concatenate([records[r][2] for r in TRAIN_RATES])
    y = np.concatenate([records[r][0].voltage for r in TRAIN_RATES])
    physical_train = np.concatenate([records[r][1]["voltage"] for r in TRAIN_RATES])
    weights = np.concatenate(
        [np.full(len(records[r][2]), 1 / len(records[r][2])) for r in TRAIN_RATES]
    )
    data_only = RidgeRegressor().fit(x, y, weights)
    residual = RidgeRegressor().fit(x, y - physical_train, weights)

    report = {
        "schema_version": 1,
        "cellkernel_version": __version__,
        "numpy_version": np.__version__,
        "experiment": "Measured LG M50 voltage prediction; complete discharge curves held out",
        "ambient_C": 25,
        "sample_period_s": dt,
        "capacity_Ah_from_training_curve": capacity,
        "calibration": "Separate 25 C pseudo-OCV record; capacity from 0.1C training discharge",
        "train_rates": list(TRAIN_RATES),
        "test_rates": list(TEST_RATES),
        "weighting": "Equal weight per training curve; metrics use every sample per curve",
        "features": list(FEATURES),
        "models": {"data_only": data_only.record(), "hybrid_residual": residual.record()},
        "sources": [
            {
                "url": url,
                "sha256": hashlib.sha256(
                    (directory / url.rsplit("/", 1)[-1]).read_bytes()
                ).hexdigest(),
            }
            for url in (reference.OCV_URL, reference.RATE_TEST_URL)
        ],
        "limitations": [
            "One cell dataset at 25 C; no claim of generalisation to new cells or temperatures.",
            "Known full initial charge; features use current-integrated model SoC, "
            "not measured SoC.",
            "Isothermal physics ignores self-heating; error includes parameter and model mismatch.",
            "Fixed feature set and alpha=0.01; no tuning on held-out curves "
            "or random sample split.",
            "The 0.5C holdout interpolates between training rates; 2C extrapolates beyond them.",
            "Resampled samples are correlated; sample count is not an independent "
            "experiment count.",
            "ML corrects terminal voltage only; it is not exported to C or used for charge limits.",
        ],
        "scores": [],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "rate",
                "time_s",
                "current_A",
                "measured_V",
                "physics_V",
                "data_only_V",
                "hybrid_V",
            ]
        )
        for rate, (segment, physical, features) in records.items():
            split = "train" if rate in TRAIN_RATES else "held_out"
            predictions = {
                "physics": physical["voltage"],
                "data_only": data_only.predict(features),
                "hybrid": physical["voltage"] + residual.predict(features),
            }
            score = {"split": split, "rate": rate, "samples": len(segment.time)}
            for name, prediction in predictions.items():
                error = (prediction - segment.voltage) * 1000.0
                score[name] = {
                    "rmse_mV": float(np.sqrt(np.mean(error**2))),
                    "mae_mV": float(np.mean(np.abs(error))),
                    "max_abs_mV": float(np.max(np.abs(error))),
                }
            report["scores"].append(score)
            writer.writerows(
                (split, rate, *row)
                for row in zip(
                    segment.time,
                    segment.current,
                    segment.voltage,
                    predictions["physics"],
                    predictions["data_only"],
                    predictions["hybrid"],
                    strict=True,
                )
            )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
