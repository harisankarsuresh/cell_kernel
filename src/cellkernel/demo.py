"""Reproducible, offline demonstration of battery charge estimation."""

from __future__ import annotations

import csv
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from . import __version__
from .codegen import generate
from .data import synthetic_drive_cycle
from .estimators import EKF
from .models import SPM
from .params import chen2020_nmc811_graphite

SCENARIOS = (
    ("high", "Gauge starts too high", 0.90, 0.001),
    ("low", "Gauge starts too low", 0.60, 0.001),
    ("noisy", "Noisier voltage sensor", 0.90, 0.010),
)


def build_demo(output: str | Path) -> dict:
    """Run three fixed synthetic scenarios and export HTML, JSON, CSV and C.

    Measurements are sampled before stepping, matching CellModel.simulate and
    EKF.run. The counting baseline must therefore integrate previous samples
    only. Metrics use the entire final ten minutes, not a downsampled plot.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cell = chen2020_nmc811_graphite()
    model = SPM(cell, dt=1.0, rom="pade", order=3)
    current = synthetic_drive_cycle(
        cell.nominal_capacity, duration=1800, peak_discharge_rate=2, seed=0
    )
    truth = model.simulate(current, soc0=0.75)
    consumed = np.r_[0.0, np.cumsum(current[:-1])] / (3600 * cell.nominal_capacity)
    project = generate(model, output / "estimator", precision="float")
    report = {
        "schema_version": 1,
        "cellkernel_version": __version__,
        "numpy_version": np.__version__,
        "data_kind": "synthetic; same physical model generates and estimates voltage",
        "cell": cell.name,
        "dt_s": model.dt,
        "duration_s": len(current) * model.dt,
        "true_initial_soc_pct": 75,
        "drive_seed": 0,
        "voltage_noise_seed": 7,
        "metric_window": "last 600 samples (10 minutes), before each model step",
        "c_export": {
            "precision": "float",
            "states": project.spec.n_states,
            "ram_per_instance_bytes": project.budget.ram_bytes,
            "predict_stack_bytes": project.budget.stack_bytes,
            "table_bytes": project.budget.flash_bytes,
            "verification": "not run by this demo; use cellkernel verify",
            "note": "C exports a single-update EKF; the demo uses three measurement iterations.",
        },
        "scenarios": [],
    }
    with (output / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "time_s",
                "current_A",
                "measured_voltage_V",
                "true_soc_pct",
                "counting_soc_pct",
                "estimated_soc_pct",
            ]
        )
        for key, label, initial, noise in SCENARIOS:
            measured = truth["voltage"] + np.random.default_rng(7).normal(0, noise, len(current))
            estimator = EKF(
                model,
                process_noise=EKF.suggest_process_noise(
                    model, current_std=0.05, soc_drift_per_hour=0.02
                ),
                measurement_noise=noise**2,
                initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.2),
                iterations=3,
            )
            estimated = estimator.run(current, measured, soc0=initial)["soc"]
            counting = initial - consumed
            error = 100 * (estimated[-600:] - truth["soc"][-600:])
            scenario = {
                "id": key,
                "label": label,
                "initial_soc_pct": initial * 100,
                "noise_mV": noise * 1000,
                "initial_error_pp": (initial - 0.75) * 100,
                "settled_rmse_pp": float(np.sqrt(np.mean(error**2))),
                "counting_rmse_pp": float(
                    np.sqrt(np.mean((100 * (counting[-600:] - truth["soc"][-600:])) ** 2))
                ),
                "final_error_pp": float(100 * (estimated[-1] - truth["soc"][-1])),
                "time_s": truth["time"].tolist(),
                "current_A": current.tolist(),
                "true_soc_pct": (100 * truth["soc"]).tolist(),
                "counting_soc_pct": (100 * counting).tolist(),
                "estimated_soc_pct": (100 * estimated).tolist(),
            }
            report["scenarios"].append(scenario)
            writer.writerows(
                (key, *row)
                for row in zip(
                    truth["time"],
                    current,
                    measured,
                    100 * truth["soc"],
                    100 * counting,
                    100 * estimated,
                    strict=True,
                )
            )
    payload = json.dumps(report, allow_nan=False, separators=(",", ":"))
    (output / "demo-data.json").write_text(payload + "\n", encoding="utf-8")
    template = files("cellkernel").joinpath("resources/demo.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace("__DEMO_DATA__", payload.replace("<", "\\u003c")), encoding="utf-8"
    )
    return report
