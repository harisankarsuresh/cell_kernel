"""Recover state of charge from a deliberately wrong starting point.

Generates a synthetic drive cycle, corrupts the voltage with realistic noise,
seeds three filters 15% away from the truth, and compares how they recover
against open-loop coulomb counting.

The interesting comparison is not physics-based against equivalent-circuit; it is
the single-shot extended Kalman filter against its iterated form. On this cell,
seeded high, the single-shot filter diverges and two Gauss-Newton iterations fix
it -- because the local slope of the open-circuit voltage near full charge is
about five times smaller than the average slope across the seeded error.
"""

from __future__ import annotations

import numpy as np

from cellkernel.data import synthetic_drive_cycle
from cellkernel.estimators import EKF, UKF
from cellkernel.models import SPM
from cellkernel.params import chen2020_nmc811_graphite

TRUE_SOC = 0.75
SEEDED_SOC = 0.90
NOISE_STD = 1.0e-3


def main() -> None:
    cell = chen2020_nmc811_graphite()
    model = SPM(cell, dt=1.0, rom="pade", order=3)

    current = synthetic_drive_cycle(
        cell.nominal_capacity, duration=2400.0, dt=model.dt, peak_discharge_rate=2.0
    )
    truth = model.simulate(current, soc0=TRUE_SOC)
    rng = np.random.default_rng(0)
    measured = truth["voltage"] + rng.normal(0.0, NOISE_STD, truth["voltage"].size)

    print(f"Cell: {cell.name}, {cell.nominal_capacity} Ah")
    print(
        f"Profile: {current.size} s, peak {np.max(np.abs(current)):.1f} A, "
        f"net {np.sum(current) * model.dt / 3600.0:.3f} Ah"
    )
    print(f"True state of charge: {truth['soc'][0]:.3f} -> {truth['soc'][-1]:.3f}")
    print(f"Seeded at {SEEDED_SOC:.2f}, an error of {SEEDED_SOC - TRUE_SOC:+.2f}")
    print(f"Voltage noise: {NOISE_STD * 1e3:.1f} mV rms")
    print()

    shared = dict(
        process_noise=EKF.suggest_process_noise(model, current_std=0.05, soc_drift_per_hour=0.02),
        measurement_noise=NOISE_STD**2,
        initial_covariance=EKF.suggest_initial_covariance(model, soc_std=0.2),
    )
    candidates = {
        "EKF (1 iteration)": EKF(model, **shared, iterations=1),
        "EKF (3 iterations)": EKF(model, **shared, iterations=3),
        "UKF": UKF(model, **shared),
    }

    open_loop = SEEDED_SOC - np.cumsum(current) * model.dt / (3600.0 * cell.nominal_capacity)
    settled = slice(-600, None)

    print(f"{'estimator':<22}{'final error':>14}{'rmse, last 10 min':>20}")
    print("-" * 56)
    print(
        f"{'open-loop counting':<22}{abs(open_loop[-1] - truth['soc'][-1]):>14.5f}"
        f"{float(np.sqrt(np.mean((open_loop[settled] - truth['soc'][settled]) ** 2))):>20.5f}"
    )

    results = {}
    for name, estimator in candidates.items():
        estimator.initialise(SEEDED_SOC)
        out = estimator.run(current, measured)
        results[name] = out
        final = abs(out["soc"][-1] - truth["soc"][-1])
        rmse = float(np.sqrt(np.mean((out["soc"][settled] - truth["soc"][settled]) ** 2)))
        print(f"{name:<22}{final:>14.5f}{rmse:>20.5f}")

    print()
    best = results["EKF (3 iterations)"]
    consistency = float(np.mean(best["innovation"][600:] ** 2 / best["innovation_variance"][600:]))
    print(f"Normalised innovation squared for the iterated filter: {consistency:.3f}")
    print("A well-tuned filter gives roughly 1. Much less means it is being too")
    print("cautious about a model that is better than advertised; much more means")
    print("the noise model is understated and the estimate is over-confident.")
    print()
    print(
        f"Reported uncertainty: {best['soc_std'][0]:.4f} at the start, "
        f"{best['soc_std'][-1]:.4f} at the end."
    )

    try:
        plot(truth, results, open_loop, model.dt)
    except ImportError:
        print()
        print("(install matplotlib to also write soc_estimation.png)")


def plot(truth, results, open_loop, dt: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minutes = truth["time"] / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(minutes, truth["current"], lw=0.6, color="0.35")
    axes[0].set_ylabel("current (A)")
    axes[0].set_title("Synthetic drive cycle, discharge positive")
    axes[0].grid(alpha=0.3)

    axes[1].plot(minutes, truth["soc"], "k", lw=2.2, label="truth")
    axes[1].plot(minutes, open_loop, color="0.6", ls=":", lw=1.4, label="open-loop counting")
    for name, out in results.items():
        axes[1].plot(minutes, out["soc"], lw=1.3, label=name)
    axes[1].set_ylabel("state of charge")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for name, out in results.items():
        axes[2].plot(minutes, np.abs(out["soc"] - truth["soc"]), lw=1.2, label=name)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("absolute error")
    axes[2].set_xlabel("time (minutes)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("soc_estimation.png", dpi=140)
    print()
    print("wrote soc_estimation.png")


if __name__ == "__main__":
    main()
