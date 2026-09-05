"""Compare charge recovery and held-out voltage predictions, using actual experiment outputs.

Run from the repository root after installing .[dev]. The measured-data panel
requires the separately downloaded LG M50 dataset. No measurement is invented
when that dataset is absent; the script prints the setup command and exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cellkernel.benchmark import run_benchmark
from cellkernel.demo import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/comparison"))
    args = parser.parse_args()
    try:
        benchmark = run_benchmark(args.out / "benchmark")
    except FileNotFoundError:
        print("Fetch measured data first: python -m cellkernel.data.reference")
        return
    demo = build_demo(args.out / "demo")
    try:
        import matplotlib
    except ImportError:
        print("Install plotting support: python -m pip install -e '.[plot]'")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, teal, orange, muted, paper = "#172d36", "#067565", "#a6501e", "#53676c", "#f4f6f3"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": ink,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#bdcbc3",
            "axes.titleweight": "bold",
        }
    )
    fig = plt.figure(figsize=(14, 8), facecolor=paper)
    fig.text(
        0.055,
        0.94,
        "cellkernel / BATTERY MODELLING + APPLIED ML",
        size=12,
        weight="bold",
        color=teal,
    )
    fig.text(0.055, 0.867, "Can a battery model learn from its mistakes?", size=27, weight="bold")
    fig.text(
        0.055,
        0.815,
        "Two reproducible experiments. Different questions. Both outcomes reported.",
        size=13,
        color=muted,
    )
    axes = [fig.add_axes([0.065, 0.29, 0.40, 0.405]), fig.add_axes([0.57, 0.29, 0.36, 0.405])]
    for ax in axes:
        ax.set_facecolor(paper)
    scenario = demo["scenarios"][0]
    minutes = np.asarray(scenario["time_s"]) / 60
    ax = axes[0]
    ax.set_title("01   Recover a wrong charge estimate", loc="left", pad=30, size=15)
    ax.text(
        0,
        1.045,
        "SYNTHETIC DRIVE · starts 15 percentage points high",
        transform=ax.transAxes,
        size=10,
        color=muted,
    )
    ax.plot(
        minutes, scenario["counting_soc_pct"], color=orange, ls="--", lw=1.7, label="Counting alone"
    )
    ax.plot(minutes, scenario["estimated_soc_pct"], color=teal, lw=2, label="Corrected estimate")
    ax.plot(minutes, scenario["true_soc_pct"], color=ink, ls=":", lw=1.8, label="Simulated truth")
    ax.set(xlabel="Time (minutes)", ylabel="State of charge (%)", xlim=(0, 30), ylim=(20, 100))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.text(
        0.065,
        0.18,
        f"15 pp → {scenario['settled_rmse_pp']:.2f} pp RMS",
        size=21,
        weight="bold",
        color=teal,
    )
    fig.text(
        0.065,
        0.135,
        "Error over the final 10 minutes. Same model creates\n"
        "and estimates voltage; this is not real-cell accuracy.",
        size=10,
        color=muted,
        linespacing=1.6,
    )
    heldout = [row for row in benchmark["scores"] if row["split"] == "held_out"]
    ax = axes[1]
    ax.set_title("02   Test a learned voltage correction", loc="left", pad=30, size=15)
    ax.text(
        0,
        1.045,
        "MEASURED LG M50 · train: 0.1C + 1C · 25 °C",
        transform=ax.transAxes,
        size=10,
        color=muted,
    )
    for offset, key, label, color in (
        (-0.25, "physics", "Physics", ink),
        (0, "data_only", "Data only", "#7f9295"),
        (0.25, "hybrid", "Hybrid", teal),
    ):
        values = [row[key]["rmse_mV"] for row in heldout]
        bars = ax.bar(np.arange(2) + offset, values, width=0.22, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=10)
    ax.set(
        xticks=[0, 1],
        xticklabels=["0.5C holdout\nWithin training range", "2C holdout\nBeyond training range"],
        ylabel="Voltage RMSE (mV) · lower is better",
        ylim=(0, 130),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)
    fig.text(0.57, 0.18, "ML helps at 0.5C. Hurts at 2C.", size=19, weight="bold", color=ink)
    fig.text(
        0.57,
        0.135,
        "Complete curves held out, not random time samples.\n"
        "One cell dataset; no claim of unseen-cell generalisation.",
        size=10,
        color=muted,
        linespacing=1.6,
    )
    fig.text(
        0.055,
        0.045,
        "Harisankar Suresh  ·  github.com/harisankarsuresh/cell_kernel",
        size=11,
        color=muted,
    )
    path = args.out / "comparison.png"
    fig.savefig(path, dpi=150, facecolor=paper)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
