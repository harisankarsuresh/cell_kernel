"""Measured LG M50 cell data, for checking models against reality.

Everything else in this package is checked against closed-form results, against
its own mirror, or against PyBaMM. All three answer the question "is this
implemented correctly". None of them answers "does this describe a cell".

The dataset used here is a rate test on an LG M50 -- 0.1C to 2C at four ambient
temperatures, with terminal voltage and three surface thermocouples. It is the
same cell design the Chen2020 parameter set describes, which makes it about as
favourable a comparison as is available without doing one's own teardown.

The data is **not** vendored into this repository. It belongs to the PyBOP
project, is distributed under their licence, and is fetched on demand by
:func:`download`. Tests that need it skip when it is absent.

What it shows
-------------
A literature parameter set does not reproduce an individual cell to millivolt
accuracy, and it would be misleading for this package to imply otherwise. Against
PyBaMM the single particle model agrees to 0.26 mV, which says the *implementation*
is right. Against this cell the same model is out by tens of millivolts, which
says the *parameters* describe a different unit -- a different sample of the same
design, differently aged, differently formed. Those are separate claims and they
are worth keeping separate.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "RATE_TEST_URL",
    "OCV_URL",
    "PULSE_URL",
    "DischargeSegment",
    "PulseSegment",
    "default_cache",
    "download",
    "load_ocv",
    "load_discharge",
    "load_pulse",
    "available_conditions",
    "available_pulses",
]

_BASE = "https://raw.githubusercontent.com/pybop-team/PyBOP/develop/examples/data/LG_M50_ECM/data"
RATE_TEST_URL = f"{_BASE}/LGM50_5Ah_RateTest.mat"
OCV_URL = f"{_BASE}/LGM50_5Ah_OCV.mat"
PULSE_URL = f"{_BASE}/LGM50_5Ah_Pulse.mat"

#: Ambient temperatures present in the rate test, in kelvin, by struct name.
AMBIENTS = {"T0": 273.15, "T10": 283.15, "T25": 298.15, "T45": 318.15}

#: C-rates present, by struct name.
RATES = {"cRate_0p1C": 0.1, "cRate_0p5C": 0.5, "cRate_1C": 1.0, "cRate_2C": 2.0}


@dataclass(frozen=True)
class DischargeSegment:
    """One constant-current discharge, resampled onto a uniform grid.

    Current follows this package's convention -- positive on discharge -- which
    is the opposite of the source file's, and is flipped on load.
    """

    #: Seconds from the start of the discharge.
    time: np.ndarray
    #: Amperes, positive.
    current: np.ndarray
    #: Terminal voltage, volts.
    voltage: np.ndarray
    #: Mid-cell surface temperature in kelvin, or ``None`` if not recorded.
    temperature: np.ndarray | None
    ambient: float
    c_rate: float

    @property
    def temperature_rise(self) -> float:
        """Peak minus initial surface temperature, kelvin. Zero if unrecorded."""
        if self.temperature is None:
            return 0.0
        return float(np.max(self.temperature) - self.temperature[0])


def default_cache() -> Path:
    """Where :func:`download` puts things: ``_data`` beside the project root."""
    return Path.cwd() / "_data"


def download(cache: Path | str | None = None) -> Path:
    """Fetch the dataset if it is not already present, and return its directory.

    Deliberately explicit rather than automatic. A test suite that silently
    reaches out to the network is a test suite that fails for reasons unrelated
    to the code.
    """
    directory = Path(cache) if cache is not None else default_cache()
    directory.mkdir(parents=True, exist_ok=True)
    for url in (RATE_TEST_URL, OCV_URL, PULSE_URL):
        target = directory / url.rsplit("/", 1)[-1]
        if not target.exists():
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                target.write_bytes(response.read())
    return directory


def _root(cache: Path | str | None, filename: str, key: str):
    from scipy.io import loadmat

    directory = Path(cache) if cache is not None else default_cache()
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run cellkernel.data.reference.download() first; "
            "the dataset is not distributed with this package."
        )
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)[key]


def load_ocv(cache: Path | str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Measured pseudo-open-circuit voltage at 25 C: ``(soc, volts)``.

    A pseudo-OCV, averaged over a slow charge and discharge to cancel hysteresis,
    which is what a cycler can actually produce. It is not a thermodynamic
    equilibrium curve and differs from one by a few millivolt.
    """
    struct = _root(cache, "LGM50_5Ah_OCV.mat", "LGM50_5Ah_OCV").T25
    return np.asarray(struct.refSoC, dtype=float) / 100.0, np.asarray(struct.meanOCV, dtype=float)


def available_conditions(cache: Path | str | None = None) -> list[tuple[str, str]]:
    """``(ambient, rate)`` keys present in the file, in order."""
    root = _root(cache, "LGM50_5Ah_RateTest.mat", "LGM50_5Ah_RateTest")
    found = []
    for ambient in AMBIENTS:
        if not hasattr(root, ambient):
            continue
        block = getattr(root, ambient)
        found.extend((ambient, rate) for rate in RATES if hasattr(block, rate))
    return found


def load_discharge(
    ambient: str = "T25",
    rate: str = "cRate_1C",
    dt: float = 1.0,
    cache: Path | str | None = None,
) -> DischargeSegment:
    """Extract and resample the constant-current discharge from one rate test.

    Each recording is a full cycle -- charge, rest, discharge, rest -- so the
    discharge has to be found rather than assumed. The longest continuous run of
    negative current in the source file is taken, which is unambiguous here
    because the charge leg is a third of the rate and the rests are at zero.

    Sample spacing in the source varies from one second to a minute depending on
    the rate, so the segment is resampled onto a uniform grid at ``dt``.
    """
    if ambient not in AMBIENTS:
        raise ValueError(f"unknown ambient {ambient!r}; expected one of {sorted(AMBIENTS)}")
    if rate not in RATES:
        raise ValueError(f"unknown rate {rate!r}; expected one of {sorted(RATES)}")
    root = _root(cache, "LGM50_5Ah_RateTest.mat", "LGM50_5Ah_RateTest")
    block = getattr(root, ambient, None)
    segment = getattr(block, rate, None) if block is not None else None
    if segment is None:
        raise KeyError(f"{ambient}/{rate} is not in this dataset")

    time = np.asarray(segment.timeVec, dtype=float)
    current = -np.asarray(segment.currVec, dtype=float)  # source is charge-positive
    voltage = np.asarray(segment.volVec, dtype=float)

    active = current > 0.05
    edges = np.diff(active.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, active.size]
    runs = [(a, b) for a, b in zip(starts, ends, strict=False) if b - a > 20]
    if not runs:
        raise RuntimeError(f"no discharge found in {ambient}/{rate}")
    first, last = max(runs, key=lambda r: time[r[1] - 1] - time[r[0]])

    span = time[first:last] - time[first]
    grid = np.arange(0.0, span[-1] + dt, dt)
    thermocouple = getattr(segment, "cellTemp_mid", None)
    resampled_temperature = None
    if thermocouple is not None:
        celsius = np.asarray(thermocouple, dtype=float)[first:last]
        resampled_temperature = np.interp(grid, span, celsius) + 273.15

    return DischargeSegment(
        time=grid,
        current=np.interp(grid, span, current[first:last]),
        voltage=np.interp(grid, span, voltage[first:last]),
        temperature=resampled_temperature,
        ambient=AMBIENTS[ambient],
        c_rate=RATES[rate],
    )


@dataclass(frozen=True)
class PulseSegment:
    """One current pulse and the rest that follows it, on a uniform grid.

    This is the experiment that makes a physics-based model identifiable, and the
    reason is visible in the numbers: on the reference cell a 1.5C pulse drops the
    terminal voltage 234 mV the instant it is applied and a further 35 mV over the
    next ten seconds. The first is ohmic and charge transfer, the second is
    diffusion. A constant-current discharge sums the two into a single number and
    no fit can take them apart again; here they are separated by a factor of a
    hundred in timescale and the separation is what a fit reads.
    """

    #: Seconds from the start of the pulse.
    time: np.ndarray
    #: Amperes, positive on discharge.
    current: np.ndarray
    voltage: np.ndarray
    #: Terminal voltage immediately before the pulse, after a long rest. Used to
    #: set the model's initial state rather than trusting the nominal level.
    rest_voltage: float
    ambient: float
    level: str
    #: The dataset's own name for the physical cell, such as ``Cell19``. Worth
    #: carrying: each ambient was run on a *different* set of six cells, so a
    #: trend across temperature is confounded with cell-to-cell scatter, which on
    #: this dataset is 6% of the series resistance.
    cell_name: str = ""

    @property
    def series_resistance(self) -> float:
        """Ohms, from the leading edge: the ohmic step divided by the current.

        The standard hybrid-pulse reading, and the one parameter a pulse test
        gives up without any fitting at all. Everything slower than one sample is
        excluded by construction, so this is collector, electrolyte and film
        resistance plus the fastest part of charge transfer -- not diffusion.
        """
        peak = float(np.max(self.current))
        if peak <= 0.0:  # pragma: no cover - defensive
            return float("nan")
        return self.ohmic_step / peak

    @property
    def ohmic_step(self) -> float:
        """Instantaneous voltage drop at the leading edge, in volts.

        Everything fast enough to be complete within one sample: the ohmic drop
        through collectors, electrolyte and film, plus the fastest part of charge
        transfer. At 10 Hz these are not separable from each other, but they are
        cleanly separable from diffusion.
        """
        return float(self.rest_voltage - self.voltage[0])


def available_pulses(cache: Path | str | None = None) -> list[tuple[str, str]]:
    """``(ambient, level)`` keys present in the pulse file."""
    root = _root(cache, "LGM50_5Ah_Pulse.mat", "LGM50_5Ah_Pulse")
    found = []
    for ambient in AMBIENTS:
        block = getattr(root, ambient, None)
        if block is None:
            continue
        found.extend(
            (ambient, level)
            for level in sorted(
                (name for name in dir(block) if name.startswith("SoC")),
                key=lambda name: int(name[3:]),
            )
        )
    return found


def load_pulse(
    ambient: str = "T25",
    level: str = "SoC5",
    cell_index: int = 1,
    dt: float = 0.1,
    cache: Path | str | None = None,
) -> PulseSegment:
    """Load one pulse-and-rest from the hybrid pulse power characterisation file.

    Parameters
    ----------
    ambient
        ``T0``, ``T10``, ``T25`` or ``T45``.
    level
        ``SoC1`` through ``SoC9``, descending in charge.
    cell_index
        Which of the six cells at this ambient, counted from one **by position**
        rather than by the dataset's own numbering -- each ambient was run on a
        different set of six, numbered 1-6 at 25 C but 19-24 at 0 C, and indexing
        by the vendor's name would silently fail on three temperatures out of
        four. :attr:`PulseSegment.cell_name` reports which physical cell it was.

        Comparing two of them is the cheapest available estimate of how much of a
        model's error is cell-to-cell scatter rather than anything a model could
        fix. Here it is 6% of the series resistance.
    dt
        Resample period. The source is logged near 10 Hz; anything much coarser
        blurs the leading edge, which is the part carrying the ohmic information.
    """
    root = _root(cache, "LGM50_5Ah_Pulse.mat", "LGM50_5Ah_Pulse")
    block = getattr(root, ambient, None)
    if block is None:
        raise ValueError(f"unknown ambient {ambient!r}")
    holder = getattr(block, level, None)
    if holder is None:
        raise ValueError(f"unknown level {level!r}")
    names = sorted(
        (name for name in dir(holder) if name.startswith("Cell")),
        key=lambda name: int(name[4:]),
    )
    if not 1 <= cell_index <= len(names):
        raise ValueError(f"cell_index must be 1..{len(names)} for {ambient}/{level}")
    cell_name = names[cell_index - 1]
    data = getattr(holder, cell_name).data

    time = np.asarray(data.ProgTime, dtype=float)
    current = -np.asarray(data.Current, dtype=float)  # source is charge-positive
    voltage = np.asarray(data.Voltage, dtype=float)

    active = np.flatnonzero(current > 0.05)
    if active.size == 0:
        raise RuntimeError(f"no pulse found in {ambient}/{level}")
    first = int(active[0])
    # The rested voltage is the highest of the samples up to the edge, not simply
    # the one before it. On some records the logger reports the current a sample
    # later than the voltage it caused, so the immediately preceding sample is
    # already under load -- on one level of this dataset that turned a 247 mV
    # ohmic step into 2 mV and made the segment unusable without saying so. Only
    # a discharge pulse is handled here, so the rested value is the maximum.
    rest_voltage = float(np.max(voltage[: first + 1]))

    span = time[first:] - time[first]
    order = np.argsort(span)
    span, ordered = span[order], order
    grid = np.arange(0.0, span[-1] + dt, dt)
    return PulseSegment(
        time=grid,
        current=np.interp(grid, span, current[first:][ordered]),
        voltage=np.interp(grid, span, voltage[first:][ordered]),
        rest_voltage=rest_voltage,
        ambient=AMBIENTS[ambient],
        level=level,
        cell_name=cell_name,
    )


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    where = download()
    print(f"downloaded to {where}")
    for ambient, rate in available_conditions():
        segment = load_discharge(ambient, rate, cache=where)
        rise = (
            f"{segment.temperature_rise:5.1f} K" if segment.temperature is not None else "      -"
        )
        print(
            f"  {ambient:>4s} {rate.replace('cRate_', ''):>5s}  "
            f"{segment.time[-1]:7.0f} s  {segment.current.mean():6.2f} A  "
            f"{segment.voltage[0]:.3f} -> {segment.voltage[-1]:.3f} V   rise {rise}"
        )
