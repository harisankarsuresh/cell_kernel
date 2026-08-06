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
    "DischargeSegment",
    "default_cache",
    "download",
    "load_ocv",
    "load_discharge",
    "available_conditions",
]

_BASE = "https://raw.githubusercontent.com/pybop-team/PyBOP/develop/examples/data/LG_M50_ECM/data"
RATE_TEST_URL = f"{_BASE}/LGM50_5Ah_RateTest.mat"
OCV_URL = f"{_BASE}/LGM50_5Ah_OCV.mat"

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
    for url in (RATE_TEST_URL, OCV_URL):
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
