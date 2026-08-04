"""Loading tabular cycler data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["load_csv"]

_TIME_KEYS = ("time", "test_time", "total_time", "step_time", "t", "time_s", "elapsed")
_CURRENT_KEYS = ("current", "current_a", "i", "amps", "current(a)")
_VOLTAGE_KEYS = ("voltage", "voltage_v", "v", "volts", "voltage(v)", "ewe")
_TEMPERATURE_KEYS = ("temperature", "temp", "aux_temperature", "t_cell", "temperature_c")


def _match(header: list[str], candidates: tuple[str, ...]) -> str | None:
    normalised = {h.strip().lower().replace(" ", "_").replace("[", "(").replace("]", ")"): h
                  for h in header}
    for key in candidates:
        if key in normalised:
            return normalised[key]
    for norm, original in normalised.items():
        for key in candidates:
            if norm.startswith(key):
                return original
    return None


def load_csv(
    path: str | Path,
    current_sign: str = "discharge-positive",
    resample_dt: float | None = None,
) -> dict[str, np.ndarray]:
    """Read a tabular cycler export into time, current, voltage and temperature.

    Column names are matched case-insensitively against a list of common spellings,
    so ``Current (A)``, ``current_a`` and ``I`` all work.

    Parameters
    ----------
    path
        CSV file.
    current_sign
        ``"discharge-positive"`` if the file already uses this package's
        convention, or ``"charge-positive"`` to negate on load.

        There is no default that is safe to guess. Neware and Arbin text exports
        commonly report charge as positive; some tools report an unsigned magnitude
        with the direction carried in a separate step label. Inferring the sign
        from the data is possible but unreliable -- a file that happens to contain
        only discharge is indistinguishable either way -- and getting it backwards
        produces a state-of-charge estimate that moves confidently in the wrong
        direction. It is better to require the caller to say.
    resample_dt
        If given, linearly interpolate onto a uniform grid at this period. Needed
        because every model here runs at a fixed step, and cycler logs are usually
        non-uniform: most write on a change of value or on a timer that drifts, so
        raw sample spacing varies by an order of magnitude within one file.

    Returns
    -------
    dict
        Arrays keyed ``time``, ``current``, ``voltage`` and, when present,
        ``temperature``.
    """
    if current_sign not in {"discharge-positive", "charge-positive"}:
        raise ValueError("current_sign must be 'discharge-positive' or 'charge-positive'")

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        header = list(reader.fieldnames)
        cols = {
            "time": _match(header, _TIME_KEYS),
            "current": _match(header, _CURRENT_KEYS),
            "voltage": _match(header, _VOLTAGE_KEYS),
            "temperature": _match(header, _TEMPERATURE_KEYS),
        }
        missing = [k for k in ("time", "current", "voltage") if cols[k] is None]
        if missing:
            raise ValueError(
                f"{path}: could not find column(s) for {', '.join(missing)}; "
                f"header is {header}"
            )
        rows = list(reader)

    def column(key: str) -> np.ndarray | None:
        name = cols[key]
        if name is None:
            return None
        out = np.empty(len(rows))
        for i, row in enumerate(rows):
            text = (row.get(name) or "").strip()
            out[i] = float(text) if text else np.nan
        return out

    data = {
        "time": column("time"),
        "current": column("current"),
        "voltage": column("voltage"),
    }
    temperature = column("temperature")
    if temperature is not None:
        data["temperature"] = temperature

    if current_sign == "charge-positive":
        data["current"] = -data["current"]

    finite = np.isfinite(data["time"]) & np.isfinite(data["current"]) & np.isfinite(
        data["voltage"]
    )
    for key in list(data):
        data[key] = data[key][finite]

    if resample_dt is not None:
        t = data["time"]
        grid = np.arange(t[0], t[-1] + 0.5 * resample_dt, resample_dt)
        resampled = {"time": grid}
        for key, values in data.items():
            if key != "time":
                resampled[key] = np.interp(grid, t, values)
        data = resampled

    return data
