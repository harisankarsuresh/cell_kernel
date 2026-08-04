"""Synthetic current profiles and cycler data loading.

Everything here runs without proprietary data. That is deliberate: the examples
and tests in this package have to be reproducible by someone who has no access to
a laboratory, and a library whose demonstrations depend on files nobody else has
cannot be evaluated by anyone.

For real measurements, :func:`load_csv` handles the common tabular cycler export.
It is intentionally thin -- parsing the many vendor binary formats is a solved
problem, and `cellpy <https://github.com/jepegit/cellpy>`__,
`BEEP <https://github.com/TRI-AMDD/beep>`__ and ``battery-data-standard`` all do it
better than a reimplementation here would. Convert with one of those, then bring
the result in.
"""

from __future__ import annotations

from .cycles import (
    cccv_charge,
    constant_current,
    hppc_pulses,
    rest,
    synthetic_drive_cycle,
)
from .io import load_csv

__all__ = [
    "cccv_charge",
    "constant_current",
    "hppc_pulses",
    "load_csv",
    "rest",
    "synthetic_drive_cycle",
]
