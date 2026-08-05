"""Validation of cycler-file loading.

Thin by design -- parsing vendor binary formats is a solved problem and cellpy,
BEEP and battery-data-standard all do it better than a reimplementation would.
What is here is the tabular common case, and it is tested because it is the entry
point for anyone bringing real measurements in, and because the sign convention
is the single easiest thing in this package to get wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellkernel.data import load_csv


def write(path, text: str):
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------- column matching


def test_reads_a_plain_file(tmp_path):
    path = write(
        tmp_path / "a.csv",
        """
        time,current,voltage
        0,1.0,4.10
        1,1.0,4.09
        2,1.0,4.08
        """,
    )
    data = load_csv(path)
    assert np.allclose(data["time"], [0, 1, 2])
    assert np.allclose(data["current"], [1.0, 1.0, 1.0])
    assert np.allclose(data["voltage"], [4.10, 4.09, 4.08])
    assert "temperature" not in data


@pytest.mark.parametrize(
    "header",
    [
        "Test Time (s),Current (A),Voltage (V)",
        "test_time,current_a,voltage_v",
        "TIME,I,V",
        "Time [s],Current [A],Voltage [V]",
        "total_time,Amps,Volts",
    ],
)
def test_matches_the_spellings_cyclers_actually_use(tmp_path, header):
    """Every vendor names these differently, and none of them agree."""
    path = write(tmp_path / "b.csv", f"{header}\n0,2.0,3.9\n1,2.0,3.8")
    data = load_csv(path)
    assert np.allclose(data["current"], [2.0, 2.0])
    assert np.allclose(data["voltage"], [3.9, 3.8])


def test_reads_temperature_when_present(tmp_path):
    path = write(
        tmp_path / "c.csv",
        """
        time,current,voltage,temperature
        0,1.0,4.1,298.1
        1,1.0,4.0,298.4
        """,
    )
    data = load_csv(path)
    assert np.allclose(data["temperature"], [298.1, 298.4])


def test_reports_which_column_is_missing(tmp_path):
    path = write(tmp_path / "d.csv", "time,voltage\n0,4.1\n1,4.0")
    with pytest.raises(ValueError, match="current"):
        load_csv(path)


def test_rejects_a_headerless_file(tmp_path):
    path = write(tmp_path / "e.csv", "")
    with pytest.raises(ValueError, match="header"):
        load_csv(path)


def test_tolerates_a_byte_order_mark(tmp_path):
    """Excel writes one, and it silently breaks naive header matching."""
    path = tmp_path / "f.csv"
    path.write_text("\ufefftime,current,voltage\n0,1.0,4.1\n1,1.0,4.0\n", encoding="utf-8")
    data = load_csv(path)
    assert np.allclose(data["current"], [1.0, 1.0])


# ---------------------------------------------------------------- sign handling


def test_charge_positive_files_are_negated(tmp_path):
    """The single easiest thing in this package to get wrong.

    Neware and Arbin text exports commonly report charge as positive, which is
    the opposite of the convention used throughout ``cellkernel``. Getting it
    backwards produces a state-of-charge estimate that moves confidently in the
    wrong direction, so the caller is required to say which they have rather than
    the loader guessing.
    """
    path = write(tmp_path / "g.csv", "time,current,voltage\n0,-1.0,4.1\n1,-1.0,4.0")
    as_is = load_csv(path, current_sign="discharge-positive")
    flipped = load_csv(path, current_sign="charge-positive")
    assert np.allclose(as_is["current"], [-1.0, -1.0])
    assert np.allclose(flipped["current"], [1.0, 1.0])


def test_rejects_an_unknown_sign_convention(tmp_path):
    path = write(tmp_path / "h.csv", "time,current,voltage\n0,1.0,4.1")
    with pytest.raises(ValueError, match="current_sign"):
        load_csv(path, current_sign="whatever")


# -------------------------------------------------------------------- cleaning


def test_rows_with_missing_values_are_dropped(tmp_path):
    path = write(
        tmp_path / "i.csv",
        """
        time,current,voltage
        0,1.0,4.10
        1,,4.09
        2,1.0,4.08
        """,
    )
    data = load_csv(path)
    assert data["time"].size == 2
    assert np.allclose(data["time"], [0, 2])


def test_temperature_is_dropped_alongside_its_row(tmp_path):
    """Every column must stay the same length, or downstream indexing is wrong."""
    path = write(
        tmp_path / "j.csv",
        """
        time,current,voltage,temperature
        0,1.0,4.10,298
        1,,4.09,299
        2,1.0,4.08,300
        """,
    )
    data = load_csv(path)
    assert data["temperature"].size == data["time"].size == 2
    assert np.allclose(data["temperature"], [298, 300])


# ------------------------------------------------------------------ resampling


def test_resampling_puts_the_record_on_a_uniform_grid(tmp_path):
    """Needed because every model here runs at a fixed step.

    Cycler logs are rarely uniform: most write on a change of value or on a timer
    that drifts, so raw sample spacing can vary by an order of magnitude within
    one file.
    """
    path = write(
        tmp_path / "k.csv",
        """
        time,current,voltage
        0,1.0,4.00
        1,1.0,3.98
        7,1.0,3.86
        10,1.0,3.80
        """,
    )
    data = load_csv(path, resample_dt=1.0)
    assert np.allclose(np.diff(data["time"]), 1.0)
    assert data["time"][0] == pytest.approx(0.0)
    assert data["time"][-1] == pytest.approx(10.0)
    # Linear interpolation between the 1 s and 7 s samples.
    assert data["voltage"][4] == pytest.approx(3.98 + (3.86 - 3.98) * 3.0 / 6.0)


def test_resampling_carries_temperature_too(tmp_path):
    path = write(
        tmp_path / "l.csv",
        """
        time,current,voltage,temperature
        0,1.0,4.0,300
        4,1.0,3.9,304
        """,
    )
    data = load_csv(path, resample_dt=2.0)
    assert np.allclose(data["temperature"], [300.0, 302.0, 304.0])


def test_resampling_preserves_the_charge_passed(tmp_path):
    """The quantity that must survive a regrid, since it sets state of charge."""
    rng = np.random.default_rng(0)
    times = np.sort(rng.choice(np.arange(0, 600), size=200, replace=False)).astype(float)
    current = 2.0 + np.sin(times / 60.0)
    lines = ["time,current,voltage"]
    lines += [f"{t},{i},{3.9}" for t, i in zip(times, current, strict=True)]
    path = write(tmp_path / "m.csv", "\n".join(lines))

    raw = load_csv(path)
    resampled = load_csv(path, resample_dt=1.0)
    before = np.trapezoid(raw["current"], raw["time"])
    after = np.trapezoid(resampled["current"], resampled["time"])
    assert after == pytest.approx(before, rel=2e-3)
