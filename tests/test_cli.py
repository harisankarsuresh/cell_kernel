"""Validation of the command-line interface.

The CLI is how anyone tries the package before reading any of it, so a broken
subcommand costs more than its share. These tests drive ``main`` directly with
argument vectors rather than spawning subprocesses, which keeps them fast enough
to run on every commit.
"""

from __future__ import annotations

import csv

import pytest

from cellkernel.cli import main
from cellkernel.verify import find_compiler

needs_cc = pytest.mark.skipif(find_compiler() is None, reason="no C compiler on PATH")


def run(*argv: str) -> int:
    return main(list(argv))


# ------------------------------------------------------------------- plumbing


def test_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run("--version")
    assert excinfo.value.code == 0
    assert "cellkernel" in capsys.readouterr().out


def test_a_command_is_required():
    with pytest.raises(SystemExit) as excinfo:
        run()
    assert excinfo.value.code != 0


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        run("chargify")


# ----------------------------------------------------------------------- roms


def test_roms_reports_every_family(capsys):
    assert run("roms") == 0
    out = capsys.readouterr().out
    for family in ("pade", "spectral", "fv", "poly"):
        assert family in out


def test_roms_honours_the_chemistry_choice(capsys):
    assert run("roms", "--chemistry", "lfp") == 0
    assert "lfp" in capsys.readouterr().out


def test_roms_rejects_an_unknown_chemistry():
    with pytest.raises(SystemExit):
        run("roms", "--chemistry", "sodium")


# ------------------------------------------------------------------- simulate


def test_simulate_writes_a_readable_csv(tmp_path, capsys):
    out = tmp_path / "record.csv"
    assert run("simulate", "--out", str(out), "--duration", "120") == 0
    assert "wrote" in capsys.readouterr().out

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 100
    assert set(rows[0]) == {"time", "current", "voltage", "soc"}
    assert all(2.0 < float(row["voltage"]) < 4.5 for row in rows)


def test_simulate_output_round_trips_through_the_loader(tmp_path):
    """What the CLI writes must be what the loader reads. It is the obvious
    pairing, and the obvious thing to leave broken."""
    from cellkernel.data import load_csv

    out = tmp_path / "record.csv"
    assert run("simulate", "--out", str(out), "--duration", "60") == 0
    data = load_csv(out)
    assert data["time"].size > 50
    assert data["voltage"].min() > 2.0


def test_simulate_to_stdout(capsys):
    assert run("simulate", "--duration", "10") == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("time,current,voltage")


def test_simulate_noise_is_reproducible(tmp_path):
    first, second = tmp_path / "1.csv", tmp_path / "2.csv"
    for path in (first, second):
        assert (
            run(
                "simulate",
                "--out",
                str(path),
                "--duration",
                "30",
                "--noise",
                "0.005",
                "--seed",
                "7",
            )
            == 0
        )
    assert first.read_text() == second.read_text()


# ------------------------------------------------------------------- generate


def test_generate_emits_a_buildable_project(tmp_path, capsys):
    out = tmp_path / "gen"
    assert run("generate", str(out), "--precision", "float") == 0
    for name in ("cellkernel_estimator.h", "cellkernel_estimator.c", "Makefile", "BUDGET.txt"):
        assert (out / name).is_file(), name
    assert "flash" in capsys.readouterr().out


def test_generate_scheduled_emits_the_other_estimator(tmp_path, capsys):
    out = tmp_path / "sched"
    assert run("generate", str(out), "--precision", "float", "--scheduled") == 0
    assert (out / "cellkernel_scheduled.c").is_file()
    assert not (out / "cellkernel_estimator.c").exists()
    captured = capsys.readouterr()
    # The parameter set carries no activation energies, so the CLI substitutes
    # representative ones and must say so rather than silently producing a
    # schedule that interpolates between identical matrices.
    assert "activation energies" in captured.err


def test_generate_respects_the_rom_choice(tmp_path):
    """Different families give different state counts, which the header records."""
    import re

    def state_count(directory) -> int:
        header = (directory / "cellkernel_estimator.h").read_text()
        match = re.search(r"#define\s+CK_N_STATES\s+(\d+)", header)
        assert match, "header must declare CK_N_STATES"
        return int(match.group(1))

    assert run("generate", str(tmp_path / "poly"), "--rom", "poly") == 0
    assert run("generate", str(tmp_path / "pade"), "--rom", "pade", "--order", "4") == 0
    # Two states per electrode for the moment closure, four for a fourth-order Pade.
    assert state_count(tmp_path / "poly") == 4
    assert state_count(tmp_path / "pade") == 8


def test_generate_rejects_an_unknown_rom():
    with pytest.raises(SystemExit):
        run("generate", "out", "--rom", "chebyshev")


# --------------------------------------------------------------------- verify


@needs_cc
def test_verify_compiles_and_reports(tmp_path, capsys):
    out = tmp_path / "verified"
    assert run("verify", str(out), "--precision", "double", "--duration", "120") == 0
    report = capsys.readouterr().out
    assert "PASS" in report
    assert "generated C vs NumPy mirror" in report


def test_charge_reports_a_grid(capsys):
    assert run("charge", "--max-c-rate", "2.0") == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip().startswith("0.")]
    assert len(lines) >= 5
    rates = [float(value) for line in lines for value in line.split()[1:]]
    assert all(0.0 <= rate <= 2.0 + 1e-9 for rate in rates), "must respect the ceiling"


def test_charge_accepts_a_single_temperature(capsys):
    assert run("charge", "--temperature", "263.15") == 0
    assert "-10C" in capsys.readouterr().out


def test_charge_is_stricter_in_the_cold(capsys):
    """The whole point of the subcommand, checked rather than assumed."""
    run("charge", "--temperature", "263.15", "313.15")
    rows = [
        line.split()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("0.7")
    ]
    assert rows, "expected a row at 70% state of charge"
    cold, warm = float(rows[0][1]), float(rows[0][2])
    assert cold < warm


def test_age_reports_the_u_shape(capsys):
    assert run("age", "--cycles", "50") == 0
    out = capsys.readouterr().out
    assert "plating" in out
    assert "interphase" in out
    assert "Best at" in out


def test_temperature_dependent_commands_warn_about_borrowed_parameters(capsys):
    """Silently substituting activation energies would be the worse failure."""
    run("charge", "--temperature", "298.15")
    assert "activation energies" in capsys.readouterr().err


def test_verify_without_a_compiler_fails_helpfully(tmp_path, capsys, monkeypatch):
    """A missing toolchain is a setup problem, and should read like one."""
    monkeypatch.setattr("cellkernel.verify.find_compiler", lambda: None)
    code = run("verify", str(tmp_path / "x"), "--compiler", "")
    assert code != 0
    assert "compiler" in capsys.readouterr().err.lower()
