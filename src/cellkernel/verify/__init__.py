"""Compile the generated C, replay it, and compare against Python.

Generating code is a claim; this module is the evidence. It compiles the emitted
sources with warnings as errors, drives the resulting binary with a current
profile, runs the same profile through Python, and reports where the two diverge.

The comparison is deliberately made in three legs, because lumping them together
would hide which one actually matters:

**C against its Python mirror.** Both evaluate the same lookup tables, the same
``asinh`` formulation and the same loop order, so any disagreement beyond
round-off is a code-generation defect. In double precision this should sit near
1e-14; in single precision near 1e-6 relative. This is the leg that validates the
generator.

**The mirror against the table-backed model.** Zero by construction if the mirror
is faithful. It exists to catch a mirror that has drifted from the model it claims
to reproduce.

**The table-backed model against the full model.** This is the price of
representing a potential as a lookup table, and it is a modelling choice rather
than an error -- but it belongs in the error budget, stated in millivolts, not
left implicit.

Separating them matters in practice. A 3 mV discrepancy between generated C and a
reference model is unremarkable if it is all lookup-table resolution and alarming
if it is arithmetic, and the aggregate number alone cannot tell you which.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..codegen import GeneratedProject
from ..codegen.spec import ReferenceEstimator, table_backed_model

__all__ = ["VerificationReport", "find_compiler", "compile_project", "verify"]


def find_compiler() -> str | None:
    """Locate a usable C compiler, or return ``None``.

    Checks ``CC`` first so a caller can pin a specific toolchain, then the usual
    names. ``cl`` is excluded: it does not accept the GCC-style flags the
    generated Makefile uses, and supporting it would mean maintaining a second
    flag set for a case that adds nothing to the verification argument.
    """
    candidates = []
    env_cc = os.environ.get("CC")
    if env_cc:
        candidates.append(env_cc)
    candidates += ["cc", "gcc", "clang"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of a generated-versus-Python comparison."""

    compiler: str
    precision: str
    mode: str
    n_samples: int
    n_states: int

    #: Generated C against its NumPy mirror. The code-generation error.
    max_voltage_error_vs_mirror: float
    max_soc_error_vs_mirror: float

    #: NumPy mirror against the table-backed Python model. Should be ~0.
    max_voltage_error_mirror_vs_model: float

    #: Table-backed against full-fidelity model. The lookup-table price.
    max_voltage_error_table_vs_exact: float

    #: End to end: generated C against the full-fidelity Python model.
    max_voltage_error_total: float

    table_error_volts: float
    compile_warnings: str

    @property
    def passed(self) -> bool:
        """Whether the code-generation legs are within tolerance.

        Judges only what the generator is responsible for. The lookup-table leg is
        a deliberate approximation and is reported, not gated -- tightening it is a
        matter of raising ``table_points``, not of fixing a defect.

        Single-precision tolerances are looser by design. The covariance
        propagation accumulates rounding over a triple loop, and the emitted
        constants are themselves rounded to 24 bits, so agreement at the 1e-5 V
        level is the expected outcome rather than a shortfall -- and it is two
        orders of magnitude below the measurement noise of any real front end.
        """
        tol_v = 1e-9 if self.precision == "double" else 5e-4
        tol_soc = 1e-10 if self.precision == "double" else 1e-4
        ok = self.max_voltage_error_vs_mirror < tol_v and self.max_soc_error_vs_mirror < tol_soc
        if self.mode == "openloop":
            ok = ok and self.max_voltage_error_mirror_vs_model < 1e-9
        return ok

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"verification {verdict}  ({self.precision}, {self.n_states} states, "
            f"{self.n_samples} samples, {self.mode}, {Path(self.compiler).name})",
            "",
            "  code generation fidelity",
            f"    generated C vs NumPy mirror, voltage   {self.max_voltage_error_vs_mirror:.3e} V",
            f"    generated C vs NumPy mirror, SoC       {self.max_soc_error_vs_mirror:.3e}",
        ]
        if self.mode == "openloop":
            lines += [
                f"    mirror vs table-backed model, voltage  "
                f"{self.max_voltage_error_mirror_vs_model:.3e} V",
                "",
                "  deliberate approximation",
                f"    lookup table vs analytic fit, voltage  "
                f"{self.max_voltage_error_table_vs_exact:.3e} V"
                f"  ({self.max_voltage_error_table_vs_exact * 1e3:.3f} mV)",
                f"    table interpolation error, worst       {self.table_error_volts * 1e3:.3f} mV",
                "",
                "  end to end",
                f"    generated C vs full Python model        "
                f"{self.max_voltage_error_total:.3e} V"
                f"  ({self.max_voltage_error_total * 1e3:.3f} mV)",
            ]
        else:
            lines += [
                "",
                "  the remaining legs are not evaluated in filter mode: the mirror",
                "  applies Kalman corrections while the comparison model runs open",
                "  loop, so the two are different algorithms and would diverge for",
                "  reasons that say nothing about code generation. Run open loop to",
                "  measure model fidelity, and filter mode to exercise the Kalman",
                "  path -- which the first leg above does.",
            ]
        return "\n".join(lines)


def compile_project(project: GeneratedProject, compiler: str | None = None) -> tuple[Path, str]:
    """Compile the harness. Returns the executable path and any warning text.

    Compiled with ``-Wall -Wextra -Wpedantic -Werror``. Treating warnings as
    errors is not decoration: an implicit conversion or an uninitialised read in
    generated numerical code is a defect, and it is far cheaper to fail here than
    to chase a millivolt bias later.
    """
    cc = compiler or find_compiler()
    if cc is None:
        raise RuntimeError("no C compiler found; set CC or install gcc or clang")
    out = project.directory / ("ck_harness.exe" if os.name == "nt" else "ck_harness")
    cmd = [
        cc,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-o",
        str(out),
        str(project.directory / "ck_harness.c"),
        str(project.directory / "cellkernel_estimator.c"),
        "-lm",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(
            "generated C failed to compile:\n" + " ".join(cmd) + "\n" + done.stdout + done.stderr
        )
    return out, (done.stderr or "").strip()


def run_harness(
    executable: Path,
    current: np.ndarray,
    voltage: np.ndarray,
    initial_soc: float,
    mode: str = "openloop",
) -> dict[str, np.ndarray]:
    """Drive the compiled harness with a current and voltage sequence."""
    buffer = io.StringIO()
    # Written with 17 significant digits rather than repr(): under NumPy 2 the
    # repr of a scalar is 'np.float64(1.5)', which scanf cannot parse, and 17
    # digits round-trips a double exactly anyway.
    for i_k, v_k in zip(
        np.asarray(current).reshape(-1), np.asarray(voltage).reshape(-1), strict=True
    ):
        buffer.write(f"{float(i_k):.17g},{float(v_k):.17g}\n")
    done = subprocess.run(
        [str(executable), mode, f"{float(initial_soc):.17g}"],
        input=buffer.getvalue(),
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"harness failed: {done.stderr}")
    reader = csv.DictReader(io.StringIO(done.stdout))
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"harness produced no output; stderr: {done.stderr}")
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def _run_mirror(
    project: GeneratedProject,
    current: np.ndarray,
    voltage: np.ndarray,
    initial_soc: float,
    mode: str,
) -> dict[str, np.ndarray]:
    """Replay the NumPy mirror with the same call sequence as the harness."""
    est = ReferenceEstimator(project.spec)
    est.init(initial_soc)
    n = current.size
    out = {"soc": np.empty(n), "voltage": np.empty(n)}
    for k in range(n):
        out["soc"][k] = est.soc()
        out["voltage"][k] = est.voltage(float(current[k]))
        est.predict(float(current[k]))
        if mode == "filter":
            est.update(float(current[k]), float(voltage[k]))
    return out


def verify(
    project: GeneratedProject,
    model,
    current: np.ndarray,
    initial_soc: float = 1.0,
    compiler: str | None = None,
    mode: str = "openloop",
) -> VerificationReport:
    """Compile, replay and compare a generated estimator against Python.

    Parameters
    ----------
    project
        Result of :func:`cellkernel.codegen.generate`.
    model
        The same :class:`~cellkernel.models.spm.SPM` the project was generated
        from.
    current
        Current profile in amperes, positive on discharge.
    initial_soc
        Starting state of charge for both implementations.
    compiler
        Override the detected compiler.
    mode
        ``"openloop"`` runs the model forward only, which is the right default:
        it isolates model and code-generation fidelity from filter dynamics, and
        an open loop cannot mask a systematic error the way a closed loop can.
        ``"filter"`` additionally exercises the Kalman path.
    """
    current = np.asarray(current, dtype=float).reshape(-1)
    executable, warnings = compile_project(project, compiler)

    # Reference voltages come from the full-fidelity model, and double as the
    # measurement the filter path is fed.
    exact = model.simulate(current, soc0=initial_soc)
    measured = exact["voltage"]

    c_out = run_harness(executable, current, measured, initial_soc, mode)
    mirror = _run_mirror(project, current, measured, initial_soc, mode)

    table_model = table_backed_model(model, project.spec)
    table_run = table_model.simulate(current, soc0=initial_soc)

    if mode == "openloop":
        mirror_vs_model = float(np.max(np.abs(mirror["voltage"] - table_run["voltage"])))
        total = float(np.max(np.abs(c_out["voltage"] - exact["voltage"])))
    else:
        mirror_vs_model = float("nan")
        total = float("nan")

    return VerificationReport(
        compiler=compiler or (find_compiler() or "cc"),
        precision=project.precision,
        mode=mode,
        n_samples=int(current.size),
        n_states=project.spec.n_states,
        max_voltage_error_vs_mirror=float(np.max(np.abs(c_out["voltage"] - mirror["voltage"]))),
        max_soc_error_vs_mirror=float(np.max(np.abs(c_out["soc"] - mirror["soc"]))),
        max_voltage_error_mirror_vs_model=mirror_vs_model,
        max_voltage_error_table_vs_exact=float(
            np.max(np.abs(table_run["voltage"] - exact["voltage"]))
        ),
        max_voltage_error_total=total,
        table_error_volts=project.budget.table_error_volts,
        compile_warnings=warnings,
    )
