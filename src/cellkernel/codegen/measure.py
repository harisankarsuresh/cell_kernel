"""Measure a generated estimator on a real ARM toolchain.

:mod:`cellkernel.codegen.budget` counts data structures and arithmetic
operations. That is exact for the tables and honest about being a model for the
rest, but it leaves out code size entirely -- and on this estimator the code
turns out to be roughly as large as the tables it reads, so a flash figure that
omits it is short by about half.

This module cross-compiles the emitted C for a Cortex-M target and reads the
section sizes out of the object file. No board is needed: the numbers come from
the linker's own accounting, which is the same accounting the firmware engineer
will do.

Instruction counts, where a QEMU build is available, come from running the
estimator on an emulated core with instruction counting enabled. Those are
*instructions retired*, not cycles -- QEMU does not model the pipeline, memory
waits or flash accelerator of any particular part. On a Cortex-M4F the two differ
by a factor of roughly one to two depending on how much of the work is
floating-point, so an instruction count is a lower bound on cycles and a
considerably better one than a hand-built model.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ArmMeasurement",
    "find_arm_toolchain",
    "find_qemu",
    "measure_arm_footprint",
    "measure_arm_instructions",
    "CORTEX_M4F",
    "CORTEX_M0",
]

#: Compiler flags for a Cortex-M4 with a single-precision FPU, the part most
#: automotive battery-management units use.
CORTEX_M4F = ("-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb")

#: A Cortex-M0+, which has no FPU at all. Included because the difference is the
#: honest answer to "can I run this on a cheap part": every floating-point
#: operation becomes a library call.
CORTEX_M0 = ("-mcpu=cortex-m0plus", "-mthumb")


@dataclass(frozen=True)
class ArmMeasurement:
    """Section sizes from a real cross-compilation, in bytes."""

    target: str
    optimisation: str
    #: Executable code.
    text_bytes: int
    #: Constant tables, which live in flash alongside the code.
    rodata_bytes: int
    #: Initialised writable data; costs both flash and RAM.
    data_bytes: int
    #: Zero-initialised writable data; costs RAM only.
    bss_bytes: int

    @property
    def flash_bytes(self) -> int:
        return self.text_bytes + self.rodata_bytes + self.data_bytes

    @property
    def ram_bytes(self) -> int:
        return self.data_bytes + self.bss_bytes

    def summary(self) -> str:
        return (
            f"{self.target} {self.optimisation}: "
            f"flash {self.flash_bytes} B "
            f"({self.text_bytes} code + {self.rodata_bytes} tables), "
            f"static RAM {self.ram_bytes} B"
        )


def find_arm_toolchain() -> str | None:
    """Locate ``arm-none-eabi-gcc``, or return ``None``.

    Checks ``PATH`` first, then the default install locations of the Arm GNU
    Toolchain on Windows, because its installer does not add itself to ``PATH``.
    """
    found = shutil.which("arm-none-eabi-gcc")
    if found:
        return found
    for root in (
        Path("C:/Program Files (x86)"),
        Path("C:/Program Files"),
        Path("/usr/bin"),
        Path("/opt"),
    ):
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("Arm GNU Toolchain*/**/arm-none-eabi-gcc.exe")):
            return str(candidate)
    return None


def measure_arm_footprint(
    project: Path | str,
    target: tuple[str, ...] = CORTEX_M4F,
    optimisation: str = "-Os",
    source: str = "cellkernel_estimator.c",
    compiler: str | None = None,
) -> ArmMeasurement:
    """Cross-compile one generated source file and report its section sizes.

    Compiles rather than links, deliberately. Linking would drag in whichever
    parts of the C library the maths functions need, and how much of that is
    already present is a property of the surrounding firmware rather than of this
    estimator. What is attributable here is the estimator's own code and tables.

    Parameters
    ----------
    project
        Directory produced by :func:`~cellkernel.codegen.generate`.
    target
        Architecture flags, for example :data:`CORTEX_M4F`.
    optimisation
        Optimisation level. ``-Os`` is the usual choice for firmware; ``-O2``
        trades a little flash for speed.
    source
        Which emitted source to measure, for the scheduled variant.
    compiler
        Path to ``arm-none-eabi-gcc``. Located automatically if omitted.
    """
    cc = compiler or find_arm_toolchain()
    if cc is None:
        raise RuntimeError(
            "no arm-none-eabi-gcc found; install the Arm GNU Toolchain or pass compiler= explicitly"
        )
    directory = Path(project)
    obj = directory / f"_measure_{optimisation.lstrip('-')}.o"
    command = [
        cc,
        *target,
        optimisation,
        "-std=c99",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-c",
        str(directory / source),
        "-o",
        str(obj),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    size = Path(cc).with_name(Path(cc).name.replace("gcc", "size"))
    listing = subprocess.run(
        [str(size), "-A", str(obj)], check=True, capture_output=True, text=True
    ).stdout
    obj.unlink(missing_ok=True)

    totals = dict.fromkeys(("text", "rodata", "data", "bss"), 0)
    for line in listing.splitlines():
        match = re.match(r"^\.(text|rodata|data|bss)\S*\s+(\d+)", line.strip())
        if match:
            totals[match.group(1)] += int(match.group(2))

    return ArmMeasurement(
        target=target[0].removeprefix("-mcpu="),
        optimisation=optimisation,
        text_bytes=totals["text"],
        rodata_bytes=totals["rodata"],
        data_bytes=totals["data"],
        bss_bytes=totals["bss"],
    )


# --------------------------------------------------------------- instructions

_BENCH_SOURCE = Path(__file__).with_name("_arm_bench.c")
_BENCH_LINKER = Path(__file__).with_name("_arm_bench.ld")

#: QEMU board with a Cortex-M4, and the SysTick clock it runs at. The clock is
#: not taken on trust -- :func:`measure_arm_instructions` calibrates against it.
_QEMU_MACHINE = "mps2-an386"


def find_qemu() -> str | None:
    """Locate ``qemu-system-arm``, or return ``None``."""
    found = shutil.which("qemu-system-arm")
    if found:
        return found
    for candidate in (
        Path("C:/Program Files/qemu/qemu-system-arm.exe"),
        Path("C:/Program Files (x86)/qemu/qemu-system-arm.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _run_bench(
    project: Path,
    cc: str,
    qemu: str,
    target: tuple[str, ...],
    optimisation: str,
    source: str,
    defines: tuple[str, ...],
) -> int:
    """Build and run the bare-metal benchmark, returning its SysTick tick count."""
    elf = project / "_bench.elf"
    subprocess.run(
        [
            cc,
            *target,
            optimisation,
            "-std=c99",
            "-nostartfiles",
            "--specs=nosys.specs",
            "-T",
            str(_BENCH_LINKER),
            f"-I{project}",
            *defines,
            "-o",
            str(elf),
            str(_BENCH_SOURCE),
            str(project / source),
            "-lm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    done = subprocess.run(
        [
            qemu,
            "-M",
            _QEMU_MACHINE,
            "-cpu",
            "cortex-m4",
            "-nographic",
            "-semihosting-config",
            "enable=on,target=native",
            "-icount",
            "shift=0",
            "-kernel",
            str(elf),
        ],
        # Not check=True: QEMU reports the guest's semihosting exit status as its
        # own, so a clean run can still come back non-zero. Whether the benchmark
        # worked is decided by whether it printed a count.
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elf.unlink(missing_ok=True)
    match = re.search(r"CK_TICKS (\d+)", done.stdout + done.stderr)
    if not match:
        raise RuntimeError(
            f"benchmark produced no tick count.\nstdout: {done.stdout!r}\nstderr: {done.stderr!r}"
        )
    return int(match.group(1))


def measure_arm_instructions(
    project: Path | str,
    target: tuple[str, ...] = CORTEX_M4F,
    optimisation: str = "-Os",
    source: str = "cellkernel_estimator.c",
    compiler: str | None = None,
    qemu: str | None = None,
) -> float:
    """Instructions retired per filter step, measured on an emulated Cortex-M4.

    Three subtractions make this a measurement rather than an estimate.

    The **fixed overhead** of reset, initialisation and semihosting is removed by
    running two different step counts and differencing, so what is reported is
    the marginal cost of one more call to ``ck_step``.

    The **tick-to-instruction conversion** is calibrated rather than assumed. The
    obvious instrument, the data watchpoint cycle counter, reads zero on QEMU's
    Cortex-M boards, so SysTick is used instead -- and SysTick counts virtual
    time, which under ``-icount shift=0`` advances one nanosecond per
    instruction. Converting that to instructions needs the board's clock rate, so
    the routine measures it: a loop of exactly ``n`` assembler ``nop``\\ s is run
    at two values of ``n`` and differenced, which cancels the loop overhead and
    leaves a known instruction count against a measured tick count. It comes out
    at exactly 40, agreeing with the 25 MHz the board is documented to run at --
    but agreeing with a datasheet is a check, not a substitute.

    The **loop overhead in the calibration** is what an earlier version of this
    got wrong. A C ``for`` loop around a single ``nop`` adds an increment, a
    compare and a branch to every iteration, so the calibration came out three
    times low. The nops are emitted with the assembler's ``.rept`` now.

    Returns instructions, not cycles. QEMU models no pipeline, no flash wait
    states and no memory system, so a real Cortex-M4F will take at least this
    many cycles and generally somewhat more.
    """
    cc = compiler or find_arm_toolchain()
    if cc is None:
        raise RuntimeError("no arm-none-eabi-gcc found")
    emulator = qemu or find_qemu()
    if emulator is None:
        raise RuntimeError("no qemu-system-arm found")
    directory = Path(project)

    def ticks(*defines: str) -> int:
        return _run_bench(directory, cc, emulator, target, optimisation, source, defines)

    # Calibrate the tick length against a known number of instructions.
    low = ticks("-DCK_CALIBRATE_NOPS=64")
    high = ticks("-DCK_CALIBRATE_NOPS=192")
    if high <= low:  # pragma: no cover - defensive
        raise RuntimeError("calibration produced no measurable difference")
    per_tick = (10_000 * 128) / (high - low)

    warm = ticks("-DCK_BENCH_STEPS=200")
    hot = ticks("-DCK_BENCH_STEPS=1200")
    return (hot - warm) * per_tick / 1000.0
