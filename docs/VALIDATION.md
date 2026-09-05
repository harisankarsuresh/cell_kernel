# Verification record

Local results from 5 September 2026. Remote CI runs are reported separately in
the repository's Actions tab.

## Test environment

| Component | Version |
|---|---|
| OS | Windows |
| Python | 3.14.3 |
| NumPy / SciPy | 2.5.1 / 1.17.1 |
| PyBaMM | 26.7.1.0 |
| Host C compiler | Zig 0.16.0 C frontend |
| ARM compiler | ARM GNU 14.2 |

The measured LG M50 dataset and QEMU were available during testing.

## Automated checks

**674 passed, 1 skipped; coverage 95.75%, above the required 93%.**
The skip is `test_degradation_reads_every_model[ECM]`: an equivalent-circuit model
does not expose the electrode surface state needed for that comparison. Lint
and format checks passed. The 13 demo and benchmark tests also passed after
formatting the HTML template.

```bash
python -m pytest -o addopts=--strict-markers -q --cov=cellkernel --cov-report=term-missing --cov-fail-under=93
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

Tests cover model behaviour, generated C, parameter fitting and the measured-data
comparisons. The demo and benchmark checks include time alignment, recomputing
metrics from exported samples, wrong-start recovery, held-out label isolation,
and rejecting unsupported SPMe C export.

## Compiled C replay

`cellkernel verify build/verified --precision float` passed both modes for 900
samples with six states. Use gcc or clang, or set `CC` to a compatible compiler.

| Check | Maximum difference |
|---|---:|
| Open-loop C voltage versus Python mirror | 8.430 µV |
| Open-loop C SoC versus Python mirror | 3.512 × 10⁻⁷ (SoC fraction) |
| Filter-path C voltage versus Python mirror | 3.377 µV |
| Filter-path C SoC versus Python mirror | 5.649 × 10⁻⁶ (SoC fraction) |
| Open-loop C voltage versus full analytic Python model | 0.059 mV |

The last row includes the voltage-table approximation. These measure code
fidelity on the replay profile, not predictive accuracy on measured batteries.
[Full replay output](results/c-verification.txt).

## ARM build and emulation

Reproduced with `cellkernel measure build/arm --precision float --order 3`:

| Target | Optimisation | Flash | Code | Tables |
|---|---|---:|---:|---:|
| Cortex-M4F | `-Os` | 4,636 B | 2,124 B | 2,512 B |
| Cortex-M4F | `-O2` | 5,492 B | 2,980 B | 2,512 B |
| Cortex-M0+ | `-Os` | 4,820 B | 2,308 B | 2,512 B |
| Cortex-M0+ | `-O2` | 5,588 B | 3,076 B | 2,512 B |

QEMU counted 5,086 instructions per filter step at `-Os` and 5,201 at `-O2`.
[Full measurement output](results/arm-measurement.txt).
Timing on physical hardware has not been measured.

## Demo and package checks

- Browser checks covered scenario switching, the time slider and narrow-screen layout.
- The standalone demo uses embedded data and scripts, with no remote chart dependencies.
- A built wheel was installed into a separate target directory and used to generate the demo and C assets.
- The comparison figure was regenerated from the actual experiment outputs.

## Saved experiment results

- [Measured ML benchmark](results/benchmark.json), including source-file SHA-256 fingerprints.
- [Synthetic demo summary](results/demo-summary.json), including seeds and metric definitions.

Regenerate full-resolution samples with `cellkernel benchmark` and `cellkernel demo`.
