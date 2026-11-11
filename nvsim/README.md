# nvsim

Sub-system, system and milestone simulations for the NV DC magnetometer plan (`plan.md`).

Run from this folder, in order (each prints a table of metric, value, target and PASS, FAIL or INFO):

```bash
python -m sims.foundations   # spin model: Rabi and SQ/DQ Ramsey fits
python -m sims.subsystems    # writes out/metrics.json and out/drift_series.npz
python -m sims.system        # eta per mode, budgets, Allan, tornado, Monte Carlo, variants
python -m sims.m0 ; python -m sims.m1 ; python -m sims.m2
python -m sims.m3 ; python -m sims.m4
python -m sims.m5 ; python -m sims.m6 ; python -m sims.robust
```

All tweakable numbers are in `config.json`. Put measured values in `measured.json` (same keys as `out/metrics.json`) to override the model; overridden keys are printed. Optional `rin.csv` (frequency in Hz, RIN in 1/rtHz) replaces the configured laser RIN in M0.
