# nvsim: conclusions from the default model

Every number below comes from placeholder parameters in `config.json` (approximate literature values, tuned so results land in an interesting range). Treat them as what the model predicts, not as measurements. Replace them via `config.json` and `measured.json` as milestones deliver data.

## Sensitivity (1 to 10 Hz band, white noise plus temperature drift)

| Mode | η (nT/√Hz) | Target | Verdict |
| --- | --- | --- | --- |
| CW ODMR, FM lock-in | 0.39 | 10 | PASS |
| Pulsed ODMR | 2.9 | beat CW | FAIL |
| SQ Ramsey | 0.41 | 1 | PASS |
| SQ Ramsey with ±1 alternation | 0.57 | 1 | PASS |
| DQ Ramsey | 0.36 | 1 | PASS |

- Shot noise dominates every mode at 1 Hz. Detection electronics are the second term; residual laser RIN is negligible with 40 dB balancing.
- Temperature drift dominates at 0.1 Hz (CW and SQ about 1 nT/√Hz, against about 0.4 white).
- Pulsed ODMR loses to CW because the cycle is mostly initialisation (about 3.3 µs polarisation) and readout, which is the M3 risk in the plan.
- Monte Carlo over metric uncertainty gives about 90% probability of meeting the Ramsey target for SQ and DQ, 79% with alternation.
- Biggest levers: collection (concentrator ×2 or more laser power roughly halves η), then photon rate, contrast and T₂\* (each ×2 gives about −50%). NA 0.5 instead of 0.8 costs about 1.8× in η.

## SQ versus DQ (M5)

- DQ fringes run at 1.9 to 2× the SQ rate, and T₂\*(DQ)/T₂\*(SQ) follows 1/(2√(1−f)). The η gain of DQ over SQ grows with the common-mode fraction (1.0 at f=0, 1.13 at the default 0.3, 4 at 0.95); DQ never hurts in this model.
- **The 10× DQ drift target fails at the defaults.** Allan SQ/DQ at 1000 s is about 2.9 (analytic 2.6). With NdFeB at −0.1%/K and a 2.4 mT bias, the magnet term is the same size as the D shift. The drift map shows 10× needs a magnet tempco of about −3×10⁻⁴/K (SmCo-like) at 1 mT, or about −1×10⁻⁴/K at 2 mT, or a coil bias.
- A diamond-only ±2 K step gives −74 kHz/K in SQ and zero in DQ, as expected.
- **Hyperfine is the dominant DQ pulse error.** The ¹⁴N splitting detunes the two tones in opposite directions, costing about 0.63 of DQ fringe contrast at 8 MHz drive. Resonator roll-off, IQ imbalance, amplitude spread and the detuning spread are each below 0.011.
- BB1 and CORPSE help SQ slightly (CORPSE raises contrast from 0.90 to 0.99, at 8× pulse length), but make DQ worse, because they assume a two-level system. A three-tone drive per transition, or much larger Rabi frequency, is the likelier DQ fix.

## Milestone pass lines the model sets

- **M0:** Allowed RIN is 4×10⁻⁵ /√Hz balanced, 4×10⁻⁷ single-ended, so balancing is required. Ambient noise must be about 0.5 nT/√Hz, so a 50 nT/√Hz lab needs about 100× reference suppression. A 1000 Hz sample rate aliases the 20th mains harmonic to DC (FAIL); do not use an integer multiple of 50 Hz. A fluxgate floor of 0.33 nT/√Hz suffices.
- **M1:** Balancing needs only about 20 dB for shot ratio ≤ 2 from 10 Hz to 100 kHz (40 dB configured). Single-ended is about 14× shot. Rf = 10⁵ Ω settles within the readout window; 10⁶ Ω does not.
- **M2:** Contrast 2.7%, linewidth 1 MHz, η_CW passes. The scale-factor test passes only with a coil field of about 1.4 µT or less (FM linear range is narrow); 20 µT gives 61% error. The best bias is about 3 mT with 16 MHz minimum line gap and a 143 MHz DQ span against a 144 MHz resonator bandwidth, so DQ is right at the resonator limit.
- **M3:** Isolation of about 50 dB is enough for under 1% loss (65 dB configured). The Rabi-decay target of 5 periods is met only at about 8 MHz and above, because the 2.16 MHz hyperfine splitting dephases weaker drive. The amplifier compresses at about 7.6 W. Recalibration every 1000 s costs under 0.3%. The 2.16 MHz triplet resolves for t_π of 0.4 µs or longer.
- **M4:** Best Ramsey detuning is about 0.3 MHz at τ ≈ 0.87 µs, near a hyperfine revival, so a three-tone drive gains only about 1.1×. T₂\* is recovered within about 2%. A single-τ lock holds only to about ±0.29 MHz, so a ±50 µT (1.4 MHz) step is lost without a coarse fallback and recovered with one. Loop bandwidth is about 900 Hz (target 100 Hz) at the default averaging.
- **M6:** The per-axis vector noise is about 0.6 nT/√Hz with a well-conditioned Jacobian. A 0.5° axis error at a 3 mT bias gives about 7° heading error until the axis matrix is fitted from a multi-tilt rotation scan; afterwards it is about 0.5°, limited by bias drift (100 ppm). The reduced BOM is about $2.6k, 16 W and 300 cm³, against $6.7k, 44 W and 4300 cm³ for the full platform, with η essentially unchanged (0.36 versus 0.36 nT/√Hz).

## Model limits to keep in mind

- Static Gaussian spreads only; no dynamic spin-bath noise, no m_s = +1 level or charge-state dynamics in the rate equations.
- η is a one-sided ASD (√2 factor in both CW and pulsed formulas). ±1 alternation uses the plan's doubled time per estimate, which may be pessimistic.
- Allan deviation runs to about 1700 s, not 3600 s, since the 1 h series is too short for overlapping averages beyond that.
- BOM power and volume entries are hand-filled guesses; the plan lists only cost and milestone.
