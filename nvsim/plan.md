# NV Diamond DC Magnetometer — Experiment Plan

Nov 11, 2026 · @hills

## Objective and scope

Build one benchtop NV-ensemble platform that measures DC to ~100 Hz magnetic fields, stepping from CW ODMR to double-quantum (DQ) Ramsey without redesigning hardware. The target is a low-cost path to a compact sensor, so every milestone records cost, power and drift alongside sensitivity.

- **In scope:** CW ODMR, Rabi calibration, pulsed ODMR, single-quantum (SQ) Ramsey, DQ Ramsey, closed-loop field tracking, balanced detection.
- **Out of scope for now:** AC/RF sensing (echo, dynamical decoupling, spin-lock), miniaturised packaging, and robust/composite pulse design (parked, see the last section).
- **Working targets:** ≤10 nT/√Hz with CW ODMR, ≤1 nT/√Hz with pulsed Ramsey, and DQ drift at least 10× lower than SQ over 1 h without temperature control.
- **Assumptions:** an ensemble diamond with ~0.1–1 ppm NV is in hand; there is access to an oscilloscope, spectrum analyser and basic optics bench. Targets are first-pass estimates to revise after M2.

## System architecture

One sequencer gates the laser and microwave switch and reads the balanced detector, so CW ODMR through DQ Ramsey all run on the same hardware.

![Benchtop architecture · optics, readout, microwave, timing](architecture.png)

The green pump reaches the diamond via the dichroic; red fluorescence returns through it to the signal photodiode, balanced against a 1% laser pickoff.

Decisions to fix before buying parts:

- **IQ-capable microwave source from the start:** needed for Ramsey phase control (M4) and two-tone DQ (M5). Cheaper fallback: amplitude-modulate a carrier at D at γB.
- **Resonator bandwidth ≥ 2γB:** with a 1–3 mT bias that is ~60–170 MHz total, so the resonator is low-Q; size the amplifier accordingly.
- **Direct laser modulation:** driver rise time <50 ns avoids an acousto-optic modulator.
- **Two-level noise cancellation:** analog auto-balancing for CW, plus a per-shot late reference window in pulsed modes.

## Milestones

Seven milestones, each gated by an exit test. M0–M2 prove the optics and microwaves with CW ODMR; M3–M5 add pulsed control and compare SQ with DQ Ramsey; M6 decides what goes into the compact design.

### M0 — Bench, safety and magnetic environment

- **Do:** laser interlock and OD4+ goggles for 520 nm; RF shielding around the amplifier; non-magnetic mounts; fix the bias magnet holder.
- **Test:** laser relative intensity noise (RIN) spectrum; ambient field and gradient at the diamond with a fluxgate; 50 Hz and harmonics level.
- **Exit:** noise sources logged; a baseline to compare every later sensitivity figure against.

### M1 — Optics and balanced detection

- **Do:** excitation path, collection optics, 650 nm long-pass filter, reference pickoff, two matched photodiodes with auto-balanced transimpedance amplifier.
- **Test:** photocurrent vs laser power (saturation curve); balanced vs single-ended noise floor across 10 Hz–100 kHz; common-mode rejection; background fluorescence with the diamond removed.
- **Exit:** detection noise within 2× the shot-noise limit at the working photocurrent.

### M2 — CW ODMR

- **Do:** synthesizer, amplifier and resonator; frequency sweep; bias field to split the 8 lines; frequency-modulation lock-in on one line.
- **Test:** resonator S11 and bandwidth; contrast and linewidth vs laser and microwave power (find the optimum); response to a known coil field; noise spectrum converted to nT/√Hz.
- **Exit:** calibrated, linear response and ≤10 nT/√Hz.

### M3 — Pulsed control, Rabi and pulsed ODMR

- **Do:** pulse sequencer; direct laser-diode pulsing; RF switch; gated readout with a signal window and a late reference window.
- **Test:**
  - laser rise/fall time and RF switch isolation (target >60 dB)
  - spin polarisation time vs laser power
  - Rabi frequency vs microwave power (expect Ω ∝ √P) and Rabi decay (microwave field non-uniformity)
  - automatic π-pulse recalibration stability over 1 h
  - pulsed ODMR linewidth vs π-pulse length; resolve the 2.16 MHz ¹⁴N hyperfine lines
- **Exit:** stable automatic calibration; pulsed ODMR sensitivity better than M2.

### M4 — Single-quantum Ramsey and closed loop

- **Do:** phase control of the second π/2 pulse (IQ modulator); 0°/180° phase-alternated readout; frequency-locked loop.
- **Test:** Ramsey fringes vs τ and T2*; detuning choice around the hyperfine beating; sensitivity vs τ; lock bandwidth and range; Allan deviation over 1 h.
- **Exit:** T2* measured; ≤1 nT/√Hz; loop holds lock through Earth-field-sized steps.

### M5 — Double-quantum Ramsey

- **Do:** two-tone microwave generation (IQ sidebands or amplitude modulation at D); confirm resonator covers D ± γB.
- **Test:**
  - DQ fringes at twice the SQ precession rate
  - T2* (DQ) vs T2* (SQ) on the same diamond
  - contrast loss from two-tone pulse errors
  - deliberate ±2 K temperature step: frequency shift in SQ vs DQ
  - SQ, SQ with ±1 alternation, and DQ Allan deviation side by side
- **Exit:** a data-backed choice of SQ, SQ with alternation, or DQ for the compact design.

### M6 — Characterisation and down-select

- **Do:** vector reconstruction from the 4 NV orientations; cross-check against a reference fluxgate; measure the power budget of each subsystem.
- **Test:** scale factor and orthogonality errors; heading error on rotation; total power and volume.
- **Exit:** a reduced BOM and block diagram for the compact prototype.

## Test matrix

Use the same metric definitions across every mode so CW, pulsed, SQ and DQ results compare directly.

| Metric | How measured | First at | Target |
| --- | --- | --- | --- |
| Detection noise floor | Balanced output PSD, laser on, MW off, vs calculated shot noise | M1 | ≤2× shot noise |
| ODMR contrast C | Dip depth / off-resonant signal | M2 | ≥1% (ensemble) |
| Linewidth Δν | Lorentzian fit; CW vs pulsed | M2, M3 | Pulsed → approach 1/(πT2*) |
| Rabi frequency and decay | Population vs pulse length, sinusoid × decay fit | M3 | Decay ≥ 5 Rabi periods |
| T2* (SQ and DQ) | Ramsey fringe envelope fit | M4, M5 | Measured; DQ/SQ ratio recorded |
| Sensitivity η | Noise PSD (1–10 Hz band) / calibrated slope | M2–M5 | 10 nT/√Hz CW; 1 nT/√Hz Ramsey |
| Scale factor | Response to known Helmholtz coil field | M2 onward | Within 1% of γ = 28 GHz/T |
| Low-frequency drift | Allan deviation, 1 s to 1 h | M4, M5 | DQ ≥10× below SQ at 1000 s |
| Temperature coefficient | Frequency shift for a ±2 K step | M5 | SQ ≈ −74 kHz/K expected; DQ near zero |
| Dynamic range and bandwidth | Closed-loop tracking of coil steps and ramps | M4 | Lock across ±50 µT; ≥100 Hz bandwidth |
| Power and volume | Per-subsystem measurement | M6 | Baseline for compact design |

## Bill of materials (excluding NV diamond)

The full benchtop platform comes to roughly USD 3,500–9,500, excluding the diamond, the laser, laser safety equipment, a lab-grade reference fluxgate and standard lab instruments. Costs are approximate list-price ranges, not quotes; confirm with suppliers. "Needed by" lets purchases follow the milestones.

| Subsystem | Item | Qty | Notes | Approx. cost (USD) | Needed by |
| --- | --- | --- | --- | --- | --- |
| Optics | Laser diode mount with TEC and collimator | 1 | Holds wavelength and power stable | 150–300 | M1 |
| Optics | Laser driver, ≥10 MHz modulation input | 1 | Rise time <50 ns for pulsed modes | 150–400 | M1 (pulsing at M3) |
| Optics | High-NA aspheric lenses (NA 0.5–0.8) | 2–3 | Excitation focus and collection | 150–400 | M1 |
| Optics | Dichroic, longpass ~600 nm | 1 | Separates green pump from red fluorescence | 150–300 | M1 |
| Optics | Longpass filter, 650 nm | 1 | Blocks residual green | 100–250 | M1 |
| Optics | Beam sampler for reference pickoff, ND filter set | 1 set | Feeds the balanced reference | 150–250 | M1 |
| Optics | Breadboard 300×450 mm, posts, mounts | 1 lot | Non-magnetic (aluminium/stainless) near diamond | 800–1,500 | M1 |
| Optics | Compound parabolic concentrator | 1 | Optional collection upgrade | 100–300 | M6 |
| Detection | Matched Si PIN photodiodes | 2 | Large area for ensemble fluorescence | 10–60 | M1 |
| Detection | Auto-balanced transimpedance amplifier PCB | 1 | Hobbs-style, low-noise op-amps; build. Buy option: commercial balanced detector, ~800–1,500 | 50–150 | M1 |
| Microwave | PLL synthesizer, 2–4 GHz (ADF4351 / LMX2572 class) | 1 | FM input for CW lock-in | 30–300 | M2 |
| Microwave | IQ modulator board | 1 | Phase control (M4) and two-tone DQ (M5) | 100–300 | M4 |
| Microwave | RF switch, SPDT, fast | 1 | >60 dB isolation | 100–200 | M3 |
| Microwave | Power amplifier, 5–16 W, 2–4 GHz | 1 | Main cost lever; size after resonator test | 300–2,000 | M2 |
| Microwave | Circulator/isolator and high-power 50 Ω load | 1 each | Protects amplifier from reflection | 150–450 | M2 |
| Microwave | Resonator PCB (loop-gap or split-ring) | 2–3 designs | Bandwidth must cover D ± γB for DQ | 50–200 | M2 |
| Microwave | SMA cables, attenuators, DC blocks | 1 lot | — | 150–300 | M2 |
| Microwave | NanoVNA | 1 | Resonator S11 and bandwidth | 80–150 | M2 |
| Timing and DAQ | Red Pitaya STEMlab 125-14 | 1 | Pulse sequencing, gated ADC readout, software lock-in. Low-cost alt: Raspberry Pi Pico PIO + external ADC | 400–700 | M2 |
| Bias and calibration | NdFeB magnets with adjustable holder | 1 set | Splits the 4 orientations | 30–100 | M2 |
| Bias and calibration | 3-axis Helmholtz coils with precision current source | 1 | Known field for scale factor and tracking tests | 300–800 | M2 |
| Bias and calibration | Compact fluxgate sensor | 1 | Ambient and reference check (lab-grade unit excluded) | 100–300 | M0 |
| Thermal | Thermistors and TEC controller for diamond mount | 1 | For the ±2 K step test | 100–300 | M5 |
| Shielding | Small mu-metal shield | 1 | Optional; isolates sensor noise from environment | 500–2,000 | Optional |

Assumed available and not costed: 520 nm laser and laser safety equipment (goggles, interlock), oscilloscope (≥200 MHz), RF spectrum analyser, PC, soldering and PCB fabrication.

## Risks and mitigations

The biggest risk is low optical power: a cheap laser diode may not repolarise the ensemble fast enough for pulsed modes to beat CW.

| Risk | Effect | Mitigation | Check at |
| --- | --- | --- | --- |
| Laser too weak for fast repolarisation | Pulsed duty cycle poor; Ramsey no better than CW | Shrink the excited volume; measure polarisation time early; keep CW as fallback | M3 |
| Poor fluorescence collection (n = 2.4) | Shot-noise floor too high | High-NA asphere first; CPC or side collection as upgrade | M1 |
| Microwave field non-uniform over the volume | Fast Rabi decay, lost contrast; worse in DQ | Resonator redesign; robust pulses (parked work) | M3, M5 |
| Resonator too narrow for two tones | DQ cannot run | Specify bandwidth ≥ 2γB at design; low-Q fallback with more amplifier power | M2 |
| Microwave heating of diamond | D drifts; fake low-frequency signal | Duty-cycle limit; heat-sink mount; DQ cancels residual | M2–M5 |
| Laser intensity noise not fully cancelled | Excess noise in 1–100 Hz band | Auto-balancing plus per-shot reference window | M1, M3 |
| Hyperfine beating in Ramsey | Fringe distortion, lower slope | Choose detuning to match; three-tone drive if needed | M4 |
| Bias magnet temperature drift | Apparent field drift | Monitor magnet temperature; DQ or ±1 alternation; coil bias option | M4–M5 |

## Open decisions and parked items

- [ ] **Robust/composite pulses (parked, explore in detail):** BB1, CORPSE, adiabatic passage, optimal control. Key question: two-tone DQ fidelity under non-uniform microwave field, and the contrast recovered vs sequence overhead. Needed before M5 sign-off.
- [ ] **Diamond choice:** HPHT vs CVD, and NV density. Sets T2*, contrast and whether DQ gives a clear gain.
- [ ] **Bias field size:** larger bias separates lines cleanly but demands more resonator bandwidth for DQ.
- [ ] **Build or buy balanced detector:** DIY board vs commercial unit.
- [ ] **Sequencer platform:** Red Pitaya vs Pico PIO + ADC vs FPGA for the compact version.
- [ ] **Vector or total field** for the end application.
