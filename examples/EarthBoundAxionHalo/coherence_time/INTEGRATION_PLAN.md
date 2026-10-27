# Plan: integrate axion coherence calculations

The scripts in this folder are working prototypes; the following steps move their capabilities into the `EarthBoundAxionHalo` API.

1. **Reuse solved states.** Accept selected state coefficients and use the inherited `getStateAmplitudeSpectrum()` to obtain normalized coefficients and frequency shifts Δν = E/h. Keep Astropy units at the API boundary.
2. **Simulate the real field.** Add a field/envelope helper for A(t) = Σ cⱼ exp(−2πi Δνⱼ t) and a(t) = Re[A(t) exp(−2πi ν_a t)]/√2. The MHz carrier is resolved for display; the slow envelope is used for long correlations.
3. **Provide both calculation paths.** Implement analytic finite-mode sums and numerical time-series estimates for normalized g1(τ) and cycle-averaged g2(τ). Make phase/time averaging assumptions explicit.
4. **Return coherence times.** Allow a stated threshold convention (such as first 1/e crossing of |g1| or g2−1) and return curves, thresholds, units, and times; handle revivals in discrete spectra.
5. **Add reusable plots.** Provide a plotting function for the three-row example: real MHz oscillation; positive/mirrored field envelopes with fill; and g1(τ)/g2(τ) with thresholds and crossing markers. Save under a caller-selected `outputs/` subfolder.
6. **Include Monte Carlo and histogram support.** Allow random-phase realizations with fixed or sampled amplitudes, report the ensemble g1/g2 results, and optionally plot the batch coherence-time histogram. Validate against analytic curves and document a short usage example.
