# Plan: integrate axion coherence calculations

1. **Reuse solved states.** Add methods to `EarthBoundAxionHalo` that accept selected state coefficients and call the inherited `getStateAmplitudeSpectrum()` for normalized coefficients and frequency shifts Δν = E/h. Keep Astropy units at the API boundary.
2. **Build the real field and its envelope.** Form A(t) = Σ cⱼ exp(−2πi Δνⱼ t), then a(t) = Re[A(t) exp(−2πi ν_a t)]/√2. Use the envelope to avoid sampling the MHz carrier over long coherence intervals.
3. **Calculate correlations.** Return normalized g1(τ) from the envelope autocorrelation and cycle-averaged g2(τ) from the intensity autocorrelation. Document whether averages are over time, phase realizations, or both; support the analytic discrete-mode sum and an optional numerical/Monte Carlo estimator.
4. **Report coherence times.** Let callers choose a convention (for example, first 1/e crossing of |g1| or of g2(τ)−1). Return the curves, units, thresholds, and crossing times; note revivals for finite discrete spectra.
5. **Validate and expose.** Compare analytic sums against simulated signals and the existing examples, then document a short usage example with user-supplied states and coefficients.
