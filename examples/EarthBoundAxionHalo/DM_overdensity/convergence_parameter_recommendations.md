# Frequency-dependent solver grid recommendations

Use a **0.01% maximum state-by-state relative change** as the practical
reference-deviation target for the tracked energy, shell fraction,
Moon-enclosed mass, and Earth-surface density. This compares tested runs with
the finest or largest saved run; it is a numerical cross-check, not an absolute
error bound.

| Frequency | Suggested N | Extent (Earth radii) | Evidence and limits |
|---:|---:|---:|---|
| 1 Hz | 40,000,000 | 4,000,000 | Provisional original preset; no convergence sweep. Very expensive. |
| 10 Hz | 10,000,000 | 1,000,000 | Provisional original preset; no convergence sweep. Very expensive. |
| 100 Hz | 3,000,000 | 1,000,000 | Meets 0.01% target: worst tracked difference is 0.0092% vs N=8M; extent 1M is within 0.001% of the 4M R_earth matched-spacing run. The 8M reference took 166 s. |
| 1 kHz | 500,000 | 100,000 | Meets target: worst tracked difference is 0.0037% vs N=2M; 375k fails Earth-density convergence. Extent 100k is within 0.0001% of the 200k R_earth matched-spacing run. |
| 10 kHz | 250,000 | 10,000 | Meets target: worst tracked difference is 0.0011% vs N=2M; 125k fails Earth-density convergence. Extent 10k is within 0.001% of the 20k R_earth matched-spacing run. |
| 100 kHz | 50,000 | 1,000 | Meets target: worst tracked difference is 0.0023% vs N=200,000 and 0.00003% vs 2,000 R_earth. |
| 1 MHz | 75,000 | 100 | Meets target with margin: worst tracked difference is 0.0033% vs N=3.2M and 0.000016% vs 500 R_earth. |
| 10 MHz | 800,000 | 100 | Earth-surface density differs by at most 0.0023% vs N=3.2M, within target; shell fractions differ by up to 84.8%. Treat density ratios as stable at this metric; shell-normalized mass scales are unresolved. |
| 30 MHz | 2,000,000 | 70 | Exploratory only. All-state runs through N=16M do not converge: between 8M and 16M, worst Earth-surface density changes by about 99.98% (excluding 1s). No reliable all-state setting is established. |

For 1 Hz and 10 Hz, retaining the existing large presets is the cautious
choice until a targeted convergence run is practical. Low-frequency checks
now cover 100 Hz, 1 kHz, and 10 kHz in addition to the saved 100 kHz, 1 MHz,
10 MHz, and 30 MHz runs. At 10 MHz, Earth-surface
density is within target by 800k, but shell fractions do not; this distinction matters
because the shell fraction sets the inferred mass scale. At 30 MHz, density and
shell fraction remain unstable across the all-state checks through 16M, so no
single grid can currently be recommended for quantitative results.

The 100 Hz--10 kHz comparison used the active ``maxTotalDMmass`` cap and tracked
energy, shell fraction, total and Moon-enclosed mass, and Earth-surface density.
The detailed rows and runtime measurements are in
``outputs/low_frequency_convergence/convergence_results.csv``. Domain plots
are saved as ``100Hz_domain.png``, ``1kHz_domain.png``,
``10kHz_domain.png``, and ``low_frequency_domain_all.png`` in that folder.
``domain_convergence_summary.csv`` has the plotted median and worst-state
deviations by extent.

The multi-frequency domain plot is
`outputs/convergence/convergence_domain_all_frequencies.png`; the plotted
curves and source run settings are in
`outputs/convergence/convergence_domain_all_frequencies.csv`. Rebuild it from
saved data without running the solver:

```powershell
python -m examples.EarthBoundAxionHalo.DM_overdensity.plot_frequency_domain_convergence
```
