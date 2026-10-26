# Numerical convergence check

Ran `check_solver_convergence.py` without plotting. It compares the existing
frequency presets at 100 kHz, 1 MHz, 10 MHz, and 30 MHz, varying resolution
and domain size separately where possible. The run used `l_vals=[0, 1, 2]`,
`max_n_r=10`, the sweep's shell definition, and all listed states. Results and
individual runtimes are in `convergence_results.csv`.

Plots: `convergence_resolution.png` compares grid refinement,
`convergence_extent.png` compares domain size, and `convergence_runtime.png`
shows solve-plus-inference runtime. Convergence lines show median and
worst-state relative changes from the finest grid or largest domain tested.
The focused 30 MHz `1s` plot is `convergence_30MHz_1s_high_resolution.png`.

## Findings

- At **100 kHz** and **1 MHz**, the preset grids are stable for the reported
  energies, Moon-enclosed masses, and Earth-surface densities. Doubling `N`
  changed the largest Earth-density ratio by under 0.002% and 0.002%,
  respectively. Doubling the domain also had negligible effect at the preset.
- At **10 MHz**, doubling `N` from 100,000 to 200,000 changed the largest
  Earth-density ratio by 0.12%; shell fractions changed by at most 0.92%.
  The preset looks adequate for density at roughly the 0.2% level, though
  shell fractions are less settled.
- At **30 MHz**, most states were stable, but the `1s` Earth-density ratio
  shifted by 6.85% when `N` increased from 1,000,000 to 2,000,000 at fixed
  70 Earth radii. Its shell fraction was extremely small and varied by many
  orders of magnitude across resolutions. The larger-domain runs (140 and
  280 Earth radii, with spacing held about constant) gave stable densities,
  but do not resolve this fixed-domain resolution sensitivity. The 30 MHz
  preset is therefore not fully converged for every state.

## 30 MHz `1s` follow-up

Ran the lowest `s` state at fixed 70 Earth radii with up to 16 million grid
points. The energy changes by less than 0.0000002% across these grids, but the
shell fraction and Earth-surface density do not converge. From 8M to 16M
points, shell fraction changed from `9.07e-64` to `5.54e-54`, and density
changed from `6.84e-37` to `4.15e-30` in SHM units. The shell fraction remains
far below float64 machine epsilon. Treat the 30 MHz `1s` shell inference as
numerically unresolved even at 16M points; increasing `N` alone has not
produced a reliable result.

Targeted runtimes for `l_vals=[0]`, `max_n_r=1`, shell inference, and Earth
density query (plots excluded):

| N | Runtime |
|---:|---:|
| 1,000,000 | 1.65 s |
| 2,000,000 | 3.52 s |
| 4,000,000 | 7.28 s |
| 8,000,000 | 13.89 s |
| 16,000,000 | 27.99 s |

These timings are machine-dependent. Data are in
`convergence_30MHz_1s_high_resolution.csv`.

Measured preset baseline runtimes (solve, shell inference, and Earth-density
query; plotting excluded):

| Frequency | N | Extent (Earth radii) | Runtime |
|---|---:|---:|---:|
| 100 kHz | 100,000 | 1,000 | 2.6 s |
| 1 MHz | 100,000 | 100 | 2.9 s |
| 10 MHz | 100,000 | 100 | 3.3 s |
| 30 MHz | 1,000,000 | 70 | 17.8 s |

Runtimes depend on machine and system load.

## Practical takeaway

The tested presets are efficient and converged for most checked outputs at
100 kHz and 1 MHz. At 10 MHz, Earth-surface density is stable by 1.6 million
points, while shell fractions still vary and affect inferred mass scales. At
30 MHz, all-state density and shell fractions remain unstable through 16
million points at fixed 70 Earth radii. Very small shell probabilities are
especially sensitive because they set the inferred mass scale. This check did
not vary `l_vals` and `max_n_r` for all frequencies. The 100 Hz–10 kHz presets
were subsequently checked separately; 1 Hz and 10 Hz remain unchecked.

## 100 Hz–10 kHz follow-up

The targeted low-frequency run is reproducible with
`check_low_frequency_convergence.py`. It varied resolution and extent for the
12 requested states using `l_vals=[0, 1, 2]`, `max_n_r=10`, the sweep shell,
and the active constructor `maxTotalDMmass` cap. The checked quantities were
energy, shell fraction, total mass, Moon-enclosed mass, and Earth-surface
density. The practical target is a maximum state-by-state change below 0.01%
against the finest or largest tested reference.

- **100 Hz:** N=3M differs from the N=8M reference by at most 0.0092% in the
  tracked metrics. The 8M solve took 166 s. The 1M Earth-radius domain is
  stable against matched-spacing tests at 2M and 4M Earth radii (worst changes
  below 0.001%). This is the smallest tested N that meets the target; N=2M
  differs by up to 0.0224% in shell fraction and Moon-enclosed mass.
- **1 kHz:** N=500k differs from N=2M by at most 0.0037%. N=375k fails because
  some Earth-surface density ratios differ by more than 7%. The 100k
  Earth-radius domain is stable against the 200k matched-spacing run (worst
  tracked change below 0.0001%).
- **10 kHz:** N=250k differs from N=2M by at most 0.0011%. N=125k has an
  Earth-density deviation near 0.95%. The 10k Earth-radius domain is stable
  against the 20k matched-spacing run (worst tracked change below 0.001%).

The data are in `outputs/low_frequency_convergence/convergence_results.csv`.
Per-frequency plots are `100Hz_domain.png`, `1kHz_domain.png`, and
`10kHz_domain.png`; the combined plot is `low_frequency_domain_all.png`.
These conclusions apply to the chosen solver states and mass cap. They do not
establish convergence for 1 Hz or 10 Hz, or for different `l_vals`,
`max_n_r`, or mass constraints.

## Frequency-specific domain plot and current grid guidance

The combined domain convergence figure for all frequencies with saved domain
sweeps (100 kHz, 1 MHz, 10 MHz, and 30 MHz) is
`outputs/convergence/convergence_domain_all_frequencies.png`. It uses the
same four-metric, median/worst-state layout as the single-frequency domain
plot and includes a 0.01% reference-deviation line. The 1 MHz curves use the
expanded 70–500 Earth-radius sweep; the other curves use the saved
multi-frequency convergence runs. The low-frequency domain tests are in
`outputs/low_frequency_convergence/low_frequency_domain_all.png`.

Recommended working presets and evidence are summarized in
`convergence_parameter_recommendations.md` and reflected in
`density_by_shell_mass_sweep.py`. In brief, the 100 kHz and 1 MHz settings
meet the 0.01% criterion against saved references; 10 MHz uses the largest
tested settings; and 30 MHz remains unresolved even at 16 million points.

## Targeted resolution follow-up

The all-state resolution follow-up is in
`outputs/convergence/targeted_frequency_grid_checks.csv` and is reproducible
with `tune_frequency_grids.py`. At 10 MHz, N=800k is within 0.0023% in
Earth-surface density of the 3.2M reference; increasing to 1.6M reduces this
to 0.00046%. However, shell fractions still vary by up to 84.8%. Treat the
near-Earth density metric as stable at the 0.01% level, but do not interpret
that as convergence of shell probabilities or the inferred mass scale. The
800k solve took about 9 seconds on this machine.

At 30 MHz, increasing N from 8M to 16M still changes the worst Earth-surface
density by about 99.98% across states, and shell fractions also remain
unstable. The 16M all-state run took about 157 seconds. No all-state 30 MHz
grid recommendation meets the 0.01% target; the 2M preset is only an
exploratory runtime/accuracy compromise.

Re-run from the repository root with:

```powershell
python -m examples.EarthBoundAxionHalo.DM_overdensity.check_solver_convergence
```

Regenerate plots from the saved CSV without rerunning the solver:

```powershell
python -m examples.EarthBoundAxionHalo.DM_overdensity.plot_convergence
```

Regenerate the targeted figure with:

```powershell
python -m examples.EarthBoundAxionHalo.DM_overdensity.plot_30MHz_1s_high_resolution
```
