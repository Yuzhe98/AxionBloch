# Mass and density inferred from a measured shell

`shell_normalized_density.py` is the short example: edit quantities at the
top of the file, solve the desired states, and call the class methods.
Run from the repository root:

```powershell
python -m examples.EarthBoundAxionHalo.DM_overdensity.shell_normalized_density
```

The example specifies an axion mass, the shell mass `(0.3 +/- 4) * 1e-9 M_earth`,
and geocentric radii of 12,300 km and 384,000 km. It has no command-line parser
or `main()` function. Each selected eigenstate is an independent hypothesis.

## Class methods

- `halo.inferHaloMass(shell_mass, mass_uncertainty, radius_range, state_names)`
  infers halo masses from already-normalized wavefunctions without changing them. It returns each state's shell fraction,
  total mass over `[0, extent]`, mass inside the outer shell radius, and the
  corresponding quoted uncertainties, all masses in Earth masses. Omit `state_names` to use all solved
  states. The instance stores independent inferred mass scales for later calls.
- `halo.getEnclosedMass(radius)` returns `mass` and `uncertainty` per state.
  Omit `radius` for the total mass over the simulated domain.
- `halo.getDensity(radii)` returns angularly averaged `density` and
  `uncertainty` per state, for a scalar or array of positive radii.
- `halo.plotDensity(radii, showPlot=True)` returns `(figure, axes, profiles)`.
  Central densities and uncertainty magnitudes appear in separate panels,
  in units of `halo.rho_M_DM_SHM`. The figure is 13 cm wide at 300 dpi and
  uses `tight_layout`. Returned profiles retain physical mass-density units.
  Use `showPlot=False` to save or customize the figure without displaying it.

Masses, radii, profiles, and uncertainties are Astropy quantities throughout.
Only plotting and serialization convert to plain numerical arrays. Optional
`state_names` on the query and plot methods selects a subset of states with inferred mass scales.
Inferences retain wavefunction copies; after solving new eigenstates,
call `inferHaloMass` again to use those solutions. This API does not
change the constructor's `totalMassEnclosed`, `N_a`, or field amplitudes:
separate single-state hypotheses cannot share one total-mass normalization.

## Interpretation

For each state, `I(a,b) = integral_a^b |u(r)|^2 dr`, and

```text
scale        = M_shell / I(r_inner, r_outer)
M(<r)        = scale * I(0, r)
rho_bar(r)   = scale * |u(r)|^2 / (4*pi*r^2)
```

The overall wavefunction normalization cancels. The code integrates the
probability density `abs(u)**2` with `np.trapezoid`, retaining the interior
grid samples and linearly interpolating `u` at both interval endpoints.
Zero domain boundaries are included. Accuracy still requires grid convergence. Shell probabilities are integrated directly to avoid
subtracting nearly equal cumulative probabilities.

Quoted uncertainty propagates using the same positive factors, including
when the central mass is zero. No confidence level, positivity truncation,
detection, or statistical upper limit is inferred. The supplied constraint
is consistent with zero. Densities are angular averages; states with nonzero
angular momentum need not have isotropic local densities.

The total over the simulated domain approximates the entire halo only after
checking the outer radius. The model neglects halo self-gravity. A very small
shell fraction can imply enormous formal halo masses; those results need
numerical convergence checks and may invalidate the assumed Earth-only
potential. Density queries exclude zero and disallow extrapolation beyond
the outer boundary.

## Detailed comparison

`shell_density_mass_sweep.py` preserves the longer three-mass, six-state
comparison, resolution/extent checks, saved density tables, and JSON output.
It now uses the class API through the small compatibility wrappers in
`shell_density_helpers.py`. Outputs go to `outputs/shell_normalized/`.
The 1% convergence flag tests Moon-enclosed mass and Earth-surface density,
not the whole profile. Extremely small plotted tails can reach the
numerical noise floor. Existing output files are replaced on reruns.

The older density examples use `totalMassEnclosed` as a whole-profile mass;
they do not perform the shell-to-total conversion.
