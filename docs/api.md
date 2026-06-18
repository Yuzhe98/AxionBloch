# API Reference

All public classes and functions are documented here.  Entries are generated
automatically from source docstrings.

## Quick reference

| Class / function | Module | Purpose |
|---|---|---|
| {class}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo` | `MilkyWayAxionHalo` | SHM axion-wind model |
| {func}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.axion_lineshape` | `MilkyWayAxionHalo` | Analytical PSD lineshape (static) |
| {func}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.getAmpSpectra` | `MilkyWayAxionHalo` | Stochastic amplitude spectra |
| {func}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.getRabiFreq` | `MilkyWayAxionHalo` | rms Rabi frequency |
| {class}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo` | `GravBoundAxionHalo` | Generic TISE solver |
| {class}`~axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo` | `EarthBoundAxionHalo` | Earth-specific TISE solver |
| {class}`~axionbloch.Station.Station` | `Station` | Geographic lab location |
| {class}`~axionbloch.Sample.Sample` | `Sample` | NMR sample |
| {class}`~axionbloch.Apparatus.Magnet` | `Apparatus` | Static polarising magnet |
| {class}`~axionbloch.SimuTypes.SimuParams` | `SimuTypes` | Simulation parameter dict (TypedDict) |
| {class}`~axionbloch.SimuTypes.SimuEntry` | `SimuTypes` | Simulation + parameters pair |
| {class}`~axionbloch.SimuTools.MagField` | `SimuTools` | (Pseudo)magnetic field builder |
| {class}`~axionbloch.SimuTools.Simulations` | `SimuTools` | Multi-run simulation manager |
| {class}`~axionbloch.SimuTools.Simulation` | `SimuTools` | Single Bloch-equation simulation |

---

## axionbloch.MilkyWayAxionHalo

Models the axion dark-matter field from the Milky Way galactic halo using the
Standard Halo Model (SHM) velocity distribution.  The main class is
{class}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo`, which provides:

- Axion field properties: Compton frequency, quality factor, coherence time.
- Analytical PSD lineshapes for three coupling geometries via the static method
  {func}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.axion_lineshape`.
- Stochastic amplitude spectra for time-domain simulations via
  {func}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.getAmpSpectra`.
- rms Rabi frequency via
  {func}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.getRabiFreq`.

```{automodule} axionbloch.MilkyWayAxionHalo
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.GravBoundAxionHalo

Provides the general TISE solver
{class}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo` for axions
gravitationally bound to a compact body.  The solver:

- Builds a 1-D radial finite-difference Hamiltonian {math}`H = T + V_\mathrm{eff}`.
- Diagonalizes it with `scipy.linalg.eigh` for each angular-momentum channel *l*.
- Stores eigen-energies and normalized reduced radial wavefunctions {math}`u_{n_r l}(r)`.
- Computes the 3-D gradient of the total wavefunction toward a geographic
  {class}`~axionbloch.Station.Station` via
  {func}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo.findGradientsAtDirection`,
  and its time evolution via
  {func}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo.findGradientsOverTime`.
- Computes the axion-nucleon coupling frequency (Omega_a) from gradients via
  {func}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo.findOmega_aOverTime`,
  with RMS value computation via
  {func}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo.findRmsOmega_aOverTime`.

Sub-class {class}`~axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo` is
pre-configured with the Earth's gravitational potential.

```{automodule} axionbloch.GravBoundAxionHalo
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.EarthBoundAxionHalo

Sub-class of {class}`~axionbloch.GravBoundAxionHalo.GravBoundAxionHalo`
pre-configured with Earth's gravitational potential from the Preliminary Earth
Model (PEM) density profile.  Adds Earth-specific gradient helpers:
{func}`~axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo.findGradients`
(takes a {class}`~axionbloch.Station.Station`),
{func}`~axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo.findGradientsAtEarthSurface`,
{func}`~axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo.findGradientsWithMilkyWay`,
and the pseudomagnetic-field conversion
{func}`~axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo.getBfield`.
Also exports helper functions used to build the potential:

| Function | Description |
|----------|-------------|
| `loadPEMdata` | Load PEM density-profile data |
| `PREM_density` | PREM piecewise density at a given radius |
| `earth_grav_potential_earth_center` | Gravitational potential referenced to Earth's centre |
| `earth_grav_potential_infty` | Gravitational potential referenced to infinity |
| `get_CumulativeMass` | Cumulative enclosed mass as a function of radius |
| `plot_earth_grav_potential` | Plot density, mass, and potential profiles |

```{automodule} axionbloch.EarthBoundAxionHalo
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.Station

Defines {class}`~axionbloch.Station.Station`, which converts geographic
coordinates (latitude, longitude, elevation) to spherical coordinates
{math}`(\theta, \phi)` and the outward unit normal {math}`\hat{n}`.
{func}`~axionbloch.Station.Station.in_solarZ_frame` expresses the station
position in a solar-oriented frame at a given measurement time.

Pre-built station instances exported from this module:

`Mainz`, `Geneva`, `Baltimore`, `Tokyo`, `Mumbai`, `Sanya`, `Sydney`,
`CapeTown`, `BuenosAires`.

```{automodule} axionbloch.Station
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.Sample

Defines {class}`~axionbloch.Sample.Sample`, which encapsulates all NMR sample
properties needed for a simulation:

- Gyromagnetic ratio {math}`\gamma`, magnetic dipole moment {math}`\mu`.
- Mass density, molar mass, spin count per molecule → spin number density.
- Relaxation times {math}`T_1`, {math}`T_2`.
- Volume, temperature, and initial polarization.

Key methods:

| Method | Description |
|--------|-------------|
| `getThermalPol` | Exact thermal polarization at a given field and temperature |
| `getM0` | Magnetisation {math}`M_0` from a given polarization |
| `getM0_SPN` | Magnetisation from polarization and spin number density |
| `getM0eqb` | Equilibrium magnetization at a given field and temperature |

```{automodule} axionbloch.Sample
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.Apparatus

Defines {class}`~axionbloch.Apparatus.Magnet`, which models the static DC
polarising magnet.  Field inhomogeneity is characterised by a Lorentzian
distribution with a user-specified fractional FWHM (e.g. 2 ppm), discretised
into `numPt` spin packets.  The resulting spread of Larmor frequencies gives
the effective {math}`T_2^*` dephasing in simulations.

```{automodule} axionbloch.Apparatus
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.SimuTypes

Shared type definitions for the simulation layer.

- {class}`~axionbloch.SimuTypes.SimuParams` — `TypedDict` describing the full
  parameter set for one simulation run.  Passed as ``all_params=[params]`` to
  {class}`~axionbloch.SimuTools.Simulations`.
- {class}`~axionbloch.SimuTypes.SimuEntry` — `dataclass` pairing a completed
  {class}`~axionbloch.SimuTools.Simulation` with its
  {class}`~axionbloch.SimuTypes.SimuParams`.  Stored in
  {attr}`~axionbloch.SimuTools.Simulations.pool` after a run.

```{automodule} axionbloch.SimuTypes
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.SimuTools

The main simulation engine.  Three classes work together:

**{class}`~axionbloch.SimuTools.MagField`** builds the time-domain effective
magnetic field array.  Call one of its ``set*`` methods to fill
{attr}`~axionbloch.SimuTools.MagField.B_vec`:

| Method | Field type |
|--------|-----------|
| `setXYPulse` | Continuous circularly-polarised excitation |
| `set90DegPulse` | Calibrated 90° hard pulse |
| `setCPMGPulseTrain` | CPMG spin-echo pulse train |
| `setNPulsesArbDelay` | N pulses with arbitrary delays |
| `setAxionFields` | Stochastic axion pseudomagnetic fields |

**{class}`~axionbloch.SimuTools.Simulation`** wraps the C++ ``blochsimulation``
RK4 kernel for a single run: generates field realizations, loops over spin
packets, integrates the Bloch equations, and stores results in
{attr}`~axionbloch.SimuTools.Simulation.trjries`.

**{class}`~axionbloch.SimuTools.Simulations`** manages a collection of
`Simulation` runs.  Call {meth}`~axionbloch.SimuTools.Simulations.run` to
execute all runs and {meth}`~axionbloch.SimuTools.Simulations.saveToPkl` to
persist results.

```{automodule} axionbloch.SimuTools
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.constants

Physical constants and unit conversion factors used throughout the package.
All values are `astropy.units.Quantity` objects so unit arithmetic is fully
supported.

| Symbol | Description |
|--------|-------------|
| `gamma_p` | Proton gyromagnetic ratio (rad Hz T⁻¹) |
| `mu_p` | Proton magnetic dipole moment |
| `gamma_Xe129` | ¹²⁹Xe gyromagnetic ratio (rad Hz T⁻¹) |
| `mu_Xe129` | ¹²⁹Xe magnetic dipole moment |

```{automodule} axionbloch.constants
:members:
:undoc-members:
:show-inheritance:
```

---

## axionbloch.utils

General-purpose helpers shared across the package.

| Item | Description |
|------|-------------|
| `check(arg)` | Debug-print a variable with its name, value, and unit |
| `PhysicalObject` | Base class with `useCommonUnits()` for consistent unit display |
| `Lorentzian`, `Gaussian`, `ExpCos` | Common NMR lineshape functions |
| `estimateLorzfit`, `estimateGaussfit` | Initial-parameter estimators for curve fits |
| `getDateAndTime()` | Compact timestamp string ``YYYYMMDD_HHMMSS`` |
| `sci_fmt(x, pos)` | Scientific-notation tick formatter for matplotlib |
| `high_contrast_extended` | Colour palette for multi-line plots |

```{automodule} axionbloch.utils
:members:
:undoc-members:
:show-inheritance:
```
