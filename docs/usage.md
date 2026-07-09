# Usage

## Software architecture

axionbloch separates three concerns: experimental configuration, axion field
modelling, and numerical integration.

| Class | Role |
|-------|------|
| `Sample` | NMR sample — gyromagnetic ratio, relaxation times, volume, polarization |
| `Magnet` | Static bias field — strength, direction, field inhomogeneity (FWHM) |
| `MagField` | Effective magnetic field — combines bias, excitation, and axion pseudomagnetic fields |
| `MilkyWayAxionHalo` | Milky Way SHM axion-wind model |
| `GravBoundAxionHalo` / `EarthBoundAxionHalo` | Gravitationally bound halo (TISE solver) |
| `Simulation` | Single simulation run |
| `Simulations` | Collection of simulation runs; parallelized execution and I/O |

Numerical integration (RK4) is performed in a C++ backend (`blochSimulation`)
exposed to Python via **pybind11**, which makes large ensembles of stochastic
field realizations computationally practical.

## Milky Way axion NMR simulation

The standard workflow for a CW axion-wind NMR experiment has four steps:
define the sample, define the axion field, define the magnet, and run the
simulation.

```python
from axionbloch.dependency import *        # numpy, astropy units/constants
from axionbloch.constants import gamma_Xe129N, mu_Xe129N
from axionbloch.SimuTools import MagField, Simulations
from axionbloch.SimuTypes import SimuParams
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo

# --- 1. NMR sample (Xe-129, 50 % hyperpolarised) ---
sample = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129N,
    massDensity=3.1 * unit.g * unit.cm**(-3),
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=10 * unit.minute,
    T1=15 * unit.minute,
    vol=1 * unit.cm**3,
    mu=mu_Xe129N,
    temp=163 * unit.K,
    pol=0.5 * unit.one,            # initial hyperpolarization
)

# --- 2. Axion field (Milky Way SHM) ---
axion = MilkyWayAxionHalo(
    nu_a=1 * unit.kHz,             # Compton frequency
    g_aNN=1e-9 * unit.GeV**(-1),   # coupling constant
)

# --- 3. Polarising magnet, on resonance, 2 ppm inhomogeneity ---
magnet = Magnet(
    B0=axion.nu_a_eff / (sample.gamma / (2 * PI)),
    direction=[0, 0, 1],
    FWHM=2 * ppm,
)

# --- 4. Bundle parameters and run ---
B_a_rms = (axion.getRabiFreq() / (sample.gamma / (2 * PI))).to(unit.T)

params: SimuParams = {
    "key_info": {"nu_a": axion.nu_a},
    "axion": axion,
    "sample": sample,
    "magnet": magnet,
    "excField": MagField(),
    "B_a_rms": B_a_rms,
    "numFields": 1000,          # stochastic field realizations
    "rand_seed": 10,
    "init_M": 1 * unit.one,
    "init_M_theta": 0 * unit.rad,
    "init_M_phi": 0 * unit.rad,
    "rate": 1 * unit.Hz,
    "duration": 4000 * unit.s,
}

simulations = Simulations(all_params=[params])
simulations.run(verbose=True)

# Post-process: compute mean ± σ and plot trajectories
for item in simulations.pool:
    item.simu.keepMeanStd()
    item.simu.displayTrjries()

# Save to disk for later analysis
simulations.saveToPkl(dir="results/")
```

### Performance

The total number of RK4 integration steps is

```{math}
N_\mathrm{steps}
= \texttt{rate} \times \texttt{duration}
  \times \texttt{numFields} \times \texttt{Magnet.numPt}.
```

For the example above ({math}`N_\mathrm{steps} \approx 4 \times 10^9`) the
runtime is approximately 10 s on a modern laptop CPU, thanks to the C++
backend.

### MW axion PSD and station-aware lineshapes

The analytical SHM PSD can be evaluated without running a full simulation:

```python
import numpy as np
from astropy import units as unit
from astropy.time import Time
from axionbloch.dependency import ppm
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Boston

axion = MilkyWayAxionHalo(nu_a=1 * unit.kHz)

nu = np.linspace(0.99, 1.01, 10000) * unit.kHz
PSD = MilkyWayAxionHalo.axion_lineshape(
    v_0=axion.v_0,
    v_lab=axion.v_lab,
    nu_a=axion.nu_a,
    nu=nu,
    case="grad_perp",   # 'non-grad', 'grad_par', or 'grad_perp'
)
```

The returned array has units Hz{math}`^{-2}` and integrates to 1 over
frequency.  The static method is the most direct way to compare the three
analytical cases:

| Case | Meaning |
|------|---------|
| `non-grad` | Non-gradient coupling; independent of `alpha` |
| `grad_par` | Gradient coupling parallel to the projection axis |
| `grad_perp` | Gradient coupling perpendicular to the projection axis |

For station-specific calculations, use the helper that first computes the
laboratory kinematics and then calls the same static PSD function:

```python
axion = MilkyWayAxionHalo(nu_a=1 * unit.MHz)

result = axion.findLineshape(
    station=Boston,
    meas_time=Time("2022-01-01T00:00:00", scale="utc"),
    case="grad_perp",
    projection_axis="zenith",
    num_frequency_points=20001,
)

frequencies = result["frequencies"]
PSD = result["PSD"]
alpha = result["alpha"]
v_lab = result["v_lab_magnitude"][0]
FWHM = result["FWHM"]
FWHM_a = result["FWHM_a"]
```

`projection_axis` selects the axis used to decompose the axion-gradient /
axion-wind vector.  `grad_par` uses the component parallel to this axis, while
`grad_perp` uses the magnitude of the component perpendicular to it.  The
default `"zenith"` is the local vertical direction normal to the station's
local ground tangent plane.  It can also be `"north"`, `"west"`, `"east"`, or
another axis accepted by
{meth}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.projectHaloVelocity`.
For fixed alpha comparisons independent of station orientation, pass alpha
directly to {meth}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.axion_lineshape`.

The sampled FWHM of a PSD or power spectrum can be measured in frequency
units and as a fractional ppm width:

```python
FWHM_result = axion.findLineshapeFWHM(
    station=Boston,
    meas_time=Time("2022-01-01T00:00:00", scale="utc"),
    case="grad_par",
    projection_axis="zenith",
    spectrum="PSD",
    num_frequency_points=20001,
)

print(FWHM_result["FWHM_frequency"])
print(FWHM_result["FWHM_freq"])
print(FWHM_result["FWHM_a"].to(ppm))
print(FWHM_result["tau_a"])
```

`findLineshape()` also measures these FWHM quantities and includes them in its
result dictionary.  These helpers update `axion.FWHM_frequency`,
`axion.FWHM_freq`, `axion.FWHM`, `axion.FWHM_a`, and `axion.tau_a` when
`update=True`.  Here `FWHM` is the dimensionless fractional linewidth and
`FWHM_a` is the same quantity expressed in the package's `ppm` unit, so
{math}`\tau_a = 1 / (\pi\,\mathrm{FWHM}_a\,\nu_a)`.

Two example scripts use these helpers:

- `examples/MilkyWayAxionHalo/plot-axion-lineshape-at-station.py` plots static
  `non-grad`, `grad_par`, and `grad_perp` PSDs for `alpha = 0` and
  `alpha = pi/2`, or station-derived alpha with `--mode station`.
- `examples/MilkyWayAxionHalo/periodic-modulations.py` plots annual
  {math}`v_\mathrm{lab}`, daily {math}`\cos\alpha`, and the relative
  `grad_par`/`grad_perp` power modulation for Boston.

## NMR calibration

The package includes calibration benchmarks for verifying the numerical
integration against known analytical or experimental results.

### Pulsed NMR — free decay and spin echo

Apply a 90° pulse followed by a 180° pulse.  The simulated signal should:

- oscillate at the Larmor frequency;
- tip into the x–y plane after the 90° pulse;
- free decay with time constant {math}`T_2^*` (inhomogeneous broadening);
- refocus (echo) after the 180° pulse;
- show decaying echo amplitudes with time constant {math}`T_2`.

This calibrates magnetic pulses, Larmor precession, and transverse
relaxation.

### Continuous-wave NMR

With a weak on-resonance CW field ({math}`\gamma B T_2^* \ll 1`), the
transverse magnetization grows as

```{math}
M_\perp(t) = \gamma B\,T_2^*\!\left(1 - e^{-t/T_2^*}\right).
```

This independently calibrates Larmor precession and {math}`T_2^*`.

### Hyperpolarised sample

Set `pol > 0` in `Sample` to start from a hyperpolarised state.  The
longitudinal magnetization then decays exponentially toward equilibrium
with time constant {math}`T_1`, calibrating longitudinal relaxation.

## Geographic station

A {class}`~axionbloch.Station.Station` records the location of an
experimental site and derives spherical coordinates {math}`(\theta, \phi)`
and the outward unit normal {math}`\hat{n}`.

```python
from astropy import units as unit
from axionbloch.Station import Station

my_lab = Station(
    name="My Lab",
    NSsemisphere="N",
    EWsemisphere="E",
    latitude=52.52 * unit.deg,
    longitude=13.40 * unit.deg,
    elevation=50.0 * unit.m,
)

print(my_lab.theta.to(unit.deg))   # polar angle from north pole
print(my_lab.phi.to(unit.deg))     # azimuthal angle from prime meridian
print(my_lab.nvec)                  # Cartesian unit normal
```

Pre-built stations: `Mainz`, `Geneva`, `Baltimore`, `Boston`, `Tokyo`,
`Mumbai`, `Sanya`, `Sydney`, `CapeTown`, `BuenosAires`.

## Earth-bound axion halo

`EarthBoundAxionHalo` solves the TISE for axions gravitationally bound to
the Earth.

### Solving for bound-state wavefunctions

```python
from astropy import units as unit
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Baltimore

halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,
    N=int(2**12),
    extent=128.0 * unit.R_earth,
    verbose=True,
)

# Solve for l = 1 (p-wave), keep 64 radial eigenstates
halo.solve_TISE_3D(l_vals=[1], max_n_r=64)

halo.listEigenStates()
```

### Visualising eigenstates

```python
halo.plotEigenstate(n_r=0, l=1)       # single 2p wavefunction
halo.stackEigenStates(numStates=8)     # stacked waterfall plot
halo.plotEigenEnergiesInPot()          # energy levels in potential well
```

### Axion field gradient at a station

Continues from the block above — `halo` must already be constructed and
solved before calling `findGradients`.

```python
import numpy as np
from astropy import units as unit
from astropy.time import Time
from axionbloch.Station import Baltimore

station = Baltimore
meas_time = Time("2024-06-21T14:00:00")

r, R_r, r_line, grad_r, grad_theta, grad_phi = halo.findGradients(
    stateCoefficients={"2p": 1.0},
    station=station,
    meas_time=meas_time,
    truncRadius=2 * unit.R_earth,
    showPlot=True,
    verbose=True,
)

# Index of the radial-line point closest to Earth's surface (r = 1 R_earth)
earthRad_idx = np.argmin(np.abs(r_line - 1 * unit.R_earth))

# Scale by a_0 to obtain the physical axion field gradient at the station
print("a_0 * grad_r     =", (halo.a_0 * grad_r[earthRad_idx]).si)
print("a_0 * grad_theta =", (halo.a_0 * grad_theta[earthRad_idx]).si)
print("a_0 * grad_phi   =", (halo.a_0 * grad_phi[earthRad_idx]).si)
```

`findGradients` returns six arrays: the radial grid `r`, the total
wavefunction along the station direction (`R_r` in the return tuple for
backward compatibility), the interpolation line `r_line`, and the three
spherical-coordinate gradient components
{math}`(\partial_r\Psi,\; r^{-1}\partial_\theta\Psi,\; (r\sin\theta)^{-1}\partial_\phi\Psi)`
evaluated along the radial line pointing toward the station.

The `stateCoefficients` argument is required and must be a dictionary such as
`{"2p": 1.0}` or `{"2p": 1.0, "3p": 1.0 + 1.0j}`.  Coefficients are normalized
internally so that {math}`\sum_i |c_i|^2=1`.  This makes interference between
bound eigenstates explicit and avoids accidental ambiguity between state names
and coefficients.

By default the gradients include the first-order laboratory-frame correction
from the station's rotation through a nonrotating Earth halo. Use
`include_lorentz_boost=False` for the intrinsic spatial gradient alone, or pass
a three-component `relative_velocity` in the solar-Z Cartesian frame to model a
different halo velocity. An explicit zero velocity represents a corotating
halo. If `meas_time` is omitted, `Time.now()` is used.

To inspect the states in a superposition, use:

```python
halo.plotStateAmplitudeVsEigenEnergy(
    stateCoefficients={"1s": 1, "2p": 1 + 1j, "3p": 0.5},
)
```

The lower x-axis is the effective mass shift
{math}`m_{a,\mathrm{eff}} - m_a(v=0)` and the top x-axis is the corresponding
frequency shift {math}`\nu-\nu_a`.

### Axion-nucleon coupling frequency (Omega_a) over time

Computes the axion-nucleon coupling frequency from gradients at multiple
measurement times:

```python
import numpy as np
from astropy import units as unit
from astropy.time import Time
from axionbloch.Station import Baltimore

# Generate time array (e.g., sidereal day)
times = Time.now() + np.linspace(0, 1, 24) * unit.hour

# Compute Omega_a over time
Omega_results = halo.findOmega_aOverTime(
    stateCoefficients={"2p": 1.0},
    station=Baltimore,
    meas_times=times,
    g_aNN=1e-9 * unit.GeV**(-1),
    truncRadius=2 * unit.R_earth,
    verbose=True,
)

# Extract results
Omega_a_r = Omega_results["Omega_a_r"]
Omega_a_theta = Omega_results["Omega_a_theta"]
Omega_a_phi = Omega_results["Omega_a_phi"]
print(f"Radial component max: {np.max(np.abs(Omega_a_r))}")
```

### RMS Omega_a over time

Computes the root-mean-square Omega_a for each gradient component:

```python
rms_results = halo.findrmsOmega_aOverTime(
    stateCoefficients={"2p": 1.0},
    station=Baltimore,
    meas_times=times,
    g_aNN=1e-9 * unit.GeV**(-1),
    truncRadius=2 * unit.R_earth,
    verbose=True,
)

# Extract RMS values
rms_r = rms_results["rms_Omega_a_r"]
rms_theta = rms_results["rms_Omega_a_theta"]
rms_phi = rms_results["rms_Omega_a_phi"]
print(f"RMS Omega_a_r: {rms_r}")
print(f"RMS Omega_a_theta: {rms_theta}")
print(f"RMS Omega_a_phi: {rms_phi}")
```

## Sample definition

```python
from astropy import units as unit
from axionbloch.constants import gamma_Xe129N, mu_Xe129N
from axionbloch.Sample import Sample

sample = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129N,
    massDensity=3.1 * unit.g / unit.cm**3,
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=10 * unit.minute,
    T1=15 * unit.minute,
    vol=1 * unit.cm**3,
    mu=mu_Xe129N,
    temp=163 * unit.K,
)

pol  = sample.getThermalPol(B_pol=0.1 * unit.T)
M0   = sample.getM0eqb(B_pol=0.1 * unit.T)
```
