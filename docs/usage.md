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
| `FineGrainedAxionStream` | Single fine-grained axion stream |
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
from axionbloch.constants import gamma_Xe129, mu_Xe129
from axionbloch.SimuTools import MagField, Simulations
from axionbloch.SimuTypes import SimuParams
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo

# --- 1. NMR sample (Xe-129, 50 % hyperpolarised) ---
sample = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129,
    massDensity=3.1 * unit.g * unit.cm**(-3),
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=10 * unit.minute,
    T1=15 * unit.minute,
    vol=1 * unit.cm**3,
    mu=mu_Xe129,
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

### Axion lineshape

The analytical SHM PSD can be evaluated without running a full simulation:

```python
import numpy as np
from astropy import units as unit
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo

axion = MilkyWayAxionHalo(nu_a=1 * unit.kHz)

nu = np.linspace(0.99, 1.01, 10000) * unit.kHz
psd = MilkyWayAxionHalo.axion_lineshape(
    v_0=axion.v_0,
    v_lab=axion.v_lab,
    nu_a=axion.nu_a,
    nu=nu,
    case="grad_perp",   # 'non-grad', 'grad_par', or 'grad_perp'
)
```

The returned array has units Hz{math}`^{-2}` and integrates to 1 over
frequency.

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

Pre-built stations: `Mainz`, `Baltimore`, `Sanya`, `Geneva`, `Tokyo`,
`Mumbai`, `Sydney`, `CapeTown`, `BuenosAires`.

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
from axionbloch.Station import Baltimore

station = Baltimore

r, R_r, r_line, grad_r, grad_theta, grad_phi = halo.findGradients(
    stateNames=["2p"],
    station=station,
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

`findGradients` returns six arrays: the radial grid `r`, the radial
wavefunction `R_r`, the interpolation line `r_line`, and the three
spherical-coordinate gradient components
{math}`(\partial_r\Psi,\; r^{-1}\partial_\theta\Psi,\; (r\sin\theta)^{-1}\partial_\phi\Psi)`
evaluated along the radial line pointing toward the station.

## Sample definition

```python
from astropy import units as unit
from axionbloch.constants import gamma_Xe129, mu_Xe129
from axionbloch.Sample import Sample

sample = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129,
    massDensity=3.1 * unit.g / unit.cm**3,
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=10 * unit.minute,
    T1=15 * unit.minute,
    vol=1 * unit.cm**3,
    mu=mu_Xe129,
    temp=163 * unit.K,
)

pol  = sample.getThermalPol(B_pol=0.1 * unit.T)
M0   = sample.getM0eqb(B_pol=0.1 * unit.T)
```
