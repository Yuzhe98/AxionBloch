# Test suite

The test suite verifies both the numerical correctness of the Bloch-equation
solver and the integrity of the supporting physics models.  All tests use
**pytest** and are located in the `tests/` directory.

## Running the tests

Run the full suite:

```
pytest tests/
```

On Windows, run the full simulator-backed suite with a Python version matching
the compiled extension module.  The pure-Python MW and bound-halo tests can be
run independently in a lighter environment, but full Bloch simulation tests may
require the project CPython 3.11 environment.

Run only the calibration benchmarks:

```
pytest tests/test_calibrations_free_decay.py \
       tests/test_calibrations_cw_nmr.py \
       tests/test_calibrations_t1_relaxation.py \
       tests/test_calibrations_t1_hyperpolarized.py
```

Show diagnostic matplotlib figures for any calibration test by passing
`--show-plots`:

```
pytest tests/test_calibrations_free_decay.py --show-plots
```

If no display is available (headless server / CI), figures are saved to
temporary PNG files and opened with the OS file viewer.

---

## Shared fixtures (`conftest.py`)

`tests/conftest.py` provides fixtures and hooks used by every test module.

| Item | Description |
|------|-------------|
| `pytest_configure` | Switches matplotlib to the non-interactive `Agg` backend when `--show-plots` is not set, preventing Tk initialization in headless runs. |
| `pytest_addoption` | Registers the `--show-plots` flag. |
| `show_plots` fixture | Returns `True` when `--show-plots` is passed on the command line; injected into any test that declares it as a parameter. |

---

## Calibration tests

Calibration tests compare simulation output against closed-form analytical
solutions.  They parametrize over physically meaningful ranges of field
strength, relaxation time, and field inhomogeneity to confirm that the RK4
integrator is accurate well below a {math}`\chi^2` threshold.

All calibration tests use a single on-resonance proton spin packet
(`nFWHM = 0`) to avoid the spin-packet explosion that occurs for large
{math}`T_1 \times \text{RCF\_freq}` products, unless field inhomogeneity is
the quantity under test.  The RK4 step count is held constant across cases
by scaling `rate` and `duration` with the relevant relaxation time.

### Free-decay envelope (`test_calibrations_free_decay.py`)

**36 cases** · tolerance {math}`\chi^2 \le 10^{-4}`

After a 90° pulse the transverse envelope {math}`|M_{xy}(t)|` is compared to
the free-decay kernel

```{math}
|\mathrm{FD}_\mathrm{sub}(t)| = \left|\sum_i w_i\,e^{2\pi i\,\delta_i t}\right|
```

where {math}`\delta_i = \gamma/(2\pi)\,B_{\mathrm{spread},i}` is the detuning
of spin packet {math}`i` from the centre Larmor frequency and {math}`w_i` are
the Hamming-squared weights from the magnet model.

Parameters swept:

| Parameter | Values |
|-----------|--------|
| FWHM | 0.1, 1, 10, 20 ppm |
| RCF frequency | 1 kHz, 1 MHz, 1 GHz |
| Larmor detuning | 0, 1, 10 ppm of RCF |

Run a single case with a plot:

```
pytest "tests/test_calibrations_free_decay.py::test_free_decay_envelope[1ppm-1-1e+06Hz]" --show-plots
```

### CW NMR signal buildup (`test_calibrations_cw_nmr.py`)

**36 cases** · tolerance {math}`\chi^2 \le 10^{-4}`

A weak on-resonance continuous-wave drive is applied and
{math}`|M_{xy}(t)|` is compared to the free-decay integral formula

```{math}
|M_{xy}(t)| = \gamma B_{1,\mathrm{eff}}\left|\int_0^t \mathrm{FD}_\mathrm{sub}(t')\,dt'\right|.
```

The drive amplitude is chosen so that {math}`\gamma B_{1,\mathrm{eff}}\,t_\mathrm{end} = 5 \times 0.001\,\mathrm{rad}`, satisfying the weak-drive (linear-response) condition.

Parameters swept:

| Parameter | Values |
|-----------|--------|
| FWHM | 0.1, 1, 10, 20 ppm |
| RCF frequency | 1 kHz, 1 MHz, 1 GHz |
| Larmor detuning | 0, 1, 10 ppm of RCF |

Run a single case with a plot:

```
pytest "tests/test_calibrations_cw_nmr.py::test_cw_signal_buildup[1ppm-1-1e+06Hz]" --show-plots
```

### T1 longitudinal recovery (`test_calibrations_t1_relaxation.py`)

**9 cases** · tolerance {math}`\chi^2 \le 10^{-3}`

A 90° pulse tips {math}`M_z` to zero.  Free recovery is then compared to

```{math}
M_z(t) = 1 + (M_{z0} - 1)\,e^{-t/T_1}
```

where {math}`M_{z0}` is the actual simulated post-pulse value (absorbing any
finite-pulse T1 relaxation), and {math}`t` is measured from the end of the
pulse.

Parameters swept:

| Parameter | Values |
|-----------|--------|
| T1 | 1 ms, 1 s, 1 ks |
| RCF frequency | 1 kHz, 1 MHz, 1 GHz |

Run a single case with a plot:

```
pytest "tests/test_calibrations_t1_relaxation.py::test_t1_recovery[1.0s-1e+06Hz]" --show-plots
```

### Hyperpolarized T1 decay (`test_calibrations_t1_hyperpolarized.py`)

**36 cases** · tolerance {math}`\chi^2 \le 10^{-3}`

The sample starts in a longitudinally overpolarized state
{math}`M_z(0) = k > 1` (no transverse magnetization, no excitation pulse).
{math}`M_z` decays toward equilibrium as

```{math}
M_z(t) = 1 + (k - 1)\,e^{-t/T_1}.
```

The overpolarization factor {math}`k` is set via `sample.pol = k × pol_thermal(B_0)`.
This exercises the T1 term in the recovery direction {math}`M_z > 1 \to 1`,
complementing the thermal T1 test ({math}`M_z < 1 \to 1`).

Parameters swept:

| Parameter | Values |
|-----------|--------|
| Overpolarization factor k | 2, 10, 100, 10 000 |
| T1 | 1 ms, 1 s, 1 ks |
| RCF frequency | 1 kHz, 1 MHz, 1 GHz |

Run a single case with a plot:

```
pytest "tests/test_calibrations_t1_hyperpolarized.py::test_t1_hyperpolarized[10.0x-1s-1e+06Hz]" --show-plots
```

---

## Unit tests

### `test_Sample.py`

Tests {class}`~axionbloch.Sample.Sample` construction and key methods for a
methanol (¹H) sample, a liquid ¹²⁹Xe sample, and a negative-temperature case
that must raise `ValueError`.

| Test | What it checks |
|------|---------------|
| `test_sample_initialization_runs` | Spin number density and total spin count are finite with correct units. |
| `test_getThermalPol_runs` | Thermal polarization is finite and dimensionless. |
| `test_getThermalPol_raises_below_absolute_zero` | Negative temperature raises `ValueError`. |
| `test_getM0_runs` | Magnetization {math}`M_0` is finite with units A m⁻¹. |
| `test_getM0eqb_runs` | Equilibrium magnetization end-to-end pipeline. |

### `test_Magnet.py`

Tests {class}`~axionbloch.Apparatus.Magnet` construction, weight normalization,
and spin-packet sampling.  Pass `PRINT_RESULTS=1` to print diagnostic
histograms:

```
$env:PRINT_RESULTS="1"; pytest tests/test_Magnet.py -s
```

### `test_constants.py`

Smoke tests for `axionbloch.constants`.  Running the file directly (`python
tests/test_constants.py`) prints every constant with its type, shape, and
value via {func}`~axionbloch.utils.check`; no assertions are made.

### `test_EarthBoundAxionHalo.py`

Tests the PREM data loader and the Earth gravitational-potential helpers used
by the bound-axion Schrödinger solver.

---

## Integration tests

### `test_MilkyWayAxionHalo.py`

Tests {class}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo` initialization,
static SHM PSD lineshapes, gradient power coefficients, station/lab wind
projections, periodic lineshape modulation, Rabi-frequency computation, and
the end-to-end `Simulations` pipeline.  Accepts `PRINT_RESULTS`, `SEED`, and
`NUM_FIELD` environment variables.

### `test_MilkyWayAxionHalo_lineshape.py`

Lightweight tests for the station/time-aware MW PSD helpers.  These checks use
the Boston station and verify that
{meth}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.findLineshape`
returns a sampled PSD and power spectrum with consistent frequency grids,
station-derived wind angle {math}`\alpha`, and lab speed.  They also verify
{meth}`~axionbloch.MilkyWayAxionHalo.MilkyWayAxionHalo.findLineshapeFWHM`
for both `"PSD"` and `"power_spectrum"` inputs, including the updates to
`FWHM_frequency`, `FWHM_a`, and
{math}`\tau_a = 1 / (\pi\,\mathrm{FWHM}_a\,\nu_a)`.

### `test_MilkyWay.py` · `test_findGradients_EarthLocation.py`

Integration tests for the Milky Way axion-halo model and the
`findGradients` interface: the `Station.location` attribute
(hemisphere sign conventions, elevation, colatitude conversion) and the
`ValueError` raised when `findGradients` is called without a station.
The gradient tests also cover the required `stateCoefficients` dictionary,
default `meas_time` resolution, the first-order boost toggle and explicit
zero-velocity case, complex eigenstate superpositions, normalized coefficient
handling, Omega_a conversion from complex gradient amplitudes, and the
state-amplitude/eigen-energy plotting helper.

---

## Performance benchmarks

### `test_numpy_unit_performance.py`

Micro-benchmarks comparing `numpy` array operations with and without
`astropy.units` wrappers, used to guide performance-sensitive code paths.
