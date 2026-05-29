# Installation

## Requirements

axionbloch requires Python 3.10 or later.  The following packages are
installed automatically as dependencies:

| Package | Purpose |
|---------|---------|
| [NumPy](https://numpy.org/) | Numerical arrays |
| [SciPy](https://scipy.org/) | Sparse linear algebra, interpolation, signal processing |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [Astropy](https://www.astropy.org/) | Physical units and constants |

## Install from PyPI

```bash
pip install axionbloch
```

## Install from source

Clone the repository and install in editable mode so that local changes are
picked up immediately:

```bash
git clone https://github.com/Yuzhe98/AxionBloch.git
cd AxionBloch
pip install -e .
```

## Verify the installation

```python
import axionbloch
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from astropy import units as unit

axion = MilkyWayAxionHalo(nu_a=1 * unit.kHz)
print("nu_a =", axion.nu_a)
print("tau_a (estimated) =", axion.tau_a_est.to(unit.s))
```
