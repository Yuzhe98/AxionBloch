# AxionBloch
A package for simulating **axion**-induced spin dynamics based on **Bloch equations**, implemented in **Python**  and **C++**. 

Yuzhe Zhang - GSI HIM & Uni Mainz - yuhzhang@uni-mainz.de

📖 Documentation: [axionbloch.readthedocs.io](https://axionbloch.readthedocs.io/en/latest/)

## Features

In AxionBloch, the axion (including axionlike-particle or ALP) field acts like a **pseudomagnetic field** coupling to (nuclear) spins.

The package provides a **numerical platform** to study these subtle effects using Bloch-equation-based simulations.

## Requirements

- Python >= 3.10

### Dependencies

- **numpy** — Numerical computing
- **scipy** — Scientific computing and numerical integration
- **matplotlib** — Plotting and visualization
- **astropy** — Astronomical utilities
- **h5py** — HDF5 file I/O

## Installation

Install from [PyPI](https://pypi.org/project/axionbloch/):
```bash
pip install axionbloch
```

Or build and install from source (requires a C++17 compiler for the simulation extension):
```bash
git clone https://github.com/Yuzhe98/AxionBloch.git
cd AxionBloch
pip install .
```

## Quick Start
Examples can be found in the examples/.

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)